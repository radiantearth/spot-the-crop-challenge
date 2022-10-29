import os,sys,pickle,multiprocessing,datetime
import numpy as np, pandas as pd
import tarfile,json,rasterio
from pathlib import Path
from radiant_mlhub.client import _download as download_file
from collections import OrderedDict
from tqdm.auto import tqdm
os.environ['MLHUB_API_KEY'] = sys.argv[1] #
OUTPUT_DIR = sys.argv[2] #data/s2/train

FOLDER_BASE = 'ref_south_africa_crops_competition_v1'
DOWNLOAD_S1 = False # If you set this to true then the Sentinel-1 data will be downloaded
# Select which Sentinel-2 imagery bands you'd like to download here. 
DOWNLOAD_S2 = OrderedDict({
    'B01': True,
    'B02': True,#Blue
    'B03': True,#Green
    'B04': True,#Red
    'B05': True,
    'B06': True,
    'B07': True,
    'B08': True, #NIR
    'B8A': True, #NIR2
    'B09': True,
    'B11': True, #SWIR1
    'B12': True, #SWIR2
    'CLM': True
})
OUTPUT_DIR = f'{OUTPUT_DIR}/train'
os.makedirs(OUTPUT_DIR,exist_ok=True)
OUTPUT_DIR_BANDS = f'{OUTPUT_DIR}/bands-raw' 
os.makedirs(OUTPUT_DIR_BANDS,exist_ok=True)

def download_archive(archive_name):
    if os.path.exists(archive_name.replace('.tar.gz', '')):
        return
    
    print(f'Downloading {archive_name} ...')
    download_url = f'https://radiant-mlhub.s3.us-west-2.amazonaws.com/archives/{archive_name}'
    download_file(download_url, '.')
    print(f'Extracting {archive_name} ...')
    with tarfile.open(archive_name) as tfile:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tfile)
    os.remove(archive_name)

for split in ['train']:
    # # Download the labels
    labels_archive = f'{FOLDER_BASE}_{split}_labels.tar.gz'
    download_archive(labels_archive)
    
    ##Download Sentinel-1 data
    if DOWNLOAD_S1:
        s1_archive = f'{FOLDER_BASE}_{split}_source_s1.tar.gz'
        download_archive(s1_archive)

    for band, download in DOWNLOAD_S2.items():
        if not download:
            continue
        s2_archive = f'{FOLDER_BASE}_{split}_source_s2_{band}.tar.gz'
        download_archive(s2_archive)

def resolve_path(base, path):
    return Path(os.path.join(base, path)).resolve()
        
def load_df(collection_id):
    split = collection_id.split('_')[-2]
    collection = json.load(open(f'{collection_id}/collection.json', 'r'))
    rows = []
    item_links = []
    for link in collection['links']:
        if link['rel'] != 'item':
            continue
        item_links.append(link['href'])
        
    for item_link in item_links:
        item_path = f'{collection_id}/{item_link}'
        current_path = os.path.dirname(item_path)
        item = json.load(open(item_path, 'r'))
        tile_id = item['id'].split('_')[-1]
        for asset_key, asset in item['assets'].items():
            rows.append([
                tile_id,
                None,
                None,
                asset_key,
                str(resolve_path(current_path, asset['href']))
            ])
            
        for link in item['links']:
            if link['rel'] != 'source':
                continue
            source_item_id = link['href'].split('/')[-2]
            
            if source_item_id.find('_s1_') > 0 and not DOWNLOAD_S1:
                continue
            elif source_item_id.find('_s1_') > 0:
                for band in ['VV', 'VH']:
                    asset_path = Path(f'{FOLDER_BASE}_{split}_source_s1/{source_item_id}/{band}.tif').resolve()
                    date = '-'.join(source_item_id.split('_')[10:13])
                    
                    rows.append([
                        tile_id,
                        f'{date}T00:00:00Z',
                        's1',
                        band,
                        asset_path
                    ])
                
            if source_item_id.find('_s2_') > 0:
                for band, download in DOWNLOAD_S2.items():
                    if not download:
                        continue

                    asset_path = Path(f'{FOLDER_BASE}_{split}_source_s2_{band}/{source_item_id}_{band}.tif').resolve()
                    date = '-'.join(source_item_id.split('_')[10:13])
                    rows.append([
                        tile_id,
                        f'{date}T00:00:00Z',
                        's2',
                        band,
                        asset_path
                    ])

    return pd.DataFrame(rows, columns=['tile_id', 'datetime', 'satellite_platform', 'asset', 'file_path'])

df_train = load_df(f'{FOLDER_BASE}_train_labels')
df_train['date'] = df_train.datetime.astype(np.datetime64)
bands = [k for k,v in DOWNLOAD_S2.items() if v==True]

def extract_s2_train(tile_ids):
  fields = []
  labels = []
  dates = []
  tiles = []
  
  for tile_id in tqdm(tile_ids):
      df_tile = df_train[df_train['tile_id']==tile_id]
      tile_dates = sorted(df_tile[df_tile['satellite_platform']=='s2']['date'].unique())
      
      ARR = {}
      for band in bands:
        band_arr = []
        for date in tile_dates:
          src = rasterio.open(df_tile[(df_tile['date']==date) & (df_tile['asset']==band)]['file_path'].values[0])
          band_arr.append(src.read(1))
        
        ARR[band] = np.array(band_arr,dtype='float32')
        
      multi_band_arr = np.stack(list(ARR.values())).astype(np.float32)
      multi_band_arr = multi_band_arr.transpose(2,3,0,1) #w,h,bands,dates
      label_src = rasterio.open(df_tile[df_tile['asset']=='labels']['file_path'].values[0])
      label_array = label_src.read(1)
      field_src = rasterio.open(df_tile[df_tile['asset']=='field_ids']['file_path'].values[0])
      fields_arr = field_src.read(1) #fields in tile
      for field_id in np.unique(fields_arr):
        if field_id==0:
          continue
        mask = fields_arr==field_id
        field_label = np.unique(label_array[mask])
        field_label = [l for l in field_label if l!=0]
        if len(field_label)==1: 
          #ignore fields with multiple labels
          field_label = field_label[0]
          patch = multi_band_arr[mask]
          np.savez_compressed(f"{OUTPUT_DIR_BANDS}/{field_id}", patch)
          
          labels.append(field_label)
          fields.append(field_id)
          tiles.append(tile_id)
          dates.append(tile_dates)
  df = pd.DataFrame(dict(field_id=fields,tile_id=tiles,label=labels,dates=dates))
  return df

tile_ids = sorted(df_train.tile_id.unique())
print(f'extracting data from {len(tile_ids)} tiles for bands {bands}')

num_processes = multiprocessing.cpu_count()
print(f'processesing on : {num_processes} cpus')
pool = multiprocessing.Pool(num_processes)
tiles_per_process = len(tile_ids) / num_processes
tasks = []
for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * tiles_per_process + 1
    end_index = num_process * tiles_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    sublist = tile_ids[start_index - 1:end_index]
    tasks.append((sublist,))
    print(f"Task # {num_process} process tiles {len(sublist)}")

results = []
for t in tasks:
    results.append(pool.apply_async(extract_s2_train, t))

all_results = []
for result in results:
    df = result.get()
    all_results.append(df)

df_train_meta = pd.concat(all_results)
df_train_meta['field_id'] = df_train_meta.field_id.astype(np.int32)
df_train_meta['tile_id'] = df_train_meta.field_id.astype(np.int32)
df_train_meta['label'] = df_train_meta.label.astype(np.int32)
df_train_meta = df_train_meta.sort_values(by=['field_id']).reset_index(drop=True)
df_train_meta['label'] = df_train_meta.label - 1
df_train_meta.to_pickle(f'{OUTPUT_DIR}/field_meta_train.pkl')

print(f'Training bands saved to {OUTPUT_DIR}')
print(f'Training metadata saved to {OUTPUT_DIR}/field_meta_train.pkl')
