import pandas as pd, numpy as np
import os,sys,random,pickle,time,glob
from datetime import timedelta
from tqdm.auto import tqdm
from collections import OrderedDict

DATA_DIR = sys.argv[1]

df = pd.read_pickle(f'{DATA_DIR}/test/field_meta_test.pkl')
DIR_BANDS = f'{DATA_DIR}/test/bands-raw/' 
df['path'] = DIR_BANDS+df.field_id.astype(str)+'.npz'
df = df.sort_values(by='field_id').reset_index(drop=True)
print(df.shape)

DATA_DIR  = f'{DATA_DIR}/d38'
os.makedirs(DATA_DIR,exist_ok=True)
all_dts = sorted(['2017-04-01', '2017-04-11', '2017-04-21', '2017-05-01',
       '2017-05-11', '2017-05-21', '2017-05-31', '2017-06-10',
       '2017-06-20', '2017-06-30', '2017-07-05', '2017-07-10',
       '2017-07-15', '2017-07-20', '2017-07-25', '2017-07-30',
       '2017-08-04', '2017-08-09', '2017-08-14', '2017-08-19',
       '2017-08-24', '2017-08-29', '2017-09-08', '2017-09-18',
       '2017-09-23', '2017-09-28', '2017-10-03', '2017-10-08',
       '2017-10-13', '2017-10-18', '2017-10-23', '2017-10-28',
       '2017-11-02', '2017-11-07', '2017-11-12', '2017-11-17',
       '2017-11-22', '2017-11-27', '2017-04-04', '2017-04-14',
       '2017-04-24', '2017-05-04', '2017-05-14', '2017-05-24',
       '2017-06-03', '2017-06-13', '2017-06-23', '2017-07-03',
       '2017-07-08', '2017-07-13', '2017-07-18', '2017-07-23',
       '2017-07-28', '2017-08-02', '2017-08-07', '2017-08-12',
       '2017-08-17', '2017-08-22', '2017-08-27', '2017-09-01',
       '2017-09-06', '2017-09-11', '2017-09-21', '2017-09-26',
       '2017-10-01', '2017-10-06', '2017-10-11', '2017-10-16',
       '2017-10-21', '2017-10-26', '2017-10-31', '2017-11-05',
       '2017-11-10', '2017-11-15', '2017-11-20', '2017-11-30'])

date_id_map = OrderedDict()
for i,d in enumerate(all_dts):
    date_id_map[d] = i


dts_pairs = [(all_dts[i],all_dts[i+1]) for i in range(0,len(all_dts),2)]
dts_dict = dict({p[0]:p[1] for p in dts_pairs})
dts_dict.update({p[1]:p[0] for p in dts_pairs})
  
def get_ix_pairs(tile_dates):
  tdts = [str(t)[:10] for t in tile_dates]
  ix_pairs = []
  skip_next = False
  # for td in tdts:
  for i in range(len(tdts)):
    if skip_next:
      skip_next = False
      continue
    dt = tdts[i]
    pair_dt = dts_dict[dt]
    if pair_dt in tdts:
      skip_next=True
      ix_pairs.append((i,i+2))
    else:
      ix_pairs.append((i,i+1))
  return ix_pairs

clm_band = 12
def get_bands38(bands,ix_pairs):
    bands38 = []
    field_size = bands.shape[0]
    for p0,p1 in ix_pairs:
        b = np.mean(bands[...,p0:p1],axis=-1)
        bands38.append(b)
#     bands38 = np.array(bands38)
    bands38 = np.array(bands38).transpose(1,2,0)
    return bands38

field_ids = []
date_ids = []
ftrs = []
MAX_CHUNKS = 1
N_DATES = 38

for _,row in tqdm(df.iterrows(),total=len(df)):
    bands = np.load(row.path)['arr_0']  

   
    ix_pairs = get_ix_pairs(row.dates)
    bands = get_bands38(bands,ix_pairs)
    
    num_chunks = MAX_CHUNKS
    mn_b = np.mean(bands,axis=0)
    mdn_b = np.median(bands,axis=0)
    std_b = np.std(bands,axis=0)
    
    ftr = np.concatenate([mn_b,mdn_b,std_b]).transpose(1,0)
    ftrs.append(ftr)

    fids = np.repeat(row.field_id,ftr.shape[0])
    field_ids.append(fids)
    
    ids = [i for i in range(num_chunks)]
    dts = np.array(range(N_DATES))
    dts = np.tile(dts,num_chunks)
    date_ids.append(dts)


all_ftrs = np.concatenate(ftrs)
all_field_ids = np.concatenate(field_ids)
all_date_ids = np.concatenate(date_ids)
all_ftrs.shape,all_field_ids.shape,all_date_ids.shape

cols = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12', 'CLM']
sts = ['mn','mdn','std']
cols = [f'{c}_{s}' for s in sts for c in cols]
len(cols)

df_data = pd.DataFrame(all_ftrs,columns=cols)
df_data.insert(0,'field_id',all_field_ids)
df_data.insert(1,'date_id',all_date_ids)

df_data['field_id'] = df_data['field_id'].astype(np.int32)
df_data['date_id'] = df_data['date_id'].astype(np.int8)
df_data.to_hdf(f'{DATA_DIR}/s2_test_d38c{MAX_CHUNKS}_bands.h5',key='df')


assert (df_data==df_data.sort_values(by=['field_id','date_id'])).all().all()