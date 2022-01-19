import pandas as pd, numpy as np
import os,random,pickle,time,glob,sys
from tqdm.auto import tqdm
from collections import OrderedDict

DATA_DIR = sys.argv[1]
MAX_CHUNKS = int(sys.argv[2])

df = pd.read_pickle(f'{DATA_DIR}/train/field_meta_train.pkl')
DIR_BANDS = f'{DATA_DIR}/train/bands-raw/' 

df['path'] = DIR_BANDS+df.field_id.astype(str)+'.npz'

field_ids = []
labels = []
sub_ids = []
dates = []
ftrs = []


for _,row in tqdm(df.iterrows(),total=len(df)):
    bands = np.load(row.path)['arr_0']

    n = bands.shape[0]
    n_dates = bands.shape[2]
    num_chunks = MAX_CHUNKS
    if n<MAX_CHUNKS*2 or MAX_CHUNKS==1:
        num_chunks = 1
        mean = np.mean(bands,axis=0)
    else:
        bands = np.array_split(bands,num_chunks)
        mean = [np.mean(x,axis=0) for x in bands]
        mean = np.concatenate(mean,axis=1)

    ftr = np.concatenate([mean]).transpose(1,0)
    ftrs.append(ftr)

    fids = np.repeat(row.field_id,ftr.shape[0])
    field_ids.append(fids)
    lbls = np.repeat(row.label,ftr.shape[0])
    labels.append(lbls)
    
    ids = [i for i in range(num_chunks)]
    dts = [str(d)[:10] for d in row.dates]
    dts = np.tile(dts,num_chunks)
    dates.append(dts)
    ids = np.repeat([ids],len(row.dates))
    sub_ids.append(ids)
    

all_ftrs = np.concatenate(ftrs)
all_field_ids = np.concatenate(field_ids)
all_labels = np.concatenate(labels)

all_sub_ids = np.concatenate(sub_ids)
all_dates = np.concatenate(dates)

cols = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12', 'CLM']
df_data = pd.DataFrame(all_ftrs,columns=cols)
df_data.insert(0,'field_id',all_field_ids)
df_data.insert(1,'sub_id',all_sub_ids)
df_data.insert(2,'date',all_dates)
df_data.insert(3,'label',all_labels)

df_data['field_id'] = df_data['field_id'].astype(np.int32)
df_data['sub_id'] = df_data['sub_id'].astype(np.int8)
fn = f'{DATA_DIR}/s2_train_dxc{MAX_CHUNKS}_bands.h5'
df_data.to_hdf(fn,key='df')
print(f'saved data to {fn}')