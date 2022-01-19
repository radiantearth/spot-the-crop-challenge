
import pandas as pd, numpy as np
import os,random,pickle,time,glob,sys
from tqdm.auto import tqdm
from collections import OrderedDict

DATA_DIR = sys.argv[1]

df = pd.read_pickle(f'{DATA_DIR}/test/field_meta_test.pkl')
DIR_BANDS = f'{DATA_DIR}/test/bands-raw/' 

df['path'] = DIR_BANDS+df.field_id.astype(str)+'.npz'
field_ids = []
dates = []
ftrs = []

for _,row in tqdm(df.iterrows(),total=len(df)):
    bands = np.load(row.path)['arr_0']  

    mn_b = np.mean(bands,axis=0)
    ftr = np.concatenate([mn_b]).transpose(1,0)
    ftrs.append(ftr)

    fids = np.repeat(row.field_id,ftr.shape[0])
    field_ids.append(fids)
    

    dts = [str(d)[:10] for d in row.dates]
    dates.append(dts)
    

all_ftrs = np.concatenate(ftrs)
all_field_ids = np.concatenate(field_ids)
all_dates = np.concatenate(dates)

cols = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12', 'CLM']
df_data = pd.DataFrame(all_ftrs,columns=cols)
df_data.insert(0,'field_id',all_field_ids)
df_data.insert(1,'date',all_dates)

df_data['field_id'] = df_data['field_id'].astype(np.int32)
fn = f'{DATA_DIR}/s2_test_dxc1_bands.h5'
df_data.to_hdf(fn,key='df')
print(f'saved data to {fn}')