import pandas as pd, numpy as np
import os,sys,random,pickle,time,glob,gc
from tqdm.auto import tqdm

DATA_DIR = sys.argv[1]
FREQ = sys.argv[2]

print(f'resampling data to {FREQ}')

cols = ['label']
df_meta = pd.read_hdf(f'{DATA_DIR}/s2_train_dxc1_bands.h5',key='df')[['field_id','label']].drop_duplicates()
dfc1 = pd.read_hdf(f'{DATA_DIR}/s2_train_dxc1_bands.h5',key='df').drop(columns=cols)
dfc4 = pd.read_hdf(f'{DATA_DIR}/s2_train_dxc4_bands.h5',key='df').drop(columns=cols)
df_test = pd.read_hdf(f'{DATA_DIR}/s2_test_dxc1_bands.h5',key='df')
df_test['sub_id'] = 0

dfc1['date'] = dfc1.date.astype(np.datetime64)
dfc4['date'] = dfc4.date.astype(np.datetime64)
df_test['date'] = df_test.date.astype(np.datetime64)
print(dfc1.shape,dfc4.shape,df_test.shape)
#((4301227, 16), (16938664, 16), (1744554, 16))


def get_resampled(df,FREQ):
  dates = sorted(df.date.unique())
  df_dates = pd.DataFrame(dict(date=dates))
  grps = []
  for gb,d in tqdm(df.groupby(['field_id','sub_id'])):
    field_id,sub_id = gb
    grp = pd.merge(df_dates,d,how='left').set_index('date').resample(FREQ).mean()
    grp['field_id'] = field_id
    grp['sub_id'] = sub_id
    grps.append(grp)
  df = pd.concat(grps).reset_index()
  return df


dir = f'{DATA_DIR}/{FREQ}/'
os.makedirs(dir,exist_ok=True)
dfc1_res = get_resampled(dfc1,FREQ)
dfc4_res = get_resampled(dfc4,FREQ)
df_test_res = get_resampled(df_test,FREQ)
cols = ['field_id','date']
dfc1_res = dfc1_res.merge(df_meta).sort_values(by=['field_id','sub_id','date']).reset_index(drop=True)
dfc4_res = dfc4_res.merge(df_meta).sort_values(by=['field_id','sub_id','date']).reset_index(drop=True)

dfc1_res.to_hdf(f'{dir}/c1_{FREQ}.h5',key='df')
dfc4_res.to_hdf(f'{dir}/c4_{FREQ}.h5',key='df')
df_test_res.to_hdf(f'{dir}/test_{FREQ}.h5',key='df')
