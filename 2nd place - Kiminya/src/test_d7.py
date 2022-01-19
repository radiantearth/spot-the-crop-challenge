import pandas as pd, numpy as np
import os,sys,random,pickle,time,glob,gc,shutil
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import tsai
from tsai.all import *

from sklearn.model_selection import KFold,StratifiedKFold
import sklearn.metrics as skm
from sklearn import preprocessing

DATA_DIR = sys.argv[1]
RANDOM_STATE = FREQ = 7
SUB_ID = f'd{FREQ}c4'
SUB_DIR = f'results/{SUB_ID}'
os.makedirs(SUB_DIR,exist_ok=True)
N_FOLDS = 5
MODEL_DIR = f'models/assets_freq_{FREQ}'
MODEL_NAME = 'XceptionTime'
print(f'running inference for model D{FREQ}')

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(2)
    try:
      dls.rng.seed(seed)
    except NameError:
      pass

fix_seed(RANDOM_STATE)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dfc1 = pd.read_hdf(f'{DATA_DIR}/{FREQ}D/c1_{FREQ}D.h5',key='df')
df_test = pd.read_hdf(f'{DATA_DIR}/{FREQ}D/test_{FREQ}D.h5',key='df')

FEATURES7 = ['ARI', 'AVI', 'B01', 'B05', 'B06', 'B09', 'B11', 'B12', 'BNDVI',
             'BSI', 'CLM', 'CVI', 'FE2', 'GBNDV2', 'MSAVI', 'MYVI', 'NBR',
             'NDMI', 'NDRE', 'NDSI', 'NDVI', 'NGRDI', 'NPCRI', 'PSRI', 'WDRVI']
def get_features_s2(df,versions=[0,1,2]):
    eps=0
    ##v0
    if 0 in versions:
        df['NDVI']=(df['B08']-df['B04'])/ (df['B08']+df['B04']+eps)
        df['BNDVI']=(df['B08']-df['B02'])/ (df['B08']+df['B02']+eps)
        df['GNDVI']=(df['B08']-df['B03'])/ (df['B08']+df['B03']+eps)
        df['GBNDVI'] = (df['B8A']-df['B03']-df['B02'])/(df['B8A']+df['B03']+df['B02']+eps)
        df['GRNDVI'] = (df['B8A']-df['B03']-df['B04'])/(df['B8A']+df['B03']+df['B04']+eps)
        df['RBNDVI'] = (df['B8A']-df['B04']-df['B02'])/(df['B8A']+df['B04']+df['B02']+eps)
        df['GARI'] = (df['B08']-df['B03']+df['B02']-df['B04'])/(df['B08']-df['B03']-df['B02']+df['B04']+eps) 
        df['NBR'] = (df['B08']-df['B12'])/ (df['B08']+df['B12']+eps)
        df['NDMI'] = (df['B08']-df['B11'])/ (df['B08']+df['B11']+eps)
        df['NPCRI'] =(df['B04']-df['B02'])/ (df['B04']+df['B02']+eps)
        a = (df['B08'] * (256-df['B04']) * (df['B08']-df['B04']))
        df['AVI'] = np.sign(a) * np.abs(a)**(1/3)
        df['BSI'] = ((df['B04']+df['B11']) - (df['B08']+df['B02']))/((df['B04']+df['B11']) + (df['B08']+df['B02']) +eps) 
    ##v1
    if 1 in versions:
        a = ((256-df['B04'])*(256-df['B03'])*(256-df['B02']))
        df['SI'] = np.sign(a) * np.abs(a)**(1/3) 
        df['BRI'] = ((1/(df['B03']+eps)) - (1/(df['B05']+eps)) )/ (df['B06']+eps)
        df['MSAVI'] = ((2*df['B08']) + 1- np.sqrt( ((2*df['B08']+1)**2) - 8*(df['B08']-df['B04']) )) /2
        df['NDSI'] = (df['B11'] - df['B12'])/(df['B11']+df['B12']+eps)
        df['NDRE'] = (df['B8A'] - df['B05'])/ (df['B8A']+df['B05'] + eps)
        df['NGRDI'] = (df['B03'] - df['B05'])/ (df['B03']+df['B05'] + eps)
        df['RDVI'] = (df['B08']-df['B04'])/ np.sqrt(df['B08']+df['B04']+eps)
        df['SIPI'] = (df['B08']-df['B02'])/ (df['B08']-df['B04']+eps)
        df['PSRI'] = (df['B04']-df['B03'])/(df['B08']+eps)
        df['GCI'] = (df['B08']/(df['B03']+eps))-1
        df['GBNDV2'] =  (df['B03']-df['B02'])/ (df['B03']+df['B02']+eps)
        df['GRNDV2'] =  (df['B03']-df['B04'])/ (df['B03']+df['B04']+eps)
    ##v2
    if 2 in versions:
        df['REIP'] = 700+(40* ( ( ((df['B04']+df['B07'])/2)-df['B05'])/ (df['B06']-df['B05']+eps)))
        df['SLAVI'] = df['B08']/ (df['B04']+df['B12']+eps)
        df['TCARI'] = 3*((df['B05']-df['B04'])-(0.2*(df['B05']-df['B03']))*(df['B05']/(df['B04']+eps)))
        df['TCI'] = (1.2*(df['B05']-df['B03']))-(1.5*(df['B04']-df['B03']))*np.sqrt(df['B05']/(df['B04']+eps))
        df['WDRVI'] = ((0.1*df['B8A'])-df['B05'])/((0.1*df['B8A'])+df['B05']+eps) 
        df['ARI'] = (1/(df['B03']+eps))-(1/(df['B05']+eps))
        df['MYVI'] = (0.723 * df['B03']) - (0.597 * df['B04']) + (0.206 * df['B06']) - (0.278 * df['B8A'])
        df['FE2'] = (df['B12']/ (df['B08']+eps)) + (df['B03']/ (df['B04']+eps))
        df['CVI'] =  (df['B08']* df['B04'])/ ((df['B03']**2)+eps)
        df['VARIG'] = (df['B03'] - df['B04'])/ (df['B03']+df['B04']-df['B02'] + eps)
    
    feat_cols = sorted([c for c in df if c not in ['tile_id','field_id','sub_id','date','label']])
    drop_feats = [f for f in feat_cols if f not in FEATURES7]
    df = df.drop(columns=drop_feats)    
    for c in FEATURES7:
        df[c] = df[c].replace([-np.inf, np.inf], np.nan).astype(np.float32)
    df = df.fillna(0)
    return df

dfc1 = get_features_s2(dfc1,versions=[0,1,2])
df_test = get_features_s2(df_test,versions=[0,1,2])
feat_cols = sorted([c for c in dfc1 if c not in ['tile_id','field_id','sub_id','date','label']])

def get_dls(fold_id,assets_dir='assets'):
  df_fields = dfc1[['field_id','label']].drop_duplicates().reset_index(drop=True)
  folds = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
  indices= [(train_index, val_index) for (train_index, val_index) in folds.split(df_fields.index,df_fields.label)]
  train_index, val_index = indices[fold_id]
  fields_valid = df_fields.loc[val_index].field_id.values

  df_val = dfc1[dfc1.field_id.isin(fields_valid)].copy()
  df_test_copy = df_test.copy()
  for c in feat_cols:
    with open(f"{assets_dir}/scaler_{fold_id}_{c}.pkl", "rb") as f:
      scaler = pickle.load(f)
      df_val[c] = scaler.transform(df_val[c].values.reshape(-1, 1))
      df_test_copy[c] = scaler.transform(df_test_copy[c].values.reshape(-1, 1))

 
  X_valid = []
  y_valid=[]
  gbs = ['field_id']
  for field_id,grp in tqdm(df_val.groupby(gbs)):
    vals = grp[feat_cols].values
    X_valid.append(vals)
    y_valid.append(grp['label'].values[0])
  
  X_valid = np.asarray(X_valid)
  y_valid = np.asarray(y_valid)

  X_test = []
  for field_id,grp in tqdm(df_test_copy.groupby('field_id')):
    vals = grp[feat_cols].values
    X_test.append(vals)
  
  X_test = np.asarray(X_test)
  y_test = np.zeros_like(X_test[:,0,0])

  # print('fold: ',fold_id, X_train.shape, X_valid.shape)

  assert(sorted(np.unique(y_valid))==[i for i in range(9)])
  X, y, splits = combine_split_data([X_valid,X_test], [y_valid,y_test])
  X = X.transpose(0,2,1)

  bs = 128
  tfms  = [None, [Categorize()]]
  batch_tfms=None
  batch_tfms=[TSStandardize(by_sample=True, by_var=True)]
  dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
  dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs],num_workers=0,batch_tfms=batch_tfms)
  assert(dls.c==9)
  # assert(dls.len==38)
  print('dls: ',dls.c,dls.vars,dls.len)
  return dls,df_val

OOFS = []
PREDS = []
PREDS_TEST = []
for fold_id in range(N_FOLDS):
  fix_seed(RANDOM_STATE)
  dls,df_val = get_dls(fold_id,MODEL_DIR)
  model = build_model(eval(MODEL_NAME),dls=dls)

  learn = Learner(dls, model, metrics=[accuracy])
  learn.model_dir = MODEL_DIR

  shutil.rmtree(f'{MODEL_DIR}/model.pth',ignore_errors=True)
  model_fn = f'{MODEL_DIR}/{MODEL_NAME}_F{fold_id}.pth'
  shutil.copyfile(model_fn,f'{MODEL_DIR}/model.pth')
  learn.model.load_state_dict(torch.load(model_fn))
  

  OOFS.append(df_val)
  with learn.no_bar():
    preds = learn.get_preds(ds_idx=0)[0].numpy()
    PREDS.append(preds)
  
    preds = learn.get_preds(ds_idx=1)[0].numpy()
  PREDS_TEST.append(preds)

os.makedirs(SUB_DIR,exist_ok=True)
df_oof = pd.concat([o.drop_duplicates(subset=['field_id']) for o in OOFS])[['field_id','label']]
preds = np.concatenate(PREDS)
df_oof['pred'] = np.argmax(preds,axis=1) 
df_oof.to_csv(f'{SUB_DIR}/oof.csv')
np.savez_compressed(f'{SUB_DIR}/preds.npz',preds)
loss,acc = skm.log_loss(df_oof.label,preds), skm.accuracy_score(df_oof.label,df_oof.pred)
print(f'OOF loss: {loss} acc: {acc}')
PREDS_TEST = np.asarray(PREDS_TEST)
np.savez_compressed(f'{SUB_DIR}/test_preds.npz',PREDS_TEST)

test_preds = np.mean(PREDS_TEST,axis=0)
assert(sorted(np.unique(np.argmax(test_preds,axis=1)))==[i for i in range(9)])
label_map = {0:'Crop_Lucerne/Medics',1:'Crop_Planted pastures (perennial)',2:'Crop_Fallow',3:'Crop_Wine grapes',
             4:'Crop_Weeds',5:'Crop_Small grain grazing',6:'Crop_Wheat',7:'Crop_Canola',8:'Crop_Rooibos'}


df_preds = pd.DataFrame(test_preds)
# df_preds = df_preds.rename(columns={i:f'CROP_ID_{i+1}' for i in range(9)})
df_preds = df_preds.rename(columns={k:v for k,v in label_map.items()})
df_preds.insert(0,'Field ID',df_test.field_id.unique())
df_preds.to_csv(f'{SUB_DIR}/{SUB_ID}.csv',index=False)