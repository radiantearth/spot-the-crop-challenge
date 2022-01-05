import pandas as pd, numpy as np
import os,sys,random,pickle,time,glob,gc
from tqdm.auto import tqdm

import tsai
from tsai.all import *

from sklearn.model_selection import KFold,StratifiedKFold
import sklearn.metrics as skm
from sklearn import preprocessing

DATA_DIR = sys.argv[1]
FOLD_ID = int(sys.argv[2])
DATA_DIR = f'{DATA_DIR}/10D/'
RANDOM_STATE = FREQ = 10
MODEL_DIR = f'models/assets_freq_{FREQ}'
MODEL_NAME = 'XceptionTime'
print(f'Training FOLD {FOLD_ID} with data from {DATA_DIR}')
print(f'Models will be saved to {MODEL_DIR}')

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
      dls.rng.seed(seed)
    except NameError:
      pass

fix_seed(RANDOM_STATE)


dfc1 = pd.read_hdf(f'{DATA_DIR}c1_10D.h5',key='df')
dfc4 = pd.read_hdf(f'{DATA_DIR}c4_10D.h5',key='df')
print(dfc1.shape,dfc4.shape)

FEATURES10 = ['ARI', 'AVI', 'B01', 'B02', 'B05', 'B06', 'B09', 'B11', 'B12',
       'BNDVI', 'BSI', 'CLM', 'CVI', 'FE2', 'GBNDV2', 'GNDVI', 'GRNDV2',
       'MSAVI', 'MYVI', 'NBR', 'NDMI', 'NDRE', 'NDSI', 'NDVI', 'NGRDI',
       'NPCRI', 'PSRI', 'RBNDVI', 'SI', 'SLAVI', 'TCARI', 'TCI', 'WDRVI']


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
    drop_feats = [f for f in feat_cols if f not in FEATURES10]
    df = df.drop(columns=drop_feats)    
    for c in FEATURES10:
        df[c] = df[c].replace([-np.inf, np.inf], np.nan).astype(np.float32)
    df = df.fillna(0)
    return df

dfc1 = get_features_s2(dfc1)
dfc4 = get_features_s2(dfc4)

feat_cols = sorted([c for c in dfc1 if c not in ['tile_id','field_id','sub_id','date','date_id','label']])
print('training features: ',feat_cols)

def get_dls(fold_id,assets_dir='assets'):
    global dfc1,dfc4
    df_fields = dfc4[['field_id','label']].drop_duplicates().reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    indices= [(train_index, val_index) for (train_index, val_index) in folds.split(df_fields.index,df_fields.label)]
    train_index, val_index = indices[fold_id]
    fields_train = df_fields.loc[train_index].field_id.values
    fields_valid = df_fields.loc[val_index].field_id.values
    df_trn = dfc4[dfc4.field_id.isin(fields_train)].copy()#.sort_values(by=['field_id','date_id'])
    df_val = dfc1[dfc1.field_id.isin(fields_valid)].copy()#.sort_values(by=['field_id','date_id'])
    
    del dfc1,dfc4; gc.collect()
  
    scaler = preprocessing.MinMaxScaler()
    for c in feat_cols:
        scaler = scaler.fit(df_trn[c].values.reshape(-1, 1))
        df_trn[c] = scaler.transform(df_trn[c].values.reshape(-1, 1))
        df_val[c] = scaler.transform(df_val[c].values.reshape(-1, 1))
      
        with open(f"{assets_dir}/scaler_{fold_id}_{c}.pkl", "wb") as f:
            pickle.dump(scaler, f)

    X_train = []
    y_train=[]
    gbs = ['field_id','sub_id']
    for field_id,grp in tqdm(df_trn.groupby(gbs)):
        X_train.append(grp[feat_cols].values)
        y_train.append(grp['label'].values[0])
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    X_valid = []
    y_valid=[]
    gbs = ['field_id']
    for field_id,grp in tqdm(df_val.groupby(gbs)):
        X_valid.append(grp[feat_cols].values)
        y_valid.append(grp['label'].values[0])
    
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    print('fold: ',fold_id, X_train.shape, X_valid.shape)
    
    del df_trn,df_val; gc.collect()
    
    assert(sorted(np.unique(y_train))==sorted(np.unique(y_valid))==[i for i in range(9)])
    X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
    X = X.transpose(0,2,1)

    del X_train,X_valid; gc.collect()
    
    bs = [128,128]
    tfms  = [None, TSClassification()]
    batch_tfms=[TSStandardize(by_sample=True, by_var=True)]
    dls = get_ts_dls(X,y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=bs)
    assert(dls.c==9)
    print('dls: ',dls.c,dls.vars,dls.len)
    return dls

os.makedirs(MODEL_DIR,exist_ok=True)
fix_seed(RANDOM_STATE)
dls = get_dls(FOLD_ID,MODEL_DIR)

def get_learn():
    fix_seed(RANDOM_STATE)
    learn = ts_learner(dls, eval(MODEL_NAME),
                     metrics=accuracy,
                    cbs=[CutMix1D(1.), SaveModelCallback(monitor='valid_loss')])
    return learn

learn = get_learn()
with learn.no_bar():
  learn.lr_find()

learn = get_learn()
with learn.no_bar():
  learn.fit_one_cycle(40, lr_max=3e-3)
torch.save(learn.model.state_dict(), f'{MODEL_DIR}/{MODEL_NAME}_F{FOLD_ID}.pth')