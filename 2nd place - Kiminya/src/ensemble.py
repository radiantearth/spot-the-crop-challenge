import pandas as pd, numpy as np
import sklearn.metrics as skm

df_test = pd.read_csv('results/d10c4/d10c4.csv')

def load_preds(ID):
  oof = pd.read_csv(f'results/{ID}/oof.csv')[['field_id','label','pred']]
  preds = np.load(f'results/{ID}/preds.npz')['arr_0']
  preds = pd.DataFrame(preds)
  preds.insert(0,'field_id',oof.field_id.values)
  preds.insert(1,'label',oof.label.values)
  preds = preds.sort_values(by='field_id').reset_index(drop=True)
  preds_test = np.load(f'results/{ID}/test_preds.npz')['arr_0']
  return preds,preds_test


IDS = ['d10c4','d7c4','d5c4','d3c4','d38c4_ftrs0','d38c4_ftrs1']
WEIGHTS = [0.16115149, 0.16765035, 0.17143377, 0.17056854, 0.16575263,0.16344321]

PREDS = []
PREDS_TEST = []
cols = [i for i in range(9)]
for i in range(len(IDS)): 
  ID = IDS[i]
  preds,preds_test = load_preds(ID)
  preds[cols] = preds[cols]*WEIGHTS[i]
  PREDS.append(preds)
  PREDS_TEST.append(preds_test)
  
# preds = pd.concat(PREDS).groupby('field_id').mean().reset_index()
# preds = [PREDS[i][cols].values *WEIGHTS[i] for i in range(len(IDS))]
# preds = np.sum(preds,axis=0)

cols = ['field_id','label']
preds = pd.concat(PREDS).groupby('field_id').mean().reset_index()
df_oof = preds[cols].copy()
preds = preds.drop(columns=cols).values
df_oof['pred'] = np.argmax(preds,axis=1) 
loss,acc = skm.log_loss(df_oof.label,preds), skm.accuracy_score(df_oof.label,df_oof.pred)
print(f'Ensemble loss: {loss} acc: {acc}')


# test_preds = np.concatenate(PREDS_TEST,axis=0)
# test_preds = np.mean(test_preds,axis=0)
test_preds = [(PREDS_TEST[i]*WEIGHTS[i]).mean(axis=0) for i in range(len(IDS))]
test_preds = np.array(test_preds).sum(axis=0)
test_preds.shape,test_preds.min(),test_preds.max()

assert(sorted(np.unique(np.argmax(test_preds,axis=1)))==[i for i in range(9)])

SUB_ID='submission'
label_map = {0:'Crop_Lucerne/Medics',1:'Crop_Planted pastures (perennial)',2:'Crop_Fallow',3:'Crop_Wine grapes',
             4:'Crop_Weeds',5:'Crop_Small grain grazing',6:'Crop_Wheat',7:'Crop_Canola',8:'Crop_Rooibos'}


df_preds = pd.DataFrame(test_preds)
df_preds = df_preds.rename(columns={k:v for k,v in label_map.items()})
df_preds.insert(0,'Field ID',df_test['Field ID'].values)
df_preds.to_csv(f'results/{SUB_ID}.csv',index=False)
print(f'submission file saved to results/{SUB_ID}.csv')