### 외부 라이브러리 ###
import os
import re
import yaml
import json
import pandas as pd
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone, timedelta

with open('./config/use/use_config.yaml') as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)

all_files=sorted(os.listdir("./input/data/ensemble/ingredients/"),reverse=True)
soft_files=[]
hard_files=[]
for file in all_files:
    if 'nbest' in file:
        soft_files.append(file)
    else:
        hard_files.append(file)

weight={}
file_name=[]

files = globals()[CFG['ensemble']['option'][:4]+'_files']

if CFG['ensemble']['option'][:4]=='hard':
    for i, file in enumerate(files):
        with open(os.path.join(f"./input/data/ensemble/ingredients/{file}"), "r", encoding="utf-8") as f:
            raw = json.load(f)
        globals()[f'df{i}']=pd.DataFrame({'id':raw.keys(),f'predict{i}':raw.values()})
        weight[f'predict{i}']=float(re.sub('.json','',file.split('-')[0]))
        file_name.append(file.split('-')[0]+'-'+file.split('-')[1])
        if i==0:
            data=globals()[f'df{i}'].copy()
        elif i>0:
            data=pd.merge(data,globals()[f'df{i}'],on='id')
    predict_columns=data.columns[1:]

else:
    for i, file in enumerate(files):
        with open(os.path.join(f"./input/data/ensemble/ingredients/{file}"), "r", encoding="utf-8") as f:
            raw = json.load(f)
        globals()[f'df{i}']=pd.DataFrame({'id':raw.keys(),f'all{i}':raw.values()})
        globals()[f'df{i}'][f'predict{i}']=globals()[f'df{i}'][f'all{i}'].apply(lambda x: x[0]['text'])
        globals()[f'df{i}'][f'probability{i}']=globals()[f'df{i}'][f'all{i}'].apply(lambda x: x[0]['probability'])
        weight[f'predict{i}']=float(re.sub('.json','',file.split('-')[0]))
        file_name.append(file.split('-')[0]+'-'+file.split('-')[1])
        if i==0:
            data=globals()[f'df{i}'].copy()
        elif i>0:
            data=pd.merge(data,globals()[f'df{i}'],on='id')


def hard_voting_random(data):
    df=data.copy()
    for i in range(len(df)):
        df.loc[i,'predict']=df.loc[i,predict_columns].mode()[0]
    return df

def hard_voting_top1(data):
    df=data.copy()
    for i in range(len(df)):
        df.loc[i,'predict']=df.loc[i,predict_columns].mode()[0]
        df.loc[i,'predict_cnt']=len(df.loc[i,predict_columns].mode().values)
        df.at[i,'predict_list']=df.loc[i,predict_columns].mode().values
    
    for idx in df[df['predict_cnt']>1].index:
        for i in range(len(files)):
            if df.loc[idx,f'predict{i}'] in df.loc[idx,'predict_list']:
                df.loc[idx,'predict']=df.loc[idx,f'predict{i}']
                break
    return df

def hard_voting_weight(data):
    df=data.copy()
    for i in range(len(df)):
        df.loc[i,'predict']=df.loc[i,predict_columns].mode()[0]
        df.loc[i,'predict_cnt']=len(df.loc[i,predict_columns].mode().values)
        df.at[i,'predict_list']=df.loc[i,predict_columns].mode().values
    
    for i in df[df['predict_cnt']>1].index:
        weight_sum=defaultdict(int)
        for name, prediction in zip(predict_columns,df.loc[i,predict_columns]):
            weight_sum[prediction]+=weight[name]
        df.loc[i,'predict']=sorted(weight_sum.items(), key=lambda x: x[1], reverse=True)[0][0]
    return df

def soft_voting_sum(data):
    df=data.copy()
    for idx in range(len(df)):
        prob_sum=defaultdict(int)
        for i in range(len(files)):
            prob_sum[df.loc[idx,f'predict{i}']]+=df.loc[idx,f'probability{i}']
        df.loc[idx,'predict']=sorted(prob_sum.items(), key=lambda x: x[1], reverse=True)[0][0]
    return df

def soft_voting_avg(data):
    df=data.copy()
    for idx in range(len(df)):
        word_cnt=defaultdict(int)
        prob_sum=defaultdict(int)
        prob_avg=defaultdict(int)
        for i in range(len(files)):
            word_cnt[df.loc[idx,f'predict{i}']]+=1
            prob_sum[df.loc[idx,f'predict{i}']]+=df.loc[idx,f'probability{i}']
        for key in word_cnt.keys():
            prob_avg[key]=prob_sum[key]/word_cnt[key]
        df.loc[idx,'predict']=sorted(prob_avg.items(), key=lambda x: x[1], reverse=True)[0][0]
    return df


if __name__ == "__main__":
    final=eval(CFG['ensemble']['option'])(data)
    all_predictions = OrderedDict()
    for i in range(len(final)):
        all_predictions[final.loc[i,"id"]] = final.loc[i,"predict"]
    now = datetime.now(tz=timezone(timedelta(hours=9)))
    prediction_file=f"./input/data/ensemble/food/{now.strftime('%d%H%M%S')}_{CFG['ensemble']['option']}_{'_'.join(file_name)}.json"
    with open(prediction_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
                )