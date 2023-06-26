import os
import json
import re
import collections
import pandas as pd
from konlpy.tag import Okt 
okt=Okt()

def remove_josa(raw1, raw2):
    df1 = pd.DataFrame({'id':raw1.keys(), 'predict':raw1.values()})
    df2 = pd.DataFrame({'id':raw2.keys(), 'context':raw2.values()})
    df3=pd.merge(df1,df2,on='id',how='inner')

    for i in range(len(df3)):
        df3.loc[i,'context_mask']=df3.loc[i,'context'].replace(df3.loc[i,'predict'],' MASK')

    for i in range(len(df3)):
        for sen in df3.loc[i,'context_mask'].split('.'):
            if ' MASK' in sen:
                df3.loc[i,'sentence_mask']=sen
                break

    for i in range(len(df3)):
        df3.loc[i,'sentence']=df3.loc[i,'sentence_mask'].replace(' MASK',df3.loc[i,'predict'])

    df3['okt_sentence_mask']=df3['sentence_mask'].apply(lambda x: okt.pos(x))
    df3['okt_sentence']=df3['sentence'].apply(lambda x: okt.pos(x))
    
    for idx in range(len(df3)):
        for i,value in enumerate(df3.loc[idx,'okt_sentence_mask']):
            if 'MASK' in value[0]:
                df3.loc[idx,'start_idx']=i
                if value[0]=='MASK':    
                    if i+1<len(df3.loc[idx,'okt_sentence_mask']):
                        df3.loc[idx,'next_mask_word']=df3.loc[idx,'okt_sentence_mask'][i+1][0]
                        df3.loc[idx,'next_mask_okt']=df3.loc[idx,'okt_sentence_mask'][i+1][-1]
                    else:
                        df3.loc[idx,'next_mask_word']='exception'
                        df3.loc[idx,'next_mask_okt']='exception'
                    break
                else:
                    df3.loc[idx,'next_mask_word']=value[0].replace('MASK','')
                    df3.loc[idx,'next_mask_okt']='Alpha'
                    break


    df3['start_idx']=df3['start_idx'].astype('int')

    for i in range(len(df3)):
        if len(re.findall(f"MASK ",df3.loc[i,'context']))==0:
            df3.loc[i,'space']='N'
        else:
            df3.loc[i,'space']='Y'

    josa_L1=['은','는','을','를','이','가','의','에','로','으로','과','와','도','에서','만','이나','나','까지','부터','에게','보다','께','처럼','이라도','라도','으로서','로서','조차','만큼','같이','마저','이나마','나마','한테','더러','에게서','한테서']
    data1=df3[(df3['next_mask_word'].isin(josa_L1))&(df3['space']=='N')|(df3['next_mask_okt']=='Punctuation')].reset_index()
    data2=df3[~((df3['next_mask_word'].isin(josa_L1))&(df3['space']=='N')|(df3['next_mask_okt']=='Punctuation'))].reset_index()

    df3['last_josa_cnt']=df3['predict'].apply(lambda x: len((re.findall(f'.+(?=[은는을를이가에와]$)',x))))

    idx1=df3[~(~((df3['next_mask_word'].isin(josa_L1)) & (df3['space']=='N'))&~(df3['next_mask_okt'].isin(['Punctuation','Foreign']))&(df3['last_josa_cnt']>0))].index
    idx2=df3[~((df3['next_mask_word'].isin(josa_L1)) & (df3['space']=='N'))&~(df3['next_mask_okt'].isin(['Punctuation','Foreign']))&(df3['last_josa_cnt']>0)].index

    for i in idx2:
        df3.loc[i,'predict']=re.findall(f'.+(?=[은는을를이가에와]$)',df3.loc[i,'predict'])[0]

    final=df3[['id','predict']]

    all_rm_josa = collections.OrderedDict()
    for i in range(len(final)):
        all_rm_josa[final.loc[i,'id']]=final.loc[i,'predict']

    return all_rm_josa