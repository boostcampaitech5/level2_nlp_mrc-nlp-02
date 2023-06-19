from pororo import Pororo
import pandas as pd
import json
print('making QG func...')
# qg = Pororo(task='qg',lang='ko')
ner = Pororo(task='ner',lang='ko')
df = pd.read_json('../input/data/wikipedia_documents.json') ## pd.read_json 이용
df = df.transpose()
print('making Dataframe...')
df_wiki_ner = df.copy()
df_wiki_ner = df_wiki_ner.head(10)


def ner_dict(x):
    try:
        result_ner = ner(x)
        ids = 0
        dict_={}
        for w,ne in result_ner:
            if ne!='O':
                dict_[ne] = dict_.get(ne,[]) 
                dict_[ne].append((w,ids))
            ids+=len(w)
        return dict_
    except ValueError:
        return {}
    

print('making NER...')
df_wiki_ner['context_ner'] = df_wiki_ner['text'].apply(lambda x: ner_dict(x))


print('making csv...')
df_wiki_ner.to_csv('/opt/ml/input/data/wiki_ner.csv', sep=',', na_rep='NaN',index=False) # do not write index
print('DONE!')
