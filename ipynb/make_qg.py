from pororo import Pororo
import pandas as pd
import json
print('making QG func...')
qg = Pororo(task='qg',lang='ko')
df = pd.read_json('../input/data/wikipedia_documents.json') ## pd.read_json 이용
df = df.transpose()
print('making Dataframe...')
df_wiki_ner = df.copy()
# df_wiki_ner = df_wiki_ner.head(10)
df_wiki_ner = df_wiki_ner.groupby('title').agg({'text': lambda x: ' '.join(map(str,x)), 'corpus_source':'first', 'url':'first', 'domain':'first',  'author':'first', 'html':'first', 'document_id':lambda x:list(map(int,x))}).reset_index() 
print('making Q...')
df_wiki_ner['Q_based title'] = df_wiki_ner[['title','text']].apply(lambda x: qg(x[0],x[1],n_wrong=0), axis=1)
print('making csv...')
df_wiki_ner.to_csv('/opt/ml/input/data/wiki_QG_concat.csv', sep=',', na_rep='NaN',index=False) # do not write index
print('DONE!')
