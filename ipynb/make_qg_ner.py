# ------------------------------------------
ids_base = 0      # max entity 어떤 걸로 할 것인지 지정 (0이라면 PERSON)
range_num = 2     # 얼마나 불릴 것인지

from pororo import Pororo
import pandas as pd
import json
import random

print('making QG func...')
qg = Pororo(task='qg',lang='ko')
df = pd.read_csv('/opt/ml/input/data/wiki_ner.csv')
print('making Dataframe...')
df_wiki_ner = df.copy()
# df_wiki_ner = df_wiki_ner.head(800).copy()

ner_list = ['PERSON', 'EVENT', 'ORGANIZATION',  'DATE', 'TIME', 'LOCATION']
# ner_list = ['PERSON','EVENT']
def pop_dict(x):
    x[0][ne].remove(x[1])
    if len(x[0][ne])==0:
        del(x[0][ne])
        return x[0]
    else:
        return x[0]
def num_T(x,df):
    try: 
        return df[x].value_counts()[True] 
    except KeyError:
        return 0
    

df_wiki_ner['context_ner'] = df_wiki_ner['context_ner'].apply(lambda x: eval(x))
print(len(df_wiki_ner))

# ------------------------------------------
ids_base=0
range_num = 2


df_add = pd.DataFrame()
for i in range(range_num):
    print(i)
    num_list = []
    for ne in ner_list:
        print(ne)
        df_wiki_ner['k_'+ne] = df_wiki_ner['context_ner'].apply(lambda x: ne in (x).keys())
        num_list.append(num_T('k_'+ne,df_wiki_ner))
    print(num_list)


    print([min(round(num_list[ids_base]),num_list[0]), min(num_list[ids_base],num_list[1]), min(num_list[ids_base],num_list[2]), min(round(num_list[ids_base]*8/24),num_list[3]), min(round(num_list[ids_base]*8/24),num_list[4]), min(round(num_list[ids_base]*12/24),num_list[5]) ])
    num_list = [min(round(num_list[ids_base]),num_list[0]), min(num_list[ids_base],num_list[1]), min(num_list[ids_base],num_list[2]), min(round(num_list[ids_base]*8/24),num_list[3]), min(round(num_list[ids_base]*8/24),num_list[4]), min(round(num_list[ids_base]*12/24),num_list[5]) ]
    for (i,ne) in enumerate(ner_list):
        print(ne)
        # if num_list[i]<p_num:
        #     pass
        # else:
        df_p = df_wiki_ner[df_wiki_ner['k_'+ne]==True].sample(n=num_list[i])
        # display(df_p)
        df_p.insert(9, 'answer', df_p['context_ner'].apply(lambda x: random.choice((x)[ne])))
        df_p['context_ner']=df_p[['context_ner','answer']].apply(pop_dict,axis=1)
        # display(df_p)
        print('========================================================================\n===========================================================')
        df_add = pd.concat([df_add,df_p])



print('total_data:', len(df_add))
# print('making Q...')
# df_add['Q_based title'] = df_add[['answer','text']].apply(lambda x: qg(x[0],x[1],n_wrong=0), axis=1)
print('making csv...')
df_add.to_csv('/opt/ml/input/data/wiki_ner_a'+str(range_num)+'.csv', sep=',', na_rep='NaN',index=False) # do not write index
print('DONE!')




print(' 여기========================================================================\n========================================================================')

print('making QG func...')
qg = Pororo(task='qg',lang='ko')
print('open csv...')
df = pd.read_csv('/opt/ml/input/data/wiki_ner_a'+str(range_num)+'.csv')
print('making Dataframe...')
df_add = df.copy()
# df_wiki_ner = df_wiki_ner.head(10)

print('========================================================================\n========================================================================')

print('making Q...')
def qg_F(x):
    try:
        return qg(eval(x[0])[0] , x[1],  n_wrong=0, len_penalty=-0.2  )
    except IndexError:
        return 'NaN'
    except RuntimeError:
        return 'NaN'
print(len(df_add))    
for i in range(len(df_add)//500):
    print(i*500,(i+1)*500)
    df_tmp = df_add.iloc[i*500:(i+1)*500].copy()
    df_tmp['Q_based title'] = df_tmp[['answer','text']].apply(qg_F, axis=1)
    # df_wiki_ner['Q_based title'] = df_wiki_ner[['answer','text']].apply(lambda x: qg(eval(x[0])[0],x[1],n_wrong=0,len_penalty =-1), axis=1)
    print('making csv...')
    df_tmp.to_csv('/opt/ml/input/data/nerQG/wiki_QG_ner_'+str(range_num)+'_p'+str(i)+'.csv', sep=',', na_rep='NaN',index=False) # do not write index
    print('DONE!')

df_tmp = df_add.iloc[(i+1)*500:].copy()
df_tmp['Q_based title'] = df_tmp[['answer','text']].apply(qg_F, axis=1)
# df_wiki_ner['Q_based title'] = df_wiki_ner[['answer','text']].apply(lambda x: qg(eval(x[0])[0],x[1],n_wrong=0,len_penalty =-1), axis=1)
print('making csv...')
df_tmp.to_csv('/opt/ml/input/data/nerQG/wiki_QG_ner_'+str(range_num)+'_p'+str(i+1)+'.csv', sep=',', na_rep='NaN',index=False) # do not write index
print('DONE!')
