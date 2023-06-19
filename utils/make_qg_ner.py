# -------------config-----------------------------
# /opt/ml/input/data/nerQG/ 이 폴더에 csv 생길 예정
# 반드시 pororo가 설치된 가상 환경에서 진행해야 함

ids_base = 0      # max entity 어떤 걸로 할 것인지 지정 (0이라면 PERSON)
range_num = 2     # 얼마나 불릴 것인지 (1 이상 설정 권장)
part = True       # 멈춤 현상을 방지하기 위해 나눠서 진행함  /opt/ml/input/data/nerQG/ 폴더에 csv가 한 400개쯤 만들어짐. 싫다면 False로 해 두길 권장.
# -------------config-----------------------------


import pandas as pd
import json
import random
import os

from pororo import Pororo

print('making data folder...')
data_path = '/opt/ml/input/data/nerQ/'
if os.path.exists(data_path):
    print('floder exists...')
    pass
else:
    os.mkdir(data_path)

print('open wiki csv...')
df = pd.read_csv('/opt/ml/input/data/wiki_ner.csv')
print('making Dataframe...')
df_wiki_ner = df.copy()
# df_wiki_ner = df_wiki_ner.head(800).copy()

ner_list = ['PERSON', 'EVENT', 'ORGANIZATION',  'DATE', 'TIME', 'LOCATION']
# ner_list = ['PERSON','EVENT']
print(f"entitiy list...{ner_list}")

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



df_add = pd.DataFrame()
for i in range(range_num):
    print(i)
    num_list = []
    for ne in ner_list:
        print(ne)
        df_wiki_ner['k_'+ne] = df_wiki_ner['context_ner'].apply(lambda x: ne in (x).keys())   # 각 entity가 들어있는지를 하나의 컬럼으로 만들어 줌
        num_list.append(num_T('k_'+ne,df_wiki_ner))    # ex) 전체 데이터가 300개라면, person을 가진 문장은 30개, event를 가진 문장은 20개라면 30개 기준으로 비율 나눠주고 싶어서 먼저 20개라 30개 라는걸 담을 수 있는 list로 만들어 둠.
    print(num_list)


    print([min(round(num_list[ids_base]),num_list[0]), min(num_list[ids_base],num_list[1]), min(num_list[ids_base],num_list[2]), min(round(num_list[ids_base]*8/24),num_list[3]), min(round(num_list[ids_base]*8/24),num_list[4]), min(round(num_list[ids_base]*12/24),num_list[5]) ])
    num_list = [min(round(num_list[ids_base]),num_list[0]), min(num_list[ids_base],num_list[1]), min(num_list[ids_base],num_list[2]), min(round(num_list[ids_base]*8/24),num_list[3]), min(round(num_list[ids_base]*8/24),num_list[4]), min(round(num_list[ids_base]*12/24),num_list[5]) ]
    # 위에서 지정한 비율대로 num_list 재배치
    
    for (i,ne) in enumerate(ner_list):
        print(ne)
        # if num_list[i]<p_num:
        #     pass
        # else:
        df_p = df_wiki_ner[df_wiki_ner['k_'+ne]==True].sample(n=num_list[i])      # ner_list에 적힌 숫자대로 랜덤 샘플링, 새로운 데이터프레임 만들어 둠 / context를 뽑아오는 것
        # display(df_p)
        df_p.insert(9, 'answer', df_p['context_ner'].apply(lambda x: random.choice((x)[ne]))) # 선택된 ex) PERSON ner 중에 하나 뽑음, answer에 넣어 줌
        df_p['context_ner']=df_p[['context_ner','answer']].apply(pop_dict,axis=1)  # answer로 뽑았다면, context_ner dict에서 빼 줘야 함 (다시 뽑히는 것 방지)
        # display(df_p)
        print('========================================================================\n===========================================================')
        df_add = pd.concat([df_add,df_p])



print('total_data:', len(df_add))
# print('making Q...')
# df_add['Q_based title'] = df_add[['answer','text']].apply(lambda x: qg(x[0],x[1],n_wrong=0), axis=1)
print('making csv...')
df_add.to_csv('/opt/ml/input/data/wiki_ner_a'+str(range_num)+'.csv', sep=',', na_rep='NaN',index=False) # do not write index
print('Answer making... DONE!')




# print(' answer을 만들어 뒀으니, QG 진행\n========================================================================\n========================================================================')

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

if part:
    for i in range(len(df_add)//500):
        print(i*500,(i+1)*500)
        df_tmp = df_add.iloc[i*500:(i+1)*500].copy()
        df_tmp['Q_based title'] = df_tmp[['answer','text']].apply(qg_F, axis=1)
        # df_wiki_ner['Q_based title'] = df_wiki_ner[['answer','text']].apply(lambda x: qg(eval(x[0])[0],x[1],n_wrong=0,len_penalty =-1), axis=1)
        print('making csv...')
        df_tmp.to_csv('/opt/ml/input/data/nerQG/wiki_QG_ner_'+str(range_num)+'_p'+str(i)+'.csv', sep=',', na_rep='NaN',index=False) # do not write index
        print(f"part_{i} DONE!")

    df_tmp = df_add.iloc[(i+1)*500:].copy()
    df_tmp['Q_based title'] = df_tmp[['answer','text']].apply(qg_F, axis=1)
    # df_wiki_ner['Q_based title'] = df_wiki_ner[['answer','text']].apply(lambda x: qg(eval(x[0])[0],x[1],n_wrong=0,len_penalty =-1), axis=1)
    print('making csv...')
    df_tmp.to_csv('/opt/ml/input/data/nerQG/wiki_QG_ner_'+str(range_num)+'_p'+str(i+1)+'.csv', sep=',', na_rep='NaN',index=False) # do not write index
else:
    df_tmp = df_add.copy()
    df_tmp['Q_based title'] = df_tmp[['answer','text']].apply(qg_F, axis=1)
    # df_wiki_ner['Q_based title'] = df_wiki_ner[['answer','text']].apply(lambda x: qg(eval(x[0])[0],x[1],n_wrong=0,len_penalty =-1), axis=1)
    print('making csv...')
    df_tmp.to_csv('/opt/ml/input/data/nerQG/wiki_QG_ner_'+str(range_num)+'.csv', sep=',', na_rep='NaN',index=False) # do not write index



print('final DONE!')
