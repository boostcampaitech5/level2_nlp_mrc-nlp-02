import numpy as np
import re
import random
import Levenshtein as lv
import pandas as pd
import pickle
import torch

from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from datasets import load_from_disk
from kiwipiepy import Kiwi
from tqdm.notebook import tqdm
from collections import Counter, defaultdict, OrderedDict

kiwi = Kiwi()

GT = load_from_disk('/opt/ml/input/data/test_dataset/validation/')

NNP = []
for question in GT['question']:
    NNP.append([token.form for token in kiwi.tokenize(question) if token.tag == 'NNP'])

NNG = []
for question in GT['question']:
    NNG.append([token.form for token in kiwi.tokenize(question) if token.tag == 'NNG'])

NNG_CONCAT = []
for question in GT['question']:
    tokenized = kiwi.tokenize(question)
    temp = []
    nng_sublist = []
    for i in range(len(tokenized)):
        if tokenized[i].tag == 'NNG':
            if i != len(tokenized)-1 and tokenized[i+1].tag == 'NNG' and tokenized[i].start + tokenized[i].len == tokenized[i+1].start:
                temp.append(tokenized[i].form)
            else:
                temp.append(tokenized[i].form)
                nng_sublist.append(''.join(temp))
                temp = []
        else:
            if temp:
                nng_sublist.append(''.join(temp))
                temp = []
    NNG_CONCAT.append(nng_sublist)

def flatten_and_count(nng_list):
    """
    Flattens the input list and counts unique elements. Each sublist is treated as a document.
    """
    counter = Counter()
    for sublist in nng_list:
        counter.update(set(sublist))  
    return counter

def remove_common_words(nng_list, threshold):
    """
    Removes words that appear more than 'threshold' times in the NNG list.
    Also, returns a dictionary of the words that were removed and their corresponding indices.
    """
    counter = flatten_and_count(nng_list)
    removed_words = defaultdict(list)
    
    new_nng_list = []
    for idx, sublist in enumerate(nng_list):
        new_sublist = []
        for word in sublist:
            if counter[word] > threshold:
                removed_words[word].append(idx)
            else:
                new_sublist.append(word)
        new_nng_list.append(new_sublist)

    return new_nng_list, removed_words

n = 3  
NNG_refined, removed_words = remove_common_words(NNG, n)
NNG_CONCAT_refined, removed_words = remove_common_words(NNG_CONCAT, n)
NNG_NNP = [nng + nnp for nng, nnp in zip(NNG_refined, NNP)]
NNG_CONCAT_NNP = [nng + nnp for nng, nnp in zip(NNG_CONCAT_refined, NNP)]

STOPWORDS = list(removed_words.keys()) + ['예', '예시']

candid = pd.read_json('/opt/ml/input/data/wikipedia_documents.json').transpose()

CONTEXT_ONLYNNP = []
for nnp in tqdm(NNP):
    if nnp:
        CONTEXT_ONLYNNP.append([text for text in candid['text'] if all(word in text for word in nnp)])
    else:
        CONTEXT_ONLYNNP.append([])

CONTEXT_NNP_NNG = []
for ngnp in tqdm(NNG_NNP):
    if ngnp:
        CONTEXT_NNP_NNG.append([text for text in candid['text'] if all(word in text for word in ngnp)])
    else:
        CONTEXT_NNP_NNG.append([])

CONTEXT_NNP_CONCAT_NNG = []
for ngnp in tqdm(NNG_CONCAT_NNP):
    if ngnp:
        CONTEXT_NNP_CONCAT_NNG.append([text for text in candid['text'] if all(word in text for word in ngnp)])
    else:
        CONTEXT_NNP_CONCAT_NNG.append([])

df = pd.read_json('/opt/ml/results/20235511-ColBERT_top30_korquad_ep1_finetune_train_checkpoint_1200/test/nbest_predictions.json').transpose()

selected = df[0].tolist()
selected = [i['context'] for i in selected]

def concat_tag_in_text(text, tag = 'NNP'):
    tokenized = kiwi.tokenize(text)
    temp = []
    tag_sublist = []
    for i in range(len(tokenized)):
        if tokenized[i].tag == tag:
            temp.append(tokenized[i].form)
            if i != len(tokenized)-1 and tokenized[i+1].tag == tag:
                
                if tokenized[i].start + tokenized[i].len != tokenized[i+1].start:
                    temp.append(' ')
            else:
                tag_sublist.append(''.join(temp))
                temp = []
        else:
            if temp:
                tag_sublist.append(''.join(temp))
                temp = []
    return tag_sublist

REAL_CONCAT = [concat_tag_in_text(question) for question in tqdm(selected)]
REAL_CONCAT_unique = list(OrderedDict((tuple(x), x) for x in REAL_CONCAT).values())
REAL_CONCAT_unique_no_duplicates = [list(OrderedDict.fromkeys(x)) for x in REAL_CONCAT_unique]

with open('rc.pickle', 'wb') as f:
    pickle.dump(REAL_CONCAT_unique_no_duplicates, f)

tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/kobart-QuestionGeneration')
model = BartForConditionalGeneration.from_pretrained('Sehong/kobart-QuestionGeneration')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

with open('selected.pickle', 'wb') as f:
    pickle.dump(selected, f)

contexts_list = []
questions_list = []
answers_list = []
answer_start_indices_list = []
alter = []

for idx, context in tqdm(enumerate(selected)):
    context = re.sub('\s+', ' ', context)
    answers = REAL_CONCAT_unique_no_duplicates[idx]
    modified_contexts = []
    answer_starts = []
    
    for answer in answers:
        tmp = []
        answer_start = context.find(answer)
        if answer_start == -1:
            print(f"Couldn't find answer in context! Answer: {answer}")
            continue
        elif answer_start != (alter_answer_start := context.rfind(answer)):
            print('alternative position scanned!', answer)
            tmp.append(answer)

        
        max_context_length = 80

        
        half_length = max_context_length // 2

        
        start_index = max(0, answer_start - half_length)
        end_index = start_index + max_context_length

        
        modified_context = context[start_index:end_index]

        
        
        end_index = min(end_index, len(context))
        
        
        new_answer_start = answer_start - start_index
        new_answer_start = max(0, new_answer_start)
        
        modified_contexts.append(modified_context)
        answer_starts.append(new_answer_start)
        
        alter.append(tmp)

    contexts_list.append(modified_contexts)
    answers_list.append(answers)
    answer_start_indices_list.append(answer_starts)
    
    combined_texts = [c + ' <unused0> ' + a for c, a in zip(modified_contexts, answers)]
    inputs = tokenizer.batch_encode_plus(combined_texts, truncation=True, padding=True, return_tensors="pt")
    print('Tokenized')

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    
    summary_ids = model.generate(input_ids, max_length=30, attention_mask=attention_mask)

    
    batch_questions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
    
    questions_list.append(batch_questions)

answer_list = []

for answers_sublist, start_indices_sublist in zip(answers_list, answer_start_indices_list):
    merged_sublist = [{'text': answer, 'answer_start': start_index} 
                      for answer, start_index in zip(answers_sublist, start_indices_sublist)]
    answer_list.append(merged_sublist)

filtered_questions_list = []
filtered_answers_list = []
filtered_contexts_list = []

for questions, answers, contexts in zip(questions_list, answer_list, contexts_list):
    for q, a, c in zip(questions, answers, contexts):
        if a['text'] not in q and a['text'] not in STOPWORDS and (q.endswith('?') or q.endswith('가') or q.endswith('나')) and not q.endswith('.'):
            filtered_questions_list.append(q)
            filtered_answers_list.append(a)
            filtered_contexts_list.append(c)

new_dataset = pd.DataFrame({'context':filtered_contexts_list, 'question':filtered_questions_list, 'answers':filtered_answers_list})
new_dataset['answers'] = new_dataset['answers'].apply(lambda x: {k: [v] for k,v in x.items()})
new_dataset.to_csv('cheader.csv', encoding='utf-8-sig')

new_dataset2 = pd.read_csv('cheader.csv', encoding='utf-8-sig')
for idx in tqdm(range(len(new_dataset2['answers']))):
    answer_start = new_dataset2['answers'][idx]['answer_start'][0]
    context = new_dataset2['context'][idx]

    if answer_start == 225:
        adjustment = random.randint(-20, 0)
        new_start = max(0, answer_start + adjustment)
        context = context[-adjustment:450]

        
        new_dataset2['answers'][idx]['answer_start'][0] = new_start
        new_dataset2['context'][idx] = context

include_words = {"것", "어떤", "어느", "무엇", "이유", "인물", "누구", "이름", "사람", "부분"}

filtered_questions = []
filtered_answers = []
filtered_contexts = []

for idx in new_dataset2.index:
    question = new_dataset2.loc[idx, 'question']
    answer = new_dataset2.loc[idx, 'answers']
    context = new_dataset2.loc[idx, 'context']
    
    
    if any(word in question for word in include_words):
        filtered_questions.append(question)
        filtered_answers.append(answer)
        filtered_contexts.append(context)

new_dataset_filtered = pd.DataFrame({
    'question': filtered_questions,
    'answers': filtered_answers,
    'context': filtered_contexts
})

new_dataset2.to_csv('cheader_2.csv', encoding='utf-8-sig')
new_dataset_filtered.to_csv('cheader_filtered.csv', encoding='utf-8-sig')

filtered_questions = []
filtered_answers = []
filtered_contexts = []

for idx in new_dataset2.index:
    question = new_dataset2.loc[idx, 'question']
    answer = new_dataset2.loc[idx, 'answers']
    context = new_dataset2.loc[idx, 'context']
    
    if any(word in question for word in include_words) and len(question) <= 40:
        filtered_questions.append(question)
        filtered_answers.append(answer)
        filtered_contexts.append(context)

new_dataset_short_question_filtered = pd.DataFrame({
    'question': filtered_questions,
    'answers': filtered_answers,
    'context': filtered_contexts
})

new_dataset_short_question_filtered.to_csv('cheader_short_question_filtered.csv', encoding='utf-8-sig')

datasets = load_from_disk('/opt/ml/input/data/test_dataset/')
test_dataset = datasets["validation"].flatten_indices().to_pandas()

most_closest = []

for origin_question in tqdm(test_dataset['question']):
    distances = [(question, lv.distance(origin_question, question)) for question in new_dataset2['question']]
    distances.sort(key=lambda x: x[1])  

    
    for distance_tuple in distances[0:5]:
        most_closest.append(distance_tuple[0])

new_dataset2['answers'] = new_dataset2['answers'].apply(eval)
pseudo_example_set = set(most_closest)
stopwords = ['회사', '이후', '인물', '사용', '나라', '전투', '상대', '국가', '일반', '신체', '정치', '원인', '시기', '말', '사람', '이름', '장소', '시장', '영향', '진행', '시작', '곳', '전쟁', '후', '군사', '대상', '주장', '정부', '해', '역', '때', '존재', '발견', '문제', '지역', '자리', '제작', '소속', '영화', '설립', '발표', '당', '체포', '자신', '연도', '세기', '군', '당시', '대표', '팀', '감독', '내용', '작품', '조직', '결과', '부대', '일', '장면', '유일', '황제', '참가', '이유', '선거', '단체', '처음', '소설', '도움', '사이', '공격', '왕', '사망', '명칭', '변화', '위치', '경우', '활동', '군대', '선수', '물건', '법', '기관', '개혁', '집단', '역할', '배', '기준', '이용', '전', '책', '출간', '생각', '사건', '이전', '결정', '영어']
stopwords_set = set(stopwords)

new_dataset2['answer_text'] = new_dataset2['answers'].apply(lambda x: x['text'][0])

filtered_cheader_2 = new_dataset2[
    (new_dataset2['question'].isin(pseudo_example_set)) & 
    (~new_dataset2['answer_text'].isin(stopwords_set)) &
    (new_dataset2['answer_text'].str.len() > 1)
]

filtered_cheader_2.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'answer_text'], axis = 1, inplace=True)
filtered_cheader_2['id'] = [0] * len(filtered_cheader_2)
filtered_cheader_2.to_csv('cheader_levensheteined.csv', encoding='utf-8-sig')