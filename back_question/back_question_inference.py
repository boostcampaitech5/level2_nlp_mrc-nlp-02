import os
import pandas as pd
import pickle
import Levenshtein
import numpy as np
import re
import torch
import json

from tqdm.auto import tqdm
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from itertools import zip_longest

from datasets import load_from_disk

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

import sys
sys.path.insert(0, '/opt/ml/')

from colbert.tokenizer import *
from colbert.model import *

from rank_bm25 import BM25Okapi

def extract_numbers(s):
    return tuple(int(num) for num in re.findall(r'\d+', s))

bin_dir = './'

filename_list = [f for f in os.listdir(bin_dir) if f.endswith('.bin')]

filename_list.sort(key=extract_numbers)

print(filename_list)

questions_list = []

for filename in filename_list:
    with open(os.path.join(bin_dir, filename), 'rb') as f:
        questions_list.extend(pickle.load(f))

questions = pd.Series(questions_list)
questions.name = 'questions'
print(len(questions))

origin_dataset = pd.read_csv('/opt/ml/input/data/preprocessed_ner.csv', encoding='utf-8-sig')
dataset = origin_dataset[['context', 'answer', 'answer_start']]
dataset = dataset[dataset['answer'].notna()]

del origin_dataset

print(len(dataset))

if len(dataset) == len(questions):
    dataset = dataset.reset_index(drop=True)
    questions = questions.reset_index(drop=True)
    PRED = pd.concat([dataset, questions], axis=1, ignore_index=True)
    PRED.columns = ['context', 'answer', 'answer_start', 'question']
else:
    print("The number of rows in 'dataset' and 'questions' are not the same. Cannot concatenate.")
    
del dataset
del questions

GT = load_from_disk('/opt/ml/input/data/test_dataset')['validation'].flatten_indices().to_pandas()
GT['TFIDF-matched'] = [[]] * len(GT)

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

def custom_tokenize(text):
    
    return tokenizer.tokenize(text)

print('vectorizing')
vectorizer = TfidfVectorizer(tokenizer=custom_tokenize).fit(PRED['question'].tolist() + GT['question'].tolist())
print('pred calculating')
pred_tfidf = vectorizer.transform(PRED['question'].tolist())
print('gt calculating')
gt_tfidf = vectorizer.transform(GT['question'].tolist())

GT['TFIDF-matched'] = pd.Series(dtype=object)

for gt_idx in tqdm(range(gt_tfidf.shape[0])):
    cosine_similarities = cosine_similarity(gt_tfidf[gt_idx:gt_idx+1], pred_tfidf).flatten()

    
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    matched_indices = []
    for idx in sorted_indices:
        
        if PRED['question'].iloc[idx] not in PRED['question'].iloc[matched_indices]:
            matched_indices.append(idx)
        if len(matched_indices) == 10:  
            break

    
    GT.at[gt_idx, 'TFIDF-matched'] = matched_indices

    if (gt_idx + 1) % 10 == 0:
        print(f'Processed {gt_idx + 1} questions')

GT['BM25-matched'] = [[]] * len(GT)
bm25 = BM25Okapi(PRED['question'].tolist() + GT['question'].tolist(), tokenizer=custom_tokenize)
    
GT.to_csv('cheader.csv', encoding='utf-8-sig')
PRED.to_csv('cheader_sheet.csv', encoding='utf-8-sig')
       
