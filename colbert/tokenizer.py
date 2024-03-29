# baseline : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-11

import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

def set_columns(dataset):
    dataset = pd.DataFrame(
        {"context": dataset["context"], "query": dataset["question"], "title": dataset["title"]}
    )

    return dataset


def load_tokenizer(MODEL_NAME):
    special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def tokenize_colbert(dataset, tokenizer, corpus):

    # for inference
    if corpus == "query":
        preprocessed_data = []
        for query in dataset:
            preprocessed_data.append("[Q] " + query)

        tokenized_query = tokenizer(
            preprocessed_data, return_tensors="pt", padding='max_length', truncation=True, max_length=64
        )
        mask_token_ids = tokenized_query["input_ids"] == tokenizer.pad_token_id
        tokenized_query["input_ids"] = tokenized_query["input_ids"].where(~mask_token_ids, tokenizer.mask_token_id)
        
        tokenized_query["attention_mask"] = torch.ones_like(tokenized_query["attention_mask"])
        
        if "token_type_ids" in tokenized_query:
            tokenized_query["token_type_ids"] = torch.zeros_like(tokenized_query["token_type_ids"])
        
        return tokenized_query
    
    # for inference
    elif corpus == "pseudo_query":
        preprocessed_data = []
        for document in dataset:
            preprocessed_data.append("[Q] " + document)

        tokenized_query = tokenizer(
            preprocessed_data, return_tensors="pt", padding='max_length', truncation=True, max_length=64
        )
        mask_token_ids = tokenized_query["input_ids"] == tokenizer.pad_token_id
        tokenized_query["input_ids"] = tokenized_query["input_ids"].where(~mask_token_ids, tokenizer.mask_token_id)
        
        tokenized_query["attention_mask"] = torch.ones_like(tokenized_query["attention_mask"])
        
        if "token_type_ids" in tokenized_query:
            tokenized_query["token_type_ids"] = torch.zeros_like(tokenized_query["token_type_ids"])
        
        return tokenized_query

    elif corpus == "doc":
        if type(dataset) == str:
            preprocessed_data = '[D] '+ dataset
            
        elif type(dataset) == list:
            preprocessed_data = []
            for document in dataset:
                preprocessed_data.append("[D] " + document)
        else:
            raise TypeError('doc tokenizer\'s input should be str or list of str')
            
        tokenized_context = tokenizer(
            preprocessed_data, return_tensors="pt", padding="max_length",truncation=True,
        )
        
        # DOC는 token_type_ids 다 1로 줘 보자
        
        if "token_type_ids" in tokenized_context:
            tokenized_context["token_type_ids"] = torch.ones_like(tokenized_context["token_type_ids"])

        return tokenized_context

    elif corpus == "bm25_hard":
        preprocessed_context = []
        for context in dataset:
            preprocessed_context.append("[D] " + context)
        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        return tokenized_context
    # for train
    else:
        preprocessed_query = []
        preprocessed_context = []
        for query, context in zip(dataset["query"], dataset["context"]):
            preprocessed_context.append("[D] " + context)
            preprocessed_query.append("[Q] " + query)
        tokenized_query = tokenizer(
            preprocessed_query, return_tensors="pt", padding='max_length', truncation=True, max_length=64
        )
        
        mask_token_ids = tokenized_query["input_ids"] == tokenizer.pad_token_id
        tokenized_query["input_ids"] = tokenized_query["input_ids"].where(~mask_token_ids, tokenizer.mask_token_id)
        tokenized_query["attention_mask"] = torch.ones_like(tokenized_query["attention_mask"])        
        if "token_type_ids" in tokenized_query:
            tokenized_query["token_type_ids"] = torch.zeros_like(tokenized_query["token_type_ids"])

        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context, tokenized_query