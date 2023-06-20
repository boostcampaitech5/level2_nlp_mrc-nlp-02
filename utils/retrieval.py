import os
import re
import json
import yaml
import time
import faiss
import torch
import utils
import wandb
import pickle
import itertools
import Levenshtein
import numpy as np
import pandas as pd
import torch.nn.functional as F

from utils import utils
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from colbert.model import *
from colbert.tokenizer import *
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset, concatenate_datasets, load_from_disk
from itertools import zip_longest
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    TrainingArguments,
)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

    
class BaseRetrieval:
    def __init__(
        self,
        CFG,
        training_args,
        tokenize_fn,
        data_path: Optional[str] = "retrieval/",
        context_path: Optional[str] = "wikipedia_documents.json"
    ) -> None:
        
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러옵니다.
        """
        self.CFG = CFG
        self.tokenize_fn = tokenize_fn        
        
        self.data_path = data_path
        with open(f"/opt/ml/input/data/{context_path}", "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
    def get_embedding(self) -> None:
        raise NotImplementedError
    
    def build_faiss(self, num_clusters=64) -> None:
        raise NotImplementedError
    
    def retrieve(
        self, query_dataset: Dataset, topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_dataset (Dataset):
                여러 Query를 포함한 HF.Dataset을 받습니다.
                `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            pd.DataFrame: [description]

        Note:
            Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
            Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            
            use_fuzz: 레벤슈타인 거리 기반 유사도 계산을 통해, 
                      이미 유사한 Passage가 더 높은 점수로 retrieve 되었다면 가져오지 않도록 filtering
        """

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        alpha = 2
        with timer("query exhaustive search"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_dataset["question"], k=max(40 + topk, alpha * topk) if self.CFG['option']['use_fuzz'] else topk
            )
        for idx, example in enumerate(
            tqdm(query_dataset, desc="Ranking...: ")
        ):
            if self.CFG['option']['use_fuzz']:
                doc_scores_topk = [doc_scores[idx][0]]
                doc_indices_topk = [doc_indices[idx][0]]

                pointer = 1

                while len(doc_indices_topk) != topk:
                    is_non_duplicate = True
                    new_text_idx = doc_indices[idx][pointer]
                    new_text = self.contexts[new_text_idx]

                    for d_id in doc_indices_topk:
                        if fuzz.ratio(self.contexts[d_id], new_text) > 85:
                            is_non_duplicate = False
                            break

                    if is_non_duplicate:
                        doc_scores_topk.append(doc_scores[idx][pointer])
                        doc_indices_topk.append(new_text_idx)

                    pointer += 1

                    if pointer == max(40 + topk, alpha * topk):
                        break

                assert len(doc_indices_topk) == topk, "중복 없는 topk 추출을 위해 alpha 값을 증가시켜 주세요."
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context": " ".join(
                    [self.contexts[pid] for pid in (doc_indices_topk if self.CFG['option']['use_fuzz'] else doc_indices[idx])]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
                tmp['context_for_metric'] = [self.contexts[pid] for pid in (doc_indices_topk if self.CFG['option']['use_fuzz'] else doc_indices[idx])]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas
    
    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        raise NotImplementedError

class SparseTFIDF(BaseRetrieval):
    def __init__(
        self,
        CFG,
        training_args,
        tokenize_fn,
        data_path: Optional[str] = "retrieval/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        
        super().__init__(CFG, training_args, tokenize_fn, data_path, context_path)
        
        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2)#, max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다

    def get_embedding(self) -> None:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding_INF.bin"
        tfidfv_name = f"tfidv_INF.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                여러 Queries를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


class SparseBM25(BaseRetrieval):
    def __init__(
        self,
        CFG,
        training_args,
        tokenize_fn,
        data_path: Optional[str] = "retrieval/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(CFG, training_args, tokenize_fn, data_path, context_path)

    def get_embedding(self) -> None:

        """
        Summary:
            BM25Okapi를 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"bm25_embedding.bin"
        bm25_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("BM25 pickle load.")
        else:
            print("Build passage embedding")
            self.bm25 = BM25Okapi(self.contexts, tokenizer=self.tokenize_fn)
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("BM25 pickle saved.")

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1, for_train=False,
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                여러 Queries를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            p_embedding을 따로 저장하지 않기 때문에 TF-IDF보다 비교적 많은 시간이 소요됩니다.
            for_train 인자를 넘기면 저장할 수 있도록 기능을 추가하였습니다.
        """
        if for_train:       # dense embedding에서 호출하는 경우엔 무조건 train에 대해서만 호출한다 -> 저장된 파일을 save and load
            bm25_docs_name = "bm25_relevant_docs_for_DenseEmbedding.npy"
            bm25_docs_path = os.path.join(self.data_path, bm25_docs_name)
            print("BM25 is now used for 'Dense Embedding' training.")
            
            if os.path.isfile(bm25_docs_path):
                result = np.load(bm25_docs_path)
                print("BM25 top docs loaded!")
            
                assert result.shape[0] != 600 and result.shape[0] > 3900, "불러오려는 파일은 train에 대해 topk가 저장된 npy입니다.\n저장된 npy 파일dms 현재 valid 에 대한 파일입니다."
            
            else:
                # self.get_embedding()
                with timer("transform"):
                    tokenized_queries = [self.tokenize_fn(query) for query in queries]
                with timer("query ex search"):
                    result = np.array([self.bm25.get_scores(q) for q in tqdm(tokenized_queries, desc='SparseBM25, query에 대한 topk docs search')])
                np.save(bm25_docs_path, result)
                print("BM25 top docs saved!")
            
        else:               # dense embedding에서 호출하는 경우 외에는 변동적이어서, 저장하지 않고 진행한다.
            # self.get_embedding()
            with timer("transform"):
                tokenized_queries = [self.tokenize_fn(query) for query in queries]
            with timer("query ex search"):
                result = np.array([self.bm25.get_scores(q) for q in tqdm(tokenized_queries, desc='SparseBM25 query ex search')])
            
            
            np.save(bm25_docs_path, result)
            print("BM25 top docs saved!")

        doc_scores = []
        doc_indices = []
        for i in tqdm(range(result.shape[0]), desc='SparseBM25 get relevant doc bulk...'):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


class SparseBM25_edited(SparseBM25):
    """
    Note: BM25를 상속받으면서, 함수를 추가. 내가 gold context를 넘기면, gold가 아닌 것들로 topk를 가져오는 함수
    """
    
    def __init__(
        self,
        CFG,
        training_args,
        tokenize_fn,
        data_path: Optional[str] = "retrieval/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(CFG, training_args, tokenize_fn, data_path, context_path)
        
    def retrieve_except_gold(
       self, query_dataset: Dataset, topk: Optional[int] = 1
    ) -> List:
        """
        Note:
            topk를 가져오면 거기에 positive가 있을 수도, 없을 수도 있다. 
            topk 개만 온전히 가져가려면, 최소 topk+1개를 애초에 retrieve해야한다.
        Args:
            query_dataset: train_dataset == self.datset
        Returns:
        """
        # self.num_neg = self.num_neg + topk
        result = []
        alpha = 2
        with timer("query exhaustive search"):
            _, doc_indices = self.get_relevant_doc_bulk(
                query_dataset["question"], k=max(40 + topk, alpha * topk) if self.CFG['option']['use_fuzz'] else topk+5, for_train=True,
            )
        
        for idx, example in enumerate(
            tqdm(query_dataset, desc="Sparse retrieval except gold context: ")
        ):
            except_gold = []
            for pid in doc_indices[idx]:
                dist = Levenshtein.distance(example['context'], self.contexts[pid])
            
                if dist > min(len(example['context']), len(self.contexts[pid]))//10*2:
                    except_gold.append(self.contexts[pid])
            #test_result.extend1([[self.contexts[pid]] for pid in doc_indices[idx] if gold_context[idx] != self.contexts[pid]])
            except_gold = except_gold[:topk]
            if len(except_gold) < topk:
                print("not enough topk")
            result.append(except_gold)
        
        return result
      
      
class DenseColBERT(BaseRetrieval):
    def __init__(
        self,
        CFG,
        training_args,
        tokenize_fn,
        data_path: Optional[str] = "retrieval/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(CFG, training_args, tokenize_fn, data_path, context_path)

    def get_embedding(self) -> None:

        """
        Summary:
            Q와 D를 embedding할 기학습된 ColBERT 모델을 불러옵니다.
            만약 미리 저장된 파일이 없다면 학습을 먼저 시켜야 합니다.
        """
        MODEL = "klue/bert-base"
        # Pickle을 저장합니다.
        model_name = self.CFG['colbert_model_name']
        colbert_path = os.path.join('/opt/ml/colbert/best_model', model_name)

        if os.path.isfile(colbert_path):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model_config = AutoConfig.from_pretrained(MODEL)
            special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
            self.ret_tokenizer = AutoTokenizer.from_pretrained(MODEL)
            self.ret_tokenizer.add_special_tokens(special_tokens)
            self.model = ColbertModel.from_pretrained(MODEL)
            self.model.resize_token_embeddings(self.ret_tokenizer.vocab_size + 2)

            self.model.to(device)

            self.model.load_state_dict(torch.load(colbert_path))
            print('colbert model loaded')
            
            with torch.no_grad():
                self.model.eval()

                print("Start passage embedding.. ....")
                self.batched_p_embs = []
                P_BATCH_SIZE = 128
                # Define a generator for iterating in chunks
                def chunks(iterable, n, fillvalue=None):
                    args = [iter(iterable)] * n
                    return zip_longest(*args, fillvalue=fillvalue)

                for step, batch in enumerate(tqdm(chunks(self.contexts, P_BATCH_SIZE), total=len(self.contexts)//P_BATCH_SIZE)):
                    # The last batch can contain `None` values if the length of `context` is not divisible by 128
                    batch = [b for b in batch if b is not None]

                    # Tokenize the entire batch at once
                    p = tokenize_colbert(batch, self.ret_tokenizer, corpus="doc").to("cuda")
                    p_emb = self.model.doc(**p).to("cpu").numpy()
                    self.batched_p_embs.append(p_emb)
                
                print('p_embs_n_batches: ', len(self.batched_p_embs))
                print('in_batch_size:\n', self.batched_p_embs[0].shape)
            
        else:
            raise NameError('there is no colbert model in', str(colbert_path))

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                여러 Queries를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """
        
        with timer("transform"):
            with torch.no_grad():
                self.model.eval()
                q_seqs_val = tokenize_colbert(queries, self.ret_tokenizer, corpus="query").to("cuda")
                q_emb = self.model.query(**q_seqs_val).to("cpu")
                print('q_emb_size: \n', q_emb.size())
        with timer("query ex search"):
            dot_prod_scores = self.model.get_score(q_emb, self.batched_p_embs, eval=True)
            print(dot_prod_scores.size())

            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
            print(dot_prod_scores)
            print(rank)
            print(rank.size())
            
        return dot_prod_scores[:,:k].tolist(), rank[:,:k].tolist()

      
class DenseRetrieval(BaseRetrieval):
    def __init__(
        self,
        CFG,
        training_args,
        tokenize_fn,
        num_neg: Optional[int] = 2, # for way 2
        # num_neg: Optional[int] = 3, # for way 1
        data_path: Optional[str] = "/opt/ml/retrieval",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(CFG, training_args, tokenize_fn, data_path, context_path)
        
        self.args = TrainingArguments(
            output_dir="dense_retrieval",
            evaluation_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            # per_device_train_batch_size=6,
            # per_device_eval_batch_size=6,
            num_train_epochs=20,
            weight_decay=0.01,
            gradient_accumulation_steps=16,
            # gradient_accumulation_steps=21,
            fp16=True,      # False 버전은 구현 중에 있음. 
        )
        self.CFG = CFG
        self.training_args = training_args
        # self.model_name = CFG['model']['model_name']
        self.model_name = 'klue/bert-base' # 'klue/roberta-base'
        self.data_dir = "/opt/ml/input/data/train_dataset/"
        self.dataset = load_from_disk(self.data_dir+"train")
        self.valid_dataset = load_from_disk(self.data_dir+"validation")
        self.num_neg = num_neg
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.p_encoder = BertEncoder.from_pretrained(self.model_name).to(self.args.device)
        self.q_encoder = BertEncoder.from_pretrained(self.model_name).to(self.args.device)
        
        wiki_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            wiki_seqs['input_ids'], wiki_seqs['attention_mask'], wiki_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)
    
    ###### 새롭게 추가한 부분
    def in_batch_train(self, args=None, CFG=None):
        """
        Note: inbatch -> bm25 neg를 여기서 만들어서 추가한다.
        """
        
        if args is None:
            args = self.args
            
        ## bm25 negs
        self.bm25_topk = self.num_neg+1
        bm25_neg = []
        # 1. gold context: (전체)
        # gold_context = [context for context in dataset['context']]
        # valid_dataset = load_from_disk("/opt/ml/input/data/train_dataset/")['validation']
        
        # 2. retrieve not gold but high bm25 score docs
        bm_25_neg = SparseBM25_edited(self.CFG, self.training_args, tokenize_fn=self.tokenizer.tokenize)
        bm_25_neg.get_embedding()
        not_gold_context = bm_25_neg.retrieve_except_gold(self.dataset, topk=self.bm25_topk)    # list (전체, topk)        # bm25개수가 num_neg+1과 다를 때 추가 수정이 필요함.

        # 3. extend
        for idx, rows in enumerate(not_gold_context):
            # p_with_neg[idx].extend(not_gold_context[idx])   # (whole corpus, (1+num_neg+bm25_topk))
            # bm25_neg.append(not_gold_context[idx])            # (whole corpus, (1+num_neg+bm25_topk))
            bm25_neg.extend(not_gold_context[idx])# 1차원 버전
        
        q_seqs = self.tokenizer(self.dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = self.tokenizer(self.dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        bm25_seqs = self.tokenizer(bm25_neg, padding="max_length", truncation=True, return_tensors='pt')
        
        max_len = p_seqs['input_ids'].size(-1)
        
        bm25_seqs['input_ids'] = bm25_seqs['input_ids'].view(-1, self.bm25_topk, max_len)
        bm25_seqs['attention_mask'] = bm25_seqs['attention_mask'].view(-1, self.bm25_topk, max_len)
        bm25_seqs['token_type_ids'] = bm25_seqs['token_type_ids'].view(-1, self.bm25_topk, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],            # (3952, max_len)
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'],
            bm25_seqs['input_ids'], bm25_seqs['attention_mask'], bm25_seqs['token_type_ids'],   # (3952, 3, max_len)
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, drop_last=True)

        print("in batch dataset setting done, start training")
        ### 학습 시작
        
        _name = CFG['실험명']
        wandb.init(name=_name+'_dense_embedding', project=CFG['wandb']['project'], 
            entity=CFG['wandb']['id'])
        
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        print(f"t total is {t_total}")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=90, num_training_steps=t_total)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
        
        if args.fp16:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # Start training!
        global_step = 0
        loss_accumulate = 0.0
        
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    global_step += 1
                    
                    self.p_encoder.train()
                    self.q_encoder.train()


                    # input:(B, max_len)
                    p_inputs = {
                        'input_ids': batch[0].to(args.device),          # (B, max_len)
                        'attention_mask': batch[1].to(args.device),
                        'token_type_ids': batch[2].to(args.device)
                    }
                    #                
                    q_inputs = {    
                        'input_ids': batch[3].to(args.device),          # (B, max_len)
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
                    # (B*(num_neg), max_len=512)
                    bm25_inputs = {
                        'input_ids': batch[6].view(batch_size * self.bm25_topk, -1).to(args.device),
                        'attention_mask': batch[7].view(batch_size * self.bm25_topk, -1).to(args.device),
                        'token_type_ids': batch[8].view(batch_size * self.bm25_topk, -1).to(args.device)
                    }

                    if args.fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                            p_outputs = self.p_encoder(**p_inputs)          # (batch_size, emb_dim) -> (B, H). BERTmodel의 pooler_output임을 기억하자.
                            q_outputs = self.q_encoder(**q_inputs)          # (batch_size, emb_dim) -> (B, H)
                            bm25_outputs = self.p_encoder(**bm25_inputs)    # (batch_size*(self.bm25_topk), emb_dim)   -> (B*(N+1), H)

                            # Calculate similarity score & loss
                            p_outputs = torch.cat([p_outputs, bm25_outputs], dim=0)     # (B+B*(N+1), H)
                            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (B, H) x (H, B+B*(N+1)) = (batch_size, B+B*(N+1)). B=8이면 8+24 = 32

                            # 먼저 inbatch 로 loss 가 얼마나 떨어지고, MRR이 ㄹ얼마나 나오는지 해보자.... inbatch, B15로 하니까 1epoch 1.5분.
                            ###
                            # target: position of positive samples = diagonal element 
                            targets = torch.arange(0, args.per_device_train_batch_size).long()
                            if torch.cuda.is_available():
                                targets = targets.to('cuda')

                            sim_scores = F.log_softmax(sim_scores, dim=1)

                            loss = F.nll_loss(sim_scores, targets)
                                                       
                        # gradient accumulation을 수행하도록 loss만 축적합니다.
                        loss = loss / args.gradient_accumulation_steps
                        tepoch.set_postfix(loss=f'{str(loss.item())}')
                        
                        scaler.scale(loss).backward()
                        loss_accumulate += loss.item()
                        # scheduler.step()
                    
                    if global_step % args.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                        # optimizer.step()
                        scheduler.step()
                        scaler.update()
                        
                        self.p_encoder.zero_grad()
                        self.q_encoder.zero_grad()
                        # 
                        torch.cuda.empty_cache()
                        
                    # if global_step % (args.gradient_accumulation_steps*10) == 0:
                        wandb.log({"train/loss": loss_accumulate/args.gradient_accumulation_steps, # "train/learning_rate": args.learning_rate})
                                   "train/learning_rate": optimizer.param_groups[0]['lr']})
                        loss_accumulate = 0.0

            del p_inputs, q_inputs
        
        wandb.finish()
        
        
        
        
    #### 새롭게 추가한부분

    
        
    def prepare_negative(self, dataset=None, num_neg=2, tokenizer=None, add_bm25=False):
        """
        Note:
            Dense Embedding을 학습하기 위한 데이터세트를 만듭니다. (In-batch가 아닌, 임의로 랜덤 negs를 골라오는 방식.)
            
            num_neg 만큼 전체 corpus에서 랜덤으로 negatives를 뽑아옵니다.
            add_bm25가 켜져있다면, bm25 점수가 높은 negatives를 topk=bm25_topk 개 만큼 뽑아옵니다.
            p_with_neg는 pos + Rand Neg passage를 가지고 있는 1차원 리스트.
            bm25_neg는 topk개를 의 Neg passage를 가지고 있는 1차원 리스트.
        Args:
            dataset: train_dataset
            num_neg
            tokenizer
            add_bm25
            
        Return: None
        
        """
        
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            passage_candidatates = []
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)                  # 1차원으로 만들 때 사용 -> ((1+num_neg)*train 길이)
                    # passage_candidatates.append(c)
                    # passage_candidatates.extend(p_neg)
                    break   
            # p_with_neg.append(passage_candidatates)             # (whole corpus, (1+num_neg))
        
        self.bm25_topk = num_neg+1
        bm25_neg = []
        if add_bm25:
            # 1. gold context: (전체)
            # gold_context = [context for context in dataset['context']]
            # valid_dataset = load_from_disk("/opt/ml/input/data/train_dataset/")['validation']
            
            # 2. retrieve not gold but high bm25 score docs
            bm_25_neg = SparseBM25_edited(self.CFG, self.training_args, tokenize_fn=tokenizer.tokenize)
            bm_25_neg.get_embedding()
            not_gold_context = bm_25_neg.retrieve_except_gold(dataset, topk=self.bm25_topk)    # list (전체, topk)        # bm25개수가 num_neg+1과 다를 때 추가 수정이 필요함.
        
            # 3. extend
            for idx, rows in enumerate(not_gold_context):
                # p_with_neg[idx].extend(not_gold_context[idx])   # (whole corpus, (1+num_neg+bm25_topk))
                # bm25_neg.append(not_gold_context[idx])            # (whole corpus, (1+num_neg+bm25_topk))
                bm25_neg.extend(not_gold_context[idx])# 1차원 버전
            
            
            # 4. bm25를 추가하면 self.num_neg가 두 배가 된다.
            # self.num_neg = self.num_neg + num_neg
            # num_neg = self.num_neg

        # 2차원 -> 1차원으로 변경(for tokenize)
        # p_with_neg = list(itertools.chain(*p_with_neg))   # 이미 1차원일 경우에는 off
        # bm25_neg = list(itertools.chain(*bm25_neg))
        
        # (Question, Passage) 데이터셋
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')
        bm25_seqs = tokenizer(bm25_neg, padding="max_length", truncation=True, return_tensors='pt')
        
        max_len = p_seqs['input_ids'].size(-1)

        # p_seqs org size (C*(num_neg+1), tokenizer_max_length) -> (C, (num_neg+1), tokenizer_max_length)
        # q_seqs size (C, tokenizer_max_length=512)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        bm25_seqs['input_ids'] = bm25_seqs['input_ids'].view(-1, self.bm25_topk, max_len)
        bm25_seqs['attention_mask'] = bm25_seqs['attention_mask'].view(-1, self.bm25_topk, max_len)
        bm25_seqs['token_type_ids'] = bm25_seqs['token_type_ids'].view(-1, self.bm25_topk, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],            # 각각은 결국 (C, num_neg, max_len) 이다.   ->
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'],
            bm25_seqs['input_ids'], bm25_seqs['attention_mask'], bm25_seqs['token_type_ids'],
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size, drop_last=True)

    def get_embedding(self) -> None:
        """
        Summary:
            Passage Embedding을 만들고(train)
            Dense Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        
        p_pickle_name = "0620test8_p_dense_embedding_randneg_bm25neg_" + f"B{(self.num_neg+1)*self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps}.pth"
        q_pickle_name = "0620test8_q_dense_embedding_randneg_bm25neg_"+ f"B{(self.num_neg+1)*self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps}.pth"
        p_emb_path = os.path.join(self.data_path, p_pickle_name)
        q_emb_path = os.path.join(self.data_path, q_pickle_name)
        
        if os.path.isfile(p_emb_path) and os.path.isfile(q_emb_path):
            self.p_encoder.load_state_dict(torch.load(p_emb_path))
            self.q_encoder.load_state_dict(torch.load(q_emb_path))
            
            print("Embedding model load.")
        
        else:
            print("\nEmbeddings are not detected!! Prepare Negatives in batch...")
            print(f"Training with this data:\n{self.dataset}\n")
            print(f"inbatch try!!")
            # self.prepare_negative(self.dataset, self.num_neg, self.tokenizer, add_bm25=True)
            print("Build dense embedding...")
            # self.train(CFG=self.CFG)
            
            # way 2.
            self.in_batch_train(CFG=self.CFG)
            print("training done")
            
            torch.save(self.p_encoder.state_dict(), p_emb_path)
            torch.save(self.q_encoder.state_dict(), q_emb_path)
            print("Embedding model saved.")
            
            
        # passage 에 대해 미리 만들어둬보자...
        with torch.no_grad():
            self.p_encoder.eval()

            passage_embs = []
            for batch in tqdm(self.passage_dataloader, desc='DenseRetrieval, 전체 문서에 대한 embedding 계산 중'):
                batch = tuple(t.to(self.args.device) for t in batch)
                
                passage_inputs = {
                    'input_ids': batch[0],          # (B, max_len)
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                if self.args.fp16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        passage_emb = self.p_encoder(**passage_inputs).to('cpu')
                passage_embs.append(passage_emb)    # [(batch, H), (batch, H), ... , (last_batch, H)]
            
            stacked = torch.cat(passage_embs, dim=0).to(self.args.device)    # (num_passage=whole_corpus, emb_dim)
            
        self.passage_embs = stacked

    def train(self, args=None, CFG=None):
        _name = CFG['실험명']
        wandb.init(name=_name+'_dense_embedding', project=CFG['wandb']['project'], 
            entity=CFG['wandb']['id'])
        
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=t_total)

        if self.args.fp16:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # Start training!
        global_step = 0
        loss_accumulate = 0.0
        
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    global_step += 1
                    
                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    # input:(B, num_neg+1, max_len) -> (B*(num_neg+1), max_len=512)
                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
                    # (B, max_len=512)                    
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
                    # (B*(num_neg), max_len=512)
                    bm25_inputs = {
                        'input_ids': batch[6].view(batch_size * self.bm25_topk, -1).to(args.device),
                        'attention_mask': batch[7].view(batch_size * self.bm25_topk, -1).to(args.device),
                        'token_type_ids': batch[8].view(batch_size * self.bm25_topk, -1).to(args.device)
                    }
                    
                    if self.args.fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                            p_outputs = self.p_encoder(**p_inputs)          # (batch_size*(num_neg+1), emb_dim) -> (B*(N+1), H). BERTmodel의 pooler_output임을 기억하자.
                            q_outputs = self.q_encoder(**q_inputs)          # (batch_size*, emb_dim)            -> (B, H)
                            bm25_outputs = self.p_encoder(**bm25_inputs)    # (batch_size*(self.bm25_topk), emb_dim)   -> (B*(N+1), H)

                            # Calculate similarity score & loss
                            p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)                   # (B, N+1, H)
                            q_outputs = q_outputs.view(batch_size, 1, -1)                                  # (B, 1, H)
                            bm25_outputs = bm25_outputs.view(batch_size, self.bm25_topk, -1)               # (B, N+1, H)
                            p_outputs = torch.cat([p_outputs, bm25_outputs], dim=1)                        # (batch_size, 1+RandNeg+bm25Neg, H)
                            
                            sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  # (batch_size, 1+RandNeg+bm25Neg)
                            sim_scores = sim_scores.view(batch_size, -1)
                            sim_scores = F.log_softmax(sim_scores, dim=1)
                            
                            loss = F.nll_loss(sim_scores, targets)
                            tepoch.set_postfix(loss=f'{str(loss.item())}')
                            
                        # gradient accumulation을 수행하도록 loss만 축적합니다.
                        loss = loss / args.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                        loss_accumulate += loss.item()
                    
                    if global_step % args.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scheduler.step()
                        scaler.update()
                        
                        self.p_encoder.zero_grad()
                        self.q_encoder.zero_grad()
                        torch.cuda.empty_cache()
                        
                        wandb.log({"train/loss": loss_accumulate/args.gradient_accumulation_steps, # "train/learning_rate": args.learning_rate})
                                   "train/learning_rate": optimizer.param_groups[0]['lr']})
                        loss_accumulate = 0.0

                    del p_inputs, q_inputs
        
        wandb.finish()

    def get_relevant_doc_bulk(self, queries, k):
        """
        Note:
            전체 wiki 안에서 quries에 가장 relevant한 문서들의 유사도 점수와 idx를 topk 개 반환합니다.
        Args:
            queries:    유사한 문장을 확인하고 싶은 queries. List.
            k:          topk 개
        Return:
            topk score(2d List), topk docs idx(2d List)
        """
        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()
            
            queries_tok = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(self.args.device)
            if self.args.fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    queries_emb = self.q_encoder(**queries_tok).to(self.args.device)
                    
            # breakpoint()
            # stacked = torch.stack(passage_embs, dim=0).view(len(self.passage_dataloader.dataset), -1).to(self.args.device)  # (num_passage, emb_dim)
            # stacked = torch.cat(passage_embs, dim=0).to(self.args.device)    # (num_passage=whole_corpus, emb_dim)
            
            dot_prod_scores = torch.matmul(queries_emb, torch.transpose(self.passage_embs, 0, 1))  # (num_queries, num_passage)       # 현재 모든 score가 242.3750이다.
            # breakpoint()
            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze() # (num_queries, num_passage)
            top_k_docs_scores = torch.gather(dot_prod_scores, dim=1, index=rank)[:, :k]
            top_k_docs_indices = rank[:, :k]
        
        return top_k_docs_scores, top_k_docs_indices
    
    def get_relevant_doc(self, query, k=1):
        """
        Note:
        
        Args:
        
        Return:
        
        """
        
        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(self.args.device)
            q_emb = self.q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(self.args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = self.p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return dot_prod_scores[:k], rank[:k]

      
class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output
