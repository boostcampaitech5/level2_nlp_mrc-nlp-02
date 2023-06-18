import json
import os
import pickle
import time
import torch
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from fuzzywuzzy import fuzz
from colbert.model import *
from colbert.tokenizer import *
from transformers import AutoConfig, AutoTokenizer
from itertools import zip_longest


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
        with open(f"input/data/{context_path}", "r", encoding="utf-8") as f:
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
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                여러 Queries를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            p_embedding을 따로 저장하지 않기 때문에 TF-IDF보다 비교적 많은 시간이 소요됩니다.
        """

        with timer("transform"):
            tokenized_queries = [self.tokenize_fn(query) for query in queries]
        with timer("query ex search"):
            result = np.array([self.bm25.get_scores(q) for q in tqdm(tokenized_queries)])
        
        
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

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