import json
import numpy as np
import pandas as pd
from utils import retriever_metric, retrieval

from datasets import Dataset


class check_question_difficulties():
    def __init__(self, dataset, CFG, training_args, tokenizer, verbose=True):
        """
        Note:
            - BM25, TFIDF, 학습시킨 특정 Retriever 모델
        Args:
            dataset: 학습 데이터세트
            option:
        """
        
        self.dataset = dataset
        self.CFG = CFG
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.verbose = verbose
        
        if isinstance(self.dataset, Dataset):
            self.input_type = 'Dataset'    
        elif isinstance(self.dataset, pd.DataFrame):
            self.input_type = 'DataFrame'
        
        if self.verbose:
            print("\nCirriculum Learning option is True. Train data would be sorted.")
        
    def mark_score_with_retriever(self, option='SparseTFIDF'):
        """
        Note:
            - Reader의 입장이 아닌, Retriever의 입장에서 query를 가지고 얼마나 쉬운지/어려운지를 마킹합니다(기준=NDCG score)
            - 입력된 옵션에 따라 self.dataset에 점수를 마킹합니다.
            
        Return:
            정렬된 Dataset or DataFrame
        """
        if 'bm25' in option or 'BM25' in option:
            self.option = "SparseBM25"
        elif 'TFIDF' in option  or 'tfidf' in option:
            self.option = "SparseTFIDF"
        
        assert self.option == "SparseBM25" or self.option == "SparseTFIDF", print("구현준비중")
        # 만약 기존 모델을 사용한다고 하면, 이미 학습되어 있는 편이 편하다. 아니라면, 에러를 만드는 게 낫다.
        # 에러를 어떻게 만들지?
        # colbert는 에러를 도출해서 괜찮은데, dense는 그냥 학습시켜버린다. get_embedding에서...
        
        # get score
        retrieval_class = eval(f"retrieval.{self.option}")
        if self.verbose:
            print(f"Retriever class: retrieval.{self.option}")
        
        retriever = retrieval_class(CFG=self.CFG, training_args=self.training_args, tokenize_fn=self.tokenizer.tokenize)
        retriever.get_embedding()
        retrieved_docs = retriever.retrieve(self.dataset, topk=self.CFG['option']['top_k_retrieval'])
        
        metric_train = retriever_metric.score_retrieved_docs(dataset=self.dataset, 
                                                            topk_docs=retrieved_docs, 
                                                            mean='context', 
                                                            metric='ALL',
                                                            return_type='list')
        mrr_score, ndcg_score = metric_train.test()
        
        # store and sort
        new_mrr_name = f"{self.option}" + "_mrr_score"
        new_ndcg_name = f"{self.option}" + "_ndcg_score"
        
        if self.input_type == 'Dataset':
            self.dataset = self.dataset.add_column(new_mrr_name, mrr_score)
            self.dataset = self.dataset.add_column(new_ndcg_name, ndcg_score)
            
            self.dataset = self.dataset.sort(new_ndcg_name, reverse=True)

        else:
            self.dataset[new_mrr_name] = mrr_score
            self.dataset[new_ndcg_name] = ndcg_score
            
            self.dataset = self.dataset.sort_values(new_ndcg_name, ascending=False)
                      
        if self.verbose:
            print(f"Train data is sorted by {self.option} NDCG score... FINISHED!\n")
            
        return self.dataset

    def mark_score_with_f1(self, model_path):
        
        pass


if __name__ == '__main__':
    pass