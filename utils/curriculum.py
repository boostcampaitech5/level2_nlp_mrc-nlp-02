import json
import numpy as np
import pandas as pd
from utils import retriever_metric, retrieval

from datasets import Dataset


class check_question_difficulties():
    def __init__(self, dataset, CFG, training_args, tokenizer, option='SparseTFIDF', verbose=True):
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

        
    def mark_score(self):
        """
        Note:
            입력된 옵션에 따라 self.dataset에 점수를 마킹합니다.
            
        Return:
            정렬된 Dataset or DataFrame
        """
        pass
        # get score

        
        # store and sort
        
        # return
        
    
if __name__ == '__main__':
    pass