import re
import numpy as np

class score_retrieved_docs():
    """
    TFIDF, BM25, RanNeg 등으로 같이 학습하여 retrieved 한 docs들의 점수를 평가하는 class 선언.
    """
    def __init__(self, dataset, topk_docs, mean='context', metric='ALL'):
        self.dataset = dataset
        self.answers = self.dataset['answer']
        self.gold_contexts = self.dataset['context']
        self.topk_docs = topk_docs
        self.c_ro_a = self.gold_contexts if mean=='context' else self.answers
        self.mean = mean
        
        self.test(metric)
     
    def mean_reciprocal_rank(contexts_or_answers, topk_docs, mean='context'):
        """
        Note:
            topk개의 docs들로부터 context/answer 기준으로 MRR을 계산합니다(context/answer 가 있는 context가 몇 번째에 있는지?)
        Args:
            - contexts_or_answers: 
                context 일 경우: self.gold_context와 동일하다.
                answer 일 경우: [쿼리1 정답, 쿼리2 정답, ...], shape (쿼리 개수)
            - docs: [[쿼리1 후보 context top k개], [쿼리1 후보 context top k개], ...], shape (쿼리 개수, topk)
        Return:
            MRR metric 값
        """
        
        mrr_value = 0.0

        for idx, c_or_a in enumerate(contexts_or_answers):
            for rank, doc in enumerate(topk_docs[idx]):
                if mean == 'context':
                    # topk개의 docs들 중에서 answer를 포함하고 있는 docs가 몇 번째에 등장하는지를 계산합니다.
                    if any(re.search(c_or_a, doc)):
                        mrr_value += 1/(rank+1)
                        break
                else:
                    # topk개의 docs들 중에서 원래 query-context 쌍이었던 context가 docs에서 몇 번째에 등장하는지를 계산합니다.
                    if c_or_a == doc:
                        mrr_value += 1/(rank+1)
                        break
        
        return mrr_value

    def get_relevance_score(answers, gold_contexts, topk_docs):
        """
        Note:
            DCG, NDCG 스코어를 위해 retrieved docs의 관련도 점수를 리턴합니다.
            relevance: answer 보유=1, gold_context=3
            
        Arg:
        
        Return:
        
        """
        
        relevance_score = []
        
        for idx, answer in enumerate(answers):
            relevance_idx = []
            gold_context = gold_contexts[idx]
            
            for rank, doc in enumerate(topk_docs[idx]):
                if doc == gold_context:
                    rscore = 3
                elif any(re.search(answer, doc)):
                    rscore = 1
                else:
                    rscore = 0
                relevance_idx.append(rscore)
                
            relevance_score.append(relevance_idx)
        
        return relevance_score


    def discounted_cumulative_gain(answers, gold_contexts, topk_docs, relevance=None):
        """
        Note:
        
        Arg:
        
        Return:
        """
        
        DCG_values = []
        
        if relevance is None:
            relevance_score = get_relevance_score(answers, gold_contexts, topk_docs)
        else:
            relevance_score = relevance
        
        for idx, _ in enumerate(answers):
            DCG_value = 0.0
            
            for rank, rscore in enumerate(relevance_score[idx]):
                DCG_value += rscore/(np.log2(2+rank))
            
            DCG_values.append(DCG_value)
            
        return DCG_values

    def normaized_discounted_cumulative_gain(answers, gold_contexts, topk_docs, relevance=None):
        """
        Note:
        
        Arg:
        
        Return:
        
        """
        
        IDCG_values = []
        
        if relevance is None:
            relevance_score = get_relevance_score(answers, gold_contexts, topk_docs)
        else:
            relevance_score = relevance
        
        DCG_values = discounted_cumulative_gain(answers, gold_contexts, topk_docs, relevance=relevance_score)
        
        for idx, _ in enumerate(answers):
            IDCG_value = 0.0
            relevance_score[idx] = sorted(relevance_score[idx], reverse=True)
            
            for rank, rscore in enumerate(relevance_score[idx]):
                IDCG_value += rscore/(np.log2(2+rank))
            
            IDCG_values.append(IDCG_value)
            
        NDCG_values = [dcg/idcg for dcg, idcg in zip(DCG_values, IDCG_values)]
        NDCG_value = sum(NDCG_values)/len(NDCG_values)
        
        return NDCG_value 

    def test(self, metric='ALL'):
        
        MRR_value, NDCG_value = None, None
        
        if metric == 'NDCG':
            NDCG_value = normaized_discounted_cumulative_gain(answers=self.answers, gold_contexts=self.gold_contexts, 
                                                              topk_docs=self.topk_docs, relevance=None)
        
        elif metric == 'MRR':
            MRR_value = mean_reciprocal_rank(contexts_or_answers, topk_docs, mean=self.mean)
        
        else:
            MRR_value = mean_reciprocal_rank(contexts_or_answers=self.c_or_a, topk_docs=self.topk_docs, mean=self.mean)
            NDCG_value = normaized_discounted_cumulative_gain(answers=self.answers, gold_contexts=self.gold_contexts, 
                                                              topk_docs=self.topk_docs, relevance=None)
        
        return MRR_value, MDCG_value