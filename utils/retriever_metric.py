import re
import numpy as np

def mean_reciprocal_rank(contexts_or_answers, topk_docs, mean='context'):
    """
    Note:
        topk개의 docs들로부터 context/answer 기준으로 MRR을 계산합니다(context/answer 가 있는 context가 몇 번째에 있는지?)
    Args:
        - answers: [쿼리1 정답, 쿼리2 정답, ...], shape (쿼리 개수)
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


# 2. NDCG 구현하기 - relevance: answer보유=1, gold_context=3

def get_relevance_score(answers, gold_contexts, topk_docs):
    """
    Note:
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
