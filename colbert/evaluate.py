import json
import torch.nn.functional as F
from model import *
from tokenizer import *
import logging
import sys
import os
from typing import Callable, Dict, List, NoReturn, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, set_seed
from datasets import load_from_disk

def main():
    epoch = 6
    MODEL_NAME = "klue/bert-base"
    set_seed(42)
    datasets = load_from_disk("/opt/ml/input/data/train_dataset")
    val_dataset = pd.DataFrame(datasets["validation"])

    val_dataset = val_dataset.reset_index(drop=True)
    val_dataset = set_columns(val_dataset)

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = ColbertModel.from_pretrained(MODEL_NAME)
    # model.resize_token_embeddings(tokenizer.vocab_size + 2)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # model.load_state_dict(torch.load(f"./best_model/compare_colbert_epoch6.pth"))

    print("opening wiki passage...")
    with open("/opt/ml/input/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print("wiki loaded!!!")

    query = list(val_dataset["query"])
    ground_truth = list(val_dataset["context"])
    length = len(val_dataset)

    # batched_p_embs = []
    # with torch.no_grad():

    #     model.eval()

    #     # 토크나이저
    #     q_seqs_val = tokenize_colbert(query, tokenizer, corpus="query").to("cuda")
    #     q_emb = model.query(**q_seqs_val).to("cpu")
        
    #     del q_seqs_val

    #     print(q_emb.size())

    #     print("Start passage embedding......")
    #     p_embs = []
    #     for step, p in enumerate(tqdm(context)):
    #         p = tokenize_colbert(p, tokenizer, corpus="doc").to("cuda")
    #         p_emb = model.doc(**p).to("cpu").numpy()
    #         p_embs.append(p_emb)
    #         if (step + 1) % 200 == 0:
    #             batched_p_embs.append(p_embs)
    #             p_embs = []
    #     batched_p_embs.append(p_embs)

    # print("passage tokenizing done!!!!")
    # length = len(val_dataset["context"])

    # dot_prod_scores = model.get_score(q_emb, batched_p_embs, eval=True)

    # print(dot_prod_scores.size())

    # rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    # print(dot_prod_scores)
    
    rank = torch.load(f"/opt/ml/colbert/inference_colbert_rank_new_predict.pth")
    print(rank)
    print(rank.size())
    # torch.save(rank, f"./rank/rank_epoch{epoch}.pth")

    k = 5
    score = 0

    for idx in range(length):
        # print(dot_prod_scores[idx])
        print(rank[idx])
        print()
        for i in range(k):
            if ground_truth[idx] == context[rank[idx][i]]:
                score += 1

    print(f"{score} over {length} context found!!")
    print(f"final score is {score/length}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()