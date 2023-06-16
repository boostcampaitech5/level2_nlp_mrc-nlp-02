import pandas as pd
import torch
import json
from itertools import zip_longest

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from .tokenizer import *
from .model import *

def run_colbert_retrieval(datasets, model_args, training_args, top_k=10):
    test_dataset = datasets["validation"].flatten_indices().to_pandas()
    MODEL_NAME = "klue/bert-base"

    print("opening wiki passage...")
    with open("/opt/ml/input/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print("wiki loaded!!!")

    query = list(test_dataset["question"])

    mrc_ids = test_dataset["id"]
    length = len(test_dataset)

    if model_args.ColBERT_rank_path:
        load_rank_path = model_args.ColBERT_rank_path
        print("retriever : load rank")
        if training_args.do_eval:
            rank = torch.load(f"{load_rank_path}_eval.pth")
        else:
            print(f"{load_rank_path}_predict.pth")
            rank = torch.load(f"{load_rank_path}_predict.pth")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
        ret_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        ret_tokenizer.add_special_tokens(special_tokens)
        model = ColbertModel.from_pretrained(MODEL_NAME)
        model.resize_token_embeddings(ret_tokenizer.vocab_size + 2)

        model.to(device)

        model.load_state_dict(torch.load(model_args.retrieval_ColBERT_path))

        batched_p_embs = []

        with torch.no_grad():
            model.eval()
            
            q_seqs_val = tokenize_colbert(query, ret_tokenizer, corpus="query").to("cuda")
            print(len(q_seqs_val['input_ids'][0]))
            q_emb = model.query(**q_seqs_val).to("cpu")
            
            del q_seqs_val
            
            print(q_emb.size())


            print("Start passage embedding.. ....")
            batched_p_embs = []
            P_BATCH_SIZE = 128
            # Define a generator for iterating in chunks
            def chunks(iterable, n, fillvalue=None):
                args = [iter(iterable)] * n
                return zip_longest(*args, fillvalue=fillvalue)

            for step, batch in enumerate(tqdm(chunks(context, P_BATCH_SIZE), total=len(context)//P_BATCH_SIZE)):
                # The last batch can contain `None` values if the length of `context` is not divisible by 128
                batch = [b for b in batch if b is not None]

                # Tokenize the entire batch at once
                p = tokenize_colbert(batch, ret_tokenizer, corpus="doc").to("cuda")
                p_emb = model.doc(**p).to("cpu").numpy()

                del p

                batched_p_embs.append(p_emb)
            
            print(len(batched_p_embs), batched_p_embs[0].size())
            
        if training_args.do_eval:
            total_score_for_eval = []
            for num in range(0, 4000, 400):
                total_score_for_eval.append((dot_prod_scores_eval:=model.get_score(q_emb[num : num+400], batched_p_embs, eval=True)))
                print(f'from {num} to {num+4000}:', dot_prod_scores_eval.size())
            dot_prod_scores = torch.cat(total_score_for_eval, dim=0)
            print(dot_prod_scores.size())
            
        else:
            dot_prod_scores = model.get_score(q_emb, batched_p_embs, eval=True)
            print(dot_prod_scores.size())

        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        print(dot_prod_scores)
        print(rank)
        if training_args.do_eval:
            torch.save(rank, f"{model_args.save_ColBERT_rank_path}_eval.pth")
        else:
            torch.save(rank, f"{model_args.save_ColBERT_rank_path}_predict.pth")
    print(rank.size())
    print(length)

    passages = []

    for idx in range(length):
        passage = ""
        for i in range(top_k):
            passage += context[rank[idx][i]]
            passage += " "
        passages.append(passage)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = pd.DataFrame({"question": query, "id": mrc_ids, "context": passages})

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        df = pd.DataFrame(
            {
                "question": query,
                "id": mrc_ids,
                "context": passages,
                "answers": test_dataset["answers"],
            }
        )
    else:
        raise ValueError

    # complete_datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return df