
# baseline : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-11

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from datasets import load_from_disk
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizer import *
from model import *
import json
import pickle

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

BM25_USED = True
pretrain = False

def main():
    
    set_seed(42)
    
    batch_size = 15
    data_path = "../input/data/train_dataset"
    load_weight_path = '/opt/ml/colbert/best_model/compare_colbert_pretrain_v3_7.pth'   
    lr = 2e-6
    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=6,
        weight_decay=0.01,
        warmup_steps=900
    )

    MODEL_NAME = "klue/bert-base"
    tokenizer = load_tokenizer(MODEL_NAME)

    if pretrain == True:
        train_dataset = pd.read_csv('/opt/ml/input/data/aug_v1.csv')
        print('pre_duplicate_remove:', len(train_dataset))
        train_dataset.drop_duplicates(subset=['context'], inplace=True) # in_batch의 취약점 개선하기 위함
        print('post_duplicate_remove:', len(train_dataset))
        train_dataset['context_length'] = train_dataset['context'].apply(len)
        train_dataset = train_dataset.sort_values('context_length', ascending=False)
        train_dataset = train_dataset.head(40000)
        train_dataset = train_dataset.drop(columns=['context_length'])
        print('post_matching_distribution:', len(train_dataset))
        print('shortest context is', train_dataset['context'].apply(len).min())
        train_dataset['title'] = [None] * len(train_dataset)
    else:
        datasets = load_from_disk(data_path)
        train_dataset = pd.DataFrame(datasets["train"])
        validation_dataset = pd.DataFrame(datasets["validation"])
        train_dataset = pd.concat([train_dataset, validation_dataset])
        train_dataset = train_dataset.reset_index(drop=True)

    if BM25_USED:
        context1_name = "bm25rank_contexts1.pickle"
        context2_name = "bm25rank_contexts2.pickle"
        comtext1_path = os.path.join('/opt/ml/colbert/', context1_name)
        comtext2_path = os.path.join('/opt/ml/colbert/', context2_name)

        if os.path.isfile(comtext1_path) and os.path.isfile(comtext1_path):
            with open(comtext1_path, "rb") as file:
                bm25rank_contexts1 = pickle.load(file)
            with open(comtext2_path, "rb") as file:
                bm25rank_contexts2 = pickle.load(file)          
            print("bm25 hard negative loaded.")
        else:
            # bm25 hard negative
            with open("bm25rank_dict.pickle", "rb") as fr:
                bm25rank_dict = pickle.load(fr)
                
            bm25rank_contexts1 = []
            bm25rank_contexts2 = []
            topk = len(eval(list(bm25rank_dict.values())[0]))
            
            # 지금 row의 ['id']에는 train_dataset의 한 샘플에 대한 id가 들어있고, (ex. mrc-1-000067)
            # 그 id를 key로 하고, value로 topk개의 bm25 ranked wiki 문서들이 들어있음
            # 그 중, bm25가 높음 wikidata임에도 row['answers']['text'], 즉 정답이 포함되어있지 않다면
            # hard negative    
            for idx, row in tqdm(list(train_dataset.iterrows()), desc="sampling hard negatives..."):
                pointer = 0
                flag = 0
                while pointer < topk:
                    if row['answers']['text'][0] not in (hard_neg := eval(bm25rank_dict[row["id"]])[pointer]):
                        if flag == 0:
                            bm25rank_contexts1.append(hard_neg)
                            flag = 1
                        else:
                            bm25rank_contexts2.append(hard_neg)
                            flag = 2
                            break
                    pointer += 1 
                if flag != 2:
                    print('Don\'t have enough samples', idx)
                    hard_neg = eval(bm25rank_dict[row["id"]])
                    bm25rank_contexts2.append(hard_neg[0] if hard_neg[0] != row['context'] else hard_neg[1])
                    
            with open('/opt/ml/colbert/bm25rank_contexts1.pickle', 'wb') as f:
                pickle.dump(bm25rank_contexts1, f, pickle.HIGHEST_PROTOCOL)
            with open('/opt/ml/colbert/bm25rank_contexts2.pickle', 'wb') as f:
                pickle.dump(bm25rank_contexts2, f, pickle.HIGHEST_PROTOCOL)

    train_dataset = set_columns(train_dataset)

    print("dataset tokenizing.......")
    # 토크나이저
    train_context, train_query = tokenize_colbert(train_dataset, tokenizer, corpus="both")
    if BM25_USED:
        train_bm25_1 = tokenize_colbert(bm25rank_contexts1, tokenizer, corpus="bm25_hard")
        train_bm25_2 = tokenize_colbert(bm25rank_contexts2, tokenizer, corpus="bm25_hard")

        train_dataset = TensorDataset(
            train_context["input_ids"],
            train_context["attention_mask"],
            train_context["token_type_ids"],
            train_query["input_ids"],
            train_query["attention_mask"],
            train_query["token_type_ids"],
            train_bm25_1["input_ids"],
            train_bm25_1["attention_mask"],
            train_bm25_1["token_type_ids"],
            train_bm25_2["input_ids"],
            train_bm25_2["attention_mask"],
            train_bm25_2["token_type_ids"],
        )
    else:
        train_dataset = TensorDataset(
            train_context["input_ids"],
            train_context["attention_mask"],
            train_context["token_type_ids"],
            train_query["input_ids"],
            train_query["attention_mask"],
            train_query["token_type_ids"],
        )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)

    if load_weight_path:
        model.load_state_dict(torch.load(load_weight_path))
    model.to(device)

    print("model train...")
    trained_model = train(args, train_dataset, model)
    torch.save(trained_model.state_dict(), f"./best_model/colbert_pretrain_v3_finetune.pth")


def train(args, dataset, model):

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size
    )

    # Optimizer 세팅
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training 시작
    global_step = 0

    model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        print(f"epoch {epoch}")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss = 0

        for step, batch in enumerate(epoch_iterator):
            model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }
            if BM25_USED:
                n_inputs_1 = {
                    "input_ids": batch[6],
                    "attention_mask": batch[7],
                    "token_type_ids": batch[8],
                }

                n_inputs_2 = {
                    "input_ids": batch[9],
                    "attention_mask": batch[10],
                    "token_type_ids": batch[11],
                }


            # outputs with similiarity score
            if BM25_USED:
                outputs = model(p_inputs, q_inputs, (n_inputs_1, n_inputs_2))
            else:
                outputs = model(p_inputs, q_inputs, None)
            # target: position of positive samples = diagonal element
            targets = torch.arange(0, outputs.shape[0]).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")
                
            sim_scores = F.log_softmax(outputs, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            torch.cuda.empty_cache()
        final_loss = total_loss / len(dataset)
        print("total_loss :", final_loss)
        torch.save(model.state_dict(), f"./best_model/compare_colbert_pretrain_v3_finetune_{epoch+1}.pth")

    return model


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()