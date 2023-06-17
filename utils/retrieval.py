import os
import time
import torch
import wandb
import pickle
import Levenshtein
import pandas as pd
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm
from contextlib import contextmanager
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")
    
    
class SparseBM25_edited(SparseBM25):
    """
    Note: BM25를 상속받으면서, 함수를 추가. 내가 gold context를 넘기면, gold가 아닌 것들로 topk를 가져오는 함수
    """
    
    def __init__(
        self,
        CFG,
        tokenize_fn,
        data_path: Optional[str] = "retrieval/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(CFG, tokenize_fn, data_path, context_path)
        
    def retrieve_except_gold(
       self, query_dataset: Dataset, topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        # want to get List of strings
        """
        Note:
            topk를 가져오면 거기에 positive가 있을 수도, 없을 수도 있다. 
            topk 개만 온전히 가져가려면, 최소 topk+1개를 애초에 retrieve해야한다.
        Args:
        
        Returns:
        """
        self.num_neg = self.num_neg + topk
        result = []
        alpha = 2
        with timer("query exhaustive search"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_dataset["question"], k=max(40 + topk, alpha * topk) if self.CFG['option']['use_fuzz'] else topk+1
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
            result.append(except_gold)
        
        return result
    

class DenseRetrieval(BaseRetrieval):
    def __init__(
        self,
        CFG,
        tokenize_fn,
        num_neg: Optional[int] = 5,
        data_path: Optional[str] = "../retrieval/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(CFG, tokenize_fn, data_path, context_path)
        
        self.args = TrainingArguments(
            output_dir="dense_retrieval_1",
            evaluation_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3, 
            num_train_epochs=1,
            weight_decay=0.01,
        )
        # self.model_name = CFG['model']['model_name']
        self.model_name = 'klue/bert-base'
        self.data_dir = "../input/data/train_dataset/train"
        self.dataset = load_from_disk(self.data_dir)
        self.num_neg = num_neg
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.p_encoder = BertEncoder.from_pretrained(self.model_name).to(dense_args.device)
        self.q_encoder = BertEncoder.from_pretrained(self.model_name).to(dense_args.device)
        
        self.prepare_in_batch_negative(self.dataset, num_neg, self.tokenizer)

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None, add_bm25):
        # random negatives -> in-batch   
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        if add_bm25:
            # p_with_neg: (전체, num_neg+1)
            # 1. gold context: (전체)
            gold_context = [context for context in dataset['context']]
            
            # 2. retrieve not gold but high bm25 docs
            bm_25_edited = SparseBM25_edited(CFG, tokenize_fn=tokenizer, data_path="../data")
            not_gold_context = bm_25_edited.retrieve_except_gold(gold_context, valid_data, topk=num_neg)    # list (전체, num_neg)
        
            # 3. extend
            for idx, rows in enumerate(p_with_neg):
                p_with_neg.extend(not_gold_context[idx])
        
        # (Question, Passage) 데이터셋
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')
        
        max_len = p_seqs['input_ids'].size(-1)
        
        # p_seqs org size (B*(num_neg+1), tokenizer_max_length) -> (B, (num_neg+1), tokenizer_max_length)
        # q_seqs size (B, tokenizer_max_length=512)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size, drop_last=True)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=True)

    def get_embedding(self) -> None:
        """
        Summary:
            Passage Embedding을 만들고(train)
            Dense Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        
        p_pickle_name = f"p_dense_embedding_randneg.bin"
        q_pickle_name = f"q_dense_embedding.bin"
        p_emb_path = os.path.join(self.data_path, p_pickle_name)
        q_emb_path = os.path.join(self.data_path, q_pickle_name)

        if os.path.isfile(p_emb_path) and os.path.isfile(q_emb_path):
            with open(p_emb_path, "rb") as file:
                self.p_encoder = pickle.load(file)
            with open(q_emb_path, "rb") as file:
                self.q_encoder = pickle.load(file)
            print("Embedding pickle load.")
        
        else:
            print("Build dense embedding...")
            self.train(CFG=self.CFG)
            
            with open(p_emb_path, "wb") as file:
                pickle.dump(self.p_encoder, file)
            with open(q_emb_path, "wb") as file:
                pickle.dump(self.q_encoder, file)
            print("Embedding pickle saved.")

    def train(self, args=None, CFG=None):
        folder_name, save_path = utils.get_folder_name(CFG)
        wandb.init(name=folder_name+'dense_embedding', project=CFG['wandb']['project'], 
            entity=CFG['wandb']['id'], dir=save_path)
        
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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

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

                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    # input:(B, num_neg+1, max_len) -> (B*(num_leg+1), max_len=512)
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
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1
                    loss_accumulate += loss.item()
                    
                    if global_step % 20 == 0:
                        wandb.log({"train/loss": loss_accumulate/20})
                        loss_accumulate = 0.0
                    
                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
        
        wandb.finish()

    def get_relevant_doc_bulk(self, queries, k):
        """
        Note:
        
        Args:
        
        Return:
        
        """
        
        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()
            
            queries_tok = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(self.args.device)
            queries_emb = self.q_encoder(**queries_tok).to(self.args.device)
            
            passage_embs = []
            for batch in tqdm(self.passage_dataloader, desc='queries retrieve 중'):
                batch = tuple(t.to(self.args.device) for t in batch)
                
                passage_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                passage_emb = self.p_encoder(**passage_inputs).to('cpu')
                passage_embs.append(passage_emb)    # [(batch, H), (batch, H), ...]
            
            stacked = torch.stack(passage_embs, dim=0).view(len(self.passage_dataloader.dataset)//self.args.per_device_train_batch_size*self.args.per_device_train_batch_size, -1).to(self.args.device)  # (num_passage, emb_dim)
            dot_prod_scores = torch.matmul(queries_emb, torch.transpose(stacked, 0, 1))  # (num_queries, num_passage)
            
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
    

if __name__ == "__main__":
    # setting
    data_dir = "../input/data/train_dataset/train"
    model_checkpoint = 'klue/bert-base'

    data = load_from_disk(data_dir)
    data = data.train_test_split(test_size=0.02, seed=42)
    train_dataset, valid_dataset = data['train'], data['test']
    
    dense_args = TrainingArguments(
        output_dir="dense_retrieval_1",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3, 
        num_train_epochs=1,
        weight_decay=0.01,
        # evaluation_strategy = 'step',
        eval_steps=0.25,
        fp16=True,
        report_to="wandb",
    )
    
    # 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(dense_args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(dense_args.device)
    
    # retriever = DenseRetrieval(args=dense_args, 
    #                            dataset=train_dataset, 
    #                            num_neg=6,
    #                            tokenizer=tokenizer, 
    #                            p_encoder=p_encoder, 
    #                            q_encoder=q_encoder)
    
    
    # retriever.train()
    # df = retriever.retrieve(valid_dataset, topk=2, args=dense_args)
    
    # main_process test
    with open('../config/use/use_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
    retriever = DenseRetrieval(CFG, tokenizer)
    retriever.get_embedding()
    
    # # retrieve 테스트
    # print(valid_dataset['question'][:10])
    queries = valid_dataset['question'][:5]
    score, docs = retriever.get_relevant_doc_bulk(queries, k=10)
    # for query in range(len(queries)):
    #     for i, idx in enumerate(docs.tolist()[query]):
    #         print(f"Top-{i + 1}th Passage (Index {idx})")
    #         print(retriever.dataset['context'][idx])