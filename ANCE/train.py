"""

Note: See more details in
    - https://github.com/juyongjiang/Awesome-ANCE/blob/6ec62eae20950d1c9c6bc5714483f9163935d3c9/inferencer.py#L42
    - https://github.com/juyongjiang/Awesome-ANCE/blob/master/models.py#L78
    - https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/roberta/modeling_roberta.py#L1173
    - https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/modeling_outputs.py#L196
        

"""
import os
import faiss
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertPreTrainedModel, 
    RobertaPreTrainedModel,
    RobertaForSequenceClassification,
    RobertaModel,
)


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        # assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0] # [batch, len, dim] -> # [batch, 0, dim] -> # [batch, dim] using the first [cls] as the final output

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

## FirstP
# (query_data[0], query_data[1], # content, mask
#  pos_data[0], pos_data[1],
#  neg_data[0], neg_data[1],) 
class NLL(EmbeddingMixin):
    def forward(self, query_ids, attention_mask_q, 
                      input_ids_a=None, attention_mask_a=None, # positive passage
                      input_ids_b=None, attention_mask_b=None, # negative passage
                      is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        # get the dense representation of query, postive passage, negtive passage
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        # nll loss
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1), (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1) # apply in the dim=1 
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)

class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # last linear transfer layer
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights) # initialize all layers' parameters weight

    def query_emb(self, input_ids, attention_mask):
        # roberta accepts input_ids, and attention_mask for each sequence, i.e., [token_id1, token_id2, ...], [1,1,1, ..., 0,0,0]
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask) # (B, S, H)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)              # (B, 0, H) mean 옵션이 안 켜져있으면 CLS 에 대해서만. 
        query1 = self.norm(self.embeddingHead(full_emb))                            # linear layer, following layerNorm
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    

def update_new_embedding(model, input, is_query_inference=True):
    embedding, embedding2id = [], []
    
    # 토크나이징
    input_tok = tokenizer(input, padding="max_length", truncation=True, return_tensors='pt')
    input_idx = torch.tensor([idx for idx, _ in enumerate(input)])

    # dataloader 생성
    dataset = TensorDataset(
        input_tok['input_ids'], input_tok['attention_mask'], input_tok['token_type_ids'],   # (전체, max_len)
        input_idx   # (전체, 1)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    # batch 기준으로 임베딩 계산
    model.eval()

    for batch in tqdm(dataloader, desc="Inferencing", position=0, leave=True):
        # batch: all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor [index of dataset]
        idxs = batch[3].detach().numpy()  # [#B]
        batch = tuple(t.to(args.device) for t in batch)

        # fp16도 나중에 추가하자..
        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(), "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs) # query1 = self.norm(self.embeddingHead(full_emb)) # linear layer, following layerNorm
            else:
                embs = model.module.body_emb(**inputs)  # 어떤 것이던, (B, H) 형태.

        embs = embs.detach().cpu().numpy() # detach: avoid gradient backward anymore
        # check for multi chunk output for long sequence
        if len(embs.shape) == 3: # [batchS, chunk_factor, embeddingS]
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)                  # embedding=[(B, H), ...]

    embedding = np.concatenate(embedding, axis=0)   # [전체, H]
    embedding2id = np.concatenate(embedding2id, axis=0)
    
    return embedding, embedding2id

def generate_nagatives(model, texts):
    
    pass

def generate_nagative_ids(new_p_embs, new_q_embs, new_p_embs_ids, I):
    neg_data_idxes = []
    
    return neg_data_idxes


def make_next_dataset(quries, passages, neg_ids):
    
    pass


def train():
    ## 1. batch
    while global_step < args.max_steps:
        
        # if some steps
        if global_step % args.gradient_accumulation_steps == 0:
            # update p, q embeddings
            # p, q = (len(train), ) -> new_p_embs (len(train), S, H), new_p_embs_ids (len(train))
            new_p_embs, new_p_embs_ids = update_new_embedding(ANCE_model, train_passages, is_query_inference=False)
            new_q_embs, new_q_embs_ids = update_new_embedding(ANCE_model, train_queries)
            
            # search best ann negs
            dim = new_p_embs.shape[1]   # H = hidden_dim = 768
            cpu_index = faiss.IndexFlatIP(dim)
            cpu_index.add(new_p_embs)
            _, I = cpu_index.search(new_q_embs, args.top_k)
            
            # generate new train_dataset
            query_negative_passage_ids = generate_nagative_ids(new_p_embs, new_q_embs, new_p_embs_ids, I) # {'queryid': neg_id, ...}
            
            next_train_dataset = make_next_dataset(train_queries, train_passages, query_negative_passage_ids)
            
            """
                query_negative_paassage = {}
                for q in range(I.shape[0]):
                    top_ann_pid = I[q, :args.negative_sample+1] # +1 for sure
                    
                    for idx in top_ann_pid:
                        neg_cnt +=1
                        neg_pid = new_p_embs_ids[idx]
                        query_negative_pasage[new_pid].append()
                        if neg_cnt >= args.negative_sample:
                            break
                return query_negative_passage
            """
            # next_train_dataset = f3(query_negative_passage)
            train_dataloader = DataLoader(next_train_dataset)
            train_dataloader_iter = iter(train_dataloader)

            # maybe you can re warmup schedulers here, too.
        
        # new batch
        batch = next(train_dataloader_iter)
        
        batch = tuple(t.to(args.device) for t in batch)
        step += 1
        
        # input
        inputs = {"query_ids": batch[0].long(),   "attention_mask_q": batch[1].long(),
                "input_ids_a": batch[3].long(), "attention_mask_a": batch[4].long(),
                "input_ids_b": batch[6].long(), "attention_mask_b": batch[7].long()}
        
        # output
        ouput = ANCE_model(**inputs)   # (B, S, H)
        
        # loss
        loss = output[0]
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        loss.backward()
        
        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # maybe eval here
            
            if global_step % args.save_steps == 0:
                # save checkpoint
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                
                
if __name__ == '__main__':
    model_name = 'klue/roberta-base'

    ANCE_config = AutoConfig.from_pretrained(model_name)
    ANCE_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ANCE_model = RobertaDot_NLL_LN(ANCE_config)