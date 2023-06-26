"""

Note: See more details in
    - https://github.com/juyongjiang/Awesome-ANCE/blob/6ec62eae20950d1c9c6bc5714483f9163935d3c9/inferencer.py#L42
    - https://github.com/juyongjiang/Awesome-ANCE/blob/master/models.py#L78
    - https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/roberta/modeling_roberta.py#L1173
    - https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/modeling_outputs.py#L196
        

"""
import os
import faiss
import wandb
import torch
import Levenshtein
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
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer): # judge whether obj exist attr (layer)
            obj = getattr(obj, layer) # get the value of layer
        else:
            return None
    return obj

def optimizer_group(model):
    optimizer_grouped_parameters = []
    layer_optim_params = set()
    
    for layer_name in ["roberta.embeddings", "score_out", "downsample1", "downsample2", "downsample3"]:
        layer = getattr_recursive(model, layer_name)
        if layer is not None:
            optimizer_grouped_parameters.append({"params": layer.parameters()}) # [{"params": layer.parameters()}, ...]
            for p in layer.parameters():
                layer_optim_params.add(p)
                
    if getattr_recursive(model, "roberta.encoder.layer") is not None:
        for layer in model.roberta.encoder.layer:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
                
    optimizer_grouped_parameters.append({"params": [p for p in model.parameters() if p not in layer_optim_params]})
    
    return optimizer_grouped_parameters


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
        # nll loss
        # targets = torch.zeros(q_embs.size(0)).long().to('cuda:0')
        # loss = F.nll_loss(lsm, targets)
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
    

def update_new_embedding(args, model, input, tokenizer, is_query_inference=True):
    embedding, embedding2id = [], []
    
    # 토크나이징
    input_tok = tokenizer(input, padding="max_length", truncation=True, return_tensors='pt')
    input_idx = torch.tensor([idx for idx, _ in enumerate(input)])  # input 위치 기억

    # dataloader 생성
    dataset = TensorDataset(
        input_tok['input_ids'], input_tok['attention_mask'], input_tok['token_type_ids'],   # (전체, max_len)
        input_idx   # (전체, 1)
    )
    dataloader = DataLoader(dataset, batch_size=256)
    
    # batch 기준으로 임베딩 계산
    model.eval()

    for batch in tqdm(dataloader, desc="embedding updating...", position=0, leave=True):
        # batch: all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor [index of dataset]
        idxs = batch[3].detach().numpy()  # [#B]
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(), "attention_mask": batch[1].long()}
            if args.fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    if is_query_inference:
                        embs = model.query_emb(**inputs) # query1 = self.norm(self.embeddingHead(full_emb)) # linear layer, following layerNorm
                    else:
                        embs = model.body_emb(**inputs)  # 어떤 것이던, (B, H) 형태.

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

# 기존 gold passage는 아닌 negative를 선택해야 한다.
def generate_nagative_ids(args, new_p_embs_ids, new_q_embs_ids, positive_passage, I):
    """
    Note:
        faiss search로 찾아낸 가까운 임베딩 I로부터 negative sample을 뽑아냅니다.
        
    Arguments:
        - new_p_embs_ids: 새로이 업데이트 된 passages들의 id
        - new_q_embs_ids: 새로이 업데이트 된 queries들의 id
        
    Return:
        - 
    """
    query_negative_passage = {}
    gold_passages = positive_passage   # {query id: gold id1, gold id2, ... }
    
    # I 로부터 top K개 negatives 추출
    for query_idx in range(I.shape[0]):
        query_id = new_q_embs_ids[query_idx]

        # 추출한 negs는 기존 gold가 아닌 경우에만 ~
        positive_pid = new_p_embs_ids[query_id]
        selected_ann_idx = I[query_id, :args.select_topK + args.select_alpha]    # 예비용 alpha 개
        
        # append
        query_negative_passage[query_idx] = []
        neg_cnt = 0
        for idx in selected_ann_idx:
            neg_pid = new_p_embs_ids[idx]
            
            # Levenshtein distance -> 문맥적 내용이 거의 비슷한 중복 문서 제거
            neg_passage = positive_passage[neg_pid]
            pos_passage = positive_passage[positive_pid]
            dist = Levenshtein.distance(neg_passage, pos_passage)      

            if neg_pid == positive_pid or dist < min(len(neg_passage), len(pos_passage))//10*2:
                continue
            if neg_cnt >= args.negative_samples:
                break
            
            query_negative_passage[query_idx].append(neg_pid)
            neg_cnt += 1    
    
    return query_negative_passage # {query_id: nag_ids_1, neg_ids_2, ...}


def make_next_dataset(tokenizer, queries, passages, neg_ids):
    """
    Note:   다음으로 학습에 사용될 Dataset을 생성합니다. 토크나이징 - 데이터세트 생성 과정을 거칩니다.
    
    Arguments:
        - queries
        - passages
        - neg_ids
    
    return:
    """
    # ANCE에서는 {}{}{} 이 triplet으로 어떻게 데이터세트로 만들ㅇ었지?
    # 막 tensordataset으로 뭘 하던데... 그냥 split해서 읽어들인거 int형태로 바꾸고 tensor dataset으로 반환한다.
    
    # 텐서화.. 으음. 클래스화 해서 ANCE는 아예 custom dataset을 만들었구나.
    neg_passages = [passages[ids[0]] for _, ids in neg_ids.items()]
    breakpoint()
    query_data = get_tokenized(tokenizer, queries)
    pos_data = get_tokenized(tokenizer, passages)
    neg_data = get_tokenized(tokenizer, neg_passages)

    return TensorDataset(query_data[0], query_data[1], query_data[2],
                         pos_data[0], pos_data[1], pos_data[2],
                         neg_data[0], neg_data[1], neg_data[2]) # qid, pos_pid, and neg_pid are not needed. 


def get_tokenized(tokenizer, input):
    embed = tokenizer(input, padding="max_length", truncation=True, return_tensors='pt')
    
    input_ids = embed['input_ids']
    attention_masks = embed['attention_mask']
    token_type_ids = embed['token_type_ids']
    input_to_id = torch.tensor([idx for idx, _ in enumerate(input)])  # input 위치 기억
    
    # 그냥 (, , , )으로 리턴하는 것과 무슨 차이가 있을까?
    # return TensorDataset(input_ids, attention_masks, token_type_ids, input_to_id)
    return (input_ids, attention_masks, token_type_ids, input_to_id)


def train(args, CFG, model, tokenizer, train_data):
    # setup
    train_passages = train_data['context']
    train_queries = train_data['question']
    _name = CFG['실험명']
    
    wandb.init(name=_name+'_ance_training', project=CFG['wandb']['project'], 
               entity=CFG['wandb']['id'])
    
    # for saving
    output_dir = "./ance_pretrained/"
    
    # optimizer, scheduler
    optimizer_grouped_parameters = optimizer_group(model)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # training start!
    model.zero_grad()
    step, global_step = 0, -1
    # _max_step = (len(train_data)//args.per_device_train_batch_size)//args.gradient_accumulation_steps * args.num_train_epochs
    # _update_step = (len(train_data)//args.per_device_train_batch_size)//args.gradient_accumulation_steps
    # print(f"t_total: {t_total}, total step: None, update step: 2")
    
    # get first train dataloader
    
    # while global_step < args.max_steps:
    _update_step = 4
    train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
    for _ in train_iterator:
        global_step += 1
        
    # while global_step < t_total:
        
        # if some steps
        # if step % args.gradient_accumulation_steps == 0:
        if global_step % _update_step == 0:
            # update p, q embeddings
            # p, q = (len(train), ) -> new_p_embs (len(train), S, H), new_p_embs_ids (len(train))
            new_p_embs, new_p_embs_ids = update_new_embedding(args, model, train_passages, tokenizer, is_query_inference=False)
            new_q_embs, new_q_embs_ids = update_new_embedding(args, model, train_queries, tokenizer)

            # search best ann negs
            dim = new_p_embs.shape[1]   # H = hidden_dim = 768
            cpu_index = faiss.IndexFlatIP(dim)
            cpu_index.add(new_p_embs)
            _, I = cpu_index.search(new_q_embs, args.select_topK)
            
            # generate new train_dataset
            query_negative_passage_ids = generate_nagative_ids(args, new_p_embs_ids, new_q_embs_ids, train_passages, I) # {'queryid': neg_id, ...}
            
            next_train_dataset = make_next_dataset(tokenizer, train_queries, train_passages, query_negative_passage_ids)
            
            train_dataloader = DataLoader(next_train_dataset, batch_size=args.per_device_train_batch_size)
            train_dataloader_iter = iter(train_dataloader)
            # maybe you can re warmup schedulers here, too.
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * _update_step
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
            
        # 대신 이 방식은 중간에 dataloader를 바꿀 수 없다.
        with tqdm(train_dataloader, unit='batch') as tepoch:
            for batch in tepoch:
                step += 1
                model.train()

                # new batch
                # batch = next(train_dataloader_iter)
            
                batch = tuple(t.to(args.device) for t in batch)
                # step += 1
            
                # input
                inputs = {"query_ids": batch[0].long(), "attention_mask_q": batch[1].long(),
                        "input_ids_a": batch[3].long(), "attention_mask_a": batch[4].long(),
                        "input_ids_b": batch[6].long(), "attention_mask_b": batch[7].long()}
                
                # output
                if args.fp16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = model(**inputs)   
                else:
                    output = model(**inputs)
                
                # loss
                loss = output[0]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                tepoch.set_postfix(loss=f'{str(loss.item())}')
                
                scaler.scale(loss).backward()
                # tepoch.set_postfix(loss=f'{str(loss.item())}')
                if step % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    
                    scheduler.step()    # Update learning rate schedule
                    scaler.update()

                    model.zero_grad()
                    model.train()
                    torch.cuda.empty_cache()
                    
                    # global_step += 1
                    # print(f"loss: {loss}")
                    wandb.log({"train/loss": loss, # "train/learning_rate": args.learning_rate})
                                "train/learning_rate": optimizer.param_groups[0]['lr']})

                    # maybe eval here
                    
                if step % args.save_steps == 0:
                    # save checkpoint
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    print(f"ance save checkpoint ...")
                
    wandb.finish()
    return model
                
if __name__ == '__main__':
    model_name = 'klue/roberta-base'

    ANCE_config = AutoConfig.from_pretrained(model_name)
    ANCE_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ANCE_model = RobertaDot_NLL_LN(ANCE_config)