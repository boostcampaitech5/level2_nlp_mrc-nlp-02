import os
import evaluate
import torch
import pandas as pd

from datetime import datetime, timezone, timedelta
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_from_disk
from koeda import EDA


class Printer():
    def __init__(self):
        self.count = 1
        self.order = ''
        self.output = "!LOGGER: {0:<30}"
    
    def start(self, order):
        self.order = order
        print(self.output.format(f"{self.count} Start. >> {self.order}"))

    def done(self):
        print(self.output.format(f"{self.count} Done. >> {self.order}"))
        self.count += 1



class NewDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        super().__init__(tokenizer, pad_to_multiple_of)

    def __call__(self, features):
        batch = super().__call__(features)

        if "masked_lm_labels" in batch:
             batch["masked_lm_labels"] = [torch.tensor(label, dtype=torch.long) for label in batch["masked_lm_labels"]]


        return batch
    

def get_folder_name(CFG):
    """
    고유값을 생성해 실험 결과 기록할 폴더를 생성
    """
    now = datetime.now(tz=timezone(timedelta(hours=9)))
    folder_name = f"{now.strftime('%d%H%M%S')}-{CFG['실험명']}"
    save_path = f"./results/{folder_name}"
    CFG['save_path'] = save_path
    os.makedirs(save_path)
    os.makedirs(save_path + '/train')
    os.makedirs(save_path + '/test')

    return folder_name, save_path


def compute_metrics(p):
    metric = evaluate.load("squad")
    
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def get_dataset_after_EDA():
    """
    Easy Data Augmentation 적용 후 train_dataset을 가져오기
    """
    datasets = load_from_disk("input/data/train_dataset/")
    columns = ['title', 'context', 'question', 'id', 'answers', 'document_id']
    # train/valid 분리
    train = pd.DataFrame({'title':datasets['train']['title'], 'context':datasets['train']['context'], 'question':datasets['train']['question'], 'id':datasets['train']['id'], 'answers':datasets['train']['answers'], 'document_id':datasets['train']['document_id'], '__index_level_0__':datasets['train']['__index_level_0__']})
    valid = pd.DataFrame({'title':datasets['validation']['title'], 'context':datasets['validation']['context'], 'question':datasets['validation']['question'], 'id':datasets['validation']['id'], 'answers':datasets['validation']['answers'], 'document_id':datasets['validation']['document_id'], '__index_level_0__':datasets['validation']['__index_level_0__']})
    # answer_text 분리
    train['answer_text'] = train['answers'].apply(lambda x: x['text'][0])
    # start_idx 분리
    train['start_idx'] = train['answers'].apply(lambda x: x['answer_start'][0])
    # answer_text를 스페셜 토큰 [ANSWER]로 변환
    def create_new_context(df):
        new_contexts = []

        for idx, row in df.iterrows():
            start_idx = row['start_idx']
            end_idx = start_idx + len(row['answer_text'])
            
            new_context = row['context'][:start_idx] + "[ANSWER]" + row['context'][end_idx:]
            new_contexts.append(new_context)
        
        return new_contexts
    train['context_new'] = create_new_context(train)
    # Easy Data Augmentation 적용
    eda = EDA()
    train['context_eda'] = train['context_new'].apply(lambda x: eda(x))
    # Easy Data Augmentation 적용 된 거 중 [ANSWER] 토큰이 살아 남은 INDEX 추출
    train_select_idx = train['context_eda'].str.extract("(\[ANSWER\])").dropna().index
    # aug train/valid 생성
    aug_train = train.iloc[train_select_idx].copy()
    # [ANSWER]를 원래대로 되돌리기
    def answer_token2answer_text(df):
        contexts = []

        for idx, row in df.iterrows():
            context = row['context_eda'].replace("[ANSWER]", row['answer_text'])
            contexts.append(context)

        return contexts
    aug_train['context'] = answer_token2answer_text(aug_train)
    
    # 최종 데이터 저장
    final_train = pd.concat([train[columns], aug_train[columns]], axis=0)
    
    final = DatasetDict({
        'train': Dataset.from_pandas(final_train),
        'validation': Dataset.from_pandas(valid)
    })

    return final