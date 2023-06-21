import os
import evaluate
import torch

from datetime import datetime, timezone, timedelta
from transformers import DataCollatorWithPadding


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
    if CFG['CL'] == 'extract':
        os.makedirs(save_path + '/prediction_train')

    return folder_name, save_path

def compute_metrics(p):
    metric = evaluate.load("squad")
    
    return metric.compute(predictions=p.predictions, references=p.label_ids)

class NewDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        super().__init__(tokenizer, pad_to_multiple_of)

    def __call__(self, features):
        batch = super().__call__(features)

        if "masked_lm_labels" in batch:
             batch["masked_lm_labels"] = [torch.tensor(label, dtype=torch.long) for label in batch["masked_lm_labels"]]


        return batch