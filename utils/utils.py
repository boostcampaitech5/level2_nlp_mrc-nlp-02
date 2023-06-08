import os
import evaluate

from datetime import datetime, timezone, timedelta


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

    return folder_name, save_path

def compute_metrics(p):
    metric = evaluate.load("squad")
    
    return metric.compute(predictions=p.predictions, references=p.label_ids)