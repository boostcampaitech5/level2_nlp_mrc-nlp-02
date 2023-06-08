import os
import yaml
import wandb

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed
)

### 우리가 만든 모듈 ###
from utils import utils, data_controller
from input.code.trainer_qa import QuestionAnsweringTrainer
from input.code.utils_qa import postprocess_qa_predictions

import warnings
warnings.filterwarnings('ignore')


### MAIN ###
printer = utils.Printer()
# config 불러오기
printer.start('config 불러오기')
with open('config/use/use_config.yaml') as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)
with open('config/use/use_trainer_args.yaml') as f:
    TRAIN_ARGS = yaml.load(f, Loader=yaml.FullLoader)
printer.done()
# transformers에서 seed 고정하기
printer.start('SEED 고정하기')
set_seed(CFG['seed'])
TRAIN_ARGS['seed'] = CFG['seed']
printer.done()

if __name__ == "__main__":
    # 실험 폴더 생성
    printer.start('실험 폴더 생성')
    folder_name, save_path = utils.get_folder_name(CFG)
    TRAIN_ARGS['output_dir'] = save_path + "/train"
    printer.done()
    # wandb 설정
    wandb.init(name=folder_name, project=CFG['wandb']['project'], 
               config=CFG, entity=CFG['wandb']['id'], dir=save_path)
    # 데이터셋 가져오기
    printer.start('train/test 데이터셋 가져오기')
    train_dataset = load_from_disk('input/data/train_dataset')
    test_dataset = load_from_disk('input/data/test_dataset')
    printer.done()
    # Trainer의 Args 객체 가져오기
    printer.start('Trainer Args 가져오기')
    training_args = TrainingArguments(**TRAIN_ARGS)
    printer.done()

    # config, tokenizer, model 가져오기
    printer.start('HuggingFace에서 모델 및 토크나이저 가져오기')
    config = AutoConfig.from_pretrained(CFG['model']['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(CFG['model']['model_name'], use_fast=True) # rust tokenizer if use_fast == True else python tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(CFG['model']['model_name'], config=config)
    printer.done()

    # 토큰화를 위한 파라미터 설정
    printer.start('토큰화를 위한 파라밈터 설정')
    CFG['tokenizer']['max_seq_length'] = min(CFG['tokenizer']['max_seq_length'],
                                             tokenizer.model_max_length)
    fn_kwargs = {
        'tokenizer': tokenizer,
        'pad_on_right': tokenizer.padding_side == "right", # Padding에 대한 옵션을 설정합니다. | (question|context) 혹은 (context|question)로 세팅 가능합니다.
        "CFG": CFG,
    }
    printer.done()

    # train/valid 데이터셋 정의
    printer.start('train/valid 데이터셋 정의')
    train_data = train_dataset['train']
    val_data = train_dataset['validation']
    print(train_data)
    print(val_data)
    printer.done()

    # 데이터 토큰나이징
    printer.start("train 토크나이징")
    train_data = train_data.map(
        data_controller.train_tokenizing,
        batched=True,
        num_proc=None,
        remove_columns=train_data.column_names,
        load_from_cache_file=not False,
        fn_kwargs=fn_kwargs
    )
    printer.done()
    printer.start("val 토크나이징")
    val_data = val_data.map(
        data_controller.val_tokenizing,
        batched=True,
        num_proc=None,
        remove_columns=val_data.column_names,
        load_from_cache_file=not False,
        fn_kwargs=fn_kwargs
    )
    printer.done()

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Trainer 초기화
    printer.start("Trainer 초기화")
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=CFG['tokenizer']['max_answer_length'],
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex['answers']}
                for ex in train_dataset["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        eval_examples=train_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=utils.compute_metrics,
    )
    printer.done()

    # Training
    printer.start("학습중...")
    train_result = trainer.train()
    trainer.save_model()
    printer.done()
    printer.start("모델 및 metrics 저장")
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_data)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        for key, value in sorted(train_result.metrics.items()):
            writer.write(f"{key} = {value}\n")

    # State 저장
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )
    printer.done()

    # val 평가
    printer.start("val 평가")
    if training_args.do_eval:
        metrics = trainer.evaluate()

        metrics["val_samples"] = len(val_data)

        trainer.log_metrics("val", metrics)
        trainer.save_metrics("val", metrics)
    printer.done()

    # predict 단계
    training_args.output_dir = save_path + '/test'