### 외부 라이브러리 ###
import os
import yaml
import wandb
import pandas as pd

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    load_from_disk
)
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed
)
from tqdm.auto import tqdm

### 우리가 만든 라이브러리 ###
from utils import utils, data_controller, retrieval
from input.code.trainer_qa import QuestionAnsweringTrainer
from input.code.utils_qa import postprocess_qa_predictions
from models.models import *

import warnings
warnings.filterwarnings('ignore')


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
TRAIN_ARGS['per_device_train_batch_size'] = CFG['option']['batch_size']
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
    printer.done()
    # Trainer의 Args 객체 가져오기
    printer.start('Trainer Args 가져오기')
    training_args = TrainingArguments(**TRAIN_ARGS)
    printer.done()

    # config, tokenizer, model 가져오기
    printer.start('HuggingFace에서 모델 및 토크나이저 가져오기')
    config = AutoConfig.from_pretrained(CFG['model']['model_name'])
    if 'bert' in CFG['model']['model_name'] and 'roberta' not in CFG['model']['model_name']:
        config.num_attention_heads = CFG['model']['num_attention_heads']
        config.attention_probs_dropout_prob = CFG['model']['attention_probs_dropout_prob']
        config.num_hidden_layers = CFG['model']['num_hidden_layers']
        config.hidden_dropout_prob = CFG['model']['hidden_dropout_prob']
    tokenizer = AutoTokenizer.from_pretrained(CFG['model']['model_name'], use_fast=True) # rust tokenizer if use_fast == True else python tokenizer
    model_class = eval(CFG['model']['select_option'][CFG['model']['option']])
    model = model_class.from_pretrained(CFG['model']['model_path'], config=config)
    printer.done()

    # 토큰화를 위한 파라미터 설정
    printer.start('토큰화를 위한 파라미터 설정')
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
    if CFG['model']['pretrain']:
        aug_df = pd.read_csv('/opt/ml/input/data/' + CFG['model']['pretrain'])
        aug_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
        aug_df['id'] = aug_df['id'].apply(lambda x:str(x))
        aug_df['answers'] = aug_df['answers'].apply(eval)
        new_data = Dataset.from_pandas(aug_df)
        train_data = new_data
        print('Pretrain with this new dataset\n\n')
        print(train_data)
    else:
        train_data = train_dataset['train']
        print('finetuning or just train with original dataset')
        print(train_data)
    val_data = train_dataset['validation']
    print(val_data)
    printer.done()

    # 데이터 토큰나이징
    printer.start("train 토크나이징")
    fn_kwargs['column_names']= train_data.column_names
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
    fn_kwargs['column_names']= val_data.column_names
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
    metrics = trainer.evaluate()

    metrics["val_samples"] = len(val_data)

    trainer.log_metrics("val", metrics)
    trainer.save_metrics("val", metrics)
    printer.done()

    # predict 단계
    training_args.do_eval = False
    training_args.do_predict = True
    training_args.output_dir = save_path + '/test'
    test_dataset = load_from_disk('input/data/test_dataset')
    
    # retrieval 단계
    retrieval_class = eval(f"retrieval.{CFG['retrieval_list'][CFG['retrieval_name']]}")
    retriever = retrieval_class(CFG=CFG, training_args = training_args, tokenize_fn=tokenizer.tokenize)
    retriever.get_embedding()

    printer.start("top-k 추출하기")
    df = retriever.retrieve(test_dataset['validation'], topk=CFG['option']['top_k_retrieval'])
    printer.done()

    printer.start("context가 추가된 test dataset 선언")
    f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    test_dataset = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    printer.done()

    # reader 단계
    test_data = test_dataset['validation']
    printer.start("test 토크나이징")
    fn_kwargs['column_names']= test_data.column_names
    test_data = test_data.map(
        data_controller.val_tokenizing,
        batched=True,
        num_proc=None,
        remove_columns=test_data.column_names,
        load_from_cache_file=not False,
        fn_kwargs=fn_kwargs
    )
    printer.done()
    # printer.start("test를 위한 trainer 초기화")
    # # Trainer 초기화
    # trainer = QuestionAnsweringTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=None,
    #     eval_dataset=test_data,
    #     eval_examples=test_dataset["validation"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     post_process_function=post_processing_function,
    #     compute_metrics=utils.compute_metrics,
    # )
    # printer.done()
    printer.start("predict 수행중...")
    predictions = trainer.predict(
        test_dataset=test_data,
        test_examples=test_dataset['validation']
    )
    printer.done()