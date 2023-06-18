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
from utils import utils, data_controller, retriever_metric
from input.code.trainer_qa import QuestionAnsweringTrainer
from input.code.utils_qa import postprocess_qa_predictions
from input.code.retrieval import SparseRetrieval

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
    for key in ['num_attention_heads', 'attention_probs_dropout_prob', 'num_hidden_layers', 'hidden_dropout_prob']:
        config[key] = CFG['model'][key]
    tokenizer = AutoTokenizer.from_pretrained(CFG['model']['model_name'], use_fast=True) # rust tokenizer if use_fast == True else python tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(CFG['model']['model_name'], config=config)
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
    train_data = train_dataset['train']
    val_data = train_dataset['validation']
    print(train_data)
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
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer, data_path='input/data', 
    )
    retriever.get_sparse_embedding()

    printer.start("top-k 추출하기")
    df = retriever.retrieve(test_dataset['validation'], topk=CFG['option']['top_k_retrieval'])
    printer.done()

    # retriever 성능 비교하기 - 현재 valid 데이터에 대해서만 평가. train에 대해서 진행하고 싶다면 주석을 풀어주세요.
    # TF-IDF기준 train 약 40-50초, valid 약 5초
    if CFG['option']['check_retrieved_score']:
        printer.start("Retrieved docs 성능평가")
        
        # train 데이터로 retrieve한 문서 점수 평가
        # df_for_train = retriever.retrieve(train_dataset['train'], topk=CFG['option']['top_k_retrieval'])
        # metric_train = retriever_metric.score_retrieved_docs(dataset=train_dataset['train'], topk_docs=df_for_train, mean='context', metric='ALL')
        # train_mrr, train_ndcg = metric_train.test()
        
        # valid 데이터로 retrieve한 문서 점수 평가
        df_for_valid = retriever.retrieve(train_dataset['validation'], topk=CFG['option']['top_k_retrieval'])
        metric_valid = retriever_metric.score_retrieved_docs(dataset=train_dataset['validation'], topk_docs=df_for_valid, mean='context', metric='ALL')
        valid_mrr, valid_ndcg = metric_valid.test()
        
        # 점수 기록
        score_save_path = save_path + 'train/'
        with open(score_save_path + "valid_retrieve_score.txt", "w") as file:
            file.write(f"Valid 데이터에 대한 MRR@topk, NDCG@topk 점수를 기록합니다.\n\n")
            file.write("MRR score : " + str(value1) + "\n")
            file.write("NDCG value: " + str(value2) + "\n")
            
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
    printer.start("test를 위한 trainer 초기화")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=test_data,
        eval_examples=test_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=utils.compute_metrics,
    )
    printer.done()
    printer.start("predict 수행중...")
    predictions = trainer.predict(
        test_dataset=test_data,
        test_examples=test_dataset['validation']
    )
    printer.done()