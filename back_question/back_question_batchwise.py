import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import pandas as pd
from tqdm.auto import tqdm
import pickle

tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/kobart-QuestionGeneration')
model = BartForConditionalGeneration.from_pretrained('Sehong/kobart-QuestionGeneration')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

origin_dataset = pd.read_csv('/opt/ml/input/data/preprocessed_ner.csv', encoding = 'utf-8-sig')
dataset = origin_dataset[['context','answer']][1800000:]
dataset = dataset[dataset['answer'].notna()]

del origin_dataset

question = []

BATCH_SIZE = 50  

contexts = dataset['context'].tolist()
answers = dataset['answer'].tolist()
combined_texts = [c + ' <unused0> ' + a for c, a in tqdm(zip(contexts, answers), total=len(contexts))]
inputs = tokenizer.batch_encode_plus(combined_texts, truncation=True, padding=True, return_tensors="pt")
print('tokenized_set')

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

input_ids_batches = input_ids.split(BATCH_SIZE)
attention_mask_batches = attention_mask.split(BATCH_SIZE)

questions = []

for input_ids, attention_mask in tqdm(zip(input_ids_batches, attention_mask_batches), total=len(input_ids_batches)):
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    
    summary_ids = model.generate(input_ids, max_length=60, attention_mask=attention_mask)

    
    batch_questions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
    
    questions.extend(batch_questions)

with open('./questions_batch18toall.bin', 'wb') as f:
    pickle.dump(questions, f)
