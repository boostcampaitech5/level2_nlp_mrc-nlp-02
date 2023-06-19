import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import pandas as pd
from tqdm.auto import tqdm
import pickle
from torch.nn.utils.rnn import pad_sequence

tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/kobart-QuestionGeneration')
model = BartForConditionalGeneration.from_pretrained('Sehong/kobart-QuestionGeneration')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


origin_dataset = pd.read_csv('/opt/ml/input/data/preprocessed_ner.csv', encoding = 'utf-8-sig')
dataset = origin_dataset[['context','answer']]
dataset = dataset[dataset['answer'].notna()]

del origin_dataset

question = []

BATCH_SIZE = 20  # adjust to your GPU capacity
PADDING_TOKEN_ID = tokenizer.pad_token_id

# Prepare batches
inputs = [tokenizer.encode(row['context'] + ' <unused0> ' + row['answer']) for _, row in dataset.iterrows()]
input_batches = [inputs[i:i+BATCH_SIZE] for i in range(0, len(inputs), BATCH_SIZE)]

questions = []

for input_batch in tqdm(input_batches):
    # Pad sequences to the same length
    input_ids = pad_sequence([torch.tensor(seq) for seq in input_batch], padding_value=PADDING_TOKEN_ID, batch_first=True)

    # Move to GPU
    input_ids = input_ids.to(device)

    # Generate summaries
    summary_ids = model.generate(input_ids, max_length=60, pad_token_id=PADDING_TOKEN_ID)

    # Decode generated sequences, ignoring padding tokens
    batch_questions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
    
    questions.extend(batch_questions)

with open('./questions_batch.bin', 'wb') as f:
    pickle.dump(questions, f)