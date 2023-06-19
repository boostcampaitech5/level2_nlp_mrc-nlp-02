import torch
import numpy as np

def train_tokenizing(examples, tokenizer, pad_on_right, CFG, column_names):
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=CFG['tokenizer']['max_seq_length'],
        stride=CFG['tokenizer']['doc_stride'],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False if 'roberta' in CFG['model']['model_name'] else True, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
        padding="max_length" if CFG['tokenizer']['pad_to_max_length'] else False,
    )
    if CFG['model']['option'] == 'question_masking':
        for idx in range(len(tokenized_examples["input_ids"])):
            # Create a probability matrix with the same length as the current input_ids
            probability_matrix = torch.full((len(tokenized_examples["input_ids"][idx]), ), 0.15)
            # Get the special tokens mask for the current input_ids
            special_tokens_mask = tokenizer.get_special_tokens_mask(tokenized_examples["input_ids"][idx], already_has_special_tokens=True)
            # Update the probability matrix with the special tokens mask
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            # Get the sequence ids for the current example
            sequence_ids = tokenized_examples.sequence_ids(idx)
            # Replace None values with 0
            sequence_ids = [0 if v is None else v for v in sequence_ids]
            # Convert sequence_ids back to tensor
            sequence_ids = torch.tensor(sequence_ids, dtype=torch.bool)
            # Update the probability matrix with the sequence ids
            probability_matrix.masked_fill_(sequence_ids, value=0.0)  # apply 0 probability for context tokens
            # Calculate the masked indices
            masked_indices = torch.bernoulli(probability_matrix).bool()
            # Apply masking and create labels
            masked_input_ids = np.where(masked_indices, tokenizer.convert_tokens_to_ids(tokenizer.mask_token), tokenized_examples["input_ids"][idx])
            labels = np.where(~masked_indices, -100, tokenized_examples["input_ids"][idx])
            tokenized_examples["input_ids"][idx] = masked_input_ids.tolist()
            # Ensure labels are of the same length as input_ids
            labels_padded = [-100] * len(tokenized_examples["input_ids"][idx])
            labels_padded[:len(labels)] = labels.tolist()

            tokenized_examples["masked_lm_labels"].append(labels_padded)
            

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
    # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 데이터셋에 "start position", "enc position" label을 부여합니다.
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]

        # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # text에서 정답의 Start/end character index
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # text에서 current span의 Start token index
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # text에서 current span의 End token index
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def val_tokenizing(examples, tokenizer, pad_on_right,  CFG, column_names):
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=CFG['tokenizer']['max_seq_length'],
        stride=CFG['tokenizer']['doc_stride'],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False if 'roberta' in CFG['model']['model_name'] else True, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
        padding="max_length" if CFG['tokenizer']['pad_to_max_length'] else False,
    )

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
    # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples