import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class CNN_BLOCK(nn.Module):
    def __init__(self, CFG):
        super(CNN_BLOCK, self).__init__()
        T = CFG['tokenizer']['max_seq_length']
        H = 768

        self.conv1 = nn.Conv1d(T, T * 2, 3, stride=1, padding='same')
        self.conv2 = nn.Conv1d(T * 2, T, 1, stride=1, padding='same')
        self.layernorm = nn.LayerNorm(H)

    def forward(self, x):
        _x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x) + _x
        x = self.layernorm(x)

        return x


class AutoModelForQuestionAnsweringAndCNN(AutoModelForQuestionAnswering):
    def __init__(self, config, CFG):
        super().__init__(config)
        self.init_weights()

        self.cnn_block_1 = CNN_BLOCK(CFG)
        self.cnn_block_2 = CNN_BLOCK(CFG)
        self.cnn_block_3 = CNN_BLOCK(CFG)
        self.cnn_block_4 = CNN_BLOCK(CFG)
        self.cnn_block_5 = CNN_BLOCK(CFG)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        masked_lm_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        ### BERT OUTPUT ###
        sequence_output = outputs[0]
        
        ### CNN 연산 ###
        sequence_output = self.cnn_block_1(sequence_output)
        sequence_output = self.cnn_block_2(sequence_output)
        sequence_output = self.cnn_block_3(sequence_output)
        sequence_output = self.cnn_block_4(sequence_output)
        sequence_output = self.cnn_block_5(sequence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )