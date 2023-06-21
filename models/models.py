import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForQuestionAnswering, AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class CNN_BLOCK(nn.Module):
    def __init__(self, T, H):
        super(CNN_BLOCK, self).__init__()
        
        self.conv1 = nn.Conv1d(T, T * 2, 3, stride=1, padding='same')
        self.conv2 = nn.Conv1d(T * 2, T, 1, stride=1, padding='same')
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.relu(output) + x
        output = self.layer_norm(output)

        return x


class AutoModelForQuestionAnsweringAndCNN(nn.Module):
    def __init__(self, CFG, config):
        super(AutoModelForQuestionAnsweringAndCNN, self).__init__()
        self.config = config
        self.CFG = CFG
        T = CFG['tokenizer']['max_seq_length']
        H = config.hidden_size

        self.PLM = AutoModel.from_pretrained(CFG['model']['model_name'])
        self.cnn_block_1 = CNN_BLOCK(T, H)
        self.cnn_block_2 = CNN_BLOCK(T, H)
        self.cnn_block_3 = CNN_BLOCK(T, H)
        self.cnn_block_4 = CNN_BLOCK(T, H)
        self.cnn_block_5 = CNN_BLOCK(T, H)
        self.qa_outputs = nn.Linear(H, 2)
    
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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.PLM(
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
        
class AutoModelForQuestionAnsweringAndMLM(nn.Module):
    def __init__(self, CFG, config):
        super(AutoModelForQuestionAnsweringAndMLM, self).__init__()
        self.config = config
        self.CFG = CFG

        self.PLM = AutoModel.from_pretrained(CFG['model']['model_name'])
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        self.mlm_outputs = nn.Linear(self.config.hidden_size, config.vocab_size)

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

        outputs = self.PLM(
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

        sequence_output = outputs[0]

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

        if masked_lm_labels is not None:
            # Compute the MLM loss
            mlm_loss_fct = CrossEntropyLoss(ignore_index=-100)
            prediction_scores = self.mlm_outputs(sequence_output)
            mlm_loss = mlm_loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = total_loss + (mlm_loss * 0.5) if total_loss is not None else mlm_loss

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


class AutoModelForQuestionAnsweringAndCustomLoss(AutoModelForQuestionAnswering):
    def __init__(self, config, CFG):
        super().__init__(config)
        self.init_weights()

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

        sequence_output = outputs[0]

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

            alpha = 0.3
            gamma = 2
            # p 구하기
            start_soft = F.softmax(start_logits, dim=1)
            end_soft = F.softmax(end_logits, dim=1)
            # weight 구하기
            start_weight = torch.pow(1.0 - start_soft, gamma)
            end_weight = torch.pow(1.0 - end_soft, gamma)
            # alpha 구하기
            start_soft_idx = torch.argmax(start_soft, axis=1)
            end_soft_idx = torch.argmax(end_soft, axis=1)
            start_same = start_soft_idx == start_positions
            end_same = end_soft_idx == end_positions

            start_alpha, end_alpha = [], []
            for i in range(len(start_same)):
                start_alpha.append(alpha if start_same[i] else 1 - alpha)
                end_alpha.append(alpha if end_same[i] else 1 - alpha)
            start_alpha = torch.tensor(start_alpha)
            end_alpha = torch.tensor(end_alpha)
            # focal loss 구하기
            start_focal_losses = -(start_alpha * start_weight * torch.log(start_soft))
            end_focal_losses =  -(end_alpha * end_weight * torch.log(end_soft))
            # start_loss, end_loss 구하기
            start_loss = torch.mean(start_focal_losses)
            end_loss = torch.mean(end_focal_losses)

            # loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
            # start_loss_list = loss_fct(start_logits, start_positions)
            # end_loss_list = loss_fct(end_logits, end_positions)
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