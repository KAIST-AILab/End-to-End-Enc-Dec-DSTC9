from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_utils import SequenceSummary


class GPT2End2EndModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.detection_head = SequenceSummary(config)
        self.selection_head = SequenceSummary(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.task = 'generation'  # None, 'detection', 'selection', 'generation'

    def get_task(self):
        return self.task

    def set_task(self, task=None):
        if task not in ['detection', 'selection', 'generation']:
            task = None
        self.task = task

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            reduction='mean',
            use_cache=True,
    ):
        """
        input_ids: `torch.LongTensor` of shape (bach_size, sequence_length) for detection and generation and (batch_size, num_candidates, sequence_length) for selection.
            Input sequence tokens.
        mc_token_ids: `torch.LongTensor` of shape (batch_size,) for detection and (batch_size, num_candidates) for selection.
            Indices of the classification token in each input sequence for detection or selection task.
        labels: `torch.LongTensor` of shape (bach_size,) for detection and selection and (batch_size, sequence_length) for generation.
            Labels for each task.
            For detection task, labels should be 0 or 1.
            For selection task, labels should be in [0, num_candidates) where `num_candidates` is the size of the second dimension of the input tensors.
            For generation task, all labels set to ``-100`` are ignored (masked).

    Return:
        loss (optional, returned when `labels` is provided): scalar `torch.FloatTensor`.
            Loss for the task.
        prediction_scores: `torch.FloatTensor` of shape (batch_size,) for detection, (batch_size, num_candidates) for selection and (batch_size, sequence_length, config.vocab_size) for generation.
            Prediction scores for the task.
        past: `List[torch.FloatTensor]` of length `config.n_layers`.
            Contains pre-computed hidden-states (key and values in the attention blocks).
        hidden_states (optional, returned when `config.output_hidden_states=True`): Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer).
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (optional, returned when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        """
        if self.task == 'selection':
            batch_size, n_cand = input_ids.shape[:2]
            input_ids = input_ids.view(batch_size * n_cand, -1)
            token_type_ids = token_type_ids.view(batch_size * n_cand, -1)

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        if self.task is None:
            return transformer_outputs
        hidden_states = transformer_outputs[0]

        if self.task == 'detection':
            outputs = self._forward_detection_head(
                hidden_states, mc_token_ids, labels, reduction)
        elif self.task == 'selection':
            hidden_states = hidden_states.view(
                batch_size, n_cand, -1, hidden_states.shape[-1])
            outputs = self._forward_selection_head(
                hidden_states, mc_token_ids, labels, reduction)
        else:
            outputs = self._forward_generation_head(
                hidden_states, labels, reduction)
        outputs = outputs + transformer_outputs[1:]
        # (loss), logits, presents, (all hidden_states), (attentions)
        return outputs

    def _forward_detection_head(self, hidden_states, mc_token_ids, labels=None, reduction='mean'):
        logits = self.detection_head(hidden_states, mc_token_ids).squeeze(-1)
        outputs = (logits,)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(reduction=reduction)
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs

    def _forward_selection_head(self, hidden_states, mc_token_ids, labels=None, reduction='mean'):
        logits = self.selection_head(hidden_states, mc_token_ids).squeeze(-1)
        outputs = (logits,)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(reduction=reduction)
            loss = loss_fct(logits, labels.float())
            outputs = (loss,) + outputs
        return outputs

    def _forward_generation_head(self, hidden_states, labels=None, reduction='mean'):
        logits = self.lm_head(hidden_states)
        outputs = (logits,)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            _logits = shift_logits.view(-1, shift_logits.size(-1))
            _labels = shift_labels.view(-1)

            loss_fct = CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(_logits, _labels)
            outputs = (loss,) + outputs
        return outputs
