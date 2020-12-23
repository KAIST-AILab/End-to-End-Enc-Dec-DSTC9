from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_utils import SequenceSummary


class T5DoubleHeadsModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        config.summary_type = 'first'
        config.summary_use_proj = True
        config.summary_activation = None
        config.summary_proj_to_labels = True
        config.summary_first_dropout = 0.1
        self.multiple_choice_head = SequenceSummary(config)
        self.mc_loss_fct = CrossEntropyLoss

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_past_key_value_states=None,
            use_cache=False,
            lm_labels=None,
            mc_labels=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            encoder_only=False,
            reduction='mean',
    ):
        r"""
        input_ids: `torch.LongTensor` of shape (batch_size, num_candidates, sequence_length):
            Input sequence tokens for encoder.
        decoder_input_ids: `torch.LongTensor` of shape (batch_size * num_candidates, sequence_length):
            Input sequence tokens for decoder.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_candidates, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the multiple choice classification loss.
                Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
                of the input tensors. (see `input_ids` above)
        encoder_only (`bool`):
                Skips the generation if set.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
            Classification loss (cross entropy).
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.
        """
        if input_ids is not None:
            batch_size, num_candidates, _ = input_ids.shape
            input_ids = input_ids.view(-1, input_ids.size(-1))

        # Encode if needed (training, first prediction pass)
        mc_outputs = ()
        if encoder_outputs is not None:
            hidden_states = encoder_outputs[0]
        else:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )
            hidden_states = encoder_outputs[0]

            mc_logits = self.multiple_choice_head(
                hidden_states).view(batch_size, num_candidates)
            mc_outputs = (mc_logits,)
            if mc_labels is not None:
                loss_fct = self.mc_loss_fct(reduction=reduction)
                loss = loss_fct(mc_logits, mc_labels)
                mc_outputs = (loss,) + mc_outputs
            if encoder_only:
                return mc_outputs + encoder_outputs

        # Decode
        if lm_labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        decoder_outputs = (lm_logits,) + mc_outputs + decoder_outputs[1:]
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(reduction=reduction, ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs + encoder_outputs


class T5DoubleHeadsMultiLabelModel(T5DoubleHeadsModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        config.summary_type = 'first'
        config.summary_use_proj = True
        config.summary_activation = None
        config.summary_proj_to_labels = True
        config.summary_first_dropout = 0.1
        self.multiple_choice_head = SequenceSummary(config)
        self.mc_loss_fct = BCEWithLogitsLoss

        self.init_weights()
