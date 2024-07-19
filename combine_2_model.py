from transformers import PreTrainedModel,LlamaForCausalLM,Cache
from torch import nn
from typing import Optional, List,Tuple
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from copy import deepcopy
class CombinedModelForCausalLM(PreTrainedModel):
    _tied_weights_keys = ["combined_lm_head.weight"]

    def __init__(self, model1:PreTrainedModel, model2:PreTrainedModel):
        config=deepcopy(model1.config)
        config._attn_implementation="eager"
        super().__init__(config)
        self.model1 = model1
        self.model2 = model2
        self.config=model1.config
        self.combined_lm_head=nn.Linear(model1.config.hidden_size+model2.config.hidden_size,model1.config.vocab_size,dtype=torch.float, bias=False)
        self.vocab_size=model1.config.vocab_size
        #self.post_init()

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):

        if past_key_values:
            past_key_values1=past_key_values[0]
            past_key_values2=past_key_values[1]
        else:
            past_key_values1=None
            past_key_values2=None

        outputs1 = self.model1(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               past_key_values=past_key_values1,
                               use_cache=use_cache,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        outputs2 = self.model2(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               past_key_values=past_key_values2,
                               use_cache=use_cache,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        #make sure the outputs are on the same device
        #outputs2.to(outputs1.device)

        #concatenate the last hidden states
        combined_hidden_states=torch.cat([outputs1.last_hidden_state,outputs2.last_hidden_state],dim=-1)
        #pass through the combined_lm_head
        logits=self.combined_lm_head(combined_hidden_states).float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(outputs1.past_key_values,outputs2.past_key_values),
            hidden_states=(outputs1.hidden_states,outputs2.hidden_states),
            attentions=(outputs1.attentions,outputs2.attentions),
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            past_key_values_all=past_key_values
            past_key_values=past_key_values[0]
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        else:
            past_key_values_all=None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values_all,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


