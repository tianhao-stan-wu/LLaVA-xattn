# llama_xattn_modeling.py
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlamaXAttnConfig(LlamaConfig):
    model_type = "llama_xattn"


class LlamaXAttnModel(LlamaModel):
    config_class = LlamaXAttnConfig


class LlamaXAttnForCausalLM(LlamaForCausalLM):
    config_class = LlamaXAttnConfig


# class CrossAttentionLayer(nn.Module):
#     def __init__(self, hidden_size, num_attention_heads):
#         super().__init__()
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(hidden_size, self.all_head_size)
#         self.key = nn.Linear(hidden_size, self.all_head_size)
#         self.value = nn.Linear(hidden_size, self.all_head_size)
#         self.out = nn.Linear(hidden_size, hidden_size)

#         self.dropout = nn.Dropout(0.1)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states, context_states, attention_mask=None):
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(context_states)
#         mixed_value_layer = self.value(context_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)

#         if attention_mask is not None:
#             attention_scores = attention_scores + attention_mask

#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         attention_probs = self.dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         output = self.out(context_layer)
#         return output

# class LlamaXAttnModel(LlamaModel):
#     config_class = LlamaXAttnConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.cross_attention_layers = nn.ModuleList([CrossAttentionLayer(config.hidden_size, config.num_attention_heads) for _ in range(config.num_hidden_layers)])

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         context_input=None,
#         context_attention_mask=None
#     ):
#         # Get the hidden states from the original LlamaModel
#         outputs = super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = outputs[0]

#         # Apply cross-attention layers
#         for i, layer in enumerate(self.cross_attention_layers):
#             hidden_states = layer(hidden_states, context_input, context_attention_mask)

#         if not return_dict:
#             return (hidden_states,) + outputs[1:]

#         return CausalLMOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

# class LlamaXAttnForCausalLM(LlamaForCausalLM):
#     config_class = LlamaXAttnConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = LlamaXAttnModel(config)

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         context_input=None,
#         context_attention_mask=None
#     ):
#         return self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             context_input=context_input,
#             context_attention_mask=context_attention_mask
#         )

