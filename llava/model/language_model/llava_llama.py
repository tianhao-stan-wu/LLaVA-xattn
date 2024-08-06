# #    Copyright 2023 Haotian Liu
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.


# from typing import List, Optional, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import CrossEntropyLoss

# from transformers import AutoConfig, AutoModelForCausalLM, \
#                          LlamaConfig, LlamaModel, LlamaForCausalLM

# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.generation.utils import GenerateOutput

# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# # xattn imports
# import os
# import math
# from einops import rearrange, repeat
# from einops_exts import rearrange_many
# from torch import einsum, nn

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention
# from transformers.cache_utils import Cache, DynamicCache
# from transformers.modeling_outputs import BaseModelOutputWithPast
# from transformers.utils import logging
# from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# from .cache import StaticCache


# logger = logging.get_logger(__name__)


# def check_invalid_values(hidden_states):

#     invalid = False
#     if torch.isinf(hidden_states).any():
#         print("hidden_states contains 'inf' values.")
#         invalid = True
#     if torch.isnan(hidden_states).any():
#         print("hidden_states contains 'nan' values.")
#         invalid = True
#     if (hidden_states < 0).any():
#         print("hidden_states contains elements less than 0.")
#         invalid = True
#     return invalid


# def count_nan_values(tensor):

#     nan_count = torch.isnan(tensor).sum().item()
#     return nan_count


# def exists(val):
#     return val is not None


# def FeedForward(dim, mult=4):
#     inner_dim = int(dim * mult)
#     return nn.Sequential(
#         nn.LayerNorm(dim),
#         nn.Linear(dim, inner_dim, bias=False),
#         nn.GELU(),
#         nn.Linear(inner_dim, dim, bias=False),
#     )


# # gated cross attention
# class MaskedCrossAttention(nn.Module):
#     def __init__(
#         self,
#         *,
#         text_dim,
#         vision_dim,
#         head_dim=128,
#         num_heads=32,
#         only_attend_immediate_media=False,
#     ):
#         super().__init__()
#         self.scale = head_dim**-0.5
#         self.heads = num_heads
#         inner_dim = head_dim * num_heads

#         self.norm = nn.LayerNorm(text_dim)

#         self.to_q = nn.Linear(text_dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(vision_dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, text_dim, bias=False)

#         # Xavier Initialization
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_kv.weight)
#         nn.init.xavier_uniform_(self.to_out.weight)

#         # whether for text to only attend to immediate preceding image, or all previous images
#         self.only_attend_immediate_media = only_attend_immediate_media

#     def forward(self, x, media, media_locations=None, use_cached_media=False):
#         """
#         Args:
#             x (torch.Tensor): text features
#                 shape (B, T_txt, D_txt)
#             media (torch.Tensor): image features
#                 shape (B, T_img, n, D_img) where n is the dim of the latents
#             media_locations: boolean mask identifying the media tokens in x
#                 shape (B, T_txt)
#             use_cached_media: bool
#                 If true, treat all of x as if they occur after the last media
#                 registered in media_locations. T_txt does not need to exactly
#                 equal media_locations.shape[1] in this case
#         """

#         if not use_cached_media:
#             assert (
#                 media_locations.shape[1] == x.shape[1]
#             ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"

#         T_txt = x.shape[1]
#         # media has shape [16, 576, 4096], add dimension n
#         media = media.unsqueeze(2)
#         # now media has shape [16, 576, 1, 4096]
#         _, T_img, n = media.shape[:3]
#         h = self.heads

#         x = self.norm(x)
#         # torch.set_printoptions(threshold=torch.numel(x))  # Set the print threshold to the number of elements in the tensor
#         # print(x)
#         # torch.set_printoptions(profile="default")  # Reset print options to default after printing
#         # num_zeros = (x == 0).sum().item()
#         # print("Number of 0.0000e+00 values:", num_zeros)
#         print("x 142:", count_nan_values(x))


#         q = self.to_q(x)
#         print("q 172:", count_nan_values(q))
#         media = rearrange(media, "b t n d -> b (t n) d")

#         k, v = self.to_kv(media).chunk(2, dim=-1)
#         print("k 176:", count_nan_values(k))
#         print("v 176:", count_nan_values(v))
#         q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

#         q = q * self.scale
#         # print("q 181:", count_nan_values(q))

#         sim = einsum("... i d, ... j d -> ... i j", q, k)
#         # print("sim 184:", count_nan_values(sim))

#         if exists(media_locations):
#             media_time = torch.arange(T_img, device=x.device) + 1

#             if use_cached_media:
#                 # text time is set to the last cached media location
#                 text_time = repeat(
#                     torch.count_nonzero(media_locations, dim=1),
#                     "b -> b i",
#                     i=T_txt,
#                 )
#             else:
#                 # at each boolean of True, increment the time counter (relative to media time)
#                 text_time = media_locations.cumsum(dim=-1)

#             # text time must equal media time if only attending to most immediate image
#             # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
#             mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

#             text_to_media_mask = mask_op(
#                 rearrange(text_time, "b i -> b 1 i 1"),
#                 repeat(media_time, "j -> 1 1 1 (j n)", n=n),
#             )
#             sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)
#         # print("attn 212:", count_nan_values(attn))

#         if exists(media_locations) and self.only_attend_immediate_media:
#             # any text without a preceding media needs to have attention zeroed out
#             text_without_media_mask = text_time == 0
#             text_without_media_mask = rearrange(
#                 text_without_media_mask, "b i -> b 1 i 1"
#             )
#             attn = attn.masked_fill(text_without_media_mask, 0.0)
#             print("only_attend_immediate_media should be false")

#         out = einsum("... i j, ... j d -> ... i d", attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         print("205 out:", count_nan_values(out))

#         return self.to_out(out)

    
# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# class LlamaCrossAttention(LlamaAttention):

#     def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
#         super().__init__(config, layer_idx)
#         self.layer_idx = layer_idx

#     def forward(
#         self,
#         vision_input: torch.Tensor,
#         hidden_states: torch.Tensor,
#     ) ->torch.Tensor:

#         v_bsz, v_q_len, _ = vision_input.size()
#         bsz, q_len, _ = hidden_states.size()

#         if(self.layer_idx == 0):
#             print(hidden_states.shape)
#             print(hidden_states[0])
#             print(vision_input.shape)
#             print(vision_input[0])

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(vision_input)
#         value_states = self.v_proj(vision_input)
        
#         if(self.layer_idx == 0):
#             print("query_states", count_nan_values(query_states))
#             print("key_states", count_nan_values(key_states))
#             print("value_states", count_nan_values(value_states))

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(v_bsz, v_q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(v_bsz, v_q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = key_states.shape[-2]

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                 f" {attn_weights.size()}"
#             )

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#         attn_output = self.o_proj(attn_output)

#         return attn_output
     

# class GatedCrossAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         *,
#         text_dim,
#         vision_dim,
#         head_dim=128,
#         num_heads=32,
#         ff_mult=4,
#         only_attend_immediate_media=False,
#         # config,
#         # layer_idx
#     ):
#         super().__init__()
#         self.attn = MaskedCrossAttention(
#             text_dim=text_dim,
#             vision_dim=vision_dim,
#             head_dim=head_dim,
#             num_heads=num_heads,
#             only_attend_immediate_media=only_attend_immediate_media,
#         )
#         # self.attn = LlamaCrossAttention(config, layer_idx)
#         self.attn_gate = nn.Parameter(torch.tensor([0.0]))

#         self.ff = FeedForward(text_dim, mult=ff_mult)
#         self.ff_gate = nn.Parameter(torch.tensor([0.0]))

#     def forward(
#         self,
#         x, # language feature
#         media, # visual feature
#         media_locations=None,
#         use_cached_media=False,
#     ):
#         # x = (
#         #     self.attn(
#         #         vision_input=media,
#         #         hidden_states=x
#         #     )
#         #     * self.attn_gate.tanh()
#         #     + x
#         # )
#         x = (
#             self.attn(
#                 x,
#                 media,
#                 media_locations=media_locations,
#                 use_cached_media=use_cached_media,
#             )
#             * self.attn_gate.tanh()
#             + x
#         )
#         x = self.ff(x) * self.ff_gate.tanh() + x

#         # if(check_invalid_values(x)):
#         #     print("in GatedCrossAttentionBlock")

#         return x


# class LlamaXAttnDecoderLayer(LlamaDecoderLayer):

#     def __init__(self, config: LlamaConfig, layer_idx: int):
#         super().__init__(config, layer_idx)
        
#         self.layer_index = layer_idx

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         # x,
#         # media,
#         # media_locations,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*):
#                 attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
#                 query_sequence_length, key_sequence_length)` if default attention is used.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#         """

#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         # if(check_invalid_values(hidden_states)):
#         #     print("in LlamaXAttnDecoderLayer")

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs


# class LlamaXAttnModel(LlamaModel):
#     config_class = LlamaConfig

#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)

#         self.layers = nn.ModuleList(
#                 [LlamaXAttnDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#             )
        
#         self.gated_xattn = GatedCrossAttentionBlock(
#             text_dim=config.hidden_size,
#             vision_dim=config.hidden_size,
#             head_dim=128,
#             num_heads=32,
#             ff_mult=4,
#             # config=config,
#             # layer_idx=layer_idx,
#             only_attend_immediate_media=False
#         )
            
#         # pretrained_model_directory = "../.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
#         # print("loading vicuna weights to xattn...")

#         # import json
#         # f = open(pretrained_model_directory+"/pytorch_model.bin.index.json") 
#         # wmap = json.load(f)

#         # state_dict1 = torch.load(f"{pretrained_model_directory}/pytorch_model-00001-of-00002.bin", map_location='cpu')
#         # state_dict2 = torch.load(f"{pretrained_model_directory}/pytorch_model-00002-of-00002.bin", map_location='cpu')
        
#         # for layer in self.layers:
#         #     layer_idx = layer.layer_index
#         #     location_q = "model.layers." + str(layer_idx) + ".self_attn.q_proj.weight"
#         #     location_k = "model.layers." + str(layer_idx) + ".self_attn.k_proj.weight"
#         #     location_v = "model.layers." + str(layer_idx) + ".self_attn.v_proj.weight"
#         #     location_o = "model.layers." + str(layer_idx) + ".self_attn.o_proj.weight"
#         #     bin = wmap["weight_map"][location_q][18]
#         #     if bin == "1":
#         #         q_proj_weight = state_dict1[location_q]
#         #         k_proj_weight = state_dict1[location_k]
#         #         v_proj_weight = state_dict1[location_v]
#         #         o_proj_weight = state_dict1[location_o]
#         #     elif bin == "2":
#         #         q_proj_weight = state_dict2[location_q]
#         #         k_proj_weight = state_dict2[location_k]
#         #         v_proj_weight = state_dict2[location_v]
#         #         o_proj_weight = state_dict2[location_o]
                
#         #     layer.gated_xattn.attn.q_proj.weight.data = q_proj_weight
#         #     layer.gated_xattn.attn.k_proj.weight.data = k_proj_weight
#         #     layer.gated_xattn.attn.v_proj.weight.data = v_proj_weight
#         #     layer.gated_xattn.attn.o_proj.weight.data = o_proj_weight

#         #     if layer_idx == 31:
#         #         print(o_proj_weight)
#         #         print("***************")
#         #         print(layer.gated_xattn.attn.o_proj.weight.data)
        
#         # print("loaded!")
#         # f.close() 
    
#     def forward(
#         self,
#         # add vision input (media)
#         media,
#         media_locations,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError(
#                 "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
#             )

#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             # print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
#             use_cache = False

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         return_legacy_cache = False
#         if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
#             return_legacy_cache = True
#             past_key_values = DynamicCache.from_legacy_cache(past_key_values)
#             logger.warning_once(
#                 "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
#                 "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
#             )
#             # print("We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)")

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )
#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)

#         causal_mask = self._update_causal_mask(
#             attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
#         )
        
#         # embed positions
#         hidden_states = inputs_embeds

#         hidden_states = self.gated_xattn(hidden_states, media, media_locations)

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for decoder_layer in self.layers:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     # xattn input,
#                     hidden_states,
#                     # media,
#                     # media_locations,
#                     causal_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                     cache_position,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     # xattn input,
#                     # x=hidden_states,
#                     hidden_states=hidden_states,
#                     # media=media,
#                     # media_locations=media_locations,
#                     attention_mask=causal_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                     cache_position=cache_position,
#                 )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         print("595 hidden_states:", count_nan_values(hidden_states))

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None
#         if return_legacy_cache:
#             next_cache = next_cache.to_legacy_cache()

#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )

#     def _update_causal_mask(
#         self,
#         attention_mask: torch.Tensor,
#         input_tensor: torch.Tensor,
#         cache_position: torch.Tensor,
#         past_key_values: Cache,
#         output_attentions: bool,
#     ):
#         # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
#         # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
#         # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
#         # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

#         if self.config._attn_implementation == "flash_attention_2":
#             if attention_mask is not None and 0.0 in attention_mask:
#                 return attention_mask
#             return None

#         # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
#         # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
#         # to infer the attention mask.
#         past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#         using_static_cache = isinstance(past_key_values, StaticCache)

#         # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
#         if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
#             if AttentionMaskConverter._ignore_causal_mask_sdpa(
#                 attention_mask,
#                 inputs_embeds=input_tensor,
#                 past_key_values_length=past_seen_tokens,
#                 is_training=self.training,
#             ):
#                 return None

#         dtype, device = input_tensor.dtype, input_tensor.device
#         min_dtype = torch.finfo(dtype).min
#         sequence_length = input_tensor.shape[1]
#         if using_static_cache:
#             target_length = past_key_values.get_max_length()
#         else:
#             target_length = (
#                 attention_mask.shape[-1]
#                 if isinstance(attention_mask, torch.Tensor)
#                 else past_seen_tokens + sequence_length + 1
#             )

#         if attention_mask is not None and attention_mask.dim() == 4:
#             # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
#             if attention_mask.max() != 0:
#                 raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
#             causal_mask = attention_mask
#         else:
#             causal_mask = torch.full(
#                 (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
#             )
#             if sequence_length != 1:
#                 causal_mask = torch.triu(causal_mask, diagonal=1)
#             causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
#             causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
#             if attention_mask is not None:
#                 causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
#                 mask_length = attention_mask.shape[-1]
#                 padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
#                 padding_mask = padding_mask == 0
#                 causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
#                     padding_mask, min_dtype
#                 )
#         if (
#             self.config._attn_implementation == "sdpa"
#             and attention_mask is not None
#             and attention_mask.device.type == "cuda"
#             and not output_attentions
#         ):
#             # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
#             # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
#             # Details: https://github.com/pytorch/pytorch/issues/110213
#             causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

#         return causal_mask
    

# class LlamaXAttnForCausalLM(LlamaForCausalLM):
#     config_class = LlamaConfig

#     def __init__(self, config):
#         super().__init__(config)

#         self.model = LlamaXAttnModel(config)

#     def forward(
#         self,
#         # vision inputs
#         media,
#         media_locations,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
        
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             media=media,
#             media_locations=media_locations,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#         hidden_states = outputs[0]
#         if self.config.pretraining_tp > 1:
#             lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
#             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
#             logits = torch.cat(logits, dim=-1)
#         else:
#             logits = self.lm_head(hidden_states)
#         logits = logits.float()

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )




# class LlavaConfig(LlamaConfig):
#     model_type = "llava_llama"


# class LlavaLlamaModel(LlavaMetaModel, LlamaXAttnModel):
#     config_class = LlavaConfig

#     def __init__(self, config: LlamaConfig):
#         super(LlavaLlamaModel, self).__init__(config)


# class LlavaLlamaForCausalLM(LlamaXAttnForCausalLM, LlavaMetaForCausalLM):
#     config_class = LlavaConfig

#     def __init__(self, config):
#         super(LlamaXAttnForCausalLM, self).__init__(config)
#         self.model = LlavaLlamaModel(config)
#         self.pretraining_tp = config.pretraining_tp
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.image_id = 3027

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_model(self):
#         return self.model

#     def forward(
#         self,
#         media: Optional[torch.FloatTensor] = None,
#         media_locations: Optional[torch.LongTensor] = None,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         image_sizes: Optional[List[List[int]]] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:

#         if inputs_embeds is None:
#             (
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 inputs_embeds,
#                 labels,
#                 # ToDo: prepare vision embedding
#                 media,
#                 media_locations
#             ) = self.prepare_inputs_labels_for_multimodal(
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 labels,
#                 images,
#                 image_sizes
#             )

#         return super().forward(
#             media=media,
#             media_locations=media_locations,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         )

#     @torch.no_grad()
#     def generate(
#         self,
#         inputs: Optional[torch.Tensor] = None,
#         images: Optional[torch.Tensor] = None,
#         image_sizes: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#         position_ids = kwargs.pop("position_ids", None)
#         attention_mask = kwargs.pop("attention_mask", None)
#         if "inputs_embeds" in kwargs:
#             raise NotImplementedError("`inputs_embeds` is not supported")

#         if images is not None:
#             (
#                 inputs,
#                 position_ids,
#                 attention_mask,
#                 _,
#                 inputs_embeds,
#                 _,
#                 media,
#                 media_locations
#             ) = self.prepare_inputs_labels_for_multimodal(
#                 inputs,
#                 position_ids,
#                 attention_mask,
#                 None,
#                 None,
#                 images,
#                 image_sizes=image_sizes
#             )
#         else:
#             inputs_embeds = self.get_model().embed_tokens(inputs)


#         kwargs['media'] = media
#         kwargs['media_locations'] = media_locations

#         return super().generate(
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             inputs_embeds=inputs_embeds,
#             **kwargs
#         )

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
#                                       inputs_embeds=None, **kwargs):
#         images = kwargs.pop("images", None)
#         image_sizes = kwargs.pop("image_sizes", None)
#         media = kwargs.pop("media", None)
#         media_locations = kwargs.pop("media_locations", None)
#         inputs = super().prepare_inputs_for_generation(
#             input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
#         )
#         if images is not None:
#             inputs['images'] = images
#         if image_sizes is not None:
#             inputs['image_sizes'] = image_sizes
#         if media is not None:
#             inputs['media'] = media
#         if media_locations is not None:
#             inputs['media_locations'] = media_locations
#         return inputs

# AutoConfig.register("llava_llama", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)



# ********************************************************
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)