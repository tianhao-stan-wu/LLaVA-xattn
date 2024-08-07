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



# xattn is adapted from openflamingo
"""
Based on: https://github.com/mlfoundations/open_flamingo
"""

from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, media):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt 4098)
            media (torch.Tensor): image features
                shape (B, T_img, D_img 64) where n is the dim of the latents    
        """

        print("x.shape:", x.shape)
        print("media.shape:", media.shape)

        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        print("q.shape:", q.shape)
        print("k.shape:", k.shape)

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x, # language feature
        media, # visual feature
    ):
        x = (
            self.attn(
                x,
                media,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x


def build_mm_resampler():

    modules = [nn.Linear(4096, 1024)]
    modules.append(nn.GELU())
    modules.append(nn.Linear(1024, 256))
    modules.append(nn.GELU())
    modules.append(nn.Linear(256, 64))
    return nn.Sequential(*modules)


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

        # add a two-layer MLP that compresses img features to reduce computational complexity
        # in cross-attention layer
        self.mm_resampler = build_mm_resampler()
        self.mm_xattn = GatedCrossAttentionBlock(dim=4096, dim_visual=64)

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