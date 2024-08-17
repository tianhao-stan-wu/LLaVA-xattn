# xattn is adapted from openflamingo
"""
Based on: https://github.com/mlfoundations/open_flamingo
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn


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

        self.text_norm = nn.LayerNorm(dim)
        # self.image_norm = nn.LayerNorm(dim_visual)

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

        h = self.heads

        x = self.text_norm(x)
        # media = self.image_norm(media)

        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

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
        layer_idx,
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
        self.layer_idx = layer_idx

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

