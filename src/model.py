"""Transformer encoder-decoder, implemented from scratch.

Built from nn.Linear / nn.LayerNorm / nn.Dropout / nn.Embedding only. No
nn.Transformer, no nn.MultiheadAttention, no scaled_dot_product_attention --
the attention arithmetic is written out.

Architecture follows Vaswani et al. 2017, "Attention Is All You Need"
(arXiv:1706.03762), with two deliberate departures:

  * Pre-norm residual blocks rather than post-norm. Post-norm at 6+6 layers
    needs a carefully tuned warmup to avoid diverging early; pre-norm trains
    stably across a much wider range of settings.
  * Source, target and output-projection weights are tied into a single
    matrix. With a joint vocabulary this is the mechanism that shares
    representations between the two translation directions, and it removes
    ~16M parameters that 140k sentence pairs cannot support.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import PAD_ID  # noqa: E402


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention over h heads."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        """(B, L, D) -> (B, H, L, d_head)"""
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x):
        """(B, H, L, d_head) -> (B, L, D)"""
        b, h, l, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, h * d)

    def forward(self, query, key, value, mask=None, cache=None):
        """mask broadcasts to (B, H, Lq, Lk); True means *keep*."""
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))

        # Incremental decoding: append this step's k/v to the running cache.
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        return self.w_o(self._merge(out))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions (Vaswani et al. eq. 1)."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x, offset=0):
        return self.dropout(x + self.pe[:, offset : offset + x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, h, h, src_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, mem_mask, cache=None):
        h = self.norm1(x)
        x = x + self.dropout(
            self.self_attn(h, h, h, tgt_mask, cache=cache["self"] if cache else None)
        )
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, mem_mask))
        h = self.norm3(x)
        x = x + self.dropout(self.ff(h))
        return x


def padding_mask(seq, pad_id=PAD_ID):
    """(B, L) -> (B, 1, 1, L); True where the token is real."""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(size, device):
    """(1, 1, L, L) lower-triangular; True where attention is allowed."""
    return torch.ones(size, size, dtype=torch.bool, device=device).tril().unsqueeze(0).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
        d_ff=2048,
        dropout=0.3,
        max_len=128,
        tie_embeddings=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, max_len=max_len + 8, dropout=dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.out_proj.weight = self.embed.weight

        self._init_parameters()

    def _init_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embed" not in name and "out_proj" not in name:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD_ID].fill_(0)

    # -- forward passes ----------------------------------------------------

    def encode(self, src):
        mask = padding_mask(src)
        x = self.pos(self.embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, mask)
        return self.enc_norm(x), mask

    def decode(self, tgt_in, memory, mem_mask, offset=0, caches=None):
        b, l = tgt_in.shape
        x = self.pos(self.embed(tgt_in) * math.sqrt(self.d_model), offset=offset)

        if caches is None:
            tgt_mask = causal_mask(l, tgt_in.device) & padding_mask(tgt_in)
        else:
            # One step at a time: attend to everything cached so far.
            tgt_mask = None

        for i, layer in enumerate(self.decoder):
            x = layer(x, memory, tgt_mask, mem_mask,
                      cache=caches[i] if caches is not None else None)
        return self.dec_norm(x)

    def forward(self, src, tgt_in):
        memory, mem_mask = self.encode(src)
        h = self.decode(tgt_in, memory, mem_mask)
        return self.out_proj(h)

    def new_caches(self, n_layers=None):
        n = n_layers or len(self.decoder)
        return [{"self": {"k": None, "v": None}} for _ in range(n)]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Tied weights are shared, so named_parameters already counts them once.
        return total


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with uniform label smoothing, ignoring padding.

    Returns the summed loss and the number of scored tokens so that the
    training loop can normalise per token rather than per batch -- with
    token-budget batching the sentence count per step varies a lot.
    """

    def __init__(self, vocab_size, smoothing=0.1, ignore_index=PAD_ID):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        logits = logits.view(-1, self.vocab_size)
        target = target.reshape(-1)

        keep = target != self.ignore_index
        n_tokens = int(keep.sum())
        if n_tokens == 0:
            return logits.sum() * 0.0, 0

        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss[keep].sum(), n_tokens
