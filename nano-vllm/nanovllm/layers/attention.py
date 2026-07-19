"""Paged attention for nano-vllm, V100/sm_70 friendly.

Replaces the flash_attn based kernels with PyTorch scaled_dot_product_attention
(flash_attn is unavailable on sm_70). Supports:
  * varlen prefill (cu_seqlens_q/cu_seqlens_k, causal), incl. prefix-cache via block_tables
  * paged decode (gather K/V from block-paged cache using context_lens + block_tables)
KV-cache store is implemented in pure torch (the original triton kernel is not
guaranteed to compile on sm_70).
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """Scatter freshly computed K/V into the paged cache.

    key/value: [N, num_kv_heads, head_dim]
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping: [N] flat physical slot index (= block_id*block_size + offset), -1 = skip
    """
    block_size = k_cache.shape[1]
    valid = slot_mapping != -1
    if not valid.any():
        return
    slots = slot_mapping[valid].long()
    k = key[valid]
    v = value[valid]
    block_idx = (slots // block_size)
    block_off = (slots % block_size)
    k_cache[block_idx, block_off] = k
    v_cache[block_idx, block_off] = v


class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    # ---- helpers ----------------------------------------------------------
    def _gather_kv(self, block_table_row, seqlen, k_cache, v_cache):
        """Gather K/V [seqlen, num_kv_heads, head_dim] from a paged cache for one seq."""
        block_size = k_cache.shape[1]
        n_blocks = (seqlen + block_size - 1) // block_size
        bt = block_table_row[:n_blocks].long()
        k = k_cache[bt].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen]
        v = v_cache[bt].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen]
        return k, v

    def _expand_gqa(self, t):
        """[seq, num_kv_heads, hd] -> [seq, num_heads, hd] for GQA."""
        if self.num_kv_heads == self.num_heads:
            return t
        rep = self.num_heads // self.num_kv_heads
        return t.repeat_interleave(rep, dim=1)

    def _sdpa(self, q, k, v, causal):
        # Deterministic scaled-dot-product attention (manual, no SDPA atomic
        # nondeterminism). q: [Sq, H, D]; k,v: [Sk, H, D] -> returns [Sq, H, D].
        q = q.transpose(0, 1)            # [H, Sq, D]
        k = k.transpose(0, 1)            # [H, Sk, D]
        v = v.transpose(0, 1)            # [H, Sk, D]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [H, Sq, Sk]
        if causal:
            sq, sk = scores.size(-2), scores.size(-1)
            i = torch.arange(sq, device=scores.device).unsqueeze(-1)
            j = torch.arange(sk, device=scores.device).unsqueeze(0)
            offset = sk - sq                       # prefix-cache: q is a suffix of k
            mask = j <= (i + offset)               # query t attends to keys <= offset + t
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, v)       # [H, Sq, D]
        return o.transpose(0, 1)        # [Sq, H, D]

    # ---- main forward -----------------------------------------------------
    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            return self._prefill(q, k, v, context, k_cache, v_cache)
        return self._decode(q, k, v, context, k_cache, v_cache)

    def _prefill(self, q, k, v, context, k_cache, v_cache):
        cu_q = context.cu_seqlens_q
        cu_k = context.cu_seqlens_k
        num_seqs = cu_q.shape[0] - 1
        # If prefix-cache is active, K/V live in the paged cache; gather full keys per seq.
        if context.block_tables is not None:
            k_list, v_list = [], []
            for i in range(num_seqs):
                sl = int(cu_k[i + 1] - cu_k[i])
                ki, vi = self._gather_kv(context.block_tables[i], sl, k_cache, v_cache)
                k_list.append(ki)
                v_list.append(vi)
            k = torch.cat(k_list, dim=0)
            v = torch.cat(v_list, dim=0)

        outs = []
        for i in range(num_seqs):
            q_i = q[int(cu_q[i]):int(cu_q[i + 1])]
            k_i = k[int(cu_k[i]):int(cu_k[i + 1])]
            v_i = v[int(cu_k[i]):int(cu_k[i + 1])]
            k_i = self._expand_gqa(k_i)
            v_i = self._expand_gqa(v_i)
            outs.append(self._sdpa(q_i, k_i, v_i, causal=True))
        return torch.cat(outs, dim=0)

    def _decode(self, q, k, v, context, k_cache, v_cache):
        # q: [num_seqs, num_heads, head_dim]; gather full K/V per seq from paged cache.
        num_seqs = q.shape[0]
        outs = []
        for i in range(num_seqs):
            sl = int(context.context_lens[i])
            ki, vi = self._gather_kv(context.block_tables[i], sl, k_cache, v_cache)
            ki = self._expand_gqa(ki)
            vi = self._expand_gqa(vi)
            outs.append(self._sdpa(q[i:i + 1], ki, vi, causal=False))
        return torch.cat(outs, dim=0)
