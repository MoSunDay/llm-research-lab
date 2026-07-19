"""Paged attention for nano-vllm, V100/sm_70 friendly.

Replaces the flash_attn based kernels with PyTorch scaled_dot_product_attention
(flash_attn is unavailable on sm_70). Supports:
  * varlen prefill (cu_seqlens_q/cu_seqlens_k, causal), incl. prefix-cache via block_tables
  * paged decode (gather K/V from block-paged cache using context_lens + block_tables)
KV-cache store and the paged-decode path are implemented in pure torch and are
CUDA-graph safe (no data-dependent CPU syncs). The original flash_attn/triton kernels
are unavailable on sm_70.
"""
import os
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context

from nanovllm.layers.triton_kernels import paged_decode


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """Scatter freshly computed K/V into the paged cache (CUDA-graph safe).

    key/value: [N, num_kv_heads, head_dim]
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping: [N] flat physical slot index (= block_id*block_size + offset), -1 = skip.
    No data-dependent Python branching / CPU sync, so this is legal inside graph capture.
    """
    block_size = k_cache.shape[1]
    valid = slot_mapping != -1                       # bool tensor (no sync)
    slots = torch.where(valid, slot_mapping, torch.zeros_like(slot_mapping)).long()
    block_idx = slots // block_size
    block_off = slots % block_size
    mask = valid.view(-1, *([1] * (key.dim() - 1)))
    cur_k = k_cache[block_idx, block_off]
    cur_v = v_cache[block_idx, block_off]
    k_cache[block_idx, block_off] = torch.where(mask, key, cur_k)
    v_cache[block_idx, block_off] = torch.where(mask, value, cur_v)


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
        # Paged decode. Primary path: Triton FlashAttention-style kernel that reads only
        # the real keys per seq (no padding to maxlen). ~7-20x faster than the torch
        # vectorized path which pads every seq to num_blocks*block_size. Pure device ops
        # (no CPU sync) => CUDA-graph safe. Falls back to torch if dtype/block mismatch.
        block_size = k_cache.shape[1]
        bt = context.block_tables
        cl = context.context_lens
        if (q.dtype == torch.float16 and (block_size & (block_size - 1)) == 0
                and block_size >= 16 and bt is not None and q.is_cuda
                and not os.environ.get("NANOVLLM_DETERMINISTIC_ATTENTION")):
            # Triton paged decode: ~7-20x faster per attention call than the torch path
            # (online softmax in fp32; ~8e-3 vs the cuBLAS path). Empirically bit-exact
            # for greedy: matches nanoGPT 960/960 tokens. Works in eager AND CUDA-graph
            # mode (q is materialized contiguous so capture is safe). Set
            # NANOVLLM_DETERMINISTIC_ATTENTION=1 to force the cuBLAS torch fallback.
            return paged_decode(q, k_cache, v_cache, bt, cl, self.scale)
        return self._decode_torch(q, k, v, context, k_cache, v_cache)

    def _decode_torch(self, q, k, v, context, k_cache, v_cache):
        # Vectorized paged decode (fallback). q: [B, num_heads, head_dim]. Gathers every
        # seq's full K/V in one advanced-index op and runs a single batched attention. No
        # Python loop and no CPU sync => CUDA-graph safe. Pads to maxlen so it does extra
        # (wasted) work for short contexts.
        bt = context.block_tables.long()              # [B, P]
        cl = context.context_lens                     # [B]
        B = q.shape[0]
        block_size = k_cache.shape[1]
        maxlen = bt.shape[1] * block_size             # padded context length (static)
        k_all = k_cache[bt].reshape(B, maxlen, self.num_kv_heads, self.head_dim)
        v_all = v_cache[bt].reshape(B, maxlen, self.num_kv_heads, self.head_dim)
        if self.num_kv_heads != self.num_heads:       # GQA expand along head dim
            rep = self.num_heads // self.num_kv_heads
            k_all = k_all.repeat_interleave(rep, dim=2)
            v_all = v_all.repeat_interleave(rep, dim=2)
        scores = torch.einsum("bhd,bphd->bhp", q, k_all) * self.scale   # [B, H, maxlen]
        pos = torch.arange(maxlen, device=q.device)
        valid = (pos.unsqueeze(0) < cl.unsqueeze(1)).unsqueeze(1)        # [B, 1, maxlen]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhp,bphd->bhd", attn, v_all)                # [B, H, head_dim]
        return out
