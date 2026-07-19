"""Triton kernels for nano-vllm (sm_70/V100 compatible).

paged_decode: FlashAttention-style online-softmax paged KV-cache decode attention.
One program per (seq, query-head); iterates cache blocks in BLOCK-sized chunks,
processing only the real keys (no padding to max context). Cuts K/V traffic vs the
torch vectorized path which pads to maxlen = num_blocks * block_size.

Numerical model: mirrors the fp16 reference (cuBLAS fp16-GEMM with fp32 accumulate):
QK and PV products are computed in fp16 (matching torch einsum/nanoGPT matmul rounding)
while the softmax statistics (max, sum) and accumulators stay fp32 for stability.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, KCache, VCache, Out, BlockTables, ContextLens, scale,
    stride_qs, stride_qh,
    stride_kblk, stride_kcbs, stride_kkvh, stride_khd,
    stride_vblk, stride_vcbs, stride_vkvh, stride_vhd,
    stride_bts,
    GROUP: tl.constexpr, HEAD_DIM: tl.constexpr,
    CACHE_BLOCK: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    INNER: tl.constexpr, BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP
    seq_len = tl.load(ContextLens + seq)
    offs_d = tl.arange(0, HEAD_DIM)
    qh = tl.load(Q + seq * stride_qs + head * stride_qh + offs_d)          # fp16 [HEAD_DIM]
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for b in tl.static_range(MAX_BLOCKS):
        block_id = tl.load(BlockTables + seq * stride_bts + b)
        for ci in tl.static_range(INNER):
            off = ci * BLOCK + tl.arange(0, BLOCK)
            key_pos = b * CACHE_BLOCK + off
            valid = key_pos < seq_len
            k_ptrs = (KCache + block_id * stride_kblk + off[:, None] * stride_kcbs
                      + kv_head * stride_kkvh + offs_d[None, :])
            v_ptrs = (VCache + block_id * stride_vblk + off[:, None] * stride_vcbs
                      + kv_head * stride_vkvh + offs_d[None, :])
            kh = tl.load(k_ptrs, mask=valid[:, None], other=0.0)            # fp16 [BLOCK, HEAD_DIM]
            vh = tl.load(v_ptrs, mask=valid[:, None], other=0.0)            # fp16 [BLOCK, HEAD_DIM]
            # fp16 products, fp32 accumulate (matches cuBLAS fp16 GEMM rounding)
            scores = tl.sum((qh[None, :] * kh).to(tl.float32), axis=1) * scale
            scores = tl.where(valid, scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum((p[:, None].to(tl.float16) * vh).to(tl.float32), axis=0)
            m_i = m_new
    out = acc / l_i
    tl.store(Out + seq * stride_qs + head * stride_qh + offs_d, out.to(tl.float16))


def paged_decode(q, k_cache, v_cache, block_tables, context_lens, scale):
    """Paged decode attention via Triton.

    q: [B, num_heads, head_dim] (fp16)
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim] (fp16)
    block_tables: [B, P] int32, context_lens: [B] int32
    Returns: [B, num_heads, head_dim] fp16.
    """
    B, H, D = q.shape
    q = q.contiguous()   # materialize qkv slice view (stride 3*H*D) -> contiguous [B,H,D]
    nkv = k_cache.shape[2]
    cache_block = k_cache.shape[1]
    max_blocks = block_tables.shape[1]
    GROUP = H // nkv
    BLOCK = 64 if cache_block >= 64 else cache_block
    assert cache_block % BLOCK == 0, "cache block_size must be a multiple of BLOCK"
    o = torch.empty_like(q)
    grid = (B, H)
    _paged_decode_kernel[grid](
        q, k_cache, v_cache, o, block_tables, context_lens, scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0),
        GROUP=GROUP, HEAD_DIM=D, CACHE_BLOCK=cache_block, MAX_BLOCKS=max_blocks,
        INNER=cache_block // BLOCK, BLOCK=BLOCK,
    )
    return o
