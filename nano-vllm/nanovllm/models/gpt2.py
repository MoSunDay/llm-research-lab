"""nano-vllm backend for nanoGPT/HF GPT-2 (124M), TP-friendly.

Differences vs Qwen3 backend:
  * Standard LayerNorm with bias (not RMSNorm).
  * Learned absolute position embeddings wpe (no RoPE).
  * MLP is a plain GELU MLP (ColumnParallelLinear -> GELU -> RowParallelLinear).
  * Vocab embeddings (wte/wpe) and lm_head are REPLICATED (GPT-2 vocab 50257 is
    not divisible by TP=3); only attention QKV/O and MLP fc/proj are sharded.
  * GELU uses the exact erf variant (nn.GELU) to match nanoGPT model.py.
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import QKVParallelLinear, ColumnParallelLinear, RowParallelLinear
from nanovllm.utils.context import get_context


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (GPT-2 uses bias=True, eps=1e-5)."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class GPT2Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        tp_size = torch.distributed.get_world_size()
        self.total_num_heads = config.n_head
        assert self.total_num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.total_num_heads   # GPT-2 is MHA
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        hidden_size = config.n_embd
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim,
            self.total_num_heads, self.total_num_kv_heads, bias=True,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=True,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # GPT-2 uses learned positions -> no rotary embedding applied here.
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class GPT2MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        intermediate_size = 4 * config.n_embd
        self.fc = ColumnParallelLinear(hidden_size, intermediate_size, bias=True)
        self.proj = RowParallelLinear(intermediate_size, hidden_size, bias=True)
        self.act = nn.GELU()   # exact erf variant, matches nanoGPT model.py

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


class GPT2DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LayerNorm(config.n_embd, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.self_attn = GPT2Attention(config)
        self.post_attention_layernorm = LayerNorm(config.n_embd, eps=getattr(config, "layer_norm_epsilon", 1e-5))
        self.mlp = GPT2MLP(config)

    def forward(self, positions, hidden_states):
        # GPT-2 pre-norm with explicit (non-fused) residual additions.
        h = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.self_attn(positions, h)
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(h)
        return hidden_states


class GPT2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        # REPLICATED token + position embeddings (vocab 50257 not divisible by TP=3).
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.n_positions, config.n_embd)
        self.layers = nn.ModuleList([GPT2DecoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.n_embd, eps=getattr(config, "layer_norm_epsilon", 1e-5))

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids) + self.pos_embeddings(positions)
        for layer in self.layers:
            hidden = layer(positions, hidden)
        hidden = self.norm(hidden)
        return hidden


class GPT2ForCausalLM(nn.Module):
    # Map the split QKV names written by convert_gpt2.py onto the fused qkv_proj.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config):
        super().__init__()
        self.model = GPT2Model(config)
        # lm_head is tied to the (replicated) token embedding; no separate param.

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = hidden_states[last_indices].contiguous()
        else:
            x = hidden_states
        return F.linear(x, self.model.embed_tokens.weight)
