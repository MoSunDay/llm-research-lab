"""Convert a nanoGPT fine-tuned checkpoint to nano-vllm format (reverse conversion).

nanoGPT uses nn.Linear (not HF Conv1D), so weights are already in [out,in] layout
— no transpose needed (unlike convert_gpt2.py). The fused c_attn is split into
separate q/k/v projections. lm_head is tied to wte (skipped). Output: safetensors
+ config.json + tokenizer files, ready for nano-vllm LLM(model_path).

Inverse of convert_gpt2.py: this goes nanoGPT ckpt.pt -> nano-vllm, whereas
convert_gpt2.py goes HF GPT-2 -> nano-vllm. The only difference between the two
source formats is that nanoGPT's nn.Linear weights need no transpose (HF stores
Conv1D weights in [in,out] which must be transposed).
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import save_file

# GPT-2's real vocabulary size. nanoGPT pads this up to 50304 (next multiple of
# 64) for kernel efficiency, so the saved wte has 50304 rows. nano-vllm's GPT-2
# backend expects the unpadded 50257 — the trailing padding rows are unused.
GPT2_VOCAB_SIZE = 50257
TARGET_DTYPE = torch.float16


def main():
    ap = argparse.ArgumentParser(
        description="Convert a nanoGPT fine-tuned checkpoint to nano-vllm format."
    )
    ap.add_argument(
        "--src", required=True,
        help="nanoGPT checkpoint dir (containing ckpt.pt)",
    )
    ap.add_argument(
        "--dst", required=True,
        help="output dir for the nano-vllm model",
    )
    ap.add_argument(
        "--tokenizer", default="/root/models/gpt2-hf",
        help="tokenizer source dir (GPT-2 HF files)",
    )
    args = ap.parse_args()
    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    # ---- load the nanoGPT checkpoint -------------------------------------
    ckpt_path = os.path.join(src, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    print(f"loading {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch without the weights_only kwarg
        ckpt = torch.load(ckpt_path, map_location="cpu")

    # The nanoGPT checkpoint dict is {'model': state_dict, 'model_args': {...}, ...}.
    # Fall back to treating the whole thing as a raw state_dict if needed.
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
        model_args = ckpt.get("model_args", {}) or {}
    else:
        sd = ckpt
        model_args = {}

    # ---- strip the torch.compile _orig_mod. prefix ----------------------
    sd = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in sd.items()
    }

    # ---- derive architecture dims from model_args (fallback: the keys) ---
    n_layer = model_args.get("n_layer")
    n_head = model_args.get("n_head")
    n_embd = model_args.get("n_embd")
    block_size = model_args.get("block_size")
    bias = model_args.get("bias", True)

    if n_layer is None:
        idxs = set()
        for k in sd:
            m = re.match(r"transformer\.h\.(\d+)\.", k)
            if m:
                idxs.add(int(m.group(1)))
        n_layer = (max(idxs) + 1) if idxs else 0
    # n_embd can be inferred from the wte shape.
    if n_embd is None:
        n_embd = sd["transformer.wte.weight"].shape[1]
    if n_head is None:
        raise ValueError(
            "could not determine n_head from model_args; please pass a checkpoint "
            "that records model_args (n_head=...)"
        )
    if block_size is None:
        block_size = sd["transformer.wpe.weight"].shape[0]
    if not bias:
        print("NOTE: model_args.bias=False — synthesizing zero biases for nano-vllm")
    print(f"architecture: n_layer={n_layer} n_head={n_head} n_embd={n_embd} "
          f"block_size={block_size} bias={bias}")

    # ---- helpers ---------------------------------------------------------
    def has(key):
        return key in sd

    def linear_bias(key, weight):
        """Return a Linear bias, or zero-init matching out_features (weight.shape[0])."""
        if has(key):
            return sd[key].to(TARGET_DTYPE).contiguous()
        # nn.Linear bias has shape [out_features]; the weight is [out_features, in_features].
        n_out = weight.shape[0]
        print(f"  WARN: {key} missing (bias=False) -> zeros [{n_out}]")
        return torch.zeros(n_out, dtype=TARGET_DTYPE)

    def layernorm_bias(key, weight):
        """Return a LayerNorm bias, or zero-init matching the weight's full shape."""
        if has(key):
            return sd[key].to(TARGET_DTYPE).contiguous()
        shp = tuple(weight.shape)
        print(f"  WARN: {key} missing (bias=False) -> zeros {list(shp)}")
        return torch.zeros(shp, dtype=TARGET_DTYPE)

    tensors = {}

    # ---- embeddings ------------------------------------------------------
    # Token embeddings: nanoGPT pads vocab to 50304; truncate to GPT-2's 50257.
    wte = sd["transformer.wte.weight"]
    if wte.shape[0] != GPT2_VOCAB_SIZE:
        print(f"truncating wte {list(wte.shape)} -> [{GPT2_VOCAB_SIZE}, {wte.shape[1]}]")
        wte = wte[:GPT2_VOCAB_SIZE]
    tensors["model.embed_tokens.weight"] = wte.to(TARGET_DTYPE).contiguous()

    # Position embeddings: [block_size, n_embd] — no padding to undo.
    tensors["model.pos_embeddings.weight"] = (
        sd["transformer.wpe.weight"].to(TARGET_DTYPE).contiguous()
    )

    # ---- final LayerNorm -------------------------------------------------
    ln_f_w = sd["transformer.ln_f.weight"].to(TARGET_DTYPE).contiguous()
    tensors["model.norm.weight"] = ln_f_w
    tensors["model.norm.bias"] = layernorm_bias("transformer.ln_f.bias", ln_f_w)

    # ---- per-layer weights ----------------------------------------------
    for i in range(n_layer):
        p = f"model.layers.{i}"

        # LayerNorms: ln_1 -> input_layernorm, ln_2 -> post_attention_layernorm
        for ln_src, ln_dst in (
            ("ln_1", "input_layernorm"),
            ("ln_2", "post_attention_layernorm"),
        ):
            w = sd[f"transformer.h.{i}.{ln_src}.weight"].to(TARGET_DTYPE).contiguous()
            tensors[f"{p}.{ln_dst}.weight"] = w
            tensors[f"{p}.{ln_dst}.bias"] = layernorm_bias(
                f"transformer.h.{i}.{ln_src}.bias", w
            )

        # Fused c_attn -> split q/k/v. nanoGPT stores nn.Linear [3*n_embd, n_embd],
        # already in [out, in] layout, so NO transpose (the key difference from
        # convert_gpt2.py which transposes HF's Conv1D weights).
        c_attn_w = sd[f"transformer.h.{i}.attn.c_attn.weight"]  # [3*n_embd, n_embd]
        qw, kw, vw = (
            t.to(TARGET_DTYPE).contiguous()
            for t in c_attn_w.split(n_embd, dim=0)
        )
        c_attn_b_key = f"transformer.h.{i}.attn.c_attn.bias"
        if has(c_attn_b_key):
            qb, kb, vb = (
                t.to(TARGET_DTYPE).contiguous()
                for t in sd[c_attn_b_key].split(n_embd, dim=0)
            )
        else:
            print(f"  WARN: layer {i} attn.c_attn.bias missing (bias=False) -> zeros")
            # NOTE: each must be a distinct tensor (safetensors rejects shared storage).
            qb = torch.zeros(n_embd, dtype=TARGET_DTYPE)
            kb = torch.zeros(n_embd, dtype=TARGET_DTYPE)
            vb = torch.zeros(n_embd, dtype=TARGET_DTYPE)
        for name, w, b in (
            ("q_proj", qw, qb),
            ("k_proj", kw, kb),
            ("v_proj", vw, vb),
        ):
            tensors[f"{p}.self_attn.{name}.weight"] = w
            tensors[f"{p}.self_attn.{name}.bias"] = b

        # Attention output projection (nn.Linear, no transpose).
        o_w = (
            sd[f"transformer.h.{i}.attn.c_proj.weight"].to(TARGET_DTYPE).contiguous()
        )
        tensors[f"{p}.self_attn.o_proj.weight"] = o_w
        tensors[f"{p}.self_attn.o_proj.bias"] = linear_bias(
            f"transformer.h.{i}.attn.c_proj.bias", o_w
        )

        # MLP fc (nn.Linear, no transpose).
        fc_w = (
            sd[f"transformer.h.{i}.mlp.c_fc.weight"].to(TARGET_DTYPE).contiguous()
        )
        tensors[f"{p}.mlp.fc.weight"] = fc_w
        tensors[f"{p}.mlp.fc.bias"] = linear_bias(
            f"transformer.h.{i}.mlp.c_fc.bias", fc_w
        )

        # MLP proj (nn.Linear, no transpose).
        proj_w = (
            sd[f"transformer.h.{i}.mlp.c_proj.weight"].to(TARGET_DTYPE).contiguous()
        )
        tensors[f"{p}.mlp.proj.weight"] = proj_w
        tensors[f"{p}.mlp.proj.bias"] = linear_bias(
            f"transformer.h.{i}.mlp.c_proj.bias", proj_w
        )

    # lm_head is tied to wte in nanoGPT and nano-vllm uses tied embeddings, so it
    # is intentionally NOT written.

    # ---- save weights ----------------------------------------------------
    save_file(
        tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"}
    )
    print(f"wrote {len(tensors)} tensors -> {dst}/model.safetensors")

    # ---- config.json (native GPT-2 fields + nano-vllm aliases) ----------
    # Mirror convert_gpt2.py: native GPT-2 keys plus the alias keys the loader /
    # engine read (torch_dtype, max_position_embeddings, num_hidden_layers, ...).
    n_positions = block_size
    head_dim = n_embd // n_head
    cfg = {
        # native GPT-2 fields (same names/shape as HF config.json)
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "activation_function": "gelu_new",
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_positions": n_positions,
        "n_ctx": n_positions,
        "vocab_size": GPT2_VOCAB_SIZE,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        # nano-vllm alias fields (identical to convert_gpt2.py output)
        "torch_dtype": "float16",
        "max_position_embeddings": n_positions,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "hidden_size": n_embd,
        "num_key_value_heads": n_head,   # GPT-2 is MHA (kv heads == attn heads)
        "head_dim": head_dim,
    }
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote config.json (model_type={cfg['model_type']}, "
          f"dtype={cfg['torch_dtype']})")

    # ---- tokenizer files -------------------------------------------------
    if os.path.isdir(args.tokenizer):
        for fn in (
            "vocab.json", "merges.txt", "tokenizer.json",
            "tokenizer_config.json", "generation_config.json",
        ):
            s = os.path.join(args.tokenizer, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, fn))
        print(f"copied tokenizer files from {args.tokenizer}")
    else:
        print(f"WARNING: tokenizer dir {args.tokenizer!r} not found; "
              f"no tokenizer files copied")

    print("done.")


if __name__ == "__main__":
    main()
