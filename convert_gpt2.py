"""Convert HF GPT-2 (124M) weights into nano-vllm format.

- Conv1D weights (c_attn/c_proj/c_fc/mlp.c_proj) are transposed to nn.Linear layout [out,in].
- Fused c_attn is split into q_proj/k_proj/v_proj (loaded via packed_modules_mapping).
- Embeddings (wte/wpe) are replicated (GPT-2 vocab 50257 is not divisible by TP=3).
- lm_head is tied to wte, so no separate weight is stored.
- Outputs fp16 model.safetensors + augmented config.json + tokenizer files.
"""
import argparse, json, os, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

N_LAYER = 12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/models/gpt2-hf")
    ap.add_argument("--dst", default="/root/models/gpt2-nanovllm")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    tensors = {}
    with safe_open(os.path.join(src, "model.safetensors"), "pt", "cpu") as f:
        keys = set(f.keys())

        def get(name):
            assert name in keys, f"missing {name}"
            return f.get_tensor(name)

        # embeddings (replicated)
        tensors["model.embed_tokens.weight"] = get("wte.weight").to(torch.float16)
        tensors["model.pos_embeddings.weight"] = get("wpe.weight").to(torch.float16)

        # final ln
        tensors["model.norm.weight"] = get("ln_f.weight").to(torch.float16)
        tensors["model.norm.bias"] = get("ln_f.bias").to(torch.float16)

        for i in range(N_LAYER):
            p = f"model.layers.{i}"
            # layernorms
            for ln_src, ln_dst in (("ln_1", "input_layernorm"), ("ln_2", "post_attention_layernorm")):
                tensors[f"{p}.{ln_dst}.weight"] = get(f"h.{i}.{ln_src}.weight").to(torch.float16)
                tensors[f"{p}.{ln_dst}.bias"] = get(f"h.{i}.{ln_src}.bias").to(torch.float16)

            # c_attn: Conv1D [768,2304] -> Linear [2304,768] -> split q/k/v
            c_attn_w = get(f"h.{i}.attn.c_attn.weight").t().contiguous()  # [2304,768]
            c_attn_b = get(f"h.{i}.attn.c_attn.bias")                      # [2304]
            qw, kw, vw = c_attn_w.split(768, dim=0)
            qb, kb, vb = c_attn_b.split(768, dim=0)
            for name, w, b in (("q_proj", qw, qb), ("k_proj", kw, kb), ("v_proj", vw, vb)):
                tensors[f"{p}.self_attn.{name}.weight"] = w.to(torch.float16)
                tensors[f"{p}.self_attn.{name}.bias"] = b.to(torch.float16)

            # attn output projection (Conv1D -> Linear)
            tensors[f"{p}.self_attn.o_proj.weight"] = get(f"h.{i}.attn.c_proj.weight").t().contiguous().to(torch.float16)
            tensors[f"{p}.self_attn.o_proj.bias"] = get(f"h.{i}.attn.c_proj.bias").to(torch.float16)

            # mlp c_fc (Conv1D -> Linear)
            tensors[f"{p}.mlp.fc.weight"] = get(f"h.{i}.mlp.c_fc.weight").t().contiguous().to(torch.float16)
            tensors[f"{p}.mlp.fc.bias"] = get(f"h.{i}.mlp.c_fc.bias").to(torch.float16)
            # mlp c_proj (Conv1D -> Linear)
            tensors[f"{p}.mlp.proj.weight"] = get(f"h.{i}.mlp.c_proj.weight").t().contiguous().to(torch.float16)
            tensors[f"{p}.mlp.proj.bias"] = get(f"h.{i}.mlp.c_proj.bias").to(torch.float16)

    save_file(tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {len(tensors)} tensors -> {dst}/model.safetensors")

    # config.json: keep GPT-2 native fields, add vllm-alias fields + dtype
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["torch_dtype"] = "float16"
    cfg["max_position_embeddings"] = cfg["n_positions"]
    cfg["num_hidden_layers"] = cfg["n_layer"]
    cfg["num_attention_heads"] = cfg["n_head"]
    cfg["hidden_size"] = cfg["n_embd"]
    cfg["num_key_value_heads"] = cfg["n_head"]  # GPT-2 is MHA
    cfg["head_dim"] = cfg["n_embd"] // cfg["n_head"]
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"wrote config.json (model_type={cfg['model_type']}, dtype={cfg['torch_dtype']})")

    # tokenizer files
    for fn in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, fn))
    print("copied tokenizer files")

if __name__ == "__main__":
    main()
