"""Export a nanoGPT agent checkpoint into Rust nvllm-models format.

Identical whitelist to export_safetensors.py (the Rust loader is
vocab-agnostic), but defaults --data_dir to the agent data and validates
vocab_size against agent_special_tokens.AGENT_VOCAB_SIZE so a mismatched
checkpoint fails loudly. Functional.
"""
import argparse, json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import agent_special_tokens as ast  # noqa: E402

_BLOCK_RE = re.compile(
    r"^transformer\.h\.\d+\.(?:ln_1|ln_2|attn\.c_attn|attn\.c_proj|"
    r"mlp\.c_fc|mlp\.c_proj)\.(?:weight|bias)$")
_FIXED_KEYS = {"transformer.wte.weight", "transformer.wpe.weight",
               "transformer.ln_f.weight", "transformer.ln_f.bias"}


def is_kept(key):
    return key in _FIXED_KEYS or bool(_BLOCK_RE.match(key))


def is_expected_droppable(key):
    return (key == "lm_head.weight" or key.endswith(".attn.bias")
            or key.endswith(".attn.masked_bias"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="out-agent-snake")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data-agent"))
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()
    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(args.in_dir, "ckpt.pt"),
                      map_location="cpu", weights_only=False)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in ckpt["model"].items()}
    src = ckpt.get("model_args") or ckpt.get("config") or {}
    cfg = {"block_size": int(src["block_size"]), "vocab_size": int(src["vocab_size"]),
           "n_layer": int(src["n_layer"]), "n_head": int(src["n_head"]),
           "n_embd": int(src["n_embd"])}
    bias = bool(src.get("bias", True))
    if not bias:
        sys.exit("ERROR: bias=False; Rust loader requires bias=True.")
    if cfg["vocab_size"] != ast.AGENT_VOCAB_SIZE:
        sys.exit(f"ERROR: vocab_size={cfg['vocab_size']} != agent vocab "
                 f"{ast.AGENT_VOCAB_SIZE}.")
    print("config:", cfg)

    kept, dropped = {}, []
    for k, v in sd.items():
        if is_kept(k):
            kept[k] = v
        else:
            dropped.append(k)
    unexpected = [k for k in dropped if not is_expected_droppable(k)]
    print(f"kept {len(kept)} tensors; dropped {len(dropped)}: {sorted(dropped)}")
    if unexpected:
        print("WARNING: unexpectedly dropped keys:", unexpected)

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    out = {k: v.to(dtype).contiguous() for k, v in kept.items()}
    st_path = os.path.join(out_dir, "model.safetensors")
    save_file(out, st_path, metadata={"format": "pt"})
    with safe_open(st_path, framework="pt") as f:
        assert f.get_tensor(next(iter(f.keys()))).dtype == dtype, "dtype mismatch"
        nkeys = len(f.keys())
    print(f"wrote {st_path} ({nkeys} tensors, dtype={dtype})")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    src_meta = os.path.join(args.data_dir, "meta.json")
    dst_meta = os.path.join(out_dir, "meta.json")
    if os.path.isfile(src_meta):
        shutil.copyfile(src_meta, dst_meta)
        print(f"copied meta.json <- {src_meta}")
    else:
        print(f"WARNING: {src_meta} not found; meta.json NOT written.")
    print(f"param count: {sum(v.numel() for v in out.values()):,}")
    print("outputs:", st_path, os.path.join(out_dir, "config.json"), dst_meta)


if __name__ == "__main__":
    main()
