"""Dump reference logits from the nanoGPT checkpoint for Rust-vs-PyTorch parity.

Rebuilds the SAME nanoGPT model from a checkpoint, runs a fixed token sequence
on CPU (deterministic, no CUDA needed), and saves the last-position logits so a
Rust test can compare them numerically against the candle forward pass.

Outputs (in --in_dir):
  - ref_logits.npy : float32 array, shape (vocab,), logits at the last position.
  - ref_tokens.npy : int64 array, shape (seq_len,), the input token ids.
  - ref_meta.json  : {vocab_size, seq_len, dtype}.

Functional style: free functions only.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_nanogpt():
    env_path = os.environ.get("NANO_GPT_PATH", "")
    candidates = [env_path] if env_path else []
    d = SCRIPT_DIR
    for _ in range(6):
        cand = os.path.join(d, "link_repos", "nanoGPT")
        if os.path.isfile(os.path.join(cand, "model.py")):
            candidates.append(cand)
            break
        d = os.path.dirname(d)
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "model.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return importlib.import_module("model")
    sys.exit("ERROR: nanoGPT model.py not found (set NANO_GPT_PATH or place "
             "it at <repo>/link_repos/nanoGPT/).")


def load_model(ng, in_dir):
    ckpt = torch.load(os.path.join(in_dir, "ckpt.pt"),
                      map_location="cpu", weights_only=False)
    ma = ckpt.get("model_args") or ckpt["config"]
    cfg = ng.GPTConfig(
        n_layer=int(ma["n_layer"]), n_head=int(ma["n_head"]),
        n_embd=int(ma["n_embd"]), block_size=int(ma["block_size"]),
        bias=bool(ma.get("bias", True)), vocab_size=int(ma["vocab_size"]),
        dropout=0.0,
    )
    model = ng.GPT(cfg)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


def load_meta(in_dir, data_dir):
    for base in (in_dir, data_dir):
        p = os.path.join(base, "meta.json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    sys.exit("ERROR: meta.json not found in in_dir or --data_dir.")


def encode(text, stoi, block_size):
    ids = [stoi[c] for c in text if c in stoi]
    return ids[:block_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="out-gpt2-snake")
    ap.add_argument("--data_dir",
                    default=os.path.join(SCRIPT_DIR, "data-shakespeare-char"))
    ap.add_argument("--seq", default=None,
                    help="override token list (comma-separated ints)")
    ap.add_argument("--text", default="SNAKE\nHEAD=3,5\nMOVE:",
                    help="text encoded via meta.json when --seq is absent")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.in_dir, "ref_logits.npy")

    ng = load_nanogpt()
    model, cfg = load_model(ng, args.in_dir)
    meta = load_meta(args.in_dir, args.data_dir)
    stoi, itos = meta["stoi"], meta["itos"]

    if args.seq is not None:
        ids = [int(x) for x in args.seq.split(",") if x.strip() != ""]
    else:
        ids = encode(args.text, stoi, cfg.block_size)
    if len(ids) == 0:
        sys.exit("ERROR: empty input sequence (text chars not in vocab?)")
    if len(ids) > cfg.block_size:
        sys.exit(f"ERROR: seq len {len(ids)} > block_size {cfg.block_size}")

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        logits, _ = model(x)  # nanoGPT forward returns (logits, loss)
    # nanoGPT's no-target path optimizes to the last position; either way the
    # last row equals the Rust forward's last-position logits (linear lm_head).
    last = logits[0, -1, :].detach().cpu().to(torch.float32).numpy()

    np.save(out_path, last)
    np.save(os.path.join(args.in_dir, "ref_tokens.npy"),
            np.asarray(ids, dtype=np.int64))
    with open(os.path.join(args.in_dir, "ref_meta.json"), "w") as f:
        json.dump({"vocab_size": int(cfg.vocab_size), "seq_len": len(ids),
                   "dtype": "float32"}, f)

    # Parity payload for the Rust-side numerical check (ASCII JSON). The Rust
    # test reads tokens + last-position logits + W/A/S/D ids. Rust GELU is erf
    # while nanoGPT uses the tanh approx, so floats will be close but not equal.
    action_chars = ["W", "A", "S", "D"]
    action_ids = [int(stoi[c]) for c in action_chars]
    last_f32 = last.astype(np.float32)
    parity = {
        "tokens": [int(i) for i in ids],
        "last_logits": last_f32.tolist(),
        "action_chars": action_chars,
        "action_ids": action_ids,
        "note": "PyTorch nanoGPT last-position logits (tanh-GELU). "
                "Rust uses erf-GELU -> small divergence expected.",
    }
    parity_path = os.path.join(args.in_dir, "parity.json")
    with open(parity_path, "w") as f:
        json.dump(parity, f, indent=2)
    print(f"wrote {parity_path} (tokens={len(ids)}, "
          f"logits={len(parity['last_logits'])}, action_ids={action_ids})")

    top5 = np.argsort(last)[::-1][:5]
    print(f"input ids ({len(ids)}): {ids}")
    decoded = "".join(itos[i] if i < len(itos) else "?" for i in ids)
    print("decoded input:", repr(decoded))
    print("top-5 tokens at the last position:")
    for t in top5:
        ch = itos[t] if t < len(itos) else "?"
        print(f"  id={t:>3} char={ch!r} logit={last[t]:.6f}")
    print(f"wrote {out_path}, ref_tokens.npy, ref_meta.json (shape={last.shape})")


if __name__ == "__main__":
    main()
