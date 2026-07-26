"""
Loss-mask SFT trainer for the GPT-2 agent.

Trains a char-level GPT-2 (vocab=73, 12/768/12, bias=True, from scratch or
warm-started from the shakespeare checkpoint) on the agent traces produced by
prepare_agent_snake.py. Only tokens flagged mask=1 contribute to the loss
(the model's own outputs); injected prompt/tool-result tokens are ignored via
a per-target weight.

Loss (per micro-batch). nanoGPT returns full-sequence logits only when
    logits,_ = model(x, y)                    # nanoGPT needs targets to return full (B,T,V)
    ce = CE(logits, y, reduction='none')     # y = x shifted by one
    loss = (ce * w).sum() / w.sum().clamp(min=1)
where w is the mask sampled aligned to y (the target token), so mask==1 exactly
on positions the model is supposed to generate.

nanoGPT-native checkpoint; export_agent_safetensors.py converts it. Functional.
"""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import agent_special_tokens as ast  # noqa: E402


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
            import importlib
            return importlib.import_module("model")
    sys.exit("ERROR: nanoGPT model.py not found (set NANO_GPT_PATH).")


def get_batch(data_dir, split, block_size, batch_size, device, device_type):
    """Sample windows of x, y, and a target-aligned supervision mask."""
    fn = "train.bin" if split == "train" else "val.bin"
    mfn = "train.mask.bin" if split == "train" else "val.mask.bin"
    data = np.memmap(os.path.join(data_dir, fn), dtype=np.uint16, mode="r")
    mdata = np.memmap(os.path.join(data_dir, mfn), dtype=np.uint8, mode="r")
    if len(data) != len(mdata):
        raise RuntimeError(f"{fn} ({len(data)}) != {mfn} ({len(mdata)})")
    if len(data) < block_size + 1:
        raise RuntimeError(f"{fn} too small ({len(data)}) for block_size {block_size}")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    # mask aligned to the TARGET: token at offset i+1+k is supervised iff
    # mdata[i+1+k]==1, so we slice the same window as y.
    m = torch.stack([torch.from_numpy(
        mdata[i + 1:i + 1 + block_size].astype(np.float32)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        m = m.pin_memory().to(device, non_blocking=True)
    else:
        x, y, m = x.to(device), y.to(device), m.to(device)
    return x, y, m


def masked_loss(logits, y, m):
    V = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none")
    w = m.reshape(-1)
    wsum = w.sum().clamp(min=1.0)
    return (ce * w).sum() / wsum


@torch.no_grad()
def estimate_loss(model, data_dir, eval_iters, block_size, batch_size, device, device_type):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, m = get_batch(data_dir, split, block_size, batch_size, device, device_type)
            with nullcontext():
                logits, _ = model(x, y)
            losses[k] = masked_loss(logits, y, m).item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_model(ng, model_args, device, init_from=None):
    GPT, GPTConfig = ng.GPT, ng.GPTConfig
    model = GPT(GPTConfig(**model_args)).to(device)
    if init_from:
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        src = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
               for k, v in ckpt["model"].items()}
        copied = []
        for k, v in src.items():
            # skip embeddings (vocab differs 65 vs 73); keep the rest.
            if k.endswith("wte.weight") or k.endswith("wpe.weight"):
                continue
            if k in model.state_dict():
                pk = ("_orig_mod." + k) if any(s.startswith("_orig_mod.")
                      for s in model.state_dict()) else k
                tgt = model.state_dict()[pk]
                if tgt.shape == v.shape:
                    with torch.no_grad():
                        tgt.copy_(v)
                    copied.append(k)
        print(f"warm-start: copied {len(copied)} tensors from {init_from} "
              f"(embeddings reinitialized for new vocab)")
    return model


def save_checkpoint(out_dir, model, optimizer, model_args, iter_num, best_val_loss, args):
    sd = {}
    for k, v in model.state_dict().items():
        nk = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        sd[nk] = v
    checkpoint = {
        "model": sd,
        "optimizer": optimizer.state_dict(),
        "model_args": dict(model_args),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": dict(out_dir=out_dir, lr=args.lr, block_size=args.block_size,
                       n_layer=12, n_head=12, n_embd=768, bias=True,
                       vocab_size=model_args["vocab_size"], dropout=0.0,
                       batch_size=args.batch_size, grad_accum=args.grad_accum,
                       max_iters=args.max_iters, seed=args.seed,
                       masked_sft=True),
    }
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    print(f"saved checkpoint -> {os.path.join(out_dir, 'ckpt.pt')} (iter {iter_num})")


def train(args):
    torch.manual_seed(args.seed)
    ng = load_nanogpt()
    data_dir = os.path.abspath(args.data_dir)
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    if vocab_size != ast.AGENT_VOCAB_SIZE:
        sys.exit(f"ERROR: data vocab_size={vocab_size} != agent vocab "
                 f"{ast.AGENT_VOCAB_SIZE}; regenerate data.")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type == "cpu":
        print("WARNING: training on CPU; will be slow.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    block_size = args.block_size
    model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=block_size,
                      bias=True, vocab_size=vocab_size, dropout=0.0)
    model = build_model(ng, model_args, device, args.init_from)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.lr, betas=(0.9, 0.95),
        device_type=device_type)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    iter_num, best_val_loss, t0 = 0, 1e9, time.time()
    x, y, m = get_batch(data_dir, "train", block_size, args.batch_size, device, device_type)
    while True:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, data_dir, args.eval_iters, block_size,
                                   args.batch_size, device, device_type)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss or args.always_save:
                best_val_loss = min(best_val_loss, losses["val"])
                if iter_num > 0:
                    save_checkpoint(out_dir, model, optimizer, model_args,
                                    iter_num, best_val_loss, args)
        if iter_num >= args.max_iters:
            break
        last_loss = 0.0
        for _ in range(args.grad_accum):
            with nullcontext():
                logits, _ = model(x, y)
            loss = masked_loss(logits, y, m)
            (loss / args.grad_accum).backward()
            x, y, m = get_batch(data_dir, "train", block_size, args.batch_size,
                                device, device_type)
            last_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if iter_num % args.log_interval == 0:
            print(f"iter {iter_num}: train loss {last_loss:.4f}")
        iter_num += 1

    losses = estimate_loss(model, data_dir, args.eval_iters, block_size,
                           args.batch_size, device, device_type)
    best_val_loss = min(best_val_loss, losses["val"])
    save_checkpoint(out_dir, model, optimizer, model_args, iter_num, best_val_loss, args)
    print(f"FINISHED. out_dir={out_dir}  final val loss={losses['val']:.4f}  "
          f"best={best_val_loss:.4f}  ({time.time() - t0:.1f}s)")


def _bool(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SFT-train char GPT-2 agent (masked loss).")
    p.add_argument("--out_dir", default="out-agent-snake")
    p.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data-agent"))
    p.add_argument("--init_from", default=None,
                   help="warm-start from a nanoGPT ckpt (shakespeare); embeddings reinit")
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--always_save", type=_bool, default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    return p.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
