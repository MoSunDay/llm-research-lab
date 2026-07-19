"""Self-contained SFT training script: fine-tune GPT-2 (124M) via nanoGPT on the
local SFT dataset, loading weights from /root/models/gpt2-hf (no internet needed).

Produces a nanoGPT-format checkpoint (ckpt.pt) that convert_nanogpt.py can convert
back to nano-vllm format. This demonstrates the full Phase 5 pipeline:
  SFT data → nanoGPT training → reverse conversion → nano-vllm inference.

Usage:
  python sft_train.py [--max-iters 50] [--out-dir ../nanoGPT/out-sft]
"""
import argparse, math, os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/models/nanoGPT")
from model import GPT, GPTConfig
from transformers import GPT2LMHeadModel

DATA_DIR = "/root/models/nanoGPT/data/sft"
SRC = "/root/models/gpt2-hf"
BLOCK_SIZE = 256


def build_model():
    """Build nanoGPT GPT-2 from local HF weights (no download)."""
    cfg = GPTConfig(n_layer=12, n_head=12, n_embd=768,
                    block_size=1024, vocab_size=50257, bias=True, dropout=0.0)
    model = GPT(cfg)
    sd = model.state_dict()
    sd_keys = [k for k in sd if not k.endswith(".attn.bias")]
    hf = GPT2LMHeadModel.from_pretrained(SRC).eval()
    sd_hf = {k: v for k, v in hf.state_dict().items()
             if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")}
    transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                  "mlp.c_fc.weight", "mlp.c_proj.weight"]
    for k in sd_hf:
        if any(k.endswith(w) for w in transposed):
            sd[k].copy_(sd_hf[k].t())
        else:
            sd[k].copy_(sd_hf[k])
    model.load_state_dict(sd)
    del hf
    return model, cfg


def get_batch(split, batch_size, device):
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, device, batch_size, eval_iters=20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-iters", type=int, default=10)
    ap.add_argument("--out-dir", default="/root/models/nanoGPT/out-sft")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    print("[building nanoGPT model from local GPT-2 weights ...]", flush=True)
    model, cfg = build_model()
    if BLOCK_SIZE < cfg.block_size:
        model.crop_block_size(BLOCK_SIZE)
        cfg.block_size = BLOCK_SIZE
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params, block_size={BLOCK_SIZE}", flush=True)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.lr,
                                           betas=(0.9, 0.95), device_type="cuda")

    # initial loss
    losses = estimate_loss(model, device, args.batch_size, eval_iters=10)
    print(f"  init: train {losses['train']:.4f}  val {losses['val']:.4f}", flush=True)

    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        # LR schedule (linear warmup → cosine decay)
        if it < args.warmup_iters:
            lr = args.lr * it / args.warmup_iters
        else:
            progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for micro in range(args.grad_accum):
            x, y = get_batch("train", args.batch_size, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % 5 == 0 or it == 1 or it == args.max_iters:
            elapsed = time.time() - t0
            print(f"  iter {it:4d}/{args.max_iters}  loss {loss.item()*args.grad_accum:.4f}  "
                  f"lr {lr:.2e}  {elapsed:.1f}s", flush=True)

    # final eval
    losses = estimate_loss(model, device, args.batch_size, eval_iters=20)
    print(f"  final: train {losses['train']:.4f}  val {losses['val']:.4f}", flush=True)

    # save checkpoint
    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                      block_size=cfg.block_size, bias=cfg.bias, vocab_size=cfg.vocab_size,
                      dropout=cfg.dropout)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": args.max_iters,
        "best_val_loss": losses["val"],
        "config": {"init_from": "gpt2", **model_args},
    }
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[saved checkpoint → {ckpt_path}]", flush=True)


if __name__ == "__main__":
    main()
