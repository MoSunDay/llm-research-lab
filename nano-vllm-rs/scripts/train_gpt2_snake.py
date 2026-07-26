"""
GPT-2 "brain" training pipeline for the nano-vllm Snake agent.

Trains a character-level GPT-2 small model on Tiny Shakespeare (the canonical
nanoGPT char dataset) and saves a checkpoint in nanoGPT's native format so that
`export_safetensors.py` can convert it into the exact tensor layout the Rust
`nvllm-models` GPT-2 loader consumes.

Pipeline:
  1. Locate vendored nanoGPT (link_repos/nanoGPT/model.py) or $NANO_GPT_PATH.
  2. Download Tiny Shakespeare (resume-capable, stdlib urllib only -- offline-safe).
  3. Build a char-level dataset: train.bin / val.bin (uint16) + meta.json.
  4. Train GPT-2 small (12/768/12, bias=True MANDATORY) in fp32, no torch.compile.
  5. Save out_dir/ckpt.pt with clean nanoGPT state_dict keys (no _orig_mod prefix).

Functional style: free functions only, no classes of our own.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from contextlib import nullcontext

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
EXPECTED_SIZE = 1115394  # bytes of the canonical Tiny Shakespeare input.txt


# ------------------------------------------------------------- nanoGPT ----

def load_nanogpt():
    """Find and import nanoGPT's model.py. Returns the module object."""
    env_path = os.environ.get("NANO_GPT_PATH", "")
    candidates = [env_path] if env_path else []
    d = SCRIPT_DIR
    for _ in range(6):  # walk up from this script to the repo root
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
    sys.stderr.write(
        "ERROR: could not find nanoGPT model.py.\n"
        "  Set NANO_GPT_PATH=<dir containing model.py> or place nanoGPT at "
        "<repo>/link_repos/nanoGPT/.\n"
    )
    sys.exit(1)


# ----------------------------------------------------------------- data ----

def _verify_text(dest):
    with open(dest, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(64)
    if not head.startswith("First Citizen:"):
        raise RuntimeError(f"downloaded content does not look like Tiny "
                           f"Shakespeare (head={head!r})")


def download_text(url, dest, expected_size=EXPECTED_SIZE, attempts=8):
    """Download `url` to `dest` with retries + HTTP Range resume (stdlib only)."""
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and abs(os.path.getsize(dest) - expected_size) <= 16:
        print(f"already downloaded ({os.path.getsize(dest)} bytes) -> {dest}")
        _verify_text(dest)
        return dest
    timeouts = [20, 30, 40, 50, 60, 70, 80, 90]
    last_err = "no attempts yet"
    for attempt in range(attempts):
        offset = os.path.getsize(dest) if os.path.exists(dest) else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(
                req, timeout=timeouts[min(attempt, len(timeouts) - 1)]
            )
            status = getattr(resp, "status", 200)
            if offset and status != 206:  # server ignored Range -> restart
                offset, mode = 0, "wb"
            else:
                mode = "ab" if offset else "wb"
            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            resp.close()
            size = os.path.getsize(dest)
            print(f"  download attempt {attempt + 1}: {size} bytes "
                  f"(resume offset was {offset})")
            if size >= expected_size - 16:
                last_err = None
                break
            last_err = f"partial {size}/{expected_size}"
        except Exception as e:  # noqa: BLE001 - retry any network error
            last_err = repr(e)[:160]
            print(f"  download attempt {attempt + 1} failed: {last_err}")
            time.sleep(1.0)
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if abs(size - expected_size) > 16:
        raise RuntimeError(f"download incomplete: {size}B (expect ~{expected_size}); "
                           f"last error: {last_err}")
    _verify_text(dest)
    print(f"OK: Tiny Shakespeare ({size} bytes) -> {dest}")
    return dest


def build_char_dataset(text, data_dir):
    """Replicate shakespeare_char prep; write train.bin/val.bin + meta.json."""
    os.makedirs(data_dir, exist_ok=True)
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = list(chars)  # index -> char, sorted order matches the uint16 ids
    vocab_size = len(chars)
    n = len(text)
    cut = int(n * 0.9)
    train_ids = [stoi[c] for c in text[:cut]]
    val_ids = [stoi[c] for c in text[cut:]]
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(data_dir, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(data_dir, "val.bin"))
    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"vocab ({vocab_size}): {chars!r}")
    print(f"train tokens: {len(train_ids):,}  val tokens: {len(val_ids):,}  "
          f"-> {data_dir}")
    return vocab_size


def get_batch(data_dir, split, block_size, batch_size, device, device_type):
    """Mirror nanoGPT get_batch: random offsets into the .bin memmap."""
    fn = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fn), dtype=np.uint16, mode="r")
    if len(data) < block_size + 1:
        raise RuntimeError(f"{fn} has {len(data)} tokens; need >= {block_size + 1}")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, eval_iters, block_size, batch_size, device, device_type):
    out = {}
    model.eval()
    ctx = nullcontext()  # fp32, no autocast
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data_dir, split, block_size, batch_size, device, device_type)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ------------------------------------------------------------- training ----

def save_checkpoint(out_dir, model, optimizer, model_args, iter_num, best_val_loss, args):
    raw_model = model  # compile=False -> already the unwrapped module
    sd = {}
    for k, v in raw_model.state_dict().items():
        nk = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        sd[nk] = v
    checkpoint = {
        "model": sd,
        "optimizer": optimizer.state_dict(),
        "model_args": dict(model_args),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": dict(
            out_dir=out_dir, lr=args.lr, block_size=args.block_size,
            n_layer=12, n_head=12, n_embd=768, bias=True,
            vocab_size=model_args["vocab_size"], dropout=0.0,
            batch_size=args.batch_size, grad_accum=args.grad_accum,
            max_iters=args.max_iters, seed=args.seed,
        ),
    }
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    print(f"saved checkpoint -> {os.path.join(out_dir, 'ckpt.pt')} (iter {iter_num})")


def train(args):
    torch.manual_seed(args.seed)
    ng = load_nanogpt()
    GPT, GPTConfig = ng.GPT, ng.GPTConfig

    # --- data prep (idempotent) ---
    data_dir = os.path.abspath(args.data_dir)
    text_path = os.path.join(data_dir, "input.txt")
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.isfile(meta_path) or args.force_data:
        download_text(SHAKESPEARE_URL, text_path)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        vocab_size = build_char_dataset(text, data_dir)
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            vocab_size = json.load(f)["vocab_size"]
        print(f"reusing dataset in {data_dir} (vocab_size={vocab_size})")

    # --- device ---
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type == "cpu":
        print("WARNING: training on CPU; this will be slow.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- model: standard GPT-2 small, char vocab, bias=True (Rust requires it) ---
    block_size = args.block_size
    model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=block_size,
                      bias=True, vocab_size=vocab_size, dropout=0.0)
    model = GPT(GPTConfig(**model_args))
    model.to(device)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.lr, betas=(0.9, 0.95), device_type=device_type
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    iter_num, best_val_loss, t0 = 0, 1e9, time.time()
    x, y = get_batch(data_dir, "train", block_size, args.batch_size, device, device_type)
    while True:
        for pg in optimizer.param_groups:  # constant lr (keep it simple)
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
                logits, loss = model(x, y)
            x, y = get_batch(data_dir, "train", block_size, args.batch_size, device, device_type)
            (loss / args.grad_accum).backward()
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
    p = argparse.ArgumentParser(description="Train char-level GPT-2 for Snake "
                                            "(nanoGPT checkpoint format).")
    p.add_argument("--out_dir", default="out-gpt2-snake")
    p.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data-shakespeare-char"))
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--always_save", type=_bool, default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--force_data", action="store_true",
                   help="rebuild dataset even if meta.json exists")
    return p.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
