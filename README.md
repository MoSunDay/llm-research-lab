# llm-research-lab

Applied research on a tiny from-scratch LLM engine ([nano-vllm](nano-vllm/)):
constrained decoding, agentic pause/resume, and supervised fine-tuning, all
exercised end-to-end on **GPT-2 124M** running on 3× Tesla V100 (32 GB).

The engine lives in [`nano-vllm/`](nano-vllm/); the repo root holds the
weight-conversion, SFT-training, and benchmark tooling that surrounds it.

## Phases

| Phase | Commit | What it adds |
|-------|--------|--------------|
| **3 — Guided decoding** | `Phase 3` | An xgrammar-backed logits-processor hook: `SamplingParams(json_schema=…, regex=…, grammar=…, choice=…, json_object=True)`, greedy (`temperature=0`) decoding, `nanovllm/grammar.py`. |
| **4 — Agentic control** | `Phase 4` | `SequenceStatus.PAUSED` + engine `pause()`/`resume()`/`generate_stream()`/`generate_agentic()`, stop/pause strings & token ids, a **P0 KV-cache desync fix** in the pause path, a V100 (sm_70) CUDA-graph-safe attention rewrite, and `MASTER_PORT` isolation for tests. |
| **5 — SFT round-trip** | `Phase 5` | Synthesize a Q&A dataset, fine-tune GPT-2 via [nanoGPT](https://github.com/karpathy/nanoGPT), reverse-convert the checkpoint back to nano-vllm format, and verify bit-exact parity with the reference nanoGPT model. |

## Repository layout

```
nano-vllm/                 the engine (forked; extended in all 3 phases)
  nanovllm/                core package
    grammar.py             xgrammar bridge (Phase 3)
    engine/                scheduler, block_manager, llm_engine, model_runner, sequence
    layers/                attention (CUDA-graph-safe), triton_kernels, sampler, ...
  test_grammar.py          Phase 3 tests
  test_phase4.py           Phase 4 tests (incl. KV-cache regression)
  test_phase5.py           Phase 5 tests
  example*.py              runnable demos (grammar, agentic, sft)
  sft_train.py             nanoGPT SFT driver (Phase 5)
convert_gpt2.py            HF GPT-2  -> nano-vllm weights        (Phase 3 prereq)
verify_gpt2.py             conversion parity check
convert_nanogpt.py         nanoGPT ckpt -> nano-vllm weights     (Phase 5)
compare_nanogpt.py         nanoGPT-vs-nano-vllm greedy parity   (Phase 5)
nanoGPT/data/sft/prepare.py  synthetic Q&A dataset builder       (Phase 5)
BENCH.md                   benchmark methodology + results
```

## Setup (all weights are gitignored — regenerate locally)

```bash
# 1. HuggingFace GPT-2 124M -> gpt2-hf/  (huggingface_hub or git lfs)
huggingface-cli download gpt2 --local-dir gpt2-hf

# 2. Convert to nano-vllm format -> gpt2-nanovllm/
python convert_gpt2.py --src gpt2-hf --dst gpt2-nanovllm
python verify_gpt2.py            # parity check (logits/max_abs)

# 3. (Phase 5 only) SFT round-trip
git clone https://github.com/karpathy/nanoGPT nanoGPT-upstream   # vendored dep
cp nanoGPT/data/sft/prepare.py nanoGPT-upstream/data/sft/prepare.py
( cd nanoGPT-upstream/data/sft && python prepare.py )           # -> train.bin/val.bin
ln -s "$PWD/nanoGPT-upstream" /root/models/nanoGPT              # sft_train.py expects this path
python nano-vllm/sft_train.py --max-iters 50                    # -> nanoGPT/out-sft/ckpt.pt
python convert_nanogpt.py \
    --src nanoGPT/out-sft/ckpt.pt \
    --dst gpt2-sft-nanovllm \
    --tokenizer gpt2-hf                                         # -> gpt2-sft-nanovllm/
```

The SFT dataset is synthetic Q&A (arithmetic, capitals, definitions, facts)
formatted as `Q: {question}\nA: {answer}\n`, tokenized with GPT-2 BPE into
nanoGPT's `train.bin`/`val.bin` (`block_size=256`).

## Running

```bash
cd nano-vllm
python example.py           # base GPT-2 demo
python example_grammar.py   # constrained decoding (Phase 3)
python example_agentic.py   # pause/resume tool loop (Phase 4)
python example_sft.py       # SFT model demo (Phase 5)

# full test suite (needs CUDA GPU(s) + the gpt2-nanovllm weights)
python -m pytest test_grammar.py test_phase4.py test_phase5.py -q
```

## Hardware

Developed and tested on **3× Tesla V100-SXM2-32GB (sm_70)**, CUDA 12.1,
PyTorch 2.5.1, Python 3.10, xgrammar 0.2.4. The V100 lacks the FlashAttention
/Triton attention kernels assumed by upstream nano-vllm, which motivated the
`layers/triton_kernels.py` + CUDA-graph-safe `attention.py` rewrite (Phase 4).

## Known limitations

- **Tensor-parallel test suite is not closed.** TP>1 paths are exercised by
  `compare_nanogpt.py` (960/960 bit-exact greedy tokens at TP=1 against the
  reference nanoGPT model) but lack a dedicated pytest regression; see
  [`BENCH.md`](BENCH.md).
- **SFT is basic instruction-tuning.** The synthetic Q&A set is small and
  narrow; the methodology (full fine-tune, fixed LR, 50 iters) is a baseline,
  not a production recipe. The fine-tuned model learns the `Q:/A:` format but
  is not a capable assistant.
- Weights and the 1.4 GB nanoGPT checkpoint are deliberately gitignored.
