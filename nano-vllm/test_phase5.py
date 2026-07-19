"""Tests for Phase 5 (nanoGPT SFT + reverse conversion).

Phase 5 demonstrates: SFT data -> nanoGPT training -> reverse conversion ->
nano-vllm inference. The reverse conversion (``convert_nanogpt.py``) is pure /
no GPU, so the bulk of these tests are unit tests that run the conversion on the
real nanoGPT SFT checkpoint and on small synthetic checkpoints, then verify the
safetensors / config.json / tokenizer output.

What makes convert_nanogpt.py different from convert_gpt2.py (the HF converter)
is that nanoGPT stores nn.Linear weights already in [out, in] layout, so the
reverse conversion does NOT transpose anything -- it only splits the fused
c_attn into q/k/v and truncates the padded vocab. These tests pin that contract.

The final test sanity-checks the pre-built SFT model artifact on disk. We
deliberately do NOT construct a second ``LLM`` here: the shared ``conftest.py``
``llm`` fixture already calls ``torch.distributed.init_process_group`` once per
process, and a second engine would re-init and crash. Loading GPT-2 weights
through the engine is already exercised by every other test module.

Run directly (``python test_phase5.py``) or via pytest.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CONVERT_SCRIPT = "/root/models/convert_nanogpt.py"
SFT_CKPT_DIR = "/root/models/nanoGPT/out-sft"
SFT_CKPT_PATH = os.path.join(SFT_CKPT_DIR, "ckpt.pt")
SFT_MODEL_DIR = "/root/models/gpt2-sft-nanovllm"      # pre-built reverse-converted artifact
REF_MODEL_DIR = "/root/models/gpt2-nanovllm"          # convert_gpt2.py output (key-set reference)
GPT2_HF_DIR = "/root/models/gpt2-hf"                  # tokenizer source
GPT2_VOCAB_SIZE = 50257


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run_convert(src, dst, tokenizer=GPT2_HF_DIR):
    """Invoke convert_nanogpt.py as a subprocess (the real CLI entry point)."""
    out = subprocess.run(
        [sys.executable, CONVERT_SCRIPT,
         "--src", src, "--dst", dst, "--tokenizer", tokenizer],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, (
        f"convert_nanogpt.py failed (rc={out.returncode})\n"
        f"--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}"
    )
    return out


def _freest_gpu():
    """Index of the GPU with the most free memory (via nvidia-smi, no CUDA context).

    The shared ``llm`` fixture already occupies GPU 0, so a generate() subprocess
    must run on a different device or its KV-cache allocation asserts to 0.
    """
    import re
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
    best, best_free = None, -1
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        idx, free = int(parts[0]), int(parts[1])
        if free > best_free:
            best, best_free = idx, free
    return best


def _load_safetensors(path):
    """Load every tensor from a safetensors file into a plain dict."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, "pt", "cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _build_tiny_ckpt(path, n_layer=1, n_head=2, n_embd=8, block_size=4,
                     vocab=16, bias=True, prefix=""):
    """Write a minimal nanoGPT-style ``ckpt.pt`` at toy scale.

    Shapes mirror nanoGPT's Block / CausalSelfAttention / MLP but small enough
    that the conversion runs in milliseconds. Options:
      * ``bias=False`` -> omit every Linear/LayerNorm bias (sets model_args.bias)
      * ``prefix="_orig_mod."`` -> prepend the torch.compile prefix to every key

    Returns ``(state_dict, model_args)`` so callers can compare against output.
    """
    p = prefix
    sd = {}
    sd[f"{p}transformer.wte.weight"] = torch.randn(vocab, n_embd)
    sd[f"{p}transformer.wpe.weight"] = torch.randn(block_size, n_embd)
    sd[f"{p}transformer.ln_f.weight"] = torch.ones(n_embd)
    if bias:
        sd[f"{p}transformer.ln_f.bias"] = torch.zeros(n_embd)
    for i in range(n_layer):
        sd[f"{p}transformer.h.{i}.ln_1.weight"] = torch.ones(n_embd)
        sd[f"{p}transformer.h.{i}.ln_2.weight"] = torch.ones(n_embd)
        if bias:
            sd[f"{p}transformer.h.{i}.ln_1.bias"] = torch.zeros(n_embd)
            sd[f"{p}transformer.h.{i}.ln_2.bias"] = torch.zeros(n_embd)
        sd[f"{p}transformer.h.{i}.attn.c_attn.weight"] = torch.randn(3 * n_embd, n_embd)
        sd[f"{p}transformer.h.{i}.attn.c_proj.weight"] = torch.randn(n_embd, n_embd)
        sd[f"{p}transformer.h.{i}.mlp.c_fc.weight"] = torch.randn(4 * n_embd, n_embd)
        sd[f"{p}transformer.h.{i}.mlp.c_proj.weight"] = torch.randn(n_embd, 4 * n_embd)
        if bias:
            sd[f"{p}transformer.h.{i}.attn.c_attn.bias"] = torch.randn(3 * n_embd)
            sd[f"{p}transformer.h.{i}.attn.c_proj.bias"] = torch.randn(n_embd)
            sd[f"{p}transformer.h.{i}.mlp.c_fc.bias"] = torch.randn(4 * n_embd)
            sd[f"{p}transformer.h.{i}.mlp.c_proj.bias"] = torch.randn(n_embd)
    model_args = {
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "block_size": block_size, "bias": bias, "vocab_size": vocab,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": sd, "model_args": model_args}, path)
    return sd, model_args


# The real SFT checkpoint is ~1.4 GB; convert it at most once per process and
# reuse the on-disk result (and the loaded source state dict) across tests.
_CACHE = {}


def _converted_sft_dir():
    """Convert the real SFT checkpoint once, cache the output dir."""
    if "dir" in _CACHE:
        return _CACHE["dir"]
    if not os.path.exists(SFT_CKPT_PATH):
        pytest.skip(f"{SFT_CKPT_PATH} not present (Phase 5 SFT checkpoint missing)")
    dst = tempfile.mkdtemp(prefix="phase5_sft_convert_")
    _run_convert(SFT_CKPT_DIR, dst)
    _CACHE["dir"] = dst
    return dst


def _sft_tensors():
    return _load_safetensors(os.path.join(_converted_sft_dir(), "model.safetensors"))


def _sft_ckpt_sd():
    if "sd" in _CACHE:
        return _CACHE["sd"]
    if not os.path.exists(SFT_CKPT_PATH):
        pytest.skip(f"{SFT_CKPT_PATH} not present (Phase 5 SFT checkpoint missing)")
    ckpt = torch.load(SFT_CKPT_PATH, map_location="cpu", weights_only=False)
    _CACHE["sd"] = ckpt["model"]
    return _CACHE["sd"]


def _ref_keys():
    """Key set produced by convert_gpt2.py (the HF -> nano-vllm converter)."""
    return set(_load_safetensors(os.path.join(REF_MODEL_DIR, "model.safetensors")).keys())


# --------------------------------------------------------------------------- #
# Core conversion tests against the real nanoGPT SFT checkpoint (no GPU)
# --------------------------------------------------------------------------- #
def test_convert_produces_valid_safetensors():
    """End-to-end conversion of the real SFT ckpt.pt produces a valid model dir."""
    dst = _converted_sft_dir()

    # model.safetensors + config.json exist
    assert os.path.exists(os.path.join(dst, "model.safetensors"))
    assert os.path.exists(os.path.join(dst, "config.json"))

    # config.json: native GPT-2 fields + nano-vllm aliases (mirror convert_gpt2.py)
    cfg = json.load(open(os.path.join(dst, "config.json")))
    assert cfg["model_type"] == "gpt2"
    assert cfg["n_layer"] == 12
    assert cfg["n_head"] == 12
    assert cfg["n_embd"] == 768
    assert cfg["torch_dtype"] == "float16"
    assert cfg["max_position_embeddings"] == cfg["n_positions"] == 256
    assert cfg["num_hidden_layers"] == 12
    assert cfg["num_attention_heads"] == 12
    assert cfg["hidden_size"] == 768
    assert cfg["num_key_value_heads"] == 12
    assert cfg["head_dim"] == 64

    # tokenizer files copied
    for fn in ("vocab.json", "merges.txt", "tokenizer.json"):
        assert os.path.exists(os.path.join(dst, fn)), f"{fn} not copied"

    tensors = _sft_tensors()

    # every tensor is float16
    for k, v in tensors.items():
        assert v.dtype == torch.float16, f"{k} dtype {v.dtype} != float16"

    # key set is identical to convert_gpt2.py output
    assert set(tensors.keys()) == _ref_keys()

    # wte is the unpadded GPT-2 vocab (50257), not nanoGPT's padded 50304
    assert tensors["model.embed_tokens.weight"].shape[0] == GPT2_VOCAB_SIZE


def test_convert_qkv_split_correct():
    """Fused c_attn is split into q/k/v by simple dim-0 chunking (NO transpose).

    nanoGPT's CausalSelfAttention uses nn.Linear, so c_attn.weight is already
    [3*n_embd, n_embd] in [out, in] layout -- concatenating q|k|v along dim 0
    reconstructs the source exactly.
    """
    tensors = _sft_tensors()
    sd = _sft_ckpt_sd()
    n_embd = 768

    q = tensors["model.layers.0.self_attn.q_proj.weight"]
    k = tensors["model.layers.0.self_attn.k_proj.weight"]
    v = tensors["model.layers.0.self_attn.v_proj.weight"]
    assert q.shape == k.shape == v.shape == (n_embd, n_embd)

    ref_w = sd["transformer.h.0.attn.c_attn.weight"]
    assert ref_w.shape == (3 * n_embd, n_embd)
    assert torch.allclose(torch.cat([q, k, v], dim=0), ref_w.to(torch.float16), atol=1e-3)

    qb = tensors["model.layers.0.self_attn.q_proj.bias"]
    kb = tensors["model.layers.0.self_attn.k_proj.bias"]
    vb = tensors["model.layers.0.self_attn.v_proj.bias"]
    ref_b = sd["transformer.h.0.attn.c_attn.bias"]
    assert torch.allclose(torch.cat([qb, kb, vb], dim=0), ref_b.to(torch.float16), atol=1e-3)


def test_convert_no_transpose_needed():
    """o_proj / mlp weights are copied verbatim (nanoGPT is already nn.Linear).

    This is the key contrast with convert_gpt2.py, which transposes HF Conv1D
    weights. Here the converted tensor must equal the checkpoint tensor directly.
    """
    tensors = _sft_tensors()
    sd = _sft_ckpt_sd()

    o = tensors["model.layers.0.self_attn.o_proj.weight"]
    ref_o = sd["transformer.h.0.attn.c_proj.weight"]
    assert o.shape == ref_o.shape == (768, 768)
    assert torch.allclose(o, ref_o.to(torch.float16), atol=1e-3)

    fc = tensors["model.layers.0.mlp.fc.weight"]
    ref_fc = sd["transformer.h.0.mlp.c_fc.weight"]
    assert fc.shape == ref_fc.shape == (3072, 768)
    assert torch.allclose(fc, ref_fc.to(torch.float16), atol=1e-3)

    proj = tensors["model.layers.0.mlp.proj.weight"]
    ref_proj = sd["transformer.h.0.mlp.c_proj.weight"]
    assert proj.shape == ref_proj.shape == (768, 3072)
    assert torch.allclose(proj, ref_proj.to(torch.float16), atol=1e-3)


def test_convert_embeddings_match():
    """Token / position embeddings are copied (wte truncated to the real vocab)."""
    tensors = _sft_tensors()
    sd = _sft_ckpt_sd()

    wte = tensors["model.embed_tokens.weight"]
    src_wte = sd["transformer.wte.weight"]
    assert wte.shape[0] == GPT2_VOCAB_SIZE
    # source may be padded to 50304 -> only the first 50257 rows are kept
    assert torch.allclose(wte, src_wte[:GPT2_VOCAB_SIZE].to(torch.float16), atol=1e-3)

    wpe = tensors["model.pos_embeddings.weight"]
    src_wpe = sd["transformer.wpe.weight"]
    assert wpe.shape == src_wpe.shape
    assert torch.allclose(wpe, src_wpe.to(torch.float16), atol=1e-3)


def test_convert_layernorm_match():
    """All LayerNorm weight/bias tensors are copied directly."""
    tensors = _sft_tensors()
    sd = _sft_ckpt_sd()

    # final LayerNorm
    assert torch.allclose(
        tensors["model.norm.weight"], sd["transformer.ln_f.weight"].to(torch.float16), atol=1e-3)
    assert torch.allclose(
        tensors["model.norm.bias"], sd["transformer.ln_f.bias"].to(torch.float16), atol=1e-3)

    # per-layer ln_1 -> input_layernorm, ln_2 -> post_attention_layernorm
    for i in range(12):
        for ln_src, ln_dst in (("ln_1", "input_layernorm"),
                               ("ln_2", "post_attention_layernorm")):
            w = tensors[f"model.layers.{i}.{ln_dst}.weight"]
            b = tensors[f"model.layers.{i}.{ln_dst}.bias"]
            assert torch.allclose(
                w, sd[f"transformer.h.{i}.{ln_src}.weight"].to(torch.float16), atol=1e-3)
            assert torch.allclose(
                b, sd[f"transformer.h.{i}.{ln_src}.bias"].to(torch.float16), atol=1e-3)


# --------------------------------------------------------------------------- #
# Synthetic-checkpoint tests (no real SFT ckpt / no GPU needed)
# --------------------------------------------------------------------------- #
def test_convert_handles_orig_mod_prefix():
    """A torch.compile'd checkpoint (``_orig_mod.`` prefix) is handled cleanly.

    If the prefix were not stripped the converter would KeyError on
    ``transformer.wte.weight``; reaching the assertions proves stripping worked.
    """
    src = tempfile.mkdtemp(prefix="phase5_origmod_src_")
    sd, model_args = _build_tiny_ckpt(
        os.path.join(src, "ckpt.pt"), prefix="_orig_mod.")
    dst = tempfile.mkdtemp(prefix="phase5_origmod_dst_")
    _run_convert(src, dst)

    tensors = _load_safetensors(os.path.join(dst, "model.safetensors"))

    # The prefix must not leak into any output key.
    assert all("_orig_mod" not in k for k in tensors), list(tensors)
    # Standard nano-vllm keys are present.
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert f"model.layers.0.self_attn.{name}.weight" in tensors
    assert "model.embed_tokens.weight" in tensors

    # n_layer was read from model_args and the (now stripped) keys were walked.
    cfg = json.load(open(os.path.join(dst, "config.json")))
    assert cfg["n_layer"] == model_args["n_layer"] == 1

    # Values carried through correctly: q|k|v (no transpose) == prefixed c_attn.
    q = tensors["model.layers.0.self_attn.q_proj.weight"]
    k = tensors["model.layers.0.self_attn.k_proj.weight"]
    v = tensors["model.layers.0.self_attn.v_proj.weight"]
    ref = sd["_orig_mod.transformer.h.0.attn.c_attn.weight"]
    assert torch.allclose(torch.cat([q, k, v], dim=0), ref.to(torch.float16), atol=1e-3)


def test_convert_bias_false():
    """A ``bias=False`` nanoGPT checkpoint gets zero biases synthesized.

    nano-vllm's GPT-2 backend always declares bias params, so the converter must
    emit all-zero biases when the source has none.
    """
    src = tempfile.mkdtemp(prefix="phase5_nobias_src_")
    sd, model_args = _build_tiny_ckpt(os.path.join(src, "ckpt.pt"), bias=False)
    assert model_args["bias"] is False
    # precondition: the source truly has no bias tensors
    assert not any(k.endswith(".bias") for k in sd)

    dst = tempfile.mkdtemp(prefix="phase5_nobias_dst_")
    _run_convert(src, dst)

    tensors = _load_safetensors(os.path.join(dst, "model.safetensors"))

    # Every synthesized bias is all zeros.
    bias_tensors = {k: v for k, v in tensors.items() if k.endswith(".bias")}
    assert bias_tensors, "expected synthesized bias keys for bias=False model"
    for k, v in bias_tensors.items():
        assert torch.count_nonzero(v).item() == 0, f"{k} should be all zeros"

    # Weights are still carried through (not zeroed) and q|k|v reconstructs.
    assert tensors["model.layers.0.self_attn.q_proj.weight"].abs().sum().item() > 0
    q = tensors["model.layers.0.self_attn.q_proj.weight"]
    k = tensors["model.layers.0.self_attn.k_proj.weight"]
    v = tensors["model.layers.0.self_attn.v_proj.weight"]
    ref = sd["transformer.h.0.attn.c_attn.weight"]
    assert torch.allclose(torch.cat([q, k, v], dim=0), ref.to(torch.float16), atol=1e-3)

    # config.json records the architecture even for the bias-less source.
    cfg = json.load(open(os.path.join(dst, "config.json")))
    assert cfg["n_layer"] == model_args["n_layer"]


# --------------------------------------------------------------------------- #
# Pre-built SFT artifact sanity check (no second LLM, no GPU)
# --------------------------------------------------------------------------- #
def test_sft_artifact_on_disk_is_valid():
    """The pre-built reverse-converted SFT model dir is a valid nano-vllm model.

    We avoid constructing a second ``LLM``: the shared ``conftest.py`` fixture
    already calls ``torch.distributed.init_process_group`` once per process, and
    a second engine would re-init and crash. Instead we validate the on-disk
    artifact directly (config fields, tokenizer files, fp16 weights, key set
    matching the HF->nano-vllm reference, expected per-layer shapes). Loading
    these weights through the engine is covered by every other test module via
    the shared GPT-2 ``llm`` fixture.
    """
    if not os.path.isdir(SFT_MODEL_DIR):
        pytest.skip(f"{SFT_MODEL_DIR} not present (Phase 5 artifact not built)")

    st = os.path.join(SFT_MODEL_DIR, "model.safetensors")
    assert os.path.exists(st)
    tensors = _load_safetensors(st)

    # config.json: native GPT-2 fields + every nano-vllm alias the loader reads.
    cfg = json.load(open(os.path.join(SFT_MODEL_DIR, "config.json")))
    assert cfg["model_type"] == "gpt2"
    assert cfg["torch_dtype"] == "float16"
    for k in ("n_layer", "n_head", "n_embd", "max_position_embeddings",
              "num_hidden_layers", "num_attention_heads", "hidden_size",
              "num_key_value_heads", "head_dim"):
        assert k in cfg, f"{k} missing from config.json"

    # tokenizer files present
    for fn in ("vocab.json", "merges.txt", "tokenizer.json"):
        assert os.path.exists(os.path.join(SFT_MODEL_DIR, fn)), f"{fn} missing"

    # every tensor is float16
    for k, v in tensors.items():
        assert v.dtype == torch.float16, f"{k} dtype {v.dtype} != float16"

    # key set matches the HF->nano-vllm reference exactly
    assert set(tensors.keys()) == _ref_keys()

    # wte unpadded to GPT-2's real vocab
    assert tensors["model.embed_tokens.weight"].shape[0] == GPT2_VOCAB_SIZE

    # spot-check per-layer shapes the nano-vllm loader expects
    assert tensors["model.layers.0.self_attn.q_proj.weight"].shape == (768, 768)
    assert tensors["model.layers.0.self_attn.o_proj.weight"].shape == (768, 768)
    assert tensors["model.layers.0.mlp.fc.weight"].shape == (3072, 768)
    assert tensors["model.layers.0.mlp.proj.weight"].shape == (768, 3072)
    assert tensors["model.norm.weight"].shape == (768,)


def test_sft_model_generate_subprocess():
    """The reverse-converted SFT model actually generates (in a fresh process).

    nano-vllm calls ``torch.distributed.init_process_group`` exactly once per
    process, so -- like example_sft.py -- we cannot hold a second ``LLM`` in the
    test process (the shared ``llm`` fixture already owns the one allowed group).
    We therefore spawn a subprocess that loads the SFT model, generates from a
    "Q:" prompt, and prints the result. This is the only test that runs a real
    forward pass through the SFT (reverse-converted) weights, and it confirms the
    fine-tuning took effect: the model answers in the "A: ..." format it was
    trained on.
    """
    if not os.path.isdir(SFT_MODEL_DIR):
        pytest.skip(f"{SFT_MODEL_DIR} not present (Phase 5 artifact not built)")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for end-to-end SFT generation")

    script = (
        "import sys; sys.path.insert(0, '/root/models/nano-vllm')\n"
        "from nanovllm import LLM, SamplingParams\n"
        f"llm = LLM({SFT_MODEL_DIR!r}, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.7)\n"
        "out = llm.generate(['Q: What is the capital of France?\\n'],\n"
        "                   SamplingParams(temperature=0, max_tokens=24, ignore_eos=True), use_tqdm=False)\n"
        "print('RESULT_BEGIN' + out[0]['text'] + 'RESULT_END')\n"
        "llm.exit()\n"
    )
    # The test process already holds the default rendezvous port (2333) and GPU 0
    # for its own engine, so run the child on a different MASTER_PORT and on the
    # GPU that currently has the most free memory.
    child_env = dict(os.environ, MASTER_PORT="24680")
    gpu = _freest_gpu()
    if gpu is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    res = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=300, env=child_env)
    assert res.returncode == 0, (
        f"SFT generate subprocess failed (rc={res.returncode})\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")

    marker = res.stdout.split("RESULT_BEGIN", 1)[-1]
    text = marker.split("RESULT_END", 1)[0]
    assert text.strip(), f"SFT model produced empty output:\n{res.stdout}"
    # The fine-tuning taught the model the "Q: ...\nA: ..." format: given a bare
    # "Q:" prompt it should continue with "A:".
    assert "A:" in text, f"SFT model did not follow the A: format: {text!r}"


if __name__ == "__main__":
    # Pure conversion tests (no model / no GPU). Mirrors test_phase4.py: call
    # the no-argument test functions directly.
    if os.path.exists(SFT_CKPT_PATH):
        test_convert_produces_valid_safetensors()
        test_convert_qkv_split_correct()
        test_convert_no_transpose_needed()
        test_convert_embeddings_match()
        test_convert_layernorm_match()
        print("SFT conversion tests OK")
    else:
        print(f"skipping SFT conversion tests: {SFT_CKPT_PATH} not found")

    # Synthetic-checkpoint tests (always runnable, no real ckpt needed).
    test_convert_handles_orig_mod_prefix()
    test_convert_bias_false()
    print("synthetic conversion tests OK")

    # Pre-built artifact sanity check.
    if os.path.isdir(SFT_MODEL_DIR):
        test_sft_artifact_on_disk_is_valid()
        test_sft_model_generate_subprocess()
        print("SFT artifact + generate tests OK")
    else:
        print(f"skipping artifact test: {SFT_MODEL_DIR} not found")

    print("ALL PHASE 5 TESTS PASSED")
