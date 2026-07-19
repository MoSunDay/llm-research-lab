"""nano-vllm guided / structured decoding demo (Phase 3, xgrammar backend).

Shows JSON-schema, regex, choice-list, and bare-JSON-object constraints enforced
during generation via the SamplingParams guided-decoding fields. The grammar is
applied as a logits mask on rank 0 (see nanovllm/grammar.py).

The GPT-2 124M base model is not instruction-tuned, so constrained output is
*syntactically* valid (matches the schema / regex / choice) but semantically
nonsense -- which is exactly what proves the grammar is being enforced.
"""
import json
import re

from nanovllm import LLM, SamplingParams


def _strip_eos(llm, text):
    """Remove a trailing EOS marker so we can validate the constrained payload."""
    eos = llm.tokenizer.eos_token or ""
    if eos and text.endswith(eos):
        text = text[: -len(eos)]
    return text


def main():
    path = "/root/models/gpt2-nanovllm"
    llm = LLM(path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.7)
    prompt = ["The record is:"]

    # 1) JSON schema: output must be a JSON object with string "name" and int "age".
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    out = llm.generate(prompt, SamplingParams(
        temperature=0, max_tokens=64, json_schema=schema), use_tqdm=False)[0]
    payload = _strip_eos(llm, out["text"]).strip()
    parsed = json.loads(payload)  # raises if not valid JSON
    assert set(parsed) == {"name", "age"} and isinstance(parsed["name"], str) \
        and isinstance(parsed["age"], int), parsed
    print("=" * 60, "\nJSON SCHEMA  -> valid JSON object, keys =", list(parsed))
    print("  raw:", repr(payload))

    # 2) Regex: output must be one or more decimal digits.
    out = llm.generate(prompt, SamplingParams(
        temperature=0, max_tokens=24, regex=r"[0-9]+"), use_tqdm=False)[0]
    payload = _strip_eos(llm, out["text"]).strip()
    assert re.fullmatch(r"[0-9]+", payload), repr(payload)
    print("\n" + "=" * 60, "\nREGEX [0-9]+  -> matches:", repr(payload))

    # 3) Choice: output must be exactly one of the literals.
    choices = ["yes", "no", "maybe"]
    out = llm.generate(prompt, SamplingParams(
        temperature=0, max_tokens=16, choice=choices), use_tqdm=False)[0]
    payload = _strip_eos(llm, out["text"]).strip()
    assert payload in set(choices), repr(payload)
    print("\n" + "=" * 60, f"\nCHOICE {choices}  -> picked:", repr(payload))

    # 4) Bare JSON object (any valid JSON value).
    out = llm.generate(prompt, SamplingParams(
        temperature=0, max_tokens=64, json_object=True), use_tqdm=False)[0]
    payload = _strip_eos(llm, out["text"]).strip()
    json.loads(payload)  # raises if not valid JSON
    print("\n" + "=" * 60, "\nJSON OBJECT  -> valid JSON:", repr(payload))

    print("\nAll guided-decoding constraints satisfied.")


if __name__ == "__main__":
    main()
