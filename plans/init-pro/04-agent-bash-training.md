# 04 -- Agent Bash Training and Pause/Resume Loop

> Reference source: link_repos/nanoGPT/ (READ-ONLY reference -- for understanding only; never modified).

## Dataset: nanoGPT/data/agent/prepare.py

Model the new data builder on the existing `nanoGPT/data/sft/prepare.py`, whose
behavior is grounded by reading it:

- It generates synthetic but deterministic/reproducible examples
  (`random.seed(42)`), concatenates them into one text stream, tokenizes the
  whole stream with `GPT2TokenizerFast`, and writes `train.bin`/`val.bin` as
  numpy `uint16` arrays plus `meta.pkl` (`{"vocab_size": VOCAB_SIZE}`).
- It asserts `len(token_ids) >= 100_000` and `max(token_ids) < VOCAB_SIZE`.
- The files are drop-in for `nanoGPT/train.py`, which samples random contiguous
  blocks of `block_size` tokens from `train.bin`/`val.bin`.

The new `nanoGPT/data/agent/prepare.py` will follow the same shape but produce
AGENT TRACES: the model emits a `<bash>command</bash>` call, a (simulated)
`<result>...</result>` is provided, then the model continues to a final
`<answer>...</answer>`. Differences from the SFT builder:

- `VOCAB_SIZE = 50304` (not 50257) so the special-token ids 50257..50272 are
  representable in uint16 (max 65535) and correspond to real embedding rows.
- Add the special tokens (doc 03) to the tokenizer as `added_tokens` with their
  fixed ids BEFORE encoding, so call/result/answer markers become single ids.
- Emit a companion loss-mask stream (one byte per token) alongside the token
  stream: `1` on assistant-emitted tokens (call, command body, answer), `0` on
  prompt and `<result>` spans. See "Loss mask" below.
- Output the same trio `train.bin` / `val.bin` / `meta.pkl`, plus
  `train.mask.bin` / `val.mask.bin` (uint8, same length) for the mask.

`meta.pkl` becomes `{"vocab_size": 50304, "special_tokens": {name: id, ...}}`.

## Pause/resume closed loop

ASCII control flow at serve time (engine drives the model):

    user prompt
        |
        v
    [engine.generate]  model emits tokens ...
        |
        +-- emitted token == <call>/<bash> boundary? -->  STOP (treat as <pause>)
        |                                                       |
        |                                                       v
        |                                    [sandbox.run(cmd, argv, timeout)]
        |                                                       |
        |                                                       v
        |                                    inject "<result>OUT</result><resume>"
        |                                                       |
        +-----------<----------- resume generate <-------------+
        |
        v
    <answer>...</answer>   (or another <call>)

Steps:
1. Feed the prompt; generate until the model emits the call boundary (the
   `</bash>`/`</call>` close, then an implicit or explicit `<pause>`).
2. Halt. Parse the command text out of the `<bash>...</bash>` span.
3. Execute via the sandbox (below). Capture stdout (capped).
4. Append `<result>OUT</result><resume>` to the context and continue
   autoregressive generation.
5. Stop at `<answer>...</answer>` or a max-token limit.

This loop is implemented in `nvllm-engine` (orchestration) and exposed through
`nvllm-serving` (cli). The model never executes code itself; it only emits a
command string that the sandbox validates and runs.

## Bash sandbox (decided design for v1)

The decided v1 sandbox is a subprocess runner with a command allowlist and a
per-call timeout. NO container isolation in v1. Concrete controls:

- Command allowlist (enumerated). ALLOWED (read-only / pure):
  `echo`, `printf`, `cat`, `head`, `tail`, `wc`, `ls`, `grep`, `sort`, `uniq`,
  `tr`, `cut`, `fold`, `seq`, `date`.
  DENIED (anything mutating, networking, interpreters, or shells):
  `rm`, `mv`, `cp`, `mkdir`, `rmdir`, `chmod`, `chown`, `dd`, `curl`, `wget`,
  `nc`, `ssh`, `sh`, `bash`, `python`, `python3`, `node`, `make`, `git`,
  `pip`, `apt`, and anything not on the allowlist (default-deny).
- argv form, no shell. Invoke as `Command::new(bin).args([...])` (Rust
  std::process::Command), never `shell=True` and never a single string passed to
  `/bin/sh -c`. This removes shell-injection vectors (no `;`, `&&`, `` ` ``,
  `$()`, `>`, `|`).
- Per-call wall-clock timeout + kill. Spawn with a deadline; on expiry, kill the
  child and its process group; report `<result><timeout/></result>`.
- cwd confinement. Run with a fixed working directory under the repo scratch
  area; do not honor paths that escape it (resolve and check the realpath stays
  within the scratch root before executing).
- No network by default. The allowlist already excludes network tools; in
  addition, no `HOME`, no proxy, and a cleared/minimal environment.
- Output size cap. Truncate stdout/stderr to a fixed byte budget (verify exact
  cap on implementation; e.g. a few KiB) to bound context length.

Explicit v1 limitation and v2 hardening: v1 has NO container/namespacing. A bug
in the allowlist/argv/cwd checks or a future allowlisted binary could let the
model affect the host filesystem. v2 hardening items (out of scope here) include
running the subprocess inside a container or seccomp profile with a read-only
root and dropped capabilities. This is tracked as risk R-SANDBOX in doc 05.

## Loss mask

Construct a per-token mask aligned with the token stream:

- mask = 1 on tokens the model should learn to emit: the call structure
  (`<call>`, `<tool>`, `<bash>`, command body, closes), `<pause>`, and the
  `<answer>` body.
- mask = 0 on prompt/user text and on engine-injected `<result>` spans (the
  model is not trained to predict the tool output verbatim).

During training (nanoGPT/train.py path), multiply the per-token cross-entropy by
the mask and normalize by the number of unmasked tokens, so the gradient is
focused on assistant/call/answer tokens. This connects to doc 03: the special
tokens carry the structure on which the mask is built.

## Security note (per repo rules)

- Never hardcode secrets, API keys, tokens, or connection strings anywhere in
  the data builder, sandbox, or engine code. If any credential is needed it
  lives in an environment variable or a config file that is NOT committed.
- The sandbox runs commands unprivileged, with a cleared environment, so the
  model never sees host secrets.
- The data builder is synthetic and contains no real credentials or PII.
