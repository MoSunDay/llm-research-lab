# 03 -- Special Tokens and Function Calling

> Reference source: link_repos/nanoGPT/ (READ-ONLY reference -- for understanding only; never modified).

## Token-id budget

- GPT-2 BPE base vocabulary: 50257 token ids (0..50256). The highest base id,
  50256, is the `<|endoftext|>` token used by both bos and eos in
  `convert_nanogpt.py` output (`bos_token_id=50256`, `eos_token_id=50256`).
- nanoGPT pads the embedding to 50304 (next multiple of 64 past 50257) for
  matmul efficiency (model.py GPTConfig comment; `convert_nanogpt.py` truncates
  wte back to 50257 rows on output but the source pad is 50304).
- The free slots 50257..50303 (47 slots) are UNUSED by GPT-2 BPE: no BPE merge
  ever produces an id in that range. They are safe to repurpose as
  special/control tokens, and the embedding rows already exist when the model is
  trained with `vocab_size=50304`.

## Proposed special tokens (within 47 slots)

| id     | name       | purpose |
|--------|------------|---------|
| 50257  | `<call>`   | begin a function/tool call block |
| 50258  | `</call>`  | end a function/tool call block |
| 50259  | `<tool>`   | begin tool-name specifier |
| 50260  | `</tool>`  | end tool-name specifier |
| 50261  | `<arg>`    | begin a named argument |
| 50262  | `</arg>`   | end a named argument |
| 50263  | `<bash>`   | begin a bash command literal |
| 50264  | `</bash>`  | end a bash command literal |
| 50265  | `<result>` | begin a tool/command result injected by the engine |
| 50266  | `</result>`| end a tool/command result |
| 50267  | `<pause>`  | engine halts generation here (await tool execution) |
| 50268  | `<resume>` | engine resumes generation after result injection |
| 50269  | `<answer>` | begin final answer span |
| 50270  | `</answer>`| end final answer span |
| 50271  | `<think>`  | reserved: optional reasoning span |
| 50272  | `</think>` | reserved: optional reasoning span |
| 50273..50303 | (reserved, 31 slots) | future tools / control tokens |

16 ids assigned (50257..50272), 31 reserved. Total within the 47-slot budget.

## Why these slots are safe

- GPT-2 BPE encodes text into ids strictly below 50257. The 47 trailing ids are
  never emitted by the base tokenizer, so reassigning them cannot collide with
  ordinary text.
- nanoGPT trains with `vocab_size=50304`, so embedding rows 50257..50303 exist
  and are trainable. The special tokens get dedicated, learnable rows rather
  than being overloaded onto existing tokens.

## Function-call serialization format

ASCII example of a model-emitted call and the engine-injected result:

    User: How many lines contain the word "error" in /var/log/app.log?
    <call><tool>bash</tool><arg>cmd</arg>
    grep -c error /var/log/app.log
    </arg><bash>grep -c error /var/log/app.log</bash></call><pause>
    -> (engine stops, sandbox runs the command)
    -> injected: <result>17</result><resume>
    <answer>There are 17 lines containing "error".</answer>

Rules:
- A call opens with `<call>` and closes with `</call>`. The tool is named inside
  `<tool>...</tool>`. Arguments use `<arg>NAME</arg> value </arg>`.
- A self-contained bash command may be given directly inside
  `<bash>...</bash>`.
- The engine stops generation at `<pause>` (or at the call boundary), runs the
  tool, then injects `<result>...</result><resume>` and continues generation.
- `<result>` spans are produced by the engine/sandbox, never trained to be
  predicted verbatim by the model (loss is masked on them; see doc 04).

## Tokenizer integration

- Python side (training data): use `GPT2TokenizerFast` with `added_tokens` for
  each special token mapped to its fixed id 50257..50272. Set them as
  `special=True` so the tokenizer never merges across them and never emits them
  as part of ordinary BPE.
- Rust side (serving): use the `tokenizers` crate with the same added-token
  table and the same fixed ids. Encoding must never merge into a special id and
  must emit a special id only when the literal `<call>` etc. appears.
- Critical invariant: encode and decode are symmetric for special tokens, and
  the ids are identical on both sides so that tokens produced in training match
  tokens consumed at serve time. Risk: a naive BPE merge could split
  `<`+`call`+`>` into multiple ids; the `added_tokens` / special-token table
  prevents this. See risk R-TOK in doc 05.

## Training implication

Because the call/result spans have a fixed structure, training loss is masked so
that the gradient focuses on the tokens the model should emit (the call itself
and the final answer), not on prompt text or on engine-injected `<result>`
spans. The mask is `1` on assistant/call/answer tokens and `0` on prompt and
result tokens. Details and the per-token mask construction are in doc 04.
