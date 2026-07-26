"""
Special-token vocabulary + agent-trace builder for the GPT-2 agent pipeline.

Defines the 8 control tokens appended to the base character vocabulary
(read from data-shakespeare-char/meta.json). Special-token IDs are assigned
dynamically right after the base vocab, so the vocab auto-adapts when the
base character set is extended (e.g. adding code characters):

    base_size + 0: <bash>
    base_size + 1: </bash>
    base_size + 2: <pause>
    base_size + 3: <result>
    base_size + 4: </result>
    base_size + 5: <resume>
    base_size + 6: <answer>
    base_size + 7: </answer>

Also provides the loss-mask state machine: a trace is a sequence of
(text, mask_value) segments; model-output segments are supervised (mask=1),
prompt/tool-result segments are injected (mask=0). This is the single source
of truth shared by prepare_agent_snake.py, sft_train_gpt2_snake.py and the
Rust CharTokenizer (meta.json).

Pure functional style: free functions, no classes.
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_META = os.path.join(SCRIPT_DIR, "data-shakespeare-char", "meta.json")

# Control-token literals in id order. Literals MUST stay byte-identical with
# the Rust SPECIAL_LITERALS table in snake-gpt2/src/agent.rs.
_SPECIAL_LITS = [
    "<bash>",
    "</bash>",
    "<pause>",
    "<result>",
    "</result>",
    "<resume>",
    "<answer>",
    "</answer>",
]


def load_base_itos(path=BASE_META):
    """Load the base character vocabulary ordering from the shakespeare meta."""
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    itos = list(meta["itos"])
    vs = meta.get("vocab_size")
    if vs is not None and vs != len(itos):
        raise ValueError(
            f"meta.json vocab_size={vs} but itos has {len(itos)} entries")
    return itos


# Base vocab size is read once at import so special IDs and AGENT_VOCAB_SIZE
# are available without an explicit call.
_BASE_SIZE = len(load_base_itos())

SPECIALS = [(_SPECIAL_LITS[i], _BASE_SIZE + i) for i in range(len(_SPECIAL_LITS))]
SPECIAL_LITERALS = list(_SPECIAL_LITS)
SPECIAL_LIT_TO_ID = dict(SPECIALS)
AGENT_VOCAB_SIZE = _BASE_SIZE + len(SPECIALS)


def build_vocab(base_itos=None):
    """Return (itos, stoi, vocab_size) for the full agent vocabulary."""
    if base_itos is None:
        base_itos = load_base_itos()
    itos = list(base_itos)
    for lit, _id in SPECIALS:
        if _id != len(itos):
            raise ValueError(f"special {lit!r} id {_id} != {len(itos)}")
        itos.append(lit)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return itos, stoi, len(itos)


def build_meta(base_itos=None):
    """Build the meta.json dict for the agent vocab."""
    itos, stoi, vocab_size = build_vocab(base_itos)
    return {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "special_tokens": dict(SPECIAL_LIT_TO_ID),
    }


# Longest-match encode: specials sorted by descending length so e.g.
# "</bash>" matches before "<bash>" at the same cursor.
_SORTED_SPECIALS = sorted(SPECIAL_LITERALS, key=len, reverse=True)


def encode(text, stoi):
    """Tokenize text with longest-match special-token support.

    Unknown single characters are skipped (mirrors the Rust CharTokenizer).
    """
    ids = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for lit in _SORTED_SPECIALS:
            if text.startswith(lit, i):
                matched = lit
                break
        if matched is not None:
            ids.append(stoi[matched])
            i += len(matched)
            continue
        ch = text[i]
        if ch in stoi:
            ids.append(stoi[ch])
        # else: skip unknown char
        i += 1
    return ids


def decode(ids, itos):
    """Decode token ids back to a string."""
    return "".join(itos[i] if 0 <= i < len(itos) else "" for i in ids)


def build_trace(segments, stoi):
    """Tokenize a list of (text, mask_value) segments.

    Returns (tokens: list[int], mask: list[int]) of equal length. mask_value
    is 1 for model-output tokens (supervised) and 0 for injected tokens.
    """
    tokens = []
    mask = []
    for text, m in segments:
        seg_ids = encode(text, stoi)
        tokens.extend(seg_ids)
        mask.extend([m] * len(seg_ids))
    if len(tokens) != len(mask):
        raise AssertionError("token/mask length drift")
    return tokens, mask


def self_test():
    """Run internal assertions; raise on any failure."""
    itos, stoi, vocab_size = build_vocab()
    assert vocab_size == AGENT_VOCAB_SIZE, vocab_size
    for lit, _id in SPECIALS:
        assert stoi[lit] == _id, (lit, stoi[lit], _id)
        assert itos[_id] == lit, (itos[_id], lit)

    # round-trip encode/decode including specials and base chars
    s = "State:\n<bash>echo D</bash><pause><result>D</result><resume>ok"
    ids = encode(s, stoi)
    assert decode(ids, itos) == s, "round-trip failed"

    # code chars survive encoding (the whole point of the vocab extension)
    code = '<canvas id="c"></canvas><script>var x=1+2*3;</script>'
    code_ids = encode(code, stoi)
    assert decode(code_ids, itos) == code, "code char round-trip failed"

    # the result+resume injected segment must be fully mask=0; the bash+pause
    # segment fully mask=1; overall there must be supervised tokens.
    segs = [
        ("State:\n", 0),
        ("<bash>echo D</bash><pause>", 1),
        ("<result>D</result><resume>", 0),
        ("<answer>done</answer>", 1),
    ]
    tokens, mask = build_trace(segs, stoi)
    assert len(tokens) == len(mask) > 0
    assert sum(mask) > 0, "no supervised tokens"

    # locate the result segment in the token stream and assert mask==0 there
    res_ids = encode("<result>D</result><resume>", stoi)
    start = _find_subseq(tokens, res_ids)
    assert start is not None, "result segment not found in trace tokens"
    for k in range(len(res_ids)):
        assert mask[start + k] == 0, f"result token {k} should be masked out"

    # the bash open tag must be supervised
    bash_id = stoi["<bash>"]
    assert bash_id in tokens, "<bash> token missing"
    bash_pos = tokens.index(bash_id)
    assert mask[bash_pos] == 1, "<bash> must be supervised (model output)"

    print(f"self_test OK: base_vocab={_BASE_SIZE}, agent_vocab={vocab_size}, "
          f"specials={SPECIAL_LIT_TO_ID}")


def _find_subseq(haystack, needle):
    """First index of needle in haystack, or None."""
    nl = len(needle)
    for i in range(len(haystack) - nl + 1):
        if haystack[i:i + nl] == needle:
            return i
    return None


def main():
    self_test()


if __name__ == "__main__":
    main()
