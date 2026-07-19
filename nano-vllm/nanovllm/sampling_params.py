from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class SamplingParams:
    """Sampling configuration for a request.

    temperature == 0 selects deterministic greedy (argmax) decoding.
    logits_processors is a list of callables applied (rank-0 only) right before
    sampling, each with signature (token_ids: list[int], logits: torch.Tensor) ->
    torch.Tensor. This is the hook used by constrained decoding (xgrammar) and
    structured/function-calling output.

    Guided (structured) decoding (Phase 3): set exactly one of json_schema /
    regex / grammar / choice / json_object. The engine compiles an xgrammar
    matcher from the spec at add_request() time (it needs the tokenizer) and
    attaches it as a logits_processor. See nanovllm/grammar.py.
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    logits_processors: list[Callable] | None = None
    # --- guided / structured decoding (xgrammar backend) ---
    json_schema: str | dict | None = None
    regex: str | None = None
    grammar: str | None = None
    choice: list[str] | None = None
    json_object: bool = False
    guided_decoding_backend: str = "xgrammar"

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be >= 0 (0 == greedy)"
        self._validate_guided_spec()

    def _validate_guided_spec(self):
        specs = [self.json_schema, self.regex, self.grammar, self.choice, self.json_object]
        n = sum(s is not None for s in specs[:4]) + bool(self.json_object)
        assert n <= 1, (
            "At most one of json_schema/regex/grammar/choice/json_object may be set; "
            f"got {n}."
        )

    @property
    def has_guidance(self) -> bool:
        return any(s is not None for s in (self.json_schema, self.regex,
                                           self.grammar, self.choice)) or self.json_object
