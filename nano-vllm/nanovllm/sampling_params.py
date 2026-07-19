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

    Pause / agentic (Phase 4): ``pause`` (list of strings) and ``pause_token_ids``
    work like ``stop`` / ``stop_token_ids`` but instead of finishing the sequence
    they PAUSE it -- generation halts and the KV cache is retained so the request
    can be resumed (e.g. after a tool call) via ``LLMEngine.resume()``. Use
    ``generate_stream()`` or ``generate_agentic()`` for pause/resume; the batch
    ``generate()`` does not support it.
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    pause: list[str] | None = None
    pause_token_ids: list[int] | None = None
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
        # accept a single string / int for convenience
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if isinstance(self.stop_token_ids, int):
            self.stop_token_ids = [self.stop_token_ids]
        if isinstance(self.pause, str):
            self.pause = [self.pause]
        if isinstance(self.pause_token_ids, int):
            self.pause_token_ids = [self.pause_token_ids]
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

    @property
    def has_pause(self) -> bool:
        return bool(self.pause or self.pause_token_ids)

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0
