from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.tokenizer = None  # set by LLMEngine for stop-string decoding

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # stop on a stop token id (the token itself is excluded from output)
            if seq.stop_token_ids and token_id in seq.stop_token_ids:
                seq.finish_reason = "stop"
                self._finish(seq)
                continue
            # pause on a pause token id. The token is appended BEFORE pausing so
            # that seq.token_ids and the KV cache stay in lock-step: num_cached_tokens
            # already counted this token (via num_scheduled_tokens above), and a later
            # resume() recomputes num_tokens - num_cached_tokens, which would underflow
            # by one if the token were dropped. The token is hidden from the displayed
            # text by setting pause_index to the pre-token text length.
            if seq.pause_token_ids and token_id in seq.pause_token_ids:
                if self.tokenizer is not None:
                    seq.pause_index = len(self.tokenizer.decode(seq.completion_token_ids))
                seq.append_token(token_id)
                seq.pause_reason = "pause_token"
                self._pause(seq)
                continue
            seq.append_token(token_id)
            if seq.num_completion_tokens >= seq.max_tokens:
                seq.finish_reason = "length"
                self._finish(seq)
            elif not seq.ignore_eos and token_id == self.eos:
                seq.finish_reason = "eos"
                self._finish(seq)
            elif seq.stop and self.tokenizer is not None:
                text = self.tokenizer.decode(seq.completion_token_ids)
                hit = next((s2 for s2 in seq.stop if s2 and s2 in text), None)
                if hit is not None:
                    seq.stop_index = text.find(hit)
                    seq.finish_reason = "stop"
                    self._finish(seq)
            elif seq.pause and self.tokenizer is not None:
                text = self.tokenizer.decode(seq.completion_token_ids)
                search_from = seq.pause_search_offset
                suffix = text[search_from:]
                hit = next((p for p in seq.pause if p and p in suffix), None)
                if hit is not None:
                    seq.pause_index = search_from + suffix.find(hit)
                    seq.pause_reason = "pause"
                    self._pause(seq)

    def _finish(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        if seq in self.running:
            self.running.remove(seq)

    def _pause(self, seq: Sequence):
        """Pause a sequence: stop generating but RETAIN its KV-cache blocks.

        The sequence is removed from the running queue (so the scheduler won't
        step it) but its block_table stays intact for a later resume().
        """
        seq.status = SequenceStatus.PAUSED
        if seq in self.running:
            self.running.remove(seq)

    def resume(self, seq: Sequence):
        """Move a PAUSED sequence back to WAITING so it re-enters the prefill path.

        New tokens (appended by the engine before calling this) are prefilled
        with prefix-cache reuse, then decode continues. Blocks are NOT
        reallocated here -- the engine calls block_manager.ensure_blocks() first.
        """
        assert seq.status == SequenceStatus.PAUSED, f"seq {seq.seq_id} is {seq.status}, not PAUSED"
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.pause_reason = None
        self.waiting.append(seq)
