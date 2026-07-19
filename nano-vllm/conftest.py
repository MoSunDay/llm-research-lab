"""Shared pytest fixtures for nano-vllm test suite.

The LLM engine calls ``torch.distributed.init_process_group`` exactly once per
process.  Both test modules need a GPU model instance, so we share a single
session-scoped ``llm`` fixture instead of each module creating its own.
"""
import pytest

MODEL_PATH = "/root/models/gpt2-nanovllm"


@pytest.fixture(scope="session")
def llm():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for end-to-end tests")
    from nanovllm import LLM
    return LLM(MODEL_PATH, tensor_parallel_size=1,
               enforce_eager=True, gpu_memory_utilization=0.7)
