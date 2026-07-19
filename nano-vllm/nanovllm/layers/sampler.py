import torch
from torch import nn


class Sampler(nn.Module):
    """Batched sampler. temperature == 0 (per-row) => greedy argmax.

    Argmax is invariant to positive scaling, so we clamp the temperature only for
    the Gumbel-max path; greedy rows are taken from the raw-logits argmax.
    """

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy = temperatures <= 0
        if greedy.all():
            return logits.argmax(dim=-1)
        t = temperatures.clamp_min(1e-10).unsqueeze(1)
        probs = torch.softmax(logits / t, dim=-1)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        chosen = logits.argmax(dim=-1)
        return torch.where(greedy, chosen, sampled)
