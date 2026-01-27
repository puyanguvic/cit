from __future__ import annotations

from typing import Sequence, Tuple, List

import torch
import torch.nn as nn

from .train import _collate


def evaluate(
    model: nn.Module,
    pairs: Sequence[Tuple[List[int], int]],
    pad_id: int,
    max_len: int,
    device: str = "cpu",
    batch_size: int = 128,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            x, m, y = _collate(batch, pad_id, max_len)
            x, m, y = x.to(device), m.to(device), y.to(device)
            logits = model(x, m)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def evaluate_with_prefix(
    model: nn.Module,
    pairs: Sequence[Tuple[List[int], int]],
    pad_id: int,
    max_len: int,
    prefix_k: int,
    device: str = "cpu",
    batch_size: int = 128,
) -> float:
    """Evaluate using only the first prefix_k tokens of each sequence."""
    trimmed = [(ids[:prefix_k], y) for ids, y in pairs]
    return evaluate(model, trimmed, pad_id, max_len=min(max_len, prefix_k), device=device, batch_size=batch_size)
