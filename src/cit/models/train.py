from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim


class SeqClsDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[List[int], int]]):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _collate(batch, pad_id: int, max_len: int):
    ids = [b[0][:max_len] for b in batch]
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    L = max(len(s) for s in ids) if ids else 1
    L = min(L, max_len)
    x = torch.full((len(ids), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(ids), L), dtype=torch.long)
    for i, s in enumerate(ids):
        s = s[:L]
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        m[i, : len(s)] = 1
    return x, m, y


def train_compute_matched(
    model: nn.Module,
    train_pairs: Sequence[Tuple[List[int], int]],
    pad_id: int,
    max_len: int,
    total_tokens: int,
    device: str = "cpu",
    batch_size: int = 64,
    lr: float = 3e-4,
) -> nn.Module:
    """Train for a fixed *encoder-token* budget.

    total_tokens counts the total number of non-pad tokens fed into the encoder.
    We adjust the number of steps accordingly.
    """
    model = model.to(device)
    ds = SeqClsDataset(train_pairs)

    def collate(batch):
        return _collate(batch, pad_id, max_len)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    seen_tokens = 0
    model.train()
    it = iter(dl)
    while seen_tokens < total_tokens:
        try:
            x, m, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, m, y = next(it)

        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x, m)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        seen_tokens += int(m.sum().item())

    return model
