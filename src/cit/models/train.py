from __future__ import annotations

import os

from typing import Callable, List, Optional, Sequence, Tuple

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
    probe_only: bool = False,
    log_path: Optional[str] = None,
    eval_pairs: Optional[Sequence[Tuple[List[int], int]]] = None,
    eval_every_tokens: int = 200_000,
    on_log: Optional[Callable[[dict], None]] = None,
) -> nn.Module:
    """Train for a fixed *encoder-token* budget.

    total_tokens counts the total number of non-pad tokens fed into the encoder.
    We adjust the number of steps accordingly.
    """
    model = model.to(device)

    # Probe-only training: freeze the encoder backbone and only train the
    # classification head. This matches the paper's drop-in regime and
    # is dramatically faster than full fine-tuning.
    if probe_only:
        for name, p in model.named_parameters():
            # Keep only the classifier head trainable.
            p.requires_grad = name.startswith("cls.")
    ds = SeqClsDataset(train_pairs)

    def collate(batch):
        return _collate(batch, pad_id, max_len)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    # Fallback: if the model has no explicit head, train all params.
    if not params:
        params = list(model.parameters())
    opt = optim.AdamW(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    import json
    import time

    seen_tokens = 0
    last_eval_at = 0
    model.train()
    it = iter(dl)
    t0 = time.time()

    def _emit(rec: dict):
        if on_log is not None:
            on_log(rec)
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
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

        # Lightweight logging/evaluation (token-budget aligned).
        if log_path is not None or on_log is not None:
            if seen_tokens - last_eval_at >= eval_every_tokens:
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    acc = float((pred == y).float().mean().item())
                rec = {
                    "seen_tokens": int(seen_tokens),
                    "train_loss": float(loss.item()),
                    "train_acc": acc,
                    "wall_s": float(time.time() - t0),
                }
                if eval_pairs is not None and len(eval_pairs) > 0:
                    from .eval import evaluate

                    rec["val_acc"] = float(
                        evaluate(model, eval_pairs, pad_id=pad_id, max_len=max_len, device=device)
                    )
                _emit(rec)
                last_eval_at = seen_tokens

    return model
