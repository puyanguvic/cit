from __future__ import annotations

import inspect
import torch
import torch.nn as nn


class TinyEncoder(nn.Module):
    """Small encoder-only transformer for controlled experiments."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        enc_kwargs = {}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            enc_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, **enc_kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(b, L)
        x = self.tok(input_ids) + self.pos(pos)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        # mean pool over non-pad tokens
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            w = attention_mask.unsqueeze(-1).float()
            pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return self.cls(pooled)
