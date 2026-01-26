import torch
import torch.nn as nn

class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, n_classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        if attention_mask is not None:
            # TransformerEncoder uses src_key_padding_mask=True for PAD positions
            pad_mask = attention_mask == 0
        else:
            pad_mask = None
        h = self.enc(x, src_key_padding_mask=pad_mask)
        # mean pool
        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            denom = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (h * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.cls(pooled)

