import torch
from torch.utils.data import DataLoader
from .train import pad_batch

@torch.no_grad()
def evaluate(model, dataset, pad_id: int, max_len: int, batch_size: int = 64, device="cuda"):
    model.eval().to(device)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: pad_batch(b, pad_id, max_len))
    correct = 0
    total = 0
    for input_ids, attn, y in dl:
        input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
        pred = model(input_ids, attn).argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)

