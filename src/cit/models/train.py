import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from tqdm import tqdm

def pad_batch(batch, pad_id: int, max_len: int):
    ids, y = zip(*batch)
    out = []
    mask = []
    for t in ids:
        t = t[:max_len]
        m = [1]*len(t)
        if len(t) < max_len:
            t = t + [pad_id]*(max_len-len(t))
            m = m + [0]*(max_len-len(m))
        out.append(t)
        mask.append(m)
    return torch.tensor(out), torch.tensor(mask), torch.tensor(y)

def train_compute_matched(model, dataset, pad_id: int, max_len: int, total_tokens: int, lr: float = 3e-4, batch_size: int = 32, device="cuda"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: pad_batch(b, pad_id, max_len))

    seen = 0
    model.train()
    pbar = tqdm(total=total_tokens, desc="train(TotalTokens)")
    while seen < total_tokens:
        for input_ids, attn, y in dl:
            input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
            # count tokens actually used
            used = int(attn.sum().item())
            logits = model(input_ids, attn)
            loss = cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += used
            pbar.update(used)
            if seen >= total_tokens:
                break
    pbar.close()
    return model

