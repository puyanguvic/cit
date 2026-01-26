from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CITArtifact:
    vocab: Dict[str, int]         # token -> id
    inv_vocab: List[str]          # id -> token
    max_tok_len: int              # max token string length
    special_unk: str = "[UNK]"

def build_artifact(tokens: List[str]):
    vocab = {t:i for i,t in enumerate(tokens)}
    inv = tokens[:]
    max_len = max(len(t) for t in tokens)
    return CITArtifact(vocab=vocab, inv_vocab=inv, max_tok_len=max_len)

def tokenize_longest_match(s: str, art: CITArtifact) -> List[int]:
    # deterministic left-to-right longest-match over raw characters
    # NOTE: This is a simplified runtime. In practice you'd use pretokenization & boundaries.
    out = []
    i = 0
    n = len(s)
    unk_id = art.vocab.get(art.special_unk, 1)
    while i < n:
        L = min(art.max_tok_len, n - i)
        found = None
        for l in range(L, 0, -1):
            sub = s[i:i+l]
            if sub in art.vocab:
                found = art.vocab[sub]
                i += l
                break
        if found is None:
            out.append(unk_id)
            i += 1
        else:
            out.append(found)
    return out

