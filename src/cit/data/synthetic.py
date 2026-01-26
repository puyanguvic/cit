from dataclasses import dataclass
import random
import string

@dataclass
class SynthConfig:
    n_fields: int = 20
    signal_vocab: int = 20
    noise_len_min: int = 24
    noise_len_max: int = 64
    signal_pos: str = "middle"  # "prefix"|"middle"|"suffix"
    sep: str = " <SEP> "
    rec_beg: str = "<REC> "
    rec_end: str = " <END>"
    seed: int = 0

def _rand_noise(rng: random.Random, lo: int, hi: int) -> str:
    L = rng.randint(lo, hi)
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(L))

def _signal_token(i: int) -> str:
    return f"S{i}"

def make_synth_dataset(cfg: SynthConfig, n: int):
    rng = random.Random(cfg.seed)
    S = [_signal_token(i) for i in range(cfg.signal_vocab)]
    X, y = [], []
    for _ in range(n):
        s_idx = rng.randrange(cfg.signal_vocab)
        label = s_idx  # multiclass
        fields = []
        for j in range(cfg.n_fields):
            key = f"k{j}"
            val = _rand_noise(rng, cfg.noise_len_min, cfg.noise_len_max)
            fields.append((key, val))
        # place signal field
        if cfg.signal_pos == "prefix":
            pos = 0
        elif cfg.signal_pos == "suffix":
            pos = cfg.n_fields - 1
        else:
            pos = cfg.n_fields // 2
        fields[pos] = ("k_signal", S[s_idx])

        # serialize
        s = cfg.rec_beg + cfg.sep.join([f"{k}={v}" for k, v in fields]) + cfg.rec_end
        X.append(s)
        y.append(label)
    return X, y

