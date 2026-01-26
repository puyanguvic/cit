from __future__ import annotations

import random
import re
from typing import List


def drift_shuffle_fields(x: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    m = re.search(r"<REC>\s*(.*)\s*<END>", x)
    if not m:
        return x
    body = m.group(1)
    parts = [p.strip() for p in body.split("<SEP>")]
    rng.shuffle(parts)
    return "<REC> " + " <SEP> ".join(parts) + " <END>"


def drift_whitespace(x: str) -> str:
    x = x.replace("=", " = ")
    x = re.sub(r"\s+", " ", x)
    return x


def drift_insert_dummy(x: str, seed: int = 0, k: int = 2) -> str:
    rng = random.Random(seed)
    m = re.search(r"<REC>\s*(.*)\s*<END>", x)
    if not m:
        return x
    body = m.group(1)
    parts = [p.strip() for p in body.split("<SEP>")]
    for i in range(k):
        parts.append(f"dummy{i}={rng.randrange(10**9)}")
    return "<REC> " + " <SEP> ".join(parts) + " <END>"


def drift_numeric_format(x: str) -> str:
    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        return s if len(s) >= 3 else s.zfill(3)

    return re.sub(r"\b\d+\b", repl, x)


def make_variants(x: str, seed: int = 0) -> List[str]:
    return [
        x,
        drift_shuffle_fields(x, seed=seed),
        drift_whitespace(x),
        drift_insert_dummy(x, seed=seed, k=2),
        drift_numeric_format(x),
    ]
