from __future__ import annotations

"""CSIC 2010 HTTP dataset utilities.

This module is intentionally lightweight and does *not* fetch data from the
internet. Instead, it expects the user to place the CSIC 2010 raw files under
``data_dir``.

Supported layouts (any one works):

1) Two files:
   - normal.txt (or normalTraffic*.txt)
   - anomalous.txt (or anomalousTraffic*.txt)

   Each file contains one HTTP request per sample, with requests separated by
   blank lines.

2) A pre-parsed TSV/CSV with columns: text,label.

The loader parses each HTTP request into a structured record (method/path,
query/body params, selected headers) and then serializes it using the existing
``SerializeCfg`` markers (<REC>, <SEP>, ...). This makes it compatible with the
paper's drift operators and value-aware tokenization.
"""

import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .serialize import SerializeCfg, serialize_iter


_REQ_SPLIT_RE = re.compile(r"\r?\n\r?\n")


def _read_requests_from_txt(path: str) -> List[str]:
    # CSIC files typically separate requests by blank lines.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().strip()
    if not raw:
        return []
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    return chunks


def _find_first_existing(data_dir: str, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    # try glob-like matching
    for fn in os.listdir(data_dir):
        for pat in candidates:
            if "*" in pat:
                # very small pattern support
                rgx = re.compile("^" + re.escape(pat).replace("\\*", ".*") + "$")
                if rgx.match(fn):
                    return os.path.join(data_dir, fn)
    return None


def _parse_query_params(qs: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in qs.split("&"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        k = k.strip()
        if not k:
            continue
        out[k] = v.strip()
    return out


def _parse_http_request(req: str, max_headers: int = 16) -> Dict[str, str]:
    """Best-effort HTTP request parser for tokenization research.

    The goal is not full RFC compliance; the goal is to extract stable
    field/value roles that stress tokenization.
    """
    lines = [ln.rstrip("\r") for ln in req.splitlines() if ln.strip()]
    if not lines:
        return {"raw": ""}

    first = lines[0]
    method, target, version = "", "", ""
    parts = first.split()
    if len(parts) >= 2:
        method = parts[0]
        target = parts[1]
        version = parts[2] if len(parts) >= 3 else ""

    path = target
    qs = ""
    if "?" in target:
        path, qs = target.split("?", 1)

    headers: Dict[str, str] = {}
    body_lines: List[str] = []
    in_headers = True
    for ln in lines[1:]:
        if in_headers and ":" in ln:
            k, v = ln.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k and len(headers) < max_headers:
                headers[k] = v
        else:
            in_headers = False
            body_lines.append(ln)

    body = "\n".join(body_lines).strip()
    # parse form-encoded body when it looks like key=value pairs
    body_params: Dict[str, str] = {}
    if body and ("=" in body) and ("&" in body or body.count("=") >= 1):
        # avoid treating arbitrary text as params
        if len(body) <= 4096:
            body_params = _parse_query_params(body.replace("\n", "&"))

    rec: Dict[str, str] = {
        "method": method,
        "path": path,
    }
    if version:
        rec["version"] = version

    for k, v in _parse_query_params(qs).items():
        rec[f"q_{k}"] = v
    for k, v in body_params.items():
        rec[f"b_{k}"] = v
    # include a few stability-relevant headers
    for hk in sorted(headers.keys()):
        if hk in {"host", "user-agent", "content-type", "cookie", "referer"}:
            rec[f"h_{hk}"] = headers[hk]

    # Keep the raw request as a fallback signal.
    rec["raw"] = req
    return rec


def load_csic2010_http(
    data_dir: str,
    n_train: int = 20000,
    n_val: int = 5000,
    n_test: int = 5000,
    seed: int = 0,
    serialize_cfg: SerializeCfg | None = None,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Load + parse + serialize CSIC 2010 HTTP samples.

    Returns serialized texts and labels for train/val/test.
    """
    serialize_cfg = serialize_cfg or SerializeCfg()

    # Option 1: CSV/TSV with text,label
    csv_path = _find_first_existing(data_dir, ["csic2010.csv", "csic2010.tsv", "data.csv", "data.tsv"])
    if csv_path is not None:
        df = pd.read_csv(csv_path, sep="\t" if csv_path.endswith(".tsv") else ",")
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Expected columns text,label in {csv_path}")
        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
    else:
        normal_path = _find_first_existing(
            data_dir,
            ["normal.txt", "normalTrafficTraining.txt", "normalTrafficTest.txt", "normalTraffic*.txt"],
        )
        anom_path = _find_first_existing(
            data_dir,
            ["anomalous.txt", "anomalousTrafficTest.txt", "anomalousTrafficTraining.txt", "anomalousTraffic*.txt"],
        )
        if normal_path is None or anom_path is None:
            raise FileNotFoundError(
                "CSIC files not found. Place the raw CSIC 2010 files under data_dir, e.g. "
                "normalTraffic*.txt and anomalousTraffic*.txt, or provide a csic2010.csv with text,label."
            )
        normal_reqs = _read_requests_from_txt(normal_path)
        anom_reqs = _read_requests_from_txt(anom_path)
        texts = normal_reqs + anom_reqs
        labels = [0] * len(normal_reqs) + [1] * len(anom_reqs)

    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)

    def take(n: int) -> List[int]:
        nonlocal idx
        out = idx[:n]
        idx = idx[n:]
        return out

    n_total = n_train + n_val + n_test
    if n_total > len(texts):
        raise ValueError(f"Requested {n_total} samples but only {len(texts)} available")

    i_train = take(n_train)
    i_val = take(n_val)
    i_test = take(n_test)

    def build(indices: Sequence[int]) -> Tuple[List[str], List[int]]:
        reqs = [texts[i] for i in indices]
        ys = [labels[i] for i in indices]
        recs = [_parse_http_request(r) for r in reqs]
        ser = serialize_iter(recs, serialize_cfg)
        return ser, ys

    tr_x, tr_y = build(i_train)
    va_x, va_y = build(i_val)
    te_x, te_y = build(i_test)
    return tr_x, tr_y, va_x, va_y, te_x, te_y
