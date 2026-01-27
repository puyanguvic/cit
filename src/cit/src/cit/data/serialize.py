from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


@dataclass
class SerializeCfg:
    rec_beg: str = "<REC> "
    rec_end: str = " <END>"
    sep: str = " <SEP> "
    kv_sep: str = "="
    none_token: str = "<NULL>"
    # When True, escape values so they cannot accidentally contain boundary markers.
    strict_escape: bool = True


def _escape_value(v: str, cfg: SerializeCfg) -> str:
    """Escape boundary tokens inside values.

    For tabular public datasets this rarely triggers, but it makes the
    role/boundary contract *well-defined* and prevents accidental marker
    collisions (important for reproducibility and tests).
    """
    if not cfg.strict_escape:
        return v
    # escape the exact boundary markers used by the serializer
    repl = {
        cfg.rec_beg.strip(): "<ESC_REC>",
        cfg.rec_end.strip(): "<ESC_END>",
        cfg.sep.strip(): "<ESC_SEP>",
        cfg.kv_sep: "<ESC_EQ>",
    }
    out = v
    for a, b in repl.items():
        out = out.replace(a, b)
    return out


def serialize_row(row: pd.Series, cfg: SerializeCfg) -> str:
    parts: List[str] = []
    for k, v in row.items():
        if pd.isna(v):
            vs = cfg.none_token
        else:
            vs = _escape_value(str(v), cfg)
        parts.append(f"{k}{cfg.kv_sep}{vs}")
    return cfg.rec_beg + cfg.sep.join(parts) + cfg.rec_end


def serialize_df(df: pd.DataFrame, cfg: SerializeCfg) -> List[str]:
    return [serialize_row(df.iloc[i], cfg) for i in range(len(df))]


def serialize_iter(rows: Iterable[dict], cfg: SerializeCfg) -> List[str]:
    out = []
    for r in rows:
        parts = []
        for k, v in r.items():
            vs = cfg.none_token if v is None else _escape_value(str(v), cfg)
            parts.append(f"{k}{cfg.kv_sep}{vs}")
        out.append(cfg.rec_beg + cfg.sep.join(parts) + cfg.rec_end)
    return out
