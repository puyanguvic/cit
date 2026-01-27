from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_jsonable)


def save_vocab_json(vocab: Dict[str, int], path: str | Path) -> None:
    """Save a token->id mapping as JSON (stable ordering by id)."""
    inv = sorted(vocab.items(), key=lambda kv: kv[1])
    ordered = {k: int(v) for k, v in inv}
    save_json(ordered, path)


def save_run_metadata(
    outdir: str | Path,
    *,
    exp_name: str,
    args: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    outdir = Path(outdir)
    meta = {"experiment": exp_name, "args": args}
    if extra:
        meta.update(extra)
    p = outdir / "run_meta.json"
    save_json(meta, p)
    return p
