from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
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
    meta = {"experiment": exp_name, "args": args, "env": _capture_env()}
    if extra:
        meta.update(extra)
    p = outdir / "run_meta.json"
    save_json(meta, p)
    return p


def _capture_env() -> Dict[str, Any]:
    """Capture lightweight reproducibility metadata for experiment runs."""
    env: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }

    # Torch/CUDA are optional for analysis-only utilities.
    try:
        import torch  # type: ignore

        env["torch_version"] = getattr(torch, "__version__", "unknown")
        env["cuda_built"] = bool(getattr(torch.backends, "cuda", None) and torch.backends.cuda.is_built())
        # Avoid noisy CUDA init warnings in CPU-only environments.
        cuda_available = False
        if env["cuda_built"]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")
                try:
                    cuda_available = bool(torch.cuda.is_available())
                except Exception:
                    cuda_available = False
        env["cuda_available"] = bool(cuda_available)
        if cuda_available:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")
                try:
                    env["cuda_device_count"] = int(torch.cuda.device_count())
                except Exception:
                    env["cuda_device_count"] = 0
                try:
                    env["cuda_device_0"] = str(torch.cuda.get_device_name(0))
                except Exception:
                    pass
    except Exception:
        pass

    env["git"] = _try_git_info()
    return env


def _try_git_info() -> Dict[str, Any]:
    """Best-effort git revision info; returns {} if unavailable."""
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {
            "root": root,
            "sha": sha,
            "dirty": bool(status.strip()),
        }
    except Exception:
        return {}
