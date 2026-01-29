#!/usr/bin/env python3
"""Sync paper figures into ./Figs for stable LaTeX paths.

The paper sources (`paper.tex`, `appendix.tex`) include figures via paths like:
  \\includegraphics{Figs/<name>.pdf}

Experiment runners typically write figures under results/<run_root>/... or the
exporter writes them under:
  results/<run_root>/paper_artifacts/{main,appendix}/figures/

This script parses the LaTeX sources, finds the required `Figs/...` assets, and
copies the newest matching files into `./Figs/`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


_INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def _find_fig_paths(tex_path: Path) -> list[Path]:
    try:
        text = tex_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read {tex_path}: {e}") from e

    out: list[Path] = []
    for m in _INCLUDE_RE.finditer(text):
        raw = m.group(1).strip()
        # Normalize common variants.
        raw = raw.removeprefix("./")
        if raw.startswith("Figs/"):
            out.append(Path(raw[len("Figs/") :]))
    return out


def _pick_source(results_root: Path, rel: Path) -> Path | None:
    """Find a source file for a figure referenced as Figs/<rel>."""
    preferred = [
        results_root / "paper_artifacts" / "main" / "figures" / rel.name,
        results_root / "paper_artifacts" / "appendix" / "figures" / rel.name,
    ]
    for p in preferred:
        if p.exists():
            return p

    # Fallback: search anywhere under results_root and pick the newest match.
    matches = [p for p in results_root.rglob(rel.name) if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=str, default="paper", help="Folder under results/ (default: paper)")
    ap.add_argument("--outdir", type=str, default="Figs", help="Output figure folder (default: ./Figs)")
    ap.add_argument(
        "--tex",
        type=str,
        action="append",
        default=[],
        help="LaTeX file to scan for \\includegraphics{Figs/...}. Can be repeated.",
    )
    ap.add_argument("--clean", action="store_true", help="Remove stale files in outdir not referenced by TeX.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be copied, but do not write files.")
    ap.add_argument("--strict", action="store_true", help="Fail if any referenced figures cannot be found.")
    args = ap.parse_args()

    tex_paths = [Path(p) for p in (args.tex or ["paper.tex", "appendix.tex"])]
    needed: set[Path] = set()
    for p in tex_paths:
        needed.update(_find_fig_paths(p))

    if not needed:
        print("[warn] No Figs/* references found in:", ", ".join(str(p) for p in tex_paths))
        return

    results_root = Path("results") / args.run_root
    outdir = Path(args.outdir)
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied = 0
    for rel in sorted(needed, key=lambda p: str(p)):
        src = _pick_source(results_root, rel)
        if src is None:
            missing.append(f"Figs/{rel}")
            continue
        dst = outdir / rel
        if args.dry_run:
            print("[DRY] copy", src, "->", dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    if args.clean and outdir.exists():
        keep = {str((outdir / rel).resolve()) for rel in needed}
        for p in outdir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".pdf", ".png"}:
                continue
            if str(p.resolve()) not in keep:
                if args.dry_run:
                    print("[DRY] remove", p)
                else:
                    p.unlink()

    if missing:
        print("[warn] Missing figure sources for:")
        for m in missing:
            print(" -", m)
        if args.strict:
            raise SystemExit(1)

    print(f"[OK] Synced {copied} figures into: {outdir}")


if __name__ == "__main__":
    main()
