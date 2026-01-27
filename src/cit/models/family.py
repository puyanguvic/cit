"""Model-family presets for controlled scaling.

These presets are intentionally small so that full fine-tuning experiments are
feasible on a single GPU (e.g., RTX 4090) while still enabling meaningful
scaling comparisons.
"""

from __future__ import annotations


MODEL_FAMILIES: dict[str, dict] = {
    # Fast default (~1–3M params depending on vocab/max_len).
    "mini": dict(d_model=256, n_layers=4, n_heads=4),
    # Stronger but still single-GPU friendly at max_len=512.
    "small": dict(d_model=384, n_layers=6, n_heads=6),
    # Optional; use if you want a clearer scaling point.
    "medium": dict(d_model=512, n_layers=8, n_heads=8),
}


def parse_model_families(spec: str) -> list[str]:
    """Parse comma-separated family names; default to ["mini"]."""
    names = [s.strip().lower() for s in (spec or "").split(",") if s.strip()]
    return names or ["mini"]


def get_family_cfg(name: str) -> dict:
    name = name.strip().lower()
    if name not in MODEL_FAMILIES:
        raise KeyError(f"Unknown model family '{name}'. Choose from {sorted(MODEL_FAMILIES)}")
    return dict(MODEL_FAMILIES[name])
