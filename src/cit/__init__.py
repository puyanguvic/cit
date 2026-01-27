"""CIT: Controlled Interface Tokenization.

This repository provides:
- A deterministic *interface contract* (typed symbols for high-cardinality values)
- A distortion-aware vocabulary induction routine aligned with a deployed
  greedy longest-match execution policy
- Deterministic runtime tokenization (reference implementation)

The code is intentionally compact to serve as an experiment scaffold.
"""

from pathlib import Path
from pkgutil import extend_path

# Ensure imports work with the nested src/ layout used in this repo.
__path__ = extend_path(__path__, __name__)
_nested = Path(__file__).resolve().parent / "src" / "cit"
if _nested.exists():
    __path__.append(str(_nested))

__all__ = [
    "data",
    "tokenizers",
    "models",
    "utils",
]
