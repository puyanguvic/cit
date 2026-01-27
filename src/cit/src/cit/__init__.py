"""CIT: Controlled Interface Tokenization.

This repository provides:
- A deterministic *interface contract* (typed symbols for high-cardinality values)
- A distortion-aware vocabulary induction routine aligned with a deployed
  greedy longest-match execution policy
- Deterministic runtime tokenization (reference implementation)

The code is intentionally compact to serve as an experiment scaffold.
"""

__all__ = [
    "data",
    "tokenizers",
    "models",
    "utils",
]
