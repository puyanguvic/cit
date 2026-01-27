from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Contract:
    """Deterministic, versionable interface contract.

    ICML-facing version (abstract, domain-agnostic): we only enforce *opaque*
    handling for high-cardinality value spans (long numeric / alphanumeric
    strings). We intentionally avoid hard-coding domain-specific value formats
    so the contract stays tied to the paper's core
    abstraction: restricting the deployable feasible set to avoid brittle
    instance-fragment memorization.

    The contract is *part of the tokenizer artifact* and is applied identically
    at build time and runtime.
    """

    # Treat long numeric runs as opaque (high-cardinality).
    min_num_len: int = 6
    # Treat long alphanumeric runs as opaque IDs (high-cardinality).
    min_id_len: int = 12


NUM_RE = re.compile(r"\b\d{6,}\b")
ID_RE = re.compile(r"\b[A-Za-z0-9]{12,}\b")


def apply_contract(x: str, c: Contract) -> str:
    """Apply typed-symbol normalization.

    We map high-cardinality spans to typed atoms and rely on downstream
    integrity constraints (in induction + runtime) to prevent fragmentation.
    """
    x = re.sub(rf"\b\d{{{c.min_num_len},}}\b", "<NUM_LONG>", x)
    x = re.sub(rf"\b[A-Za-z0-9]{{{c.min_id_len},}}\b", "<ID_LONG>", x)
    return x


def extract_contract_markers(x: str) -> set[str]:
    """Return all <...> atoms present in the string."""
    return set(re.findall(r"<[^<>\s]+>", x))
