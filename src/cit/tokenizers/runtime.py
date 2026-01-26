from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"]
    tok_id: Optional[int] = None

    def __init__(self):
        self.children = {}
        self.tok_id = None


def build_trie(vocab: Dict[str, int]) -> TrieNode:
    root = TrieNode()
    for tok, tid in vocab.items():
        node = root
        for ch in tok:
            node = node.children.setdefault(ch, TrieNode())
        node.tok_id = tid
    return root


def tokenize_longest_match(text: str, vocab: Dict[str, int], unk_id: int = 1) -> List[int]:
    """Greedy left-to-right longest-match tokenization.

    This is deterministic and has no backtracking. Complexity is O(n * a) where
    'a' is average trie depth walked per character; for typical small vocabularies
    used in experiments it's fast enough and predictable.

    Notes:
    - We treat whitespace as hard token boundaries.
    - Contract markers like '<REC>' or '<ID_LONG>' should be included in vocab.
    """
    trie = build_trie(vocab)

    ids: List[int] = []
    for chunk in text.split():
        i = 0
        while i < len(chunk):
            node = trie
            last_id: Optional[int] = None
            last_j = i
            j = i
            while j < len(chunk) and chunk[j] in node.children:
                node = node.children[chunk[j]]
                j += 1
                if node.tok_id is not None:
                    last_id = node.tok_id
                    last_j = j
            if last_id is None:
                ids.append(unk_id)
                i += 1
            else:
                ids.append(last_id)
                i = last_j
    return ids
