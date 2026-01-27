from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


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


@dataclass(frozen=True)
class LongestMatchTokenizer:
    """Deterministic greedy longest-match tokenizer with a prebuilt trie.

    NOTE: Building the trie is O(|V| * avg_token_len). Building it *per encode*
    (as older versions did) can dominate runtime and make CIT look artificially
    slow in E3/E4. This class fixes that by compiling once and reusing.
    """

    vocab: Dict[str, int]
    unk_id: int = 1

    def __post_init__(self):
        object.__setattr__(self, "_trie", build_trie(self.vocab))

    def encode(self, text: str) -> List[int]:
        trie: TrieNode = getattr(self, "_trie")
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
                    ids.append(self.unk_id)
                    i += 1
                else:
                    ids.append(last_id)
                    i = last_j
        return ids


def tokenize_longest_match(text: str, vocab: Dict[str, int], unk_id: int = 1) -> List[int]:
    """Backward-compatible helper.

    Prefer LongestMatchTokenizer(vocab).encode(text) in any hot loop.
    """
    return LongestMatchTokenizer(vocab=vocab, unk_id=unk_id).encode(text)
