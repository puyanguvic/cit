import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Contract:
    # role markers in our serialization are already explicit (<REC>, <SEP>, <END>, '=')
    # typed-value patterns are domain-agnostic: long alnum ids and long numbers
    min_id_len: int = 12
    min_num_len: int = 6

ID_RE = re.compile(r"\b[A-Za-z0-9]{12,}\b")
NUM_RE = re.compile(r"\b\d{6,}\b")

def apply_contract(x: str, c: Contract) -> str:
    # typed integrity: replace high-cardinality blobs with typed symbols
    x = re.sub(r"\b\d{%d,}\b" % c.min_num_len, "<NUM_LONG>", x)
    x = re.sub(r"\b[A-Za-z0-9]{%d,}\b" % c.min_id_len, "<ID_LONG>", x)
    return x

