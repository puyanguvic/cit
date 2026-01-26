from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def train_bpe(texts, vocab_size: int):
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok

def train_wordpiece(texts, vocab_size: int):
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIALS)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok

def train_unigram(texts, vocab_size: int):
    tok = Tokenizer(Unigram())
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIALS)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok

