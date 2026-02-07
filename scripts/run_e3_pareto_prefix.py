"""E3: System Pareto slice via model-size scaling.

We vary encoder size (Small/Base) and evaluate accuracy vs. token length.
For a lightweight public benchmark, we use OpenML Adult.

This is intentionally minimal: it produces a CSV that you can directly copy
into the paper's Pareto table/plot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from shutil import copy2
from pathlib import Path

import torch
from tokenizers import Tokenizer

from cit.utils.seed import set_seed
from cit.data.uci import load_adult
from cit.data.serialize import SerializeCfg, serialize_df
from cit.tokenizers.hf_baselines import train_bpe, train_wordpiece, train_unigram
from cit.tokenizers.cit_contract import Contract, apply_contract
from cit.tokenizers.cit_induction import train_cit, InductionCfg
from cit.tokenizers.runtime import tokenize_longest_match
from cit.models.encoder import TinyEncoder
from cit.models.train import train_compute_matched
from cit.models.eval import evaluate
from cit.utils.artifacts import save_json, save_vocab_json, save_run_metadata
from cit.tokenizers.metrics import DistortionCfg, estimate_surrogate_distortion, estimate_rate


TOK_CACHE_SCHEMA = "e3_tokenizers_v1"


def hf_encode(tok, texts):
    return [tok.encode(t).ids for t in texts]

def _encode_utf8_bytes_single(s: str) -> list[int]:
    # Reserve 0 for pad_id in the encoder; keep byte ids in [1,256].
    return [b + 1 for b in s.encode("utf-8", errors="replace")]

def encode_utf8_bytes(texts: list[str]) -> list[list[int]]:
    return [_encode_utf8_bytes_single(t) for t in texts]


def measure_latency(model, sample_batch, device: str, iters: int = 50) -> float:
    """Return p95 forward latency in milliseconds."""
    x, m = sample_batch
    x = x.to(device)
    m = m.to(device)
    model = model.to(device)
    model.eval()

    # warmup
    for _ in range(10):
        _ = model(x, m)

    times = []
    if device.startswith("cuda") and torch.cuda.is_available():
        for _ in range(iters):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            _ = model(x, m)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x, m)
            times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return float(times[int(0.95 * len(times))])


def _hash_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        b = t.encode("utf-8", errors="replace")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def _hash_labels(labels: list[int]) -> str:
    h = hashlib.sha256()
    for y in labels:
        h.update(int(y).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def _resolve_cache_root(user_out: Path, cache_arg: str, exp_name: str) -> Path:
    if cache_arg:
        p = Path(cache_arg)
        if p.is_absolute():
            return p
        return Path("results") / p
    run_root = user_out.parts[0] if user_out.parts else user_out.name
    return Path("results") / run_root / "_tokenizer_cache" / exp_name


def _build_cache_meta(args: argparse.Namespace, Xtr: list[str], ytr: list[int]) -> dict:
    return {
        "schema": TOK_CACHE_SCHEMA,
        "dataset": "openml_adult",
        "vocab": int(args.vocab),
        "seed": int(args.seed),
        "cit_contract": {"min_id_len": 12, "min_num_len": 6},
        "cit_cfg": {
            "vocab_size": int(args.vocab),
            "seed": int(args.seed),
            "mode": "fast",
            "apply_contract_text": True,
            "enforce_equals_boundary": True,
        },
        "texts_hash": _hash_texts(Xtr),
        "labels_hash": _hash_labels(ytr),
    }


def _cache_key(meta: dict) -> str:
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_vocab_json(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): int(v) for k, v in dict(obj).items()}


def _load_contract_json(path: Path) -> Contract:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    d = dict(obj)
    return Contract(
        min_num_len=int(d.get("min_num_len", 6)),
        min_id_len=int(d.get("min_id_len", 12)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument(
        "--total-tokens",
        type=int,
        default=5_000_000,
        help="Training budget measured in *non-pad encoder tokens* (token-fair).",
    )
    ap.add_argument(
        "--model-families",
        type=str,
        default="mini,small",
        help="Comma-separated model families (mini,small,medium).",
    )
    ap.add_argument(
        "--full-finetune",
        action="store_true",
        help="Full fine-tuning of the encoder. Recommended for E3; probe-only on a random backbone collapses.",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="", help="Optional legacy CSV output path (in addition to results/...)")
    ap.add_argument("--outdir", type=str, default="runs/e3_pareto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reuse-tokenizers",
        dest="reuse_tokenizers",
        action="store_true",
        help="Reuse cached tokenizer artifacts when available (default).",
    )
    ap.add_argument(
        "--no-reuse-tokenizers",
        dest="reuse_tokenizers",
        action="store_false",
        help="Disable tokenizer artifact cache and retrain tokenizers.",
    )
    ap.add_argument(
        "--tokenizer-cache-dir",
        type=str,
        default="",
        help=(
            "Optional tokenizer cache directory. Relative paths are resolved under results/. "
            "Default: results/<run-root>/_tokenizer_cache/e3"
        ),
    )
    ap.set_defaults(reuse_tokenizers=True)
    args = ap.parse_args()

    set_seed(args.seed)

    user_out = Path(args.outdir)
    if user_out.is_absolute():
        user_out = Path(user_out.name)
    outdir = (Path("results") / user_out) / f"seed{args.seed}"

    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e3_pareto", args=vars(args))
    ds = load_adult(args.seed)
    n_classes = len(set(ds.y_train))

    ser_cfg = SerializeCfg()
    Xtr = serialize_df(ds.X_train, ser_cfg)
    Xte = serialize_df(ds.X_test, ser_cfg)

    tok_dir = outdir / "tokenizers"
    (tok_dir / "bpe").mkdir(parents=True, exist_ok=True)
    (tok_dir / "wordpiece").mkdir(parents=True, exist_ok=True)
    (tok_dir / "unigram").mkdir(parents=True, exist_ok=True)
    (tok_dir / "cit").mkdir(parents=True, exist_ok=True)
    tok_bpe = tok_dir / "bpe" / "tokenizer.json"
    tok_wp = tok_dir / "wordpiece" / "tokenizer.json"
    tok_uni = tok_dir / "unigram" / "tokenizer.json"
    tok_cit_vocab = tok_dir / "cit" / "vocab.json"
    tok_cit_contract = tok_dir / "cit" / "contract.json"
    tok_cit_log = tok_dir / "cit" / "induction_log.jsonl"

    ytr = ds.y_train.tolist()

    cache_dir: Path | None = None
    cache_paths: dict[str, Path] = {}
    cache_hit = False
    cache_meta: dict = {}

    if args.reuse_tokenizers:
        cache_root = _resolve_cache_root(user_out, args.tokenizer_cache_dir, exp_name="e3")
        cache_meta = _build_cache_meta(args, Xtr, ytr)
        cache_dir = cache_root / _cache_key(cache_meta)
        cache_paths = {
            "bpe": cache_dir / "bpe" / "tokenizer.json",
            "wordpiece": cache_dir / "wordpiece" / "tokenizer.json",
            "unigram": cache_dir / "unigram" / "tokenizer.json",
            "cit_vocab": cache_dir / "cit" / "vocab.json",
            "cit_contract": cache_dir / "cit" / "contract.json",
            "cit_log": cache_dir / "cit" / "induction_log.jsonl",
        }
        need = [cache_paths["bpe"], cache_paths["wordpiece"], cache_paths["unigram"], cache_paths["cit_vocab"], cache_paths["cit_contract"]]
        if all(p.exists() for p in need):
            bpe = Tokenizer.from_file(str(cache_paths["bpe"]))
            wp = Tokenizer.from_file(str(cache_paths["wordpiece"]))
            uni = Tokenizer.from_file(str(cache_paths["unigram"]))
            cit_vocab = _load_vocab_json(cache_paths["cit_vocab"])
            cit_contract = _load_contract_json(cache_paths["cit_contract"])
            cache_hit = True
            print(f"[tok-cache] hit: {cache_dir}")
        else:
            print(f"[tok-cache] miss: {cache_dir}")

    if not cache_hit:
        # tokenizers: keep small for speed
        bpe = train_bpe(Xtr, vocab_size=args.vocab)
        wp = train_wordpiece(Xtr, vocab_size=args.vocab)
        uni = train_unigram(Xtr, vocab_size=args.vocab)

        contract = Contract()
        cit_vocab, cit_contract = train_cit(
            Xtr,
            ytr,
            contract,
            InductionCfg(vocab_size=args.vocab, seed=args.seed),
            log_path=str(tok_cit_log),
        )

        if cache_dir is not None:
            (cache_dir / "bpe").mkdir(parents=True, exist_ok=True)
            (cache_dir / "wordpiece").mkdir(parents=True, exist_ok=True)
            (cache_dir / "unigram").mkdir(parents=True, exist_ok=True)
            (cache_dir / "cit").mkdir(parents=True, exist_ok=True)
            bpe.save(str(cache_paths["bpe"]))
            wp.save(str(cache_paths["wordpiece"]))
            uni.save(str(cache_paths["unigram"]))
            save_vocab_json(cit_vocab, cache_paths["cit_vocab"])
            save_json(cit_contract, cache_paths["cit_contract"])
            if tok_cit_log.exists():
                copy2(tok_cit_log, cache_paths["cit_log"])
            save_json({"key": _cache_key(cache_meta), "meta": cache_meta}, cache_dir / "meta.json")
            print(f"[tok-cache] saved: {cache_dir}")

    # save tokenizer artifacts for this run
    bpe.save(str(tok_bpe))
    wp.save(str(tok_wp))
    uni.save(str(tok_uni))
    save_vocab_json(cit_vocab, tok_cit_vocab)
    save_json(cit_contract, tok_cit_contract)
    if cache_hit and cache_paths.get("cit_log") and cache_paths["cit_log"].exists():
        copy2(cache_paths["cit_log"], tok_cit_log)

    tokenizers = [
        ("BPE", lambda t: hf_encode(bpe, t)),
        ("WordPiece", lambda t: hf_encode(wp, t)),
        ("Unigram", lambda t: hf_encode(uni, t)),
        ("Bytes", encode_utf8_bytes),
        ("CIT", lambda t: [tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab) for s in t]),
    ]

    # model families
    from cit.models.family import parse_model_families, get_family_cfg

    fams = parse_model_families(args.model_families)
    backbones = [(name, get_family_cfg(name)) for name in fams]

    rows = []
    dcfg = DistortionCfg(sample_size=2000, seed=args.seed, show_progress=True)
    for tok_name, enc in tokenizers:
        tr_ids = enc(Xtr)
        te_ids = enc(Xte)
        pad_id = 0
        train_pairs = list(zip(tr_ids, ytr))
        test_pairs = list(zip(te_ids, ds.y_test.tolist()))

        avg_len = sum(len(s) for s in te_ids) / len(te_ids)
        p95 = sorted([len(s) for s in te_ids])[int(0.95 * len(te_ids))]

        # interface distortion (surrogate)
        if tok_name == "CIT":
            enc_pref = lambda s: tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab)
        else:
            if tok_name == "Bytes":
                enc_pref = _encode_utf8_bytes_single
            else:
                base_tok = {"BPE": bpe, "WordPiece": wp, "Unigram": uni}[tok_name]
                enc_pref = lambda s, _tok=base_tok: _tok.encode(s).ids
        dist = estimate_surrogate_distortion(Xtr, ytr, encode_prefix=enc_pref, vocab_size=args.vocab, cfg=dcfg)

        for bb_name, bb_cfg in backbones:
            model = TinyEncoder(vocab_size=args.vocab, n_classes=n_classes, max_len=args.max_len, **bb_cfg)
            n_params = int(sum(p.numel() for p in model.parameters()))
            model = train_compute_matched(
                model,
                train_pairs,
                pad_id,
                args.max_len,
                args.total_tokens,
                device=args.device,
                probe_only=not args.full_finetune,
            )
            acc = evaluate(model, test_pairs, pad_id, args.max_len, device=args.device)

            # latency on a small batch
            # build a batch of 32 samples
            batch_ids = te_ids[:32]
            L = min(args.max_len, max(len(s) for s in batch_ids))
            x = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
            m = torch.zeros((len(batch_ids), L), dtype=torch.long)
            for i, s in enumerate(batch_ids):
                s = s[:L]
                x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                m[i, : len(s)] = 1
            p95_ms = measure_latency(model, (x, m), device=args.device)

            rows.append(
                {
                    "tokenizer": tok_name,
                    "backbone": bb_name,
                    "seed": int(args.seed),
                    "vocab": int(args.vocab),
                    "max_len": int(args.max_len),
                    "total_tokens": int(args.total_tokens),
                    "acc": acc,
                    "avg_len": avg_len,
                    "p95_len": p95,
                    "distortion_hat": float(dist),
                    "p95_latency_ms": p95_ms,
                    "n_params": n_params,
                    "params_m": float(n_params) / 1_000_000.0,
                }
            )
            print(tok_name, bb_name, "acc", acc, "p95_ms", p95_ms)

    if args.out:
        out_csv = Path(args.out)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # canonical copy inside outdir
    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows}, outdir / "results.json")


if __name__ == "__main__":
    main()
