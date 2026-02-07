"""E2: Token-fair end-to-end training on a structured public HTTP dataset (CSIC 2010).

This experiment validates CIT under a controlled end-to-end setup:
we train the same encoder family from scratch for each tokenizer, holding
architecture and token-budget constant.

The script expects the user to place CSIC 2010 raw files under --data-dir.
See cit.data.http_csic.load_csic2010_http for supported layouts.

Example:
  python scripts/run_e2_csic_http.py --data-dir data/csic2010 --device cuda --seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from shutil import copy2
from pathlib import Path
from typing import Callable, List, Tuple

from tokenizers import Tokenizer

from cit.data.http_csic import load_csic2010_http
from cit.data.drift import drift_reorder_kv_groups, drift_whitespace
from cit.tokenizers.metrics import estimate_rate
from cit.tokenizers.hf_baselines import train_bpe, train_wordpiece, train_unigram
from cit.tokenizers.cit_contract import Contract, apply_contract
from cit.tokenizers.cit_induction import InductionCfg, train_cit
from cit.tokenizers.runtime import tokenize_longest_match
from cit.models.encoder import TinyEncoder, SmallEncoder, BaseEncoder
from cit.models.train import train_compute_matched
from cit.models.eval import evaluate
from cit.utils.artifacts import save_json, save_run_metadata, save_vocab_json
from cit.utils.seed import set_seed


TOK_CACHE_SCHEMA = "e2_tokenizers_v1"


def hf_encode(tok, texts: List[str]) -> List[List[int]]:
    return [tok.encode(t).ids for t in texts]

def _encode_utf8_bytes_single(s: str) -> List[int]:
    # Reserve 0 for pad_id in the encoder; keep byte ids in [1,256].
    return [b + 1 for b in s.encode("utf-8", errors="replace")]

def encode_utf8_bytes(texts: List[str]) -> List[List[int]]:
    return [_encode_utf8_bytes_single(t) for t in texts]


def build_drift_texts(texts: List[str], seed: int) -> List[str]:
    """Build a realistic robustness slice.

    For HTTP-like records, fully shuffling *all* serialized fields is often too
    destructive (it can effectively change semantics). Here we use gentler,
    record-aware operators: reorder query/body/header groups and normalize
    whitespace.
    """
    out = []
    for i, x in enumerate(texts):
        y = drift_reorder_kv_groups(x, seed=seed + i)
        y = drift_whitespace(y)
        out.append(y)
    return out


def make_model(kind: str, vocab_size: int, n_classes: int, max_len: int):
    if kind == "tiny":
        return TinyEncoder(vocab_size=vocab_size, n_classes=n_classes, max_len=max_len)
    if kind == "small":
        return SmallEncoder(vocab_size=vocab_size, n_classes=n_classes, max_len=max_len)
    if kind == "base":
        return BaseEncoder(vocab_size=vocab_size, n_classes=n_classes, max_len=max_len)
    raise ValueError(f"Unknown model kind: {kind}")


def _parse_budget_modes(s: str) -> List[str]:
    modes: List[str] = []
    for part in s.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key in {"token", "token_fair", "token-fair"}:
            norm = "token_fair"
        elif key in {"step", "step_fair", "step-fair"}:
            norm = "step_fair"
        else:
            raise ValueError(f"Unknown budget mode: {part}")
        if norm not in modes:
            modes.append(norm)
    if not modes:
        raise ValueError("Empty --budget-modes list")
    return modes


def _avg_trunc_len(ids: List[List[int]], max_len: int) -> float:
    if not ids:
        return 0.0
    total = 0
    for s in ids:
        total += min(len(s), max_len)
    return float(total) / float(len(ids))


def _pareto_frontier(points: List[dict], *, x_key: str, y_key: str) -> List[dict]:
    """Return non-dominated points for minimizing x and maximizing y."""
    pts = sorted(points, key=lambda d: (float(d[x_key]), -float(d[y_key])))
    frontier: List[dict] = []
    best_y = float("-inf")
    for d in pts:
        y = float(d[y_key])
        if y > best_y:
            frontier.append(d)
            best_y = y
    return frontier


def _write_curves(curve_rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seen_tokens", "seen_steps", "train_loss", "train_acc", "val_acc", "wall_s", "lr"])
        for r in curve_rows:
            w.writerow(
                [
                    int(r.get("seen_tokens", 0)),
                    int(r.get("seen_steps", 0)),
                    float(r.get("train_loss", float("nan"))),
                    float(r.get("train_acc", float("nan"))),
                    float(r.get("val_acc", float("nan"))),
                    float(r.get("wall_s", float("nan"))),
                    float(r.get("lr", float("nan"))),
                ]
            )


def _hash_texts(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        b = t.encode("utf-8", errors="replace")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def _hash_labels(labels: List[int]) -> str:
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


def _build_cache_meta(args: argparse.Namespace, Xtr: List[str], ytr: List[int]) -> dict:
    return {
        "schema": TOK_CACHE_SCHEMA,
        "dataset": "csic2010_http",
        "vocab": int(args.vocab),
        "seed": int(args.seed),
        "include_raw": bool(args.include_raw),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
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
    ap.add_argument("--data-dir", type=str, required=True, help="Directory containing CSIC 2010 raw files")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument(
        "--models",
        type=str,
        default="tiny,small",
        help="Comma-separated model sizes to run: tiny,small,base",
    )
    ap.add_argument(
        "--total-tokens",
        type=int,
        default=10_000_000,
        help="Encoder-token budget per (tokenizer, model) run (token-fair training).",
    )
    ap.add_argument(
        "--budget-modes",
        type=str,
        default="token,step",
        help="Comma-separated budget modes to run: token,step",
    )
    ap.add_argument(
        "--step-fair-steps",
        type=int,
        default=0,
        help="If >0, optimizer steps for step-fair runs. If 0, derive from token budget + BPE avg length.",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="paper/e2_csic_http")
    ap.add_argument("--plot", action="store_true", help="Run plot_frontier.py after saving CSVs.")
    ap.add_argument("--plot-group-key", type=str, default="tokenizer", help="Grouping key for frontier plots.")
    ap.add_argument(
        "--include-raw",
        action="store_true",
        help="Include the raw HTTP request field in serialization (off by default for tokenizer-centric runs).",
    )
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument(
        "--auto-download",
        action="store_true",
        help="If set, download and prepare the CSIC 2010 CSVs from lexr.ai when data-dir is missing.",
    )
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
            "Default: results/<run-root>/_tokenizer_cache/e2"
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
    save_run_metadata(outdir, exp_name="e2_csic_http", args=vars(args))

    if args.auto_download:
        data_dir = Path(args.data_dir)
        have_dir = data_dir.exists()
        has_data = False
        if have_dir:
            for name in ["csic2010.csv", "csic2010.tsv", "data.csv", "data.tsv"]:
                if (data_dir / name).exists():
                    has_data = True
                    break
            if not has_data:
                for pat in ["normal.txt", "normalTraffic*.txt", "anomalous.txt", "anomalousTraffic*.txt"]:
                    if list(data_dir.glob(pat)):
                        has_data = True
                        break
        if not have_dir or not has_data:
            setup_script = Path(__file__).parent / "setup_csic2010_lexr.py"
            print(f"[auto-download] Preparing CSIC 2010 dataset under {data_dir} ...")
            subprocess.check_call([sys.executable, str(setup_script), "--out", str(data_dir)])

    Xtr, ytr, Xva, yva, Xte, yte = load_csic2010_http(
        args.data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        include_raw=bool(args.include_raw),
    )
    n_classes = len(set(ytr))

    # Class-imbalance handling (important for HTTP anomaly detection): compute
    # inverse-frequency weights so training does not collapse to majority class.
    counts = [0] * n_classes
    for yy in ytr:
        if 0 <= int(yy) < n_classes:
            counts[int(yy)] += 1
    total = max(1, sum(counts))
    class_weights = [float(total / max(1, c)) for c in counts]
    majority_class = int(max(range(n_classes), key=lambda i: counts[i])) if n_classes > 0 else 0
    majority_train_frac = float(max(counts) / total) if counts else 0.0
    majority_test_acc = float(sum(1 for yy in yte if int(yy) == majority_class) / max(1, len(yte)))

    budget_modes = _parse_budget_modes(args.budget_modes)

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

    cache_dir: Path | None = None
    cache_paths: dict[str, Path] = {}
    cache_hit = False
    cache_meta: dict = {}

    if args.reuse_tokenizers:
        cache_root = _resolve_cache_root(user_out, args.tokenizer_cache_dir, exp_name="e2")
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
        # Train tokenizers on training serialization.
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

    # Save tokenizer artifacts for the current run.
    bpe.save(str(tok_bpe))
    wp.save(str(tok_wp))
    uni.save(str(tok_uni))
    save_vocab_json(cit_vocab, tok_cit_vocab)
    save_json(cit_contract, tok_cit_contract)
    if cache_hit and cache_paths.get("cit_log") and cache_paths["cit_log"].exists():
        copy2(cache_paths["cit_log"], tok_cit_log)

    # Drift slice (role/boundary stress under the same serializer markers).
    Xte_drift = build_drift_texts(Xte, seed=args.seed)

    encoders: List[Tuple[str, Callable[[List[str]], List[List[int]]]]]
    encoders = [
        ("BPE", lambda t: hf_encode(bpe, t)),
        ("WordPiece", lambda t: hf_encode(wp, t)),
        ("Unigram", lambda t: hf_encode(uni, t)),
        ("Bytes", encode_utf8_bytes),
        ("CIT", lambda t: [tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab) for s in t]),
    ]

    model_kinds = [m.strip() for m in args.models.split(",") if m.strip()]

    encoded = {}
    for tok_name, enc in encoders:
        tr_ids = enc(Xtr)
        va_ids = enc(Xva)
        te_ids = enc(Xte)
        te_drift_ids = enc(Xte_drift)
        encoded[tok_name] = {
            "tr_ids": tr_ids,
            "va_ids": va_ids,
            "te_ids": te_ids,
            "te_drift_ids": te_drift_ids,
            "train_pairs": list(zip(tr_ids, ytr)),
            "val_pairs": list(zip(va_ids, yva)),
            "test_pairs": list(zip(te_ids, yte)),
            "drift_pairs": list(zip(te_drift_ids, yte)),
            "avg_len": float(estimate_rate(te_ids)),
            "p95_len": int(sorted([len(s) for s in te_ids])[int(0.95 * len(te_ids))]),
            "avg_len_train": float(_avg_trunc_len(tr_ids, args.max_len)),
        }

    step_budget = None
    if "step_fair" in budget_modes:
        if args.step_fair_steps > 0:
            step_budget = int(args.step_fair_steps)
        else:
            ref_tok = "BPE" if "BPE" in encoded else list(encoded.keys())[0]
            ref_len = max(1.0, encoded[ref_tok]["avg_len_train"])
            tokens_per_step = max(1.0, ref_len * float(args.batch_size))
            step_budget = max(1, int(args.total_tokens / tokens_per_step))
        print(f"[step-fair] Using total_steps={step_budget}")

    curves_root = outdir / "curves"
    rows: List[dict] = []
    rows_by_mode = {m: [] for m in budget_modes}

    for model_kind in model_kinds:
        # Conservative LR scaling for larger models to avoid collapse.
        lr_eff = float(args.lr)
        if model_kind == "base":
            lr_eff = min(lr_eff, 2e-4)
        for tok_name, _ in encoders:
            data = encoded[tok_name]
            pad_id = 0
            train_pairs = data["train_pairs"]
            val_pairs = data["val_pairs"]
            test_pairs = data["test_pairs"]
            drift_pairs = data["drift_pairs"]

            for mode in budget_modes:
                curve_rows: List[dict] = []

                def _on_log(rec: dict):
                    curve_rows.append(dict(rec))

                if mode == "token_fair":
                    total_steps = None
                    eval_every_tokens = max(50_000, args.total_tokens // 50)
                    eval_every_steps = None
                    warmup_tokens = max(50_000, int(args.total_tokens // 20))
                    warmup_steps = 0
                elif mode == "step_fair":
                    if not step_budget:
                        continue
                    total_steps = int(step_budget)
                    eval_every_tokens = max(50_000, args.total_tokens // 50)
                    eval_every_steps = max(10, total_steps // 50)
                    warmup_tokens = 0
                    warmup_steps = min(total_steps, max(10, total_steps // 20))
                else:
                    raise ValueError(f"Unknown budget mode: {mode}")

                model = make_model(model_kind, vocab_size=args.vocab, n_classes=n_classes, max_len=args.max_len)

                log_path = outdir / "train_logs" / f"{model_kind}_{tok_name}_{mode}.jsonl"
                model = train_compute_matched(
                    model,
                    train_pairs,
                    pad_id,
                    args.max_len,
                    total_tokens=args.total_tokens,
                    total_steps=total_steps,
                    device=args.device,
                    batch_size=args.batch_size,
                    lr=lr_eff,
                    probe_only=False,
                    grad_clip=float(args.grad_clip),
                    class_weights=class_weights,
                    warmup_tokens=warmup_tokens,
                    warmup_steps=warmup_steps,
                    log_path=str(log_path),
                    eval_pairs=val_pairs,
                    eval_every_tokens=eval_every_tokens,
                    eval_every_steps=eval_every_steps,
                    on_log=_on_log,
                )

                acc = evaluate(model, test_pairs, pad_id, args.max_len, device=args.device)
                drift_acc = evaluate(model, drift_pairs, pad_id, args.max_len, device=args.device)

                n_params = int(sum(p.numel() for p in model.parameters()))
                curves_path = curves_root / mode / f"curves_{model_kind}_{tok_name}.csv"
                _write_curves(curve_rows, curves_path)

                row = {
                    "dataset": "csic2010",
                    "model": model_kind,
                    "tokenizer": tok_name,
                    "budget_mode": mode,
                    "acc": float(acc),
                    "drift_acc": float(drift_acc),
                    "avg_len": float(data["avg_len"]),
                    "p95_len": int(data["p95_len"]),
                    "n_params": n_params,
                    "params_m": float(n_params) / 1_000_000.0,
                    "vocab": int(args.vocab),
                    "max_len": int(args.max_len),
                    "total_tokens": int(args.total_tokens),
                    "total_steps": int(total_steps or 0),
                    "majority_train_frac": float(majority_train_frac),
                    "majority_test_acc": float(majority_test_acc),
                    "seed": int(args.seed),
                }
                rows.append(row)
                rows_by_mode[mode].append(row)
                print(
                    f"csic2010\t{model_kind}\t{tok_name}\t{mode}\tacc={acc:.4f}\t"
                    f"drift_acc={drift_acc:.4f}\tavg_len={data['avg_len']:.1f}\tp95_len={data['p95_len']}"
                )

    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows}, outdir / "results.json")

    for mode, mode_rows in rows_by_mode.items():
        if not mode_rows:
            continue
        out_csv = outdir / f"results_{mode}.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mode_rows[0].keys()))
            w.writeheader()
            w.writerows(mode_rows)

        frontier = _pareto_frontier(mode_rows, x_key="avg_len", y_key="acc")
        out_f = outdir / f"frontier_{mode}.csv"
        with out_f.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mode_rows[0].keys()))
            w.writeheader()
            w.writerows(frontier)
        save_json({"frontier": frontier}, outdir / f"frontier_{mode}.json")

    if args.plot:
        script = Path(__file__).parent / "plot_frontier.py"
        for mode, mode_rows in rows_by_mode.items():
            if not mode_rows:
                continue
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        str(script),
                        "--run_dir",
                        str(outdir),
                        "--results_csv",
                        f"results_{mode}.csv",
                        "--group_key",
                        args.plot_group_key,
                        "--prefix",
                        mode,
                    ]
                )
            except Exception as e:
                print(f"[warn] plot_frontier failed for {mode}: {e}")

    print(f"[wrote] {outdir / 'results.csv'}")


if __name__ == "__main__":
    main()
