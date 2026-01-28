# CIT (Controlled Interface Tokenization) — experiment scaffold

This repo is a compact, reproducible code scaffold for the paper draft.
It includes:

- **CIT interface contract** (`src/cit/tokenizers/cit_contract.py`): deterministic typed-symbol normalization for *high-cardinality value spans* (long numeric / alphanumeric runs) enforced as opaque typed atoms.
- **Distortion-aware vocabulary induction** (`src/cit/tokenizers/cit_induction.py`): greedy gain–distortion selection aligned with the deployed greedy longest-match execution.
- **Deterministic runtime tokenizer** (`src/cit/tokenizers/runtime.py`).
- **Minimal encoder-only backbone** (`src/cit/models/encoder.py`) + **compute-matched training** (`src/cit/models/train.py`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This repository ignores local bytecode caches and experiment outputs via `.gitignore` (e.g., `__pycache__/`, `*.pyc`, `results/`).

## Experiments

### E0 (tokenizer playground): pretrained tokenizers on structured data

This scans off-the-shelf tokenizers (e.g., BERT, GPT-2, cl100k, Grok-1) on the paper's serialized datasets and saves token-length distribution stats.

```bash
python scripts/run_e0_tokenizer_playground.py --dataset csic2010 --data-dir data/csic2010 --auto-download
```

Customize the tokenizer list (name=repo_id pairs):

```bash
python scripts/run_e0_tokenizer_playground.py --dataset csic2010 --data-dir data/csic2010 \\\n+  --tokenizers bert-base-cased,gpt2,gpt3=Xenova/text-davinci-003,gpt4=Xenova/gpt-4,grok1=Xenova/grok-1-tokenizer
```

### E1 (synthetic structured stream + format drift variants)

```bash
python scripts/run_e1_synth.py
```

### E2 (end-to-end, token-fair): CSIC 2010 HTTP

CSIC files are not tracked in git (they live under `data/csic2010/`).

```bash
python scripts/run_e2_csic_http.py --data-dir data/csic2010 --device cuda --seed 0 --auto-download
```

### E3 (Pareto slice: model size vs accuracy/latency)

```bash
python scripts/run_e3_pareto_prefix.py --out e3_pareto.csv --device cuda
```

### Appendix: public tabular benchmarks (Adult / Credit-G)

```bash
python scripts/run_e2_uci.py --dataset adult --vocab 2048 --device cuda
python scripts/run_e2_uci.py --dataset credit-g --vocab 2048 --device cuda
```

### Appendix: Vocabulary-budget frontier (rate–distortion / accuracy)

```bash
python scripts/run_e4_frontier.py --vocabs 256,512,1024,2048,4096 --device cuda --outdir runs/e4_frontier --seed 0
```

This produces `results/runs/e4_frontier/seed0/results.csv` and `frontier.csv`, plus tokenizer artifacts under `results/runs/e4_frontier/seed0/tokenizers/vocabXXXX/`.

### (Legacy name) E5: CSIC 2010 HTTP

The CSIC experiment is E2 in the main-paper numbering. `run_e5_csic_http.py` is kept as a thin wrapper for backward compatibility.
You can let the script auto-download a public CSV mirror, or provide the raw CSIC files yourself.

Option A (auto-download):

```bash
python scripts/run_e5_csic_http.py --data-dir data/csic2010 --device cuda --seed 0 --auto-download
```

Option B (manual data placement). Place CSIC 2010 raw files under a directory such as `data/csic2010/`, e.g.:

```text
data/csic2010/
  normalTraffic*.txt
  anomalousTraffic*.txt
```

Then run:

```bash
python scripts/run_e5_csic_http.py --data-dir data/csic2010 --device cuda --seed 0
```

Outputs (tokenizers, logs, metrics) will be written under `results/paper/e2_csic_http/seed0/` by default.


## Plotting (frontier curves)

After running E4, generate paper-ready plots:

```bash
python scripts/plot_frontier.py --run_dir results/runs/e4_frontier/seed0
```

All experiment outputs are written under the top-level `results/` folder.


## One-click paper runs

All scripts write outputs under `results/`. The default one-click runner runs the main-paper E1/E2/E3 and will auto-download CSIC 2010 if missing.

```bash
python scripts/run_all.py --device cuda --seed 0 --plot
```

This will populate:
- `results/paper/e1_synth/seed0/...`
- `results/paper/e2_csic_http/seed0/...`
- `results/paper/e3_pareto/seed0/...`

To also run appendix sweeps:

```bash
python scripts/run_all.py --device cuda --seed 0 --only uci,frontier --plot
```

To disable auto-download for E2 (CSIC):

```bash
python scripts/run_all.py --device cuda --seed 0 --plot --no-auto-download
```

To run multiple seeds and aggregate:

```bash
python scripts/run_all.py --device cuda --seeds 0,1,2 --plot
python scripts/summarize_paper_results.py --run-root paper
```

To run E0 (tokenizer playground) from the runner:

```bash
python scripts/run_all.py --device cuda --seed 0 --only e0
```
