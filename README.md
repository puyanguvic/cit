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

## Experiments

### E1 (synthetic structured stream + format drift variants)

```bash
python scripts/run_e1_synth.py
```

### E2 (public OpenML datasets: Adult / Credit-G)

```bash
python scripts/run_e2_uci.py --dataset adult --vocab 2048 --device cuda
python scripts/run_e2_uci.py --dataset credit-g --vocab 2048 --device cuda
```

### E3 (Pareto slice: model size vs accuracy/latency)

```bash
python scripts/run_e3_pareto_prefix.py --out e3_pareto.csv --device cuda
```



### E4: Vocabulary-budget frontier (rate–distortion / accuracy)

```bash
python scripts/run_e4_frontier.py --vocabs 256,512,1024,2048,4096 --device cuda --outdir runs/e4_frontier --seed 0
```

This produces `runs/e4_frontier/seed0/results.csv` and `frontier.csv`, plus tokenizer artifacts under `tokenizers/vocabXXXX/`.

### E5 (end-to-end): CSIC 2010 HTTP

E5 trains the encoder end-to-end (no probe-only freezing) under a *token-fair* compute budget on a structured public HTTP dataset.
To keep the repo self-contained, the script does not download datasets automatically.

Place CSIC 2010 raw files under a directory such as `data/csic2010/`, e.g.:

```text
data/csic2010/
  normalTraffic*.txt
  anomalousTraffic*.txt
```

Then run:

```bash
python scripts/run_e5_csic_http.py --data-dir data/csic2010 --device cuda --seed 0
```

Outputs (tokenizers, logs, metrics) will be written under `results/paper/e5_csic_http/seed0/` by default.


## Plotting (frontier curves)

After running E4, generate paper-ready plots:

```bash
python scripts/plot_frontier.py --run_dir results/e4_frontier/seed0
```

All experiment outputs are written under the top-level `results/` folder.


## One-click paper runs

All scripts write outputs under `results/`.

```bash
python scripts/run_all.py --device cuda --seed 0 --plot
```

This will populate:
- `results/paper/e1_synth/seed0/...`
- `results/paper/e2_uci/seed0/...`
- `results/paper/e3_pareto/seed0/...`
- `results/paper/e4_frontier/seed0/...` (plus `frontier_*.pdf/png`)
