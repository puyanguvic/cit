# Paper artifacts (auto-generated)

## Main paper (3 experiments)
- E1 (synthetic): `main/tables/e1_synth.tex`
- E2 (CSIC HTTP): `main/tables/e2_csic_token_fair.tex`; optional convergence table `main/tables/e2_csic_convergence.tex`; scaling/frontier figures in `main/figures/e2_csic_*.pdf`
- E3 (Pareto slice): `main/tables/e3_pareto.tex` + `main/figures/e3_pareto_triptych.pdf`; optional scaling figure `main/figures/e3_budget_scaling.pdf`

## Appendix
- Tokenizer scan (E0): `appendix/tables/e0_tokenizer_playground.tex` (if present)
- E2 step-fair variant: `appendix/tables/e2_csic_step_fair.tex` + `appendix/figures/e2_csic_step_fair_frontier_acc_vs_len.pdf` (if present)
- UCI benchmarks: `appendix/tables/uci.tex` (if present)
- Vocab frontier sweep: `appendix/tables/frontier.tex` + `appendix/figures/frontier_*.pdf` (if present)

## Notes
- Tables require `\usepackage{booktabs}`.
- If multiple seeds are present, tables show mean ± std; otherwise they show the single-seed value.
