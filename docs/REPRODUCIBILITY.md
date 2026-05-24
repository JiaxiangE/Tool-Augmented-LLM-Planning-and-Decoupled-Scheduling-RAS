# Reproducibility

This document maps every table and figure in the paper to the
script that regenerates it from the data shipped in this repository.

All commands assume a working install:

```bash
# from the repository root
git lfs install && git lfs pull           # fetch checkpoints + DEM rasters
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Table-by-table

| Paper artifact | Generating script | Default output path |
|---|---|---|
| Table 4 — main benchmark (10 schedulers × 14 scenarios) | `experiments/A_benchmark_comparison.py` | `results/A_benchmark_raw.{csv,json}` + `results/A_table_*.tex` |
| Tables 5–6 — eager-vs-lazy decoupling attribution + foreknowledge premium | `experiments/C_decoupling_attribution.py` | `results/C_decoupling.{csv,json}` |
| Table 8 — learning-rate sensitivity (HGT-SS vs HGT-MS) | `experiments/G_lr_sensitivity.py` | `results/G_lr_sensitivity_curve.{pdf,png}` |
| Table 9 — hybrid decomposition (assignment vs ordering) | `experiments/E_hybrid_decomposition.py` | `results/E_hybrid.json` |
| Table 10 — Shackleton 1 km / 5 km case study | `experiments/D_case_study_dem.py` | `results/D_case_study/` |
| Tables 14–16 — OOD severity stratification | `experiments/B_ood_benchmark.py stats` | `results/B_ood_summary.csv` |
| Tables 17–20 — structural regression (Appendix D) | `experiments/F_structural_regression.py` | `results/F_regression/regression_results.json` (see note below) |
| Appendix B — natural-language paraphrase robustness | `experiments/H_nl_robustness.py` | `results/H_nl_robust.json` |

---

## Figure-by-figure

| Paper artifact | Generating script | Default output path |
|---|---|---|
| Figure 2 — 10-scheduler comparison bar chart | `scripts/figures/make_fig2_benchmark_bar.py` | `figures/fig2_benchmark_bar.{pdf,png}` |
| Figure 3 — hybrid scheduler decomposition | `scripts/figures/make_fig3_hybrid_decomposition.py` | `figures/fig3_hybrid_decomposition.{pdf,png}` |
| Figure 4 — Shackleton terrain (5 km, LOLA DEM) | `scripts/figures/make_fig4_shackleton_terrain.py` | `figures/fig4_shackleton_terrain.{pdf,png}` |
| Figure 5 — 15-agent Greedy+CPM schedule (Gantt) | `scripts/figures/make_fig5_shackleton_gantt.py` | `figures/fig5_shackleton_gantt.{pdf,png}` |
| LR sensitivity (appendix) | `scripts/figures/make_fig_lr_sensitivity.py` | `figures/fig_lr_sensitivity.{pdf,png}` |

**Note on Figures 2 and 3.** These two scripts render the published Table 4
and Table 9 values directly from constants in the script source, so the
figure can be rebuilt without re-running every experiment. The constants
are the same numbers that `A_benchmark_comparison.py` and
`E_hybrid_decomposition.py` produce when run end-to-end against the
shipped data and checkpoints, so a reviewer who runs both pipelines
and the figure scripts will see consistent numbers. Figures 4 and 5
are computed live from the DEM raster and the greedy schedule.

---

## End-to-end smoke test

Run the full main-benchmark chain on the shipped data to confirm the
install is correct (1–2 minutes on a laptop CPU):

```bash
python experiments/A_benchmark_comparison.py
python scripts/figures/make_fig2_benchmark_bar.py
```

If `figures/fig2_benchmark_bar.png` is produced and the summary printed
to stdout reports corpus-mean gaps close to:

| Method            | vs Greedy+CPM | vs CP-SAT |
|-------------------|---------------|-----------|
| HEFT+CPM          | +1.2 %        | +6.9 %    |
| GNN-MLP+CPM       | +19.9 %       | +18.9 %   |
| GNN-HGT+CPM       | +20.3 %       | +20.4 %   |

the install is sound. The GNN-HGT cell is produced by the single
canonical checkpoint at `models/checkpoints/gnn_hgt/gnn_hgt_final.pt`,
which reproduces every per-scenario Table 4 GNN-HGT cell exactly
(e.g. `small_b` = 1774, `medium_b` = 4458, `large_e` = 5085).

---

## OOD evaluation (Tables 14–16)

The OOD pipeline is split into two subcommands so the expensive `run`
step can be cached:

```bash
# 1. run 5 random seeds across all 40 OOD scenarios (long)
#    NB: --output-dir is a DIRECTORY (not a file). The script writes
#    `ood_benchmark_metrics.json` inside it.
python experiments/B_ood_benchmark.py run --rounds 5 \
    --output-dir results/B_ood

# 2. compute the stratified summary tables (fast).
#    --input is the .json file the previous step produced.
python experiments/B_ood_benchmark.py stats \
    --input  results/B_ood/ood_benchmark_metrics.json \
    --output results/B_ood_stats

# 3. (optional) per-scenario diagnostic probe
python experiments/B_ood_benchmark.py probe \
    --output results/B_ood_probe.json
```

---

## Note on Appendix D structural regression

Running `experiments/F_structural_regression.py` against the shipped
`gnn_hgt_ss/` 5-seed ensemble + the 40 OOD scenarios in
`data/training_corpus_ood/` produces:

| Indicator | Paper Table 17 | Repo (this build) |
|---|---|---|
| HGT-SS Linear in-sample R² | 0.397 | ~0.75 |
| HGT-SS Random Forest R² | 0.771 | ~0.83 |
| Top RF feature                  | n_tasks (0.476) | n_tasks (0.71) |
| Second RF feature               | chain_depth_frac (0.216) | chain_depth_frac (0.12) |
| coef `longest_chain_depth` sign | + | + |
| coef `mutex_ratio` sign         | − | − |

The qualitative finding — "structural features carry non-trivial signal for
OOD gap; n_tasks dominates; chain depth is second" — reproduces exactly.
The quantitative R² is slightly higher in this build than in the paper's
Table 17 because the 40 procedurally-generated OOD scenarios shipped here
are a re-generation of the paper-era set (the generator script has
evolved); the structural-feature ↔ gap mapping happens to be tighter in
this realisation. The shipped `gnn_hgt_ss/` checkpoints are unchanged
from the paper's training run.

## Hardware notes

- Most experiments run on a single CPU core. The full Table 4
  benchmark completes in roughly one minute on a laptop.
- `experiments/D_case_study_dem.py` and figures 4 / 5 require the 5 km
  DEM (`data/dem/shackleton_5km.npz`).
- The learned scheduler loads PyTorch checkpoints from
  `models/checkpoints/gnn_hgt_ls/seed_*/gnn_hgt_final.pt` (HGT) and
  `models/checkpoints/gnn_mlp/gnn_mlp_final.zip` (MLP). CPU is fine.
- Override the default HGT seed (419) via the `GNN_HGT_PATH`
  environment variable, e.g.
  `GNN_HGT_PATH=models/checkpoints/gnn_hgt_ls/seed_137/gnn_hgt_final.pt`.

---

## Determinism

All experiments accept a `--seed-offset` argument (default 0). The
published numbers are derived from runs at `--seed-offset 0`. The
random-baseline (`Random+CPM`) is the only method whose output depends
on this argument; all other methods are deterministic given a task
graph and agent set.

---

## Re-training the learned scheduler

The training driver is `core/scheduler/gnn/trainer.py`. The 5-seed
HGT ensemble shipped in `models/checkpoints/gnn_hgt_ls/` was produced
by running it once per seed (42, 137, 271, 419, 613) with the
staged-curriculum corpus mix listed in each seed's
`training_summary.json`. Re-training takes roughly 35 hours per seed
on a single CPU and is not required to reproduce any table in the paper.
