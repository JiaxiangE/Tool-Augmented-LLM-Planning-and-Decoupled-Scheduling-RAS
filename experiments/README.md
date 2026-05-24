# Experiments

One driver script per paper artifact. All drivers accept `--help`.

Default output goes to `./results/` (created on first run). Override with
`--output-dir` / `--output`.

---

## A — Main benchmark (Table 4)

Ten scheduling configurations on the 14 in-distribution scenarios.

```bash
python experiments/A_benchmark_comparison.py --seed-offset 0 --travel-model eager
```

Outputs: `results/A_benchmark_raw.{csv,json}`, `results/A_summary_stats.json`,
`results/A_table_{comparison,gaps}.tex`.

Wall-clock: ~1 minute on a laptop CPU.

---

## B — OOD generalization (Tables 14–16)

Three subcommands sharing a single script:

```bash
# full sweep: 40 OOD scenarios x 5 seeds x N rounds
# Important: --output-dir is a DIRECTORY; the script creates
# `ood_benchmark_metrics.json` inside it.
python experiments/B_ood_benchmark.py run \
    --output-dir results/B_ood --rounds 5

# aggregate raw JSON into the paper's stratified tables
python experiments/B_ood_benchmark.py stats \
    --input  results/B_ood/ood_benchmark_metrics.json \
    --output results/B_ood_stats

# quick diagnostic on the 10-scenario probe set
python experiments/B_ood_benchmark.py probe \
    --probe-set data/training_corpus_ood_probe_10.json \
    --output    results/B_ood_probe.json

# Then feed the raw .json into F for structural regression:
python experiments/F_structural_regression.py \
    --ood-metrics results/B_ood/ood_benchmark_metrics.json \
    --output      results/F_regression
```

Wall-clock for full `run`: 10–30 minutes depending on `--rounds`.

---

## C — Decoupling attribution (Tables 5–6)

```bash
python experiments/C_decoupling_attribution.py --seed-offset 0 --travel-model eager
```

Outputs: `results/C_decoupling.{csv,json}`, `results/C_table_attribution.tex`.

---

## D — Shackleton case study (Table 10)

```bash
# 5 km mission (15 agents + recharge logistics)
python experiments/D_case_study_dem.py \
    --scenario data/dem/real_shackleton_5km.json \
    --dem      data/dem/shackleton_5km.npz \
    --output-dir results/D_5km \
    --rounds 5 --enable-recharge

# 1 km mission (8 agents)
python experiments/D_case_study_dem.py \
    --scenario data/dem/real_shackleton_1km.json \
    --dem      data/dem/shackleton_1km.npz \
    --output-dir results/D_1km \
    --rounds 5
```

Outputs: `results/D_{1,5}km/D_dem_*_metrics.json` plus per-method Gantt PNGs.

---

## E — Hybrid scheduler decomposition (Table 9)

```bash
python experiments/E_hybrid_decomposition.py \
    --checkpoint-root models/checkpoints/gnn_hgt_ls \
    --holdout-dir     data/test_corpus_holdout \
    --output          results/E_hybrid
```

Outputs: `results/E_hybrid/hybrid_decomposition.json`.

---

## F — Structural regression on OOD gap (Appendix D)

Consumes per-cell data emitted by `B_ood_benchmark.py run`:

```bash
python experiments/F_structural_regression.py \
    --ood-metrics results/B_ood_raw.json \
    --output      results/F_regression
```

Outputs: `results/F_regression/regression_results.json`,
`results/F_regression/regression_report.md`.

---

## G — Learning-rate sensitivity (Table 8)

```bash
python experiments/G_lr_sensitivity.py --stats-dir data/lr_sensitivity
```

Outputs: `results/G_lr_sensitivity_curve.{pdf,png}`.

---

## H — NL paraphrase robustness (Appendix B)

Requires the `DASHSCOPE_API_KEY` environment variable (or equivalent
provider key — see `core/llm/llm_backend.py`).

```bash
python experiments/H_nl_robustness.py \
    --dataset       data/nl_robustness/paraphrase_dataset.json \
    --n-paraphrases 5 \
    --output-dir    results/H_nl_robustness
```

Outputs: `results/H_nl_robustness/nl_robustness_{raw,summary}.json`.

---

## Required input data

| Path | Used by | Description |
|---|---|---|
| `data/training_corpus/*.json` (14 files) | A, C | In-distribution scenarios |
| `data/training_corpus_ood/*.json` (40 files) | B (run/stats), F | OOD scenarios |
| `data/training_corpus_ood_probe_10.json` | B (probe) | 10-scenario probe set |
| `data/test_corpus_holdout/*.json` (10 files) | E | Held-out scenarios |
| `data/dem/shackleton_{1,5}km.{npz,json}` | D | DEM rasters + scenarios |
| `data/lr_sensitivity/lr_sensitivity_stats.json` | G | Pre-computed LR sweep |
| `data/nl_robustness/paraphrase_dataset.json` | H | 70 NL paraphrases |
| `models/checkpoints/gnn_hgt_ls/seed_*/gnn_hgt_final.pt` | A, B, C, D, E | HGT 5-seed |
| `models/checkpoints/gnn_mlp/gnn_mlp_final.zip` | A, D | MLP single checkpoint |
