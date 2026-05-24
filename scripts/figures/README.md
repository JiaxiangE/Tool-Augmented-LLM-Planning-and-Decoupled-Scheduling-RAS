# Figure-rendering scripts

Standalone scripts that build every figure shipped in the paper. Each script
writes both a PDF (vector) and a PNG (raster) into the top-level `figures/`
directory at the repository root.

All scripts assume the repository root is the current working directory when
invoked.

## Scripts

| Script | Paper figure | Output files |
| --- | --- | --- |
| `make_fig2_benchmark_bar.py` | Figure 2 — 10-scheduler makespan comparison | `figures/fig2_benchmark_bar.{pdf,png}` |
| `make_fig3_hybrid_decomposition.py` | Figure 3 — Hybrid mechanistic decomposition | `figures/fig3_hybrid_decomposition.{pdf,png}` |
| `make_fig4_shackleton_terrain.py` | Figure 4 — Shackleton 5 km terrain (LOLA DEM) | `figures/fig4_shackleton_terrain.{pdf,png}` |
| `make_fig5_shackleton_gantt.py` | Figure 5 — Greedy+CPM 15-agent Gantt | `figures/fig5_shackleton_gantt.{pdf,png}` |
| `make_fig_lr_sensitivity.py` | Supplementary — learning-rate sensitivity curve | `figures/fig_lr_sensitivity.{pdf,png}` |

`_common.py` is a tiny shared helper used by figures 4 and 5 (DEM loading and
the Greedy+CPM 15-agent schedule build). It is not itself a figure.

## How to run

From the repository root:

```bash
python scripts/figures/make_fig2_benchmark_bar.py
python scripts/figures/make_fig3_hybrid_decomposition.py
python scripts/figures/make_fig4_shackleton_terrain.py
python scripts/figures/make_fig5_shackleton_gantt.py
python scripts/figures/make_fig_lr_sensitivity.py
```

Each script creates `figures/` if it does not already exist.

## Data prerequisites

| Script | Required data |
| --- | --- |
| `make_fig2_benchmark_bar.py` | None (makespan values are inlined from the main benchmark table). |
| `make_fig3_hybrid_decomposition.py` | None (decomposition values are inlined from the hybrid ablation table). |
| `make_fig4_shackleton_terrain.py` | `data/dem/shackleton_5km.npz` (DEM bundle) and `data/dem/real_shackleton_5km.json` (15-agent scenario). |
| `make_fig5_shackleton_gantt.py` | Same as Figure 4. |
| `make_fig_lr_sensitivity.py` | `experiments/results/lr_sensitivity/lr_sensitivity_stats.json` (produced by the LR sensitivity experiment driver). |

Figures 4 and 5 import from `core.schema.taskgraph` and the experiment drivers
under `experiments/`; both must be importable from the repository root (which
is the default when running with `cwd = <repo root>`).
