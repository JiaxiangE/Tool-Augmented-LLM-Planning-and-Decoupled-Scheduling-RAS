"""
Figure 2 — 10-scheduler makespan comparison across three representative scenarios.

Design:
  Semantic-grouped palette to make scheduler families visually distinguishable.
    noise (gray) | coupled (warm reds) | +CPM family (cool blue sequential)
    | exact (gold) | learned (purple)
  CP-SAT on the large_e scenario is N/A (exceeds the 30-task solver gate within
  the configured time budget) and is rendered as a hatched stub bar with a
  vertical "N/A" annotation.

Layout:
  - Single figure, 11.5 x 6.2 in
  - Legend placed below the plot (2 rows x 5 cols) to avoid clipping the title
  - Value labels above each bar; y-axis headroom = 18% of max real value

Data:
  The bar heights below are the published Table 4 makespan values (seconds)
  for the three representative scenarios, transcribed as Python constants.
  Running experiments/A_benchmark_comparison.py against the shipped data and
  checkpoints reproduces these numbers (within a few seconds for the
  deterministic methods; Random+CPM varies with --seed-offset). None entries
  denote N/A bars (e.g., CP-SAT exceeds its tractability gate on large_e).

Output:
  figures/fig2_benchmark_bar.pdf
  figures/fig2_benchmark_bar.png (300 dpi)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "figures"

# Semantic-grouped 10-scheduler palette
METHODS = [
    "Random+CPM",          # gray noise floor
    "Greedy (coupled)",    # coupled family
    "HEFT (native)",
    "Greedy+CPM",          # +CPM family (light -> dark blue sequential)
    "HEFT+CPM",
    "HEFT-Insertion+CPM",
    "Auction+CPM",
    "CP-SAT",              # gold optimal
    "GNN-MLP+CPM",         # learned family
    "GNN-HGT+CPM",
]
COLORS = [
    "#bdc3c7",                                  # Random — gray
    "#fadbd8", "#c0392b",                       # Coupled — light pink, deep red
    "#aed6f1", "#5dade2", "#2874a6", "#1b4f72", # +CPM — 4-step blue sequential
    "#f1c40f",                                  # CP-SAT — gold
    "#d2b4de", "#6c3483",                       # Learned — light + deep purple
]

# Hardcoded scenario makespan data (None = N/A)
SCENARIOS = [
    ("small_b_scan_and_sample", [1774, 2397, 2397, 1774, 1774, 1774, 1774, 1774, 1774, 1774]),
    ("medium_b_dual_parallel",  [4526, 6873, 6873, 4478, 4478, 4478, 4478, 3581, 4709, 4458]),
    ("large_e_mixed_dag",       [5683, 9216, 9763, 5355, 5355, 5355, 5025, None, 5505, 5085]),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_scen = len(SCENARIOS)
    n_meth = len(METHODS)
    x = np.arange(n_scen)
    width = 0.8 / n_meth

    # y-limit from real (non-None) values so the N/A stub has a stable anchor
    all_real = [v for _, row in SCENARIOS for v in row if v is not None]
    y_top = max(all_real) * 1.18  # headroom for value labels + legend

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_ylim(0, y_top)

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        xs = x + i * width - (n_meth - 1) * width / 2
        for j, (_, row) in enumerate(SCENARIOS):
            val = row[i]
            xpos = xs[j]
            if val is None:
                # N/A stub: short hatched bar + vertical "N/A" label
                stub_h = y_top * 0.035
                ax.bar(xpos, stub_h, width * 0.9,
                       color="white", edgecolor=color, hatch="///",
                       linewidth=0.6,
                       label=method if j == 0 else "_nolegend_")
                ax.text(xpos, stub_h + y_top * 0.005, "N/A",
                        ha="center", va="bottom", fontsize=6,
                        color=color, rotation=90, fontweight="bold")
            else:
                ax.bar(xpos, val, width * 0.9,
                       color=color, edgecolor="black", linewidth=0.3,
                       label=method if j == 0 else "_nolegend_")
                ax.text(xpos, val + y_top * 0.005, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=5.8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SCENARIOS], fontsize=9)
    ax.set_xlabel("Scenario", fontsize=10)
    ax.set_ylabel("Makespan (s)", fontsize=10)
    ax.set_title("Makespan comparison across three representative scenarios "
                 "(ten scheduling configurations)",
                 fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    # Legend below the plot: 10 entries -> 2 rows x 5 cols
    ax.legend(fontsize=8, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              frameon=False, columnspacing=1.4, handletextpad=0.5)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    pdf = OUT_DIR / "fig2_benchmark_bar.pdf"
    png = OUT_DIR / "fig2_benchmark_bar.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"[Fig 2] saved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
