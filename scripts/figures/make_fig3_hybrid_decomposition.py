"""
Figure 3 — Hybrid mechanistic decomposition.

Panel (a) decomposes the makespan gap (vs. Greedy+CPM) attributable to a
learned policy's assignment choices vs. its ordering choices, by progressively
delegating responsibility back to the classical Greedy rule.

Panel (b) reports the per-topology gap of the hybrid (ordering-only) variant.

Palette (Blues monochrome, dark -> light):
    Conveys progressive constraint relaxation across the four left-panel bars.
    Two pure-white bars (0.0% gap) mark the zero baselines (Portfolio and the
    Greedy+CPM reference). The right panel uses the lightest blue
    (#aed6f1) for an independent semantic dimension (topology class).

      bar 1 (full learned output)  -> #2874a6
      bar 2 (ordering-only hybrid) -> #5dade2
      bar 3 (portfolio per-cell min) -> #ffffff
      bar 4 (Greedy+CPM reference)   -> #ffffff

Data:
  The bar heights below are the published Table 9 values, transcribed as
  Python constants. Running experiments/E_hybrid_decomposition.py against
  the shipped 5-seed HGT ensemble and the data/test_corpus_holdout/ split
  reproduces these numbers (mean over 5 seeds x 10 holdout scenarios = 50
  cells).

Output:
  figures/fig3_hybrid_decomposition.pdf
  figures/fig3_hybrid_decomposition.png (200 dpi)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "figures"

# Variant labels — no project codenames in any bar / annotation
VARIANTS = [
    ("GNN\nfull output",         169.1),
    ("Hybrid\n(ordering only)",   23.0),
    ("Portfolio\n(per-cell min)",  0.0),
    ("Greedy+CPM\n(reference)",    0.0),
]

# Blues monochrome (dark -> light) for left panel; pure white for zero bars
COLORS = ["#2874a6", "#5dade2", "#ffffff", "#ffffff"]

# Per-topology hybrid (ordering-only) gap
TOPO = [("deep_chain",      6.3),
        ("mutex_dense",    17.8),
        ("sampling_burst", 20.6),
        ("mixed",          33.7),
        ("parallel_pure",  36.8)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): variant decomposition
    names = [v[0] for v in VARIANTS]
    vals = [v[1] for v in VARIANTS]
    bars = axL.bar(names, vals, color=COLORS, width=0.62,
                   edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        axL.text(b.get_x() + b.get_width() / 2, v + 4, f"+{v:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    axL.set_ylabel("Makespan gap vs. Greedy+CPM (%)", fontsize=10)
    axL.set_title("(a) Failure decomposition: assignment vs. ordering",
                  fontsize=10)
    axL.set_ylim(0, 195)
    axL.axhline(0, color="black", linewidth=0.8)
    axL.grid(axis="y", linestyle=":", alpha=0.5)
    axL.tick_params(axis="x", labelsize=8.5)

    # Annotation: delegate assignment to Greedy rule -> 86% of gap closes
    axL.annotate("", xy=(1, 23.0), xytext=(0, 169.1),
                 arrowprops=dict(arrowstyle="->", color="#2874a6", lw=1.4))
    axL.text(0.5, 110,
             "delegate assignment\nto Greedy rule:\n-146 pp (86% of gap)",
             ha="center", va="center", fontsize=8.2, color="#2874a6",
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff",
                       ec="#2874a6", lw=0.8))
    axL.text(1.0, 56,
             "residual +23.0%\n= ordering gap (14%)",
             ha="center", va="center", fontsize=8.2, color="#2874a6")

    # Panel (b): per-topology hybrid gap
    tnames = [t[0] for t in TOPO]
    tvals = [t[1] for t in TOPO]
    barsR = axR.bar(tnames, tvals, color="#aed6f1", width=0.6,
                    edgecolor="black", linewidth=0.6)
    for b, v in zip(barsR, tvals):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.8, f"+{v:.1f}%",
                 ha="center", va="bottom", fontsize=8.5)
    axR.set_ylabel("Hybrid makespan gap vs. Greedy+CPM (%)", fontsize=10)
    axR.set_title("(b) Hybrid gap by topology class", fontsize=10)
    axR.set_ylim(0, 44)
    axR.axhline(0, color="black", linewidth=0.8)
    axR.grid(axis="y", linestyle=":", alpha=0.5)
    axR.tick_params(axis="x", labelrotation=20, labelsize=8.5)

    fig.tight_layout()
    pdf = OUT_DIR / "fig3_hybrid_decomposition.pdf"
    png = OUT_DIR / "fig3_hybrid_decomposition.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"[Fig 3] saved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
