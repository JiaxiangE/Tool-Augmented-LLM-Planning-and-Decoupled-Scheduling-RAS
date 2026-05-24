"""
Learning-rate sensitivity curve for the multi-scenario fine-tune ablation.

Reads precomputed statistics from
  data/lr_sensitivity/lr_sensitivity_stats.json
and renders a 4-point curve (one line per scenario) with 95% CI error bars.

  x-axis: 4 learning-rate configurations (e.g. 3e-4, 6e-5, 1.2e-5, plateau)
  y-axis: paired delta (MS gap - SS gap) in percentage points
  lines:  1 km Shackleton, 5 km Shackleton

Output:
  figures/fig_lr_sensitivity.pdf
  figures/fig_lr_sensitivity.png (160 dpi)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_DIR = REPO_ROOT / "data" / "lr_sensitivity"
OUT_DIR = REPO_ROOT / "figures"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = json.load(open(STATS_DIR / "lr_sensitivity_stats.json", encoding="utf-8"))

    # x positions: keep declaration order from JSON (large -> small -> plateau)
    cfgs = stats["by_lr_config"]
    x = list(range(len(cfgs)))
    x_labels = [c["label"] for c in cfgs]

    scenarios = stats["scenarios"]
    colors = {"real_shackleton_1km": "#1f77b4", "real_shackleton_5km": "#d62728"}
    markers = {"real_shackleton_1km": "o", "real_shackleton_5km": "s"}
    pretty = {"real_shackleton_1km": "1km Shackleton",
              "real_shackleton_5km": "5km Shackleton"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for scen in scenarios:
        means, cis = [], []
        for c in cfgs:
            sc = c["by_scenario"][scen]
            means.append(sc["paired_delta_mean_pp"])
            ci = sc["paired_delta_ci_95_half_pp"]
            cis.append(ci if ci is not None and not math.isnan(ci) else 0.0)
        ax.errorbar(x, means, yerr=cis, fmt=markers[scen] + "-",
                    color=colors[scen], label=pretty[scen],
                    capsize=4, linewidth=2, markersize=8)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel(r"Paired $\Delta$ = HGT-SS gap - HGT-MS gap, pp")
    ax.set_xlabel("Fine-tune learning rate configuration")
    ax.set_title("Learning-rate sensitivity of MS fine-tune (5 seeds, 95% CI)\n"
                 "positive $\\Delta$ = MS improves over SS; negative = MS harms")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    pdf = OUT_DIR / "fig_lr_sensitivity.pdf"
    png = OUT_DIR / "fig_lr_sensitivity.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=160)
    print(f"[Fig lr] saved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
