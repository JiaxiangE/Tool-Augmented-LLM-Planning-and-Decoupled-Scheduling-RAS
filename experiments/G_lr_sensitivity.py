"""
Driver G — Learning-rate sensitivity curve plot.

Reads an ``lr_sensitivity_stats.json`` file (produced by the training pipeline)
and produces a 4-point curve (one line per scenario) with 95% CI error bars.

  x-axis: 4 lr configs (e.g. 3e-4, 6e-5, 1.2e-5, plateau)
  y-axis: paired Δ (HGT-MS − HGT-SS) in pp
  lines : one per scenario (e.g. 1km Shackleton, 5km Shackleton)

Output:
  results/lr_sensitivity/lr_sensitivity_curve.png
  results/lr_sensitivity/lr_sensitivity_curve.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATS_DIR = PROJECT_ROOT / "data" / "lr_sensitivity"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"


def main():
    ap = argparse.ArgumentParser(description="Plot lr sensitivity curve")
    ap.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR,
                    help="Directory containing lr_sensitivity_stats.json "
                         "(default: data/lr_sensitivity)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Where to write the curve PDF/PNG (default: results/)")
    args = ap.parse_args()

    stats = json.load(open(args.stats_dir / "lr_sensitivity_stats.json", encoding="utf-8"))

    # x positions: keep declaration order from JSON
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
        ax.errorbar(x, means, yerr=cis, fmt=markers.get(scen, "o") + "-",
                    color=colors.get(scen, "#444444"), label=pretty.get(scen, scen),
                    capsize=4, linewidth=2, markersize=8)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel(r"Paired $\Delta$ = HGT-SS gap − HGT-MS gap, pp")
    ax.set_xlabel("Fine-tune learning rate configuration")
    ax.set_title("Learning-rate sensitivity of MS fine-tune (5 seeds, 95% CI)\n"
                 r"positive $\Delta$ = MS improves over SS; negative = MS harms")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "G_lr_sensitivity_curve.png"
    pdf = args.output_dir / "G_lr_sensitivity_curve.pdf"
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    print(f"  -> {png}")
    print(f"  -> {pdf}")


if __name__ == "__main__":
    main()
