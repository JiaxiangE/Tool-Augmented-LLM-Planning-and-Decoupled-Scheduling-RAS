"""
Figure 5 — Greedy+CPM 15-agent Gantt for the Shackleton 5 km scenario.

Banner-format single-panel figure (11 x 4.5 in). Rows = agents, grouped by
agent type (ROVER, RELAY, SAMPLER, ANALYZER). Bar color encodes agent type.

Output:
  figures/fig5_shackleton_gantt.pdf
  figures/fig5_shackleton_gantt.png (300 dpi)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_schedule, agent_type_of, AGENT_TYPE_COLOR

OUT_DIR = REPO_ROOT / "figures"


def draw_gantt(ax, agents, sched):
    items = sched.items
    makespan = max(it.end_s for it in items)
    print(f"[Fig 5] Gantt: {len(items)} tasks / {len(agents)} agents / "
          f"makespan {makespan:.0f}s")

    type_order = {"ROVER": 0, "RELAY": 1, "SAMPLER": 2, "ANALYZER": 3}
    agent_ids = sorted([a.id for a in agents],
                       key=lambda a: (type_order.get(agent_type_of(a), 9), a))
    row_of = {a: i for i, a in enumerate(agent_ids)}

    for it in items:
        r = row_of.get(it.agent)
        if r is None:
            continue
        atype = agent_type_of(it.agent)
        ax.barh(r, it.end_s - it.start_s, left=it.start_s, height=0.72,
                color=AGENT_TYPE_COLOR.get(atype, "#95a5a6"),
                edgecolor="black", linewidth=0.25)
    ax.set_yticks(range(len(agent_ids)))
    ax.set_yticklabels(agent_ids, fontsize=7.5)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_title(f"Greedy+CPM 15-agent schedule (makespan {makespan:.0f} s)",
                 fontsize=11)
    ax.set_xlim(0, makespan * 1.02)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    legend_handles = [Patch(facecolor=c, edgecolor="black", label=t)
                      for t, c in AGENT_TYPE_COLOR.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
              title="agent type", title_fontsize=9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _tg, agents, sched = load_schedule()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    draw_gantt(ax, agents, sched)
    fig.tight_layout()

    pdf = OUT_DIR / "fig5_shackleton_gantt.pdf"
    png = OUT_DIR / "fig5_shackleton_gantt.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] saved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
