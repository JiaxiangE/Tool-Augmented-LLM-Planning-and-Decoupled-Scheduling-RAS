"""
Figure 4 — Shackleton 5 km x 5 km terrain (LOLA DEM).

A single-panel terrain figure intended as a journal-column figure (6.5 x 5.8 in).

Layers:
  - elevation heatmap (matplotlib ``terrain`` colormap)
  - 30 deg traversability boundary contour (red dashed)
  - PSR (permanently shadowed region) mask in semi-transparent gray
  - task sites (deduplicated; sampling sub-steps share the same x,y)
  - base / 15-agent start (gold star at origin)

Output:
  figures/fig4_shackleton_terrain.pdf
  figures/fig4_shackleton_terrain.png (300 dpi)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_dem, load_schedule

OUT_DIR = REPO_ROOT / "figures"


def draw_terrain(ax, fig, tg):
    dem = load_dem()
    elev = dem["elevation"]
    slope = dem["slope"]
    is_psr = dem["is_psr"]
    cell = dem["cell_size"]
    ny, nx = elev.shape
    extent_km = [0, nx * cell / 1000.0, 0, ny * cell / 1000.0]
    print(f"[Fig 4] DEM {nx}x{ny}, cell={cell} m, span={extent_km[1]:.1f} km")

    im = ax.imshow(elev, origin="lower", extent=extent_km, cmap="terrain",
                   aspect="equal")
    xs = np.linspace(extent_km[0], extent_km[1], nx)
    ys = np.linspace(extent_km[2], extent_km[3], ny)
    ax.contour(xs, ys, slope, levels=[30.0], colors="#c0392b",
               linewidths=1.4, linestyles="--")
    psr_mask = np.ma.masked_where(~is_psr, np.ones_like(elev))
    ax.imshow(psr_mask, origin="lower", extent=extent_km, cmap="gray_r",
              alpha=0.35, aspect="equal", vmin=0, vmax=1)

    # task locations — dedup (sampling sub-steps share x,y)
    tlocs_all = [(n.location[0] / 1000.0, n.location[1] / 1000.0)
                 for n in tg.nodes if n.location]
    unique_locs = sorted(set(tlocs_all))
    n_total = len(tlocs_all)
    n_sites = len(unique_locs)
    if unique_locs:
        tx, ty = zip(*unique_locs)
        ax.scatter(tx, ty, s=12, c="white", edgecolors="black",
                   linewidths=0.4, zorder=3)
    # base / 15-agent start — gold star
    ax.scatter([0], [0], marker="*", s=380, c="#f1c40f", edgecolors="black",
               linewidths=0.8, zorder=4)
    ax.set_xlabel("x (km)", fontsize=10)
    ax.set_ylabel("y (km)", fontsize=10)
    ax.set_title("Shackleton 5 km x 5 km terrain (LOLA DEM)", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("elevation (m)", fontsize=9)
    handles = [
        plt.Line2D([0], [0], color="#c0392b", ls="--", lw=1.4,
                   label="30 deg traversability boundary"),
        Patch(facecolor="gray", alpha=0.35, label="PSR (shadowed)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=6,
                   label=f"task sites (n={n_sites}; n={n_total} incl. sampling sub-steps)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#f1c40f",
                   markeredgecolor="black", markersize=15,
                   label="base / 15-agent start"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.92)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tg, _agents, _sched = load_schedule()

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    draw_terrain(ax, fig, tg)
    fig.tight_layout()

    pdf = OUT_DIR / "fig4_shackleton_terrain.pdf"
    png = OUT_DIR / "fig4_shackleton_terrain.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] saved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
