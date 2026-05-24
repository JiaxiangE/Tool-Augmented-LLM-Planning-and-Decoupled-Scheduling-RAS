"""
Driver D — Real DEM (Shackleton crater rim) case study.

Uses real LOLA DEM terrain data to produce publication figures for a
Shackleton crater rim exploration scenario, plus an optional multi-scale
CLI mode (1km / 5km) honoring per-scenario agent fleets and an optional
recharge cycle.

Legacy single-scenario mode outputs (default):
  - D_terrain_heatmap.png      Elevation colormap + slope overlay + task paths
  - D_gantt_greedy.png         Greedy+CPM Gantt chart
  - D_gantt_heft.png           HEFT (native) Gantt chart
  - D_gantt_gnn.png            GNN-HGT+CPM Gantt chart (if model available)
  - D_dem_metrics.json         Raw metrics for all schedulers
  - D_table_dem.tex            LaTeX comparison table

Multi-scale CLI mode (use --scenario / --dem):
  - D_dem_{200m,1km,5km}_metrics.json   per-cell raw + summary

No LLM required.

Usage:
    # Legacy 200m Shackleton case study
    python experiments/D_case_study_dem.py

    # Multi-scale CLI mode
    python experiments/D_case_study_dem.py \
        --scenario data/dem/real_shackleton_5km.json \
        --dem data/dem/shackleton_terrain_5km.npz \
        --output-dir results/dem_5km --rounds 5 --enable-recharge
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (
    assignments_to_schedule, auction_assignments, cpsat_schedule,
    ensure_results_dir,
    gnn_custom_assignments, gnn_custom_available, gnn_mlp_assignments,
    gnn_mlp_available, greedy_assignments, greedy_coupled_schedule,
    heft_assignments, heft_schedule, latex_booktabs, random_assignments,
    save_json, save_tex, setup_matplotlib,
)
from core.schema.agent_types import AgentType
from core.schema.environment import AgentState
from core.schema.taskgraph import TaskGraph
from core.simulator.lunar_sim import LunarSimulator, SimConfig, SimResult
from core.simulator.terrain import TerrainGrid

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEM_NPZ = PROJECT_ROOT / "data" / "dem" / "shackleton_terrain.npz"
SCENARIO_JSON = PROJECT_ROOT / "data" / "dem" / "real_shackleton_rim.json"


# ── Custom agent factory for Shackleton (includes ANALYZER) ─────────────────
def make_shackleton_agents() -> list[AgentState]:
    """Create agents matching the Shackleton scenario metadata."""
    return [
        AgentState(id="rover_1",    agent_type=AgentType.ROVER,    energy_wh=5000, initial_energy_wh=5000),
        AgentState(id="rover_2",    agent_type=AgentType.ROVER,    energy_wh=5000, initial_energy_wh=5000),
        AgentState(id="relay_1",    agent_type=AgentType.RELAY,    energy_wh=5000, initial_energy_wh=5000),
        AgentState(id="sampler_1",  agent_type=AgentType.SAMPLER,  energy_wh=5000, initial_energy_wh=5000),
        AgentState(id="analyzer_1", agent_type=AgentType.ANALYZER, energy_wh=5000, initial_energy_wh=5000),
    ]


def simulate_dem(tg: TaskGraph, sched, agents: list[AgentState]) -> SimResult:
    """Simulate using real DEM terrain if available, else a synthetic fallback."""
    if DEM_NPZ.exists():
        terrain = TerrainGrid.from_npz(DEM_NPZ)
    else:
        print("  WARNING: DEM NPZ not found, using synthetic terrain")
        terrain = TerrainGrid(width=20, height=20, cell_size=10.0)
    return LunarSimulator(terrain, agents, SimConfig()).evaluate(tg, sched)


# ── Terrain Heatmap ──────────────────────────────────────────────────────────

def draw_terrain_heatmap(tg: TaskGraph, sched, save_path: Path):
    """Terrain heatmap with elevation colormap, slope contours, task locations,
    agent paths, and PSR/obstacle overlays."""
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize

    if not DEM_NPZ.exists():
        print("  SKIP terrain heatmap: DEM NPZ not found")
        return

    data = np.load(str(DEM_NPZ))
    elevation = data["elevation"]
    slope = data["slope"]
    is_psr = data["is_psr"]
    is_obstacle = data["is_obstacle"]
    cell_size = float(data["cell_size"]) if "cell_size" in data else 10.0

    h, w = elevation.shape
    extent = [0, w * cell_size, h * cell_size, 0]  # [left, right, bottom, top]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Elevation colormap
    im = ax.imshow(elevation, extent=extent, cmap="terrain", origin="upper",
                   aspect="equal")
    cbar = fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.8)

    # Slope contour overlay
    y_coords = np.arange(h) * cell_size + cell_size / 2
    x_coords = np.arange(w) * cell_size + cell_size / 2
    X, Y = np.meshgrid(x_coords, y_coords)
    cs = ax.contour(X, Y, slope, levels=[10, 20, 30], colors=["#00ff00", "#ffff00", "#ff0000"],
                    linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt="%.0f°", fontsize=6)

    # PSR overlay (semi-transparent blue)
    psr_overlay = np.ma.masked_where(~is_psr, np.ones_like(elevation))
    ax.imshow(psr_overlay, extent=extent, cmap="Blues", alpha=0.25, origin="upper")

    # Obstacle overlay (hatched red)
    obs_overlay = np.ma.masked_where(~is_obstacle, np.ones_like(elevation))
    ax.imshow(obs_overlay, extent=extent, cmap="Reds", alpha=0.4, origin="upper")

    # ── Base station ──────────────────────────────────────────────────────
    BASE_POS = (0.0, 0.0)  # all agents depart from here
    ax.scatter(*BASE_POS, marker="*", c="yellow", s=250,
               edgecolors="black", linewidths=1.0, zorder=10)
    ax.annotate("Base", BASE_POS, textcoords="offset points",
                xytext=(8, -10), fontsize=7, fontweight="bold", color="yellow",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

    # ── Task location markers ─────────────────────────────────────────────
    task_map = {n.id: n for n in tg.nodes}
    type_markers = {
        "DeployRelay": ("^", "#f28e2b", 100),
        "ExploreCell": ("o", "#4e79a7", 80),
        "TakeSample": ("s", "#b07aa1", 90),
        "AnalyzeSample": ("D", "#59a14f", 90),
    }

    for node in tg.nodes:
        if node.location:
            x, y = node.location[0], node.location[1]
            marker, color, size = type_markers.get(node.op_type, ("o", "gray", 60))
            ax.scatter(x, y, marker=marker, c=color, s=size,
                       edgecolors="black", linewidths=0.8, zorder=5)
            short = node.id.replace("sk_", "").replace("_", "\n")
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(8, 4), fontsize=5, fontweight="bold",
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # ── Agent paths (base → task1 → task2 → …) with arrows + step numbers ─
    agent_styles = {
        "rover_1":    {"color": "#1f77b4", "ls": "-",  "label": "rover_1"},
        "rover_2":    {"color": "#ff7f0e", "ls": "--", "label": "rover_2"},
        "relay_1":    {"color": "#2ca02c", "ls": "-.", "label": "relay_1"},
        "sampler_1":  {"color": "#d62728", "ls": ":",  "label": "sampler_1"},
        "analyzer_1": {"color": "#9467bd", "ls": "-",  "label": "analyzer_1"},
    }
    ARROW_STYLE = "Simple,tail_width=1.0,head_width=5,head_length=4"

    if sched:
        from collections import defaultdict
        from matplotlib.patches import FancyArrowPatch

        agent_tasks = defaultdict(list)
        for item in sorted(sched.items, key=lambda x: x.start_s):
            agent_tasks[item.agent].append(item)

        for agent_id, items in agent_tasks.items():
            style = agent_styles.get(agent_id, {"color": "gray", "ls": "-", "label": agent_id})
            color, ls = style["color"], style["ls"]

            # Build waypoint list: base → first task loc → second task loc → …
            waypoints = [BASE_POS]
            for item in items:
                node = task_map.get(item.node_id)
                if node and node.location:
                    waypoints.append((node.location[0], node.location[1]))

            # Draw arrows between consecutive waypoints
            for i in range(len(waypoints) - 1):
                x0, y0 = waypoints[i]
                x1, y1 = waypoints[i + 1]
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle=ARROW_STYLE, color=color,
                    linestyle=ls, linewidth=1.2, alpha=0.75,
                    zorder=4, mutation_scale=1,
                )
                ax.add_patch(arrow)
                # Step number at midpoint
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mx, my, str(i + 1), fontsize=5, fontweight="bold",
                        color=color, ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white",
                                  ec=color, lw=0.6, alpha=0.85),
                        zorder=6)

            # Agent name label near first task (not at base, to avoid crowding)
            if len(waypoints) >= 2:
                lx, ly = waypoints[1]
                ax.annotate(style["label"], (lx, ly), textcoords="offset points",
                            xytext=(-8, -12), fontsize=5,
                            color=color, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec=color, lw=0.4, alpha=0.7))

    # ── Communication range ───────────────────────────────────────────────
    # comm_range = 500 m, map = 200 × 200 m → circle covers entire map.
    # Draw the true 500 m arc (clipped by axes) so the reader sees the scale.
    COMM_RANGE_M = 500.0
    map_w, map_h = w * cell_size, h * cell_size
    for node in tg.nodes:
        if node.op_type == "DeployRelay" and node.location:
            rx, ry = node.location[0], node.location[1]
            circle = plt.Circle((rx, ry), COMM_RANGE_M,
                                fill=False, linestyle="--", color="#f28e2b",
                                linewidth=1.2, alpha=0.7, clip_on=True)
            ax.add_patch(circle)
            # Annotate with distance to farthest task
            max_dist = max(
                (np.sqrt((n.location[0] - rx)**2 + (n.location[1] - ry)**2)
                 for n in tg.nodes if n.location and n.id != node.id),
                default=0,
            )
            ax.annotate(f"comm range={COMM_RANGE_M:.0f}m\n(covers full map; "
                        f"max task dist={max_dist:.0f}m)",
                        (rx, ry), textcoords="offset points",
                        xytext=(10, -20), fontsize=5, color="#f28e2b",
                        fontstyle="italic",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ── Legend ────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([], [], marker="*", color="yellow", markersize=10,
                      markeredgecolor="black", linestyle="None", label="Base Station"),
        mpatches.Patch(facecolor="#4e79a7", label="Explore"),
        mpatches.Patch(facecolor="#f28e2b", label="Deploy Relay"),
        mpatches.Patch(facecolor="#b07aa1", label="Take Sample"),
        mpatches.Patch(facecolor="#59a14f", label="Analyze"),
        mpatches.Patch(facecolor="blue", alpha=0.25, label="PSR Region"),
        mpatches.Patch(facecolor="red", alpha=0.4, label="Obstacle (>30°)"),
    ]
    # Agent path legend entries
    for agent_id, style in agent_styles.items():
        legend_elements.append(
            mlines.Line2D([], [], color=style["color"], linestyle=style["ls"],
                          linewidth=1.5, label=style["label"]))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=6,
              framealpha=0.9, ncol=2)

    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)
    ax.set_title("Shackleton Crater Rim — Real LOLA DEM Terrain\n"
                 "(20×20 pixels, 10m/px, 200m×200m region)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, w * cell_size)
    ax.set_ylim(h * cell_size, 0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


def _draw_terrain_on_ax(ax, fig, tg, sched, elevation, slope, is_psr, is_obstacle,
                        cell_size, title, show_cbar=False):
    """Draw the terrain heatmap with task paths on a given axes (shared helper)."""
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    from matplotlib.patches import FancyArrowPatch
    from collections import defaultdict

    h, w = elevation.shape
    extent = [0, w * cell_size, h * cell_size, 0]

    # Elevation colormap
    im = ax.imshow(elevation, extent=extent, cmap="terrain", origin="upper", aspect="equal")
    if show_cbar:
        fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.75, pad=0.02)

    # Slope contour overlay
    y_coords = np.arange(h) * cell_size + cell_size / 2
    x_coords = np.arange(w) * cell_size + cell_size / 2
    X, Y = np.meshgrid(x_coords, y_coords)
    cs = ax.contour(X, Y, slope, levels=[10, 20, 30],
                    colors=["#00ff00", "#ffff00", "#ff0000"],
                    linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fmt="%.0f°", fontsize=5)

    # PSR overlay
    psr_overlay = np.ma.masked_where(~is_psr, np.ones_like(elevation))
    ax.imshow(psr_overlay, extent=extent, cmap="Blues", alpha=0.25, origin="upper")

    # Obstacle overlay
    obs_overlay = np.ma.masked_where(~is_obstacle, np.ones_like(elevation))
    ax.imshow(obs_overlay, extent=extent, cmap="Reds", alpha=0.4, origin="upper")

    # Base station
    BASE_POS = (0.0, 0.0)
    ax.scatter(*BASE_POS, marker="*", c="yellow", s=200,
               edgecolors="black", linewidths=1.0, zorder=10)
    ax.annotate("Base", BASE_POS, textcoords="offset points",
                xytext=(8, -10), fontsize=6, fontweight="bold", color="yellow",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

    # Task location markers
    task_map = {n.id: n for n in tg.nodes}
    type_markers = {
        "DeployRelay": ("^", "#f28e2b", 80),
        "ExploreCell": ("o", "#4e79a7", 60),
        "TakeSample": ("s", "#b07aa1", 70),
        "AnalyzeSample": ("D", "#59a14f", 70),
    }
    for node in tg.nodes:
        if node.location:
            x, y = node.location[0], node.location[1]
            marker, color, size = type_markers.get(node.op_type, ("o", "gray", 50))
            ax.scatter(x, y, marker=marker, c=color, s=size,
                       edgecolors="black", linewidths=0.8, zorder=5)
            short = node.id.replace("sk_", "").replace("_", "\n")
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(8, 4), fontsize=4.5, fontweight="bold",
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Agent paths
    agent_styles = {
        "rover_1":    {"color": "#1f77b4", "ls": "-",  "label": "rover_1"},
        "rover_2":    {"color": "#ff7f0e", "ls": "--", "label": "rover_2"},
        "relay_1":    {"color": "#2ca02c", "ls": "-.", "label": "relay_1"},
        "sampler_1":  {"color": "#d62728", "ls": ":",  "label": "sampler_1"},
        "analyzer_1": {"color": "#9467bd", "ls": "-",  "label": "analyzer_1"},
    }
    ARROW_STYLE = "Simple,tail_width=1.0,head_width=5,head_length=4"

    if sched:
        agent_tasks = defaultdict(list)
        for item in sorted(sched.items, key=lambda x: x.start_s):
            agent_tasks[item.agent].append(item)

        for agent_id, items in agent_tasks.items():
            style = agent_styles.get(agent_id, {"color": "gray", "ls": "-", "label": agent_id})
            color, ls = style["color"], style["ls"]

            waypoints = [BASE_POS]
            for item in items:
                node = task_map.get(item.node_id)
                if node and node.location:
                    waypoints.append((node.location[0], node.location[1]))

            for i in range(len(waypoints) - 1):
                x0, y0 = waypoints[i]
                x1, y1 = waypoints[i + 1]
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle=ARROW_STYLE, color=color,
                    linestyle=ls, linewidth=1.2, alpha=0.75,
                    zorder=4, mutation_scale=1,
                )
                ax.add_patch(arrow)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mx, my, str(i + 1), fontsize=4.5, fontweight="bold",
                        color=color, ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white",
                                  ec=color, lw=0.6, alpha=0.85),
                        zorder=6)

            if len(waypoints) >= 2:
                lx, ly = waypoints[1]
                ax.annotate(style["label"], (lx, ly), textcoords="offset points",
                            xytext=(-8, -12), fontsize=4.5,
                            color=color, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec=color, lw=0.4, alpha=0.7))

    # Communication range circles
    COMM_RANGE_M = 500.0
    import matplotlib.pyplot as plt
    for node in tg.nodes:
        if node.op_type == "DeployRelay" and node.location:
            rx, ry = node.location[0], node.location[1]
            circle = plt.Circle((rx, ry), COMM_RANGE_M,
                                fill=False, linestyle="--", color="#f28e2b",
                                linewidth=1.0, alpha=0.6, clip_on=True)
            ax.add_patch(circle)

    # Legend
    legend_elements = [
        mlines.Line2D([], [], marker="*", color="yellow", markersize=8,
                      markeredgecolor="black", linestyle="None", label="Base"),
        mpatches.Patch(facecolor="#4e79a7", label="Explore"),
        mpatches.Patch(facecolor="#f28e2b", label="Relay"),
        mpatches.Patch(facecolor="#b07aa1", label="Sample"),
        mpatches.Patch(facecolor="#59a14f", label="Analyze"),
        mpatches.Patch(facecolor="blue", alpha=0.25, label="PSR"),
        mpatches.Patch(facecolor="red", alpha=0.4, label="Obstacle"),
    ]
    for agent_id, style in agent_styles.items():
        legend_elements.append(
            mlines.Line2D([], [], color=style["color"], linestyle=style["ls"],
                          linewidth=1.5, label=style["label"]))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=5,
              framealpha=0.9, ncol=2)

    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlim(0, w * cell_size)
    ax.set_ylim(h * cell_size, 0)


def draw_terrain_heatmap_comparison(tg: TaskGraph, sched_left, sched_right,
                                    save_path: Path,
                                    title_left: str = "GNN-HGT+CPM",
                                    title_right: str = "CP-SAT Optimal"):
    """Side-by-side terrain heatmap: GNN (left) vs CP-SAT (right)."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    if not DEM_NPZ.exists():
        print("  SKIP terrain heatmap comparison: DEM NPZ not found")
        return

    data = np.load(str(DEM_NPZ))
    elevation = data["elevation"]
    slope = data["slope"]
    is_psr = data["is_psr"]
    is_obstacle = data["is_obstacle"]
    cell_size = float(data["cell_size"]) if "cell_size" in data else 10.0

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 8))

    _draw_terrain_on_ax(ax_l, fig, tg, sched_left, elevation, slope,
                        is_psr, is_obstacle, cell_size, title_left)
    _draw_terrain_on_ax(ax_r, fig, tg, sched_right, elevation, slope,
                        is_psr, is_obstacle, cell_size, title_right,
                        show_cbar=True)

    fig.suptitle("Shackleton Crater Rim — Real LOLA DEM Terrain  "
                 "(20×20 px, 10 m/px, 200 m × 200 m)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


# ── Gantt Chart ──────────────────────────────────────────────────────────────

OP_COLORS = {
    "ExploreCell": "#4e79a7",
    "DeployRelay": "#f28e2b",
    "TakeSample": "#b07aa1",
    "AnalyzeSample": "#59a14f",
}
DEFAULT_COLOR = "#aec7e8"


def draw_gantt(sched, tg: TaskGraph, save_path: Path, title: str = ""):
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    task_map = {n.id: n for n in tg.nodes}
    agent_ids = sorted(set(item.agent for item in sched.items))
    agent_y = {a: i for i, a in enumerate(agent_ids)}

    fig, ax = plt.subplots(figsize=(12, max(3, len(agent_ids) * 0.8 + 1)))

    used_ops = set()
    for item in sched.items:
        y = agent_y[item.agent]
        node = task_map.get(item.node_id)
        op = node.op_type if node else "Unknown"
        color = OP_COLORS.get(op, DEFAULT_COLOR)
        used_ops.add(op)

        duration = item.end_s - item.start_s
        ax.barh(y, duration, left=item.start_s, height=0.6,
                color=color, edgecolor="black", linewidth=0.5, alpha=0.85)

        if duration > sched.makespan_s * 0.03:
            short_id = item.node_id.replace("sk_", "")
            if len(short_id) > 10:
                short_id = short_id[:10]
            ax.text(item.start_s + duration / 2, y, short_id,
                    ha="center", va="center", fontsize=5, fontweight="bold")

    ax.set_yticks(range(len(agent_ids)))
    ax.set_yticklabels(agent_ids, fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_xlim(0, sched.makespan_s * 1.05)
    ax.invert_yaxis()

    legend_patches = [mpatches.Patch(color=OP_COLORS.get(op, DEFAULT_COLOR), label=op)
                      for op in sorted(used_ops)]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=6, framealpha=0.9)

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


# ── Metric Comparison Bar Chart ──────────────────────────────────────────────

def draw_metric_bars(metrics: dict, save_path: Path):
    """Grouped bar chart: makespan, violations, completion for each scheduler."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    methods = []
    makespans = []
    completions = []
    violations = []

    for key, label in [
        ("random", "Random+CPM"),
        ("greedy_coupled", "Greedy"),
        ("greedy_cpm", "Greedy+CPM"),
        ("heft_native", "HEFT"),
        ("heft_cpm", "HEFT+CPM"),
        ("auction", "Auction+CPM"),
        ("cpsat", "CP-SAT"),
        ("gnn_mlp", "GNN-MLP"),
        ("gnn_custom", "GNN-HGT"),
    ]:
        ms = metrics.get(f"{key}_makespan")
        if ms is not None:
            methods.append(label)
            makespans.append(ms)
            completions.append(metrics.get(f"{key}_completion", 1.0))
            violations.append(metrics.get(f"{key}_violations", 0))

    if not methods:
        return

    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Makespan bars
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax1.bar(x, makespans, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, makespans):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.0f}s", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7, rotation=30, ha="right")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("Makespan Comparison — Shackleton Rim", fontweight="bold", fontsize=10)

    # Violations + Completion
    bar_width = 0.35
    ax2.bar(x - bar_width / 2, violations, bar_width, label="Violations",
            color="#e15759", edgecolor="black", linewidth=0.5)
    ax2.bar(x + bar_width / 2, [c * 100 for c in completions], bar_width,
            label="Completion %", color="#59a14f", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=7, rotation=30, ha="right")
    ax2.set_ylabel("Count / Percentage")
    ax2.legend(fontsize=8)
    ax2.set_title("Quality Metrics — Shackleton Rim", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def _main_legacy_200m():
    """Single-scenario 200 m Shackleton case study (paper §5.5 figures)."""
    res_dir = ensure_results_dir()
    mlp_ok = gnn_mlp_available()
    custom_ok = gnn_custom_available()

    print(f"Case Study D — Shackleton Crater Rim (Real DEM)")
    print(f"  GNN-MLP: {'Y' if mlp_ok else 'N'} | GNN-HGT: {'Y' if custom_ok else 'N'}")
    print(f"  DEM NPZ: {'Y' if DEM_NPZ.exists() else 'N'}")
    print()

    # Load scenario
    if not SCENARIO_JSON.exists():
        print(f"ERROR: Scenario JSON not found: {SCENARIO_JSON}")
        return

    with open(SCENARIO_JSON, encoding="utf-8") as f:
        scenario = json.load(f)

    tg = TaskGraph(**scenario["taskgraph"])
    agents = make_shackleton_agents()
    n_tasks = len(tg.nodes)

    print(f"  Scenario: {tg.mission_id} ({n_tasks} tasks, 5 agents)")
    print(f"  Agent types: ROVER×2, RELAY×1, SAMPLER×1, ANALYZER×1\n")

    metrics = {"scenario": "shackleton_rim", "n_tasks": n_tasks}

    # ── Random+CPM
    print("  [1/10] Random+CPM...")
    asgn_r, wall_r = random_assignments(tg, agents)
    sched_r = assignments_to_schedule(tg, asgn_r, agents)
    sr_r = simulate_dem(tg, sched_r, agents)
    metrics["random_makespan"] = sr_r.makespan
    metrics["random_violations"] = len(sr_r.constraint_violations)
    metrics["random_completion"] = sr_r.task_completion_rate
    metrics["random_wall_s"] = round(wall_r, 4)

    # ── Greedy (coupled)
    print("  [2/10] Greedy (coupled)...")
    sched_gc, wall_gc = greedy_coupled_schedule(tg, agents)
    sr_gc = simulate_dem(tg, sched_gc, agents)
    metrics["greedy_coupled_makespan"] = sr_gc.makespan
    metrics["greedy_coupled_violations"] = len(sr_gc.constraint_violations)
    metrics["greedy_coupled_completion"] = sr_gc.task_completion_rate

    # ── Greedy+CPM
    print("  [3/10] Greedy+CPM...")
    asgn_g, wall_g = greedy_assignments(tg, agents)
    sched_g = assignments_to_schedule(tg, asgn_g, agents)
    sr_g = simulate_dem(tg, sched_g, agents)
    metrics["greedy_cpm_makespan"] = sr_g.makespan
    metrics["greedy_cpm_violations"] = len(sr_g.constraint_violations)
    metrics["greedy_cpm_completion"] = sr_g.task_completion_rate
    metrics["greedy_cpm_wall_s"] = round(wall_g, 4)

    draw_gantt(sched_g, tg, res_dir / "D_gantt_greedy.png",
               title="Greedy+CPM — Shackleton Rim")

    # ── HEFT (native)
    print("  [4/10] HEFT (native)...")
    sched_heft, wall_heft = heft_schedule(tg, agents)
    sr_heft = simulate_dem(tg, sched_heft, agents)
    metrics["heft_native_makespan"] = sr_heft.makespan
    metrics["heft_native_violations"] = len(sr_heft.constraint_violations)
    metrics["heft_native_completion"] = sr_heft.task_completion_rate
    metrics["heft_native_wall_s"] = round(wall_heft, 4)

    draw_gantt(sched_heft, tg, res_dir / "D_gantt_heft.png",
               title="HEFT (native) — Shackleton Rim")

    # ── HEFT+CPM
    print("  [5/10] HEFT+CPM...")
    asgn_hcpm, wall_hcpm = heft_assignments(tg, agents)
    sched_hcpm = assignments_to_schedule(tg, asgn_hcpm, agents)
    sr_hcpm = simulate_dem(tg, sched_hcpm, agents)
    metrics["heft_cpm_makespan"] = sr_hcpm.makespan
    metrics["heft_cpm_violations"] = len(sr_hcpm.constraint_violations)
    metrics["heft_cpm_completion"] = sr_hcpm.task_completion_rate

    # ── Auction+CPM
    print("  [6/10] Auction+CPM...")
    asgn_a, wall_a = auction_assignments(tg, agents)
    sched_a = assignments_to_schedule(tg, asgn_a, agents)
    sr_a = simulate_dem(tg, sched_a, agents)
    metrics["auction_makespan"] = sr_a.makespan
    metrics["auction_violations"] = len(sr_a.constraint_violations)
    metrics["auction_completion"] = sr_a.task_completion_rate
    metrics["auction_wall_s"] = round(wall_a, 4)

    # ── CP-SAT (optimal solver — 8 tasks is well within solvable range)
    print("  [7/10] CP-SAT...")
    sched_cs = None
    try:
        sched_cs, wall_cs = cpsat_schedule(tg, agents, timeout_s=120)
        if sched_cs:
            sr_cs = simulate_dem(tg, sched_cs, agents)
            metrics["cpsat_makespan"] = sr_cs.makespan
            metrics["cpsat_violations"] = len(sr_cs.constraint_violations)
            metrics["cpsat_completion"] = sr_cs.task_completion_rate
            metrics["cpsat_wall_s"] = round(wall_cs, 4)

            draw_gantt(sched_cs, tg, res_dir / "D_gantt_cpsat.png",
                       title="CP-SAT (optimal) — Shackleton Rim")
        else:
            print("    CP-SAT returned None")
            metrics["cpsat_makespan"] = None
    except Exception as e:
        print(f"    CP-SAT FAILED: {e}")
        metrics["cpsat_makespan"] = None

    # ── GNN-MLP+CPM
    if mlp_ok:
        print("  [8/10] GNN-MLP+CPM...")
        asgn_mlp, wall_mlp = gnn_mlp_assignments(tg, agents)
        sched_mlp = assignments_to_schedule(tg, asgn_mlp, agents)
        sr_mlp = simulate_dem(tg, sched_mlp, agents)
        metrics["gnn_mlp_makespan"] = sr_mlp.makespan
        metrics["gnn_mlp_violations"] = len(sr_mlp.constraint_violations)
        metrics["gnn_mlp_completion"] = sr_mlp.task_completion_rate
        metrics["gnn_mlp_wall_s"] = round(wall_mlp, 4)
    else:
        print("  [8/10] GNN-MLP+CPM... SKIPPED (no model)")

    # ── GNN-HGT+CPM
    if custom_ok:
        print("  [9/10] GNN-HGT+CPM...")
        asgn_h, wall_h = gnn_custom_assignments(tg, agents)
        sched_h = assignments_to_schedule(tg, asgn_h, agents)
        sr_h = simulate_dem(tg, sched_h, agents)
        metrics["gnn_custom_makespan"] = sr_h.makespan
        metrics["gnn_custom_violations"] = len(sr_h.constraint_violations)
        metrics["gnn_custom_completion"] = sr_h.task_completion_rate
        metrics["gnn_custom_wall_s"] = round(wall_h, 4)

        draw_gantt(sched_h, tg, res_dir / "D_gantt_gnn.png",
                   title="GNN-HGT+CPM — Shackleton Rim")
    else:
        print("  [9/10] GNN-HGT+CPM... SKIPPED (no model)")

    # ── Terrain Heatmap (greedy paths)
    print("\n  [10/10] Drawing terrain heatmap...")
    draw_terrain_heatmap(tg, sched_g, res_dir / "D_terrain_heatmap.png")

    # ── Side-by-side terrain heatmap: GNN-HGT vs CP-SAT
    if custom_ok and sched_cs:
        print("  Drawing terrain heatmap comparison (GNN vs CP-SAT)...")
        gnn_ms = metrics.get("gnn_custom_makespan", "?")
        cpsat_ms = metrics.get("cpsat_makespan", "?")
        draw_terrain_heatmap_comparison(
            tg, sched_h, sched_cs,
            res_dir / "D_terrain_heatmap_comparison.png",
            title_left=f"GNN-HGT+CPM ({gnn_ms:.0f}s)",
            title_right=f"CP-SAT Optimal ({cpsat_ms:.0f}s)",
        )

    # ── Metric comparison bar chart
    print("  Drawing metric comparison...")
    draw_metric_bars(metrics, res_dir / "D_dem_metrics.png")

    # ── Save metrics
    save_json(metrics, res_dir / "D_dem_metrics.json")

    # ── LaTeX table
    headers = ["Method", "Makespan (s)", "Violations", "Completion"]
    rows = []
    for key, label in [
        ("random", "Random+CPM"),
        ("greedy_coupled", "Greedy (coupled)"),
        ("greedy_cpm", "Greedy+CPM"),
        ("heft_native", "HEFT (native)"),
        ("heft_cpm", "HEFT+CPM"),
        ("auction", "Auction+CPM"),
        ("cpsat", "CP-SAT"),
        ("gnn_mlp", "GNN-MLP+CPM"),
        ("gnn_custom", "GNN-HGT+CPM"),
    ]:
        ms = metrics.get(f"{key}_makespan")
        if ms is not None:
            rows.append([
                label,
                f"{ms:.0f}",
                str(metrics.get(f"{key}_violations", 0)),
                f"{metrics.get(f'{key}_completion', 1.0):.1%}",
            ])

    tex = latex_booktabs(
        headers, rows,
        caption="Shackleton crater rim case study (real LOLA DEM terrain): "
                "scheduler comparison on an 8-task heterogeneous mission with "
                "ROVER, RELAY, SAMPLER, and ANALYZER agents.",
        label="tab:dem_case_study",
        col_fmt="lrrc",
    )
    save_tex(tex, res_dir / "D_table_dem.tex")

    # ── Summary
    print("\n  Summary:")
    for key, label in [
        ("random", "Random+CPM"),
        ("greedy_cpm", "Greedy+CPM"),
        ("heft_native", "HEFT (native)"),
        ("heft_cpm", "HEFT+CPM"),
        ("auction", "Auction+CPM"),
        ("cpsat", "CP-SAT"),
        ("gnn_custom", "GNN-HGT+CPM"),
    ]:
        ms = metrics.get(f"{key}_makespan")
        if ms is not None:
            v = metrics.get(f"{key}_violations", 0)
            c = metrics.get(f"{key}_completion", 1.0)
            print(f"    {label:20s}  makespan={ms:6.0f}s  violations={v}  completion={c:.0%}")

    print("\nDone. All figures and data saved to results/")


# ── Multi-scale CLI benchmark ────────────────────────────────────────────────

def _build_agents_from_scenario(scenario: dict) -> list[AgentState]:
    """Build the agent fleet from scenario metadata.

    Honors ``metadata.agent_fleet`` (e.g. ``{rover:5, relay:3, sampler:4,
    analyzer:3}``); falls back to a 5-agent Shackleton fleet when not present.
    """
    meta = scenario.get("metadata", {})
    fleet = meta.get("agent_fleet")
    if not fleet:
        # No explicit fleet — fall back by scale. Multi-km scenarios use the
        # "large" 8-agent fleet; the 200 m legacy scenario uses the 5-agent
        # Shackleton fleet (includes ANALYZER).
        scale = meta.get("scale", "")
        if scale in ("small", "medium", "large"):
            from common import make_agents
            return make_agents({"scale": scale})
        return make_shackleton_agents()
    type_map = {
        "rover": AgentType.ROVER, "relay": AgentType.RELAY,
        "sampler": AgentType.SAMPLER, "analyzer": AgentType.ANALYZER,
    }
    # Energy budget: large multi-km missions need a bigger battery; recharge
    # handles the rest.
    energy = float(meta.get("agent_energy_wh", 25000.0))
    agents = []
    for atype_str, count in fleet.items():
        atype = type_map.get(atype_str.lower())
        if atype is None:
            continue
        for i in range(int(count)):
            agents.append(AgentState(
                id=f"{atype_str}_{i+1}", agent_type=atype,
                position=(0.0, 0.0, 0.0),
                energy_wh=energy, initial_energy_wh=energy))
    return agents


def run_multiscale_benchmark(args):
    """9-way × N-round × {eager, lazy} benchmark on an arbitrary DEM scenario."""
    import statistics
    from core.simulator.energy_model import RechargeConfig

    proj = PROJECT_ROOT
    scen_path = Path(args.scenario)
    if not scen_path.is_absolute():
        scen_path = proj / scen_path
    dem_path = Path(args.dem)
    if not dem_path.is_absolute():
        dem_path = proj / dem_path

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = proj / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = json.load(open(scen_path, encoding="utf-8"))
    tg = TaskGraph(**scenario["taskgraph"])
    agents = _build_agents_from_scenario(scenario)
    terrain = TerrainGrid.from_npz(dem_path)

    # Recharge config
    if args.enable_recharge:
        recharge = RechargeConfig(
            enabled=True,
            threshold_frac=args.recharge_threshold,
            recharge_duration_s=args.recharge_duration,
            base_position=(0.0, 0.0, 0.0),
        )
    else:
        recharge = RechargeConfig(enabled=False)
    sim_cfg = SimConfig(recharge=recharge)

    n_tasks = len(tg.nodes)
    print(f"Multi-scale benchmark: {tg.mission_id}")
    print(f"  scenario={scen_path.name}  dem={dem_path.name}")
    print(f"  {n_tasks} tasks / {len(agents)} agents / recharge={'ON' if args.enable_recharge else 'OFF'}")
    print(f"  rounds={args.rounds}  travel_models={args.travel_models}\n")

    mlp_ok = gnn_mlp_available()
    custom_ok = gnn_custom_available()

    def _sim(sched):
        return LunarSimulator(terrain, agents, sim_cfg).evaluate(tg, sched)

    # Scheduler registry: name -> callable(seed_offset, travel_model) -> SimResult
    def _run_assignment_method(asgn_fn, travel_model):
        asgn, _ = asgn_fn()
        sched = assignments_to_schedule(tg, asgn, agents, travel_model=travel_model)
        return _sim(sched)

    methods = {
        "random_cpm":            lambda r, tm: _run_assignment_method(
            lambda: random_assignments(tg, agents, seed=1000 * r + 42), tm),
        "greedy_coupled":        lambda r, tm: _sim(greedy_coupled_schedule(tg, agents)[0]),
        "greedy_cpm":            lambda r, tm: _run_assignment_method(
            lambda: greedy_assignments(tg, agents), tm),
        "heft_native":           lambda r, tm: _sim(heft_schedule(tg, agents)[0]),
        "heft_cpm":              lambda r, tm: _run_assignment_method(
            lambda: heft_assignments(tg, agents), tm),
        "heft_insertion_cpm":    lambda r, tm: _run_assignment_method(
            lambda: __import__("common").heft_insertion_assignments(tg, agents), tm),
        "auction_cpm":           lambda r, tm: _run_assignment_method(
            lambda: auction_assignments(tg, agents), tm),
        "cpsat":                 lambda r, tm: _sim(cpsat_schedule(tg, agents)[0])
                                  if cpsat_schedule(tg, agents)[0] else None,
        "gnn_custom_cpm":        (lambda r, tm: _run_assignment_method(
            lambda: gnn_custom_assignments(tg, agents), tm)) if custom_ok else None,
    }
    if mlp_ok:
        methods["gnn_mlp_cpm"] = lambda r, tm: _run_assignment_method(
            lambda: gnn_mlp_assignments(tg, agents), tm)

    # Run all cells
    raw = {}  # (method, travel_model) -> list of per-round dicts
    for tm in args.travel_models:
        for mname, mfn in methods.items():
            if mfn is None:
                continue
            cells = []
            for r in range(args.rounds):
                try:
                    sr = mfn(r, tm)
                    if sr is None:
                        cells.append({"round": r, "makespan": None, "status": "N/A"})
                        continue
                    cells.append({
                        "round": r,
                        "makespan": round(float(sr.makespan), 2),
                        "completion": round(sr.task_completion_rate, 4),
                        "violations": len(sr.constraint_violations),
                        "n_recharges": sum(sr.n_recharges_per_agent.values()),
                        "total_recharge_s": sr.total_recharge_time_s,
                    })
                except Exception as e:
                    cells.append({"round": r, "makespan": None, "status": f"ERR: {str(e)[:80]}"})
            raw[f"{mname}|{tm}"] = cells
            ms_vals = [c["makespan"] for c in cells if isinstance(c.get("makespan"), (int, float))]
            if ms_vals:
                mean = statistics.fmean(ms_vals)
                std = statistics.stdev(ms_vals) if len(ms_vals) > 1 else 0.0
                print(f"  {mname:<22} [{tm:<5}] mean={mean:>10.0f}  std={std:>8.1f}  n={len(ms_vals)}")
            else:
                print(f"  {mname:<22} [{tm:<5}] N/A (unsolvable / errored)")

    # Aggregate + determinism check
    DETERMINISTIC = {"greedy_coupled", "greedy_cpm", "heft_native", "heft_cpm",
                     "heft_insertion_cpm", "auction_cpm", "cpsat",
                     "gnn_custom_cpm", "gnn_mlp_cpm"}
    summary = {"scenario": tg.mission_id, "n_tasks": n_tasks, "n_agents": len(agents),
               "recharge_enabled": args.enable_recharge, "rounds": args.rounds,
               "travel_models": args.travel_models, "by_cell": {}}
    determinism_violations = []
    for key, cells in raw.items():
        mname, tm = key.split("|")
        ms_vals = [c["makespan"] for c in cells if isinstance(c.get("makespan"), (int, float))]
        if not ms_vals:
            summary["by_cell"][key] = {"mean": None, "std": None, "n": 0}
            continue
        mean = round(statistics.fmean(ms_vals), 2)
        std = round(statistics.stdev(ms_vals), 4) if len(ms_vals) > 1 else 0.0
        summary["by_cell"][key] = {
            "mean": mean, "std": std, "n": len(ms_vals),
            "min": min(ms_vals), "max": max(ms_vals),
            "completion": cells[0].get("completion"),
            "n_recharges": cells[0].get("n_recharges", 0),
            "total_recharge_s": cells[0].get("total_recharge_s", 0.0),
        }
        if mname in DETERMINISTIC and std > 0.5:
            determinism_violations.append({"cell": key, "std": std})
    summary["determinism_violations"] = determinism_violations

    # Write outputs
    metrics_path = out_dir / f"D_dem_{_scale_label(n_tasks)}_metrics.json"
    json.dump({"summary": summary, "raw": raw}, open(metrics_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n  -> {metrics_path}")

    # Recharge stats report
    if args.enable_recharge:
        rc_path = out_dir / f"D_{_scale_label(n_tasks)}_recharge_stats.md"
        _write_recharge_report(rc_path, summary, raw)
        print(f"  -> {rc_path}")

    if determinism_violations:
        print(f"\n  WARNING: {len(determinism_violations)} determinism violation(s): "
              f"{determinism_violations}")
    else:
        print(f"\n  Determinism check: PASS (all deterministic schedulers std=0)")

    return summary


def _scale_label(n_tasks: int) -> str:
    if n_tasks <= 20:
        return "200m"
    if n_tasks <= 90:
        return "1km"
    return "5km"


def _write_recharge_report(path: Path, summary: dict, raw: dict):
    L = ["# Recharge Statistics Report", "",
         f"Scenario: `{summary['scenario']}` — {summary['n_tasks']} tasks, "
         f"{summary['n_agents']} agents", "",
         f"Recharge enabled: **{summary['recharge_enabled']}**", "",
         "| Method × travel_model | mean makespan | n_recharges | total_recharge_s |",
         "|---|---:|---:|---:|"]
    for key, cell in sorted(summary["by_cell"].items()):
        if cell.get("mean") is None:
            continue
        L.append(f"| {key} | {cell['mean']:.0f} | {cell.get('n_recharges', 0)} | "
                 f"{cell.get('total_recharge_s', 0.0):.0f} |")
    L.append("")
    L.append("Recharge events are inserted by the simulator when an agent's energy "
             "drops below `threshold_frac` of its initial budget; the agent travels "
             "to base, recharges for `recharge_duration_s`, and returns. The cycle's "
             "wall-clock cost is reflected in the makespan above.")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="D — DEM case study / multi-scale benchmark")
    ap.add_argument("--scenario", default=None,
                    help="Path to scenario JSON (triggers multi-scale benchmark mode)")
    ap.add_argument("--dem", default=None, help="Path to DEM NPZ")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--travel-models", nargs="*", default=["eager", "lazy"])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--enable-recharge", action="store_true")
    ap.add_argument("--recharge-threshold", type=float, default=0.20)
    ap.add_argument("--recharge-duration", type=float, default=1800.0)
    args = ap.parse_args()

    if args.scenario:
        if not args.dem or not args.output_dir:
            print("ERROR: --scenario mode requires --dem and --output-dir", file=sys.stderr)
            sys.exit(1)
        run_multiscale_benchmark(args)
    else:
        _main_legacy_200m()


if __name__ == "__main__":
    main()
