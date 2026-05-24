"""
Shared helpers for the Shackleton 5 km figure scripts (fig4 terrain + fig5 Gantt).

Both figures consume the same scenario file (``data/dem/real_shackleton_5km.json``)
and DEM bundle (``data/dem/shackleton_5km.npz``), and both render the
Greedy+CPM 15-agent schedule. The two helpers below load that data once each
so the figure-specific scripts stay focused on rendering.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

import numpy as np

from core.schema.taskgraph import TaskGraph
from experiments.common import assignments_to_schedule, greedy_assignments
from experiments.D_case_study_dem import _build_agents_from_scenario

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCEN_PATH = REPO_ROOT / "data" / "dem" / "real_shackleton_5km.json"
DEM_PATH = REPO_ROOT / "data" / "dem" / "shackleton_5km.npz"


def load_dem() -> dict:
    """Load the Shackleton 5 km DEM bundle (elevation, slope, PSR mask)."""
    z = np.load(DEM_PATH)
    return {"elevation": z["elevation"],
            "slope": z["slope"],
            "is_psr": z["is_psr"],
            "cell_size": float(z["cell_size"])}


def load_schedule() -> Tuple[TaskGraph, List, "Schedule"]:
    """Build the Greedy+CPM schedule for the 15-agent Shackleton scenario."""
    scenario = json.load(open(SCEN_PATH, encoding="utf-8"))
    tg = TaskGraph(**scenario["taskgraph"])
    agents = _build_agents_from_scenario(scenario)
    asgn, _ = greedy_assignments(tg, agents)
    sched = assignments_to_schedule(tg, asgn, agents, travel_model="eager")
    return tg, agents, sched


def agent_type_of(agent_id: str) -> str:
    return agent_id.split("_")[0].upper()


AGENT_TYPE_COLOR = {
    "ROVER":    "#2980b9",
    "RELAY":    "#27ae60",
    "SAMPLER":  "#e67e22",
    "ANALYZER": "#8e44ad",
}
