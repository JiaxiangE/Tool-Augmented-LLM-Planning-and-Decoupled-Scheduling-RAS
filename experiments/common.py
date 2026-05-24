"""
Shared utilities for the paper's experimental drivers.

Every driver script (A–H) imports from here; no scheduler or simulator
re-implementation is performed inside the experiment scripts themselves.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_DIR = PROJECT_ROOT / "data" / "training_corpus"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default GNN policy paths.
#
#   - GNN-HGT (Heterogeneous Graph Transformer + custom PPO):
#       Default: ``models/checkpoints/gnn_hgt/gnn_hgt_final.pt`` — the
#       single canonical checkpoint that reproduces the paper's Table 4
#       GNN-HGT+CPM cells (500K training steps, mean reward +1.72).
#       Override via ``GNN_HGT_PATH``.
#
#       A secondary 5-seed ensemble lives at
#       ``models/checkpoints/gnn_hgt_ls/seed_{42,137,271,419,613}/``;
#       it is used by ``B_ood_benchmark.py`` and ``E_hybrid_decomposition.py``
#       which both report ensemble statistics (see those scripts' headers).
#
#   - GNN-MLP (MaskablePPO MLP policy):
#       Single checkpoint at ``models/checkpoints/gnn_mlp/gnn_mlp_final.zip``.
#       sb3 expects the path without the .zip suffix.
import os as _os
GNN_CUSTOM_PATH = _os.environ.get(
    "GNN_HGT_PATH",
    str(PROJECT_ROOT / "models" / "checkpoints" / "gnn_hgt" / "gnn_hgt_final.pt"),
)
GNN_MLP_PATH = _os.environ.get(
    "GNN_MLP_PATH",
    str(PROJECT_ROOT / "models" / "checkpoints" / "gnn_mlp" / "gnn_mlp_final"),
)

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState
from core.schema.taskgraph import EdgeType, TaskGraph
from core.scheduler.types import Schedule, ScheduleItem
from core.simulator.lunar_sim import LunarSimulator, SimConfig, SimResult
from core.simulator.terrain import TerrainGrid

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("experiments")


# ── constants ─────────────────────────────────────────────────────────────────
SCHEDULER_NAMES_7WAY = [
    "Random+CPM",
    "Greedy (coupled)",
    "Greedy+CPM",
    "Auction+CPM",
    "CP-SAT",
    "GNN-MLP+CPM",
    "GNN-HGT+CPM",
]

SCHEDULER_NAMES_9WAY = [
    "Random+CPM",
    "Greedy (coupled)",
    "Greedy+CPM",
    "HEFT (native)",
    "HEFT+CPM",
    "Auction+CPM",
    "CP-SAT",
    "GNN-MLP+CPM",
    "GNN-HGT+CPM",
]


# ── result helpers ────────────────────────────────────────────────────────────
def ensure_results_dir(override: Optional[Path] = None) -> Path:
    """Return the results directory. If ``override`` is provided (e.g. via the
    ``--output-dir`` CLI flag), use it instead of the default ``RESULTS_DIR``."""
    d = Path(override) if override else RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs for reproducibility.

    Used to drive the 5-round multi-run benchmarks. Idempotent and safe to call
    before any scheduler/inference invocation.
    """
    import random as _rnd
    _rnd.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _t
        _t.manual_seed(seed)
        if _t.cuda.is_available():
            _t.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_paper_experiments_argparser(default_travel_model: str = "eager"):
    """Standard argparser for the multi-run protocol used by drivers A-F."""
    import argparse
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Seed offset for the random baseline. Other methods "
                        "are deterministic; the published numbers use 0.")
    p.add_argument("--travel-model", choices=["eager", "lazy"], default=default_travel_model,
                   help="CPM travel model: 'eager' (paper Eq. 8) or 'lazy' (coupled-style)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override results directory (defaults to ./results/)")
    return p


def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f"  -> {path}")


def save_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    keys = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


# ── corpus loading ────────────────────────────────────────────────────────────
def load_corpus(corpus_dir: Path = CORPUS_DIR) -> List[Dict[str, Any]]:
    """Load all JSON corpus files, sorted by name. Skips files whose name starts
    with ``_`` (reserved for sidecar artifacts)."""
    return [
        {**json.load(open(f, encoding="utf-8")), "_filename": f.name}
        for f in sorted(corpus_dir.glob("*.json"))
        if not f.name.startswith("_")
    ]


# ── agent factory ─────────────────────────────────────────────────────────────
def make_agents(meta: Dict) -> List[AgentState]:
    scale = meta.get("scale", "small")
    if scale == "small":
        return [
            AgentState(id="rover_1",   agent_type=AgentType.ROVER,   energy_wh=3000, initial_energy_wh=3000),
            AgentState(id="relay_1",   agent_type=AgentType.RELAY,   energy_wh=3000, initial_energy_wh=3000),
            AgentState(id="sampler_1", agent_type=AgentType.SAMPLER, energy_wh=3000, initial_energy_wh=3000),
        ]
    elif scale == "medium":
        return [
            AgentState(id="rover_1",   agent_type=AgentType.ROVER,   energy_wh=5000, initial_energy_wh=5000),
            AgentState(id="rover_2",   agent_type=AgentType.ROVER,   energy_wh=5000, initial_energy_wh=5000),
            AgentState(id="relay_1",   agent_type=AgentType.RELAY,   energy_wh=5000, initial_energy_wh=5000),
            AgentState(id="relay_2",   agent_type=AgentType.RELAY,   energy_wh=5000, initial_energy_wh=5000),
            AgentState(id="sampler_1", agent_type=AgentType.SAMPLER, energy_wh=5000, initial_energy_wh=5000),
        ]
    else:  # large
        return [
            AgentState(id="rover_1",   agent_type=AgentType.ROVER,   energy_wh=8000, initial_energy_wh=8000),
            AgentState(id="rover_2",   agent_type=AgentType.ROVER,   energy_wh=8000, initial_energy_wh=8000),
            AgentState(id="rover_3",   agent_type=AgentType.ROVER,   energy_wh=8000, initial_energy_wh=8000),
            AgentState(id="relay_1",   agent_type=AgentType.RELAY,   energy_wh=8000, initial_energy_wh=8000),
            AgentState(id="relay_2",   agent_type=AgentType.RELAY,   energy_wh=8000, initial_energy_wh=8000),
            AgentState(id="sampler_1", agent_type=AgentType.SAMPLER, energy_wh=8000, initial_energy_wh=8000),
            AgentState(id="sampler_2", agent_type=AgentType.SAMPLER, energy_wh=8000, initial_energy_wh=8000),
            AgentState(id="sampler_3", agent_type=AgentType.SAMPLER, energy_wh=8000, initial_energy_wh=8000),
        ]


# ── unified CPM timing (travel-aware; eager + lazy variants) ─────────────────
def _euclid_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p1, p2)))


def assignments_to_schedule(
    tg: TaskGraph,
    assignments: Dict[str, str],
    agents: List[AgentState],
    mission_id: str = "cpm",
    travel_model: Literal["eager", "lazy"] = "eager",
    default_speed_m_per_s: float = 0.5,
) -> Schedule:
    """Convert ``{task_id -> agent_id}`` into a Schedule with travel-aware CPM
    timing (paper §3.3 / §4.3.1, Eq. (8)).

    Two travel models are supported:

      - 'eager' (default, paper-compliant): an agent begins traveling to its
        next assigned task immediately on completing the previous one. Models
        offline planning with foreknowledge of all assignments::

            start = max(pred_finish, free(a) + travel)

      - 'lazy' (apples-to-apples vs. coupled): the agent only begins traveling
        after both predecessors finish AND it becomes free. Matches the
        coupled greedy/HEFT online reactive model, useful for isolating the
        pure assignment-strategy effect from the offline foreknowledge premium::

            start = max(pred_finish, free(a)) + travel
    """
    if travel_model not in ("eager", "lazy"):
        raise ValueError(f"Unknown travel_model: {travel_model!r} "
                         f"(expected 'eager' or 'lazy')")

    preds: Dict[str, set] = {n.id: set() for n in tg.nodes}
    for e in tg.edges:
        if e.type in (EdgeType.ORDER, EdgeType.DATA) and e.dst in preds:
            preds[e.dst].add(e.src)

    asgn = sorted(assignments)  # sorted for deterministic iteration order
    asgn_set = set(asgn)
    in_deg = {t: sum(1 for p in preds.get(t, set()) if p in asgn_set) for t in asgn}
    q = deque(t for t in asgn if in_deg[t] == 0)
    topo: List[str] = []
    while q:
        t = q.popleft()
        topo.append(t)
        succs = []
        for e in tg.edges:
            if e.type in (EdgeType.ORDER, EdgeType.DATA) and e.src == t and e.dst in asgn_set:
                in_deg[e.dst] -= 1
                if in_deg[e.dst] == 0:
                    succs.append(e.dst)
        for s in sorted(succs):
            q.append(s)
    topo.extend(t for t in asgn if t not in set(topo))

    task_map = {n.id: n for n in tg.nodes}
    agent_init_pos: Dict[str, Tuple[float, float, float]] = {
        a.id: tuple(a.position) for a in agents
    }
    agent_speed: Dict[str, float] = {
        a.id: float(getattr(a, "speed_m_per_s", default_speed_m_per_s)
                    or default_speed_m_per_s)
        for a in agents
    }
    last_loc: Dict[str, Tuple[float, float, float]] = {}
    end: Dict[str, float] = {}
    free: Dict[str, float] = {}
    items = []
    for t in topo:
        ag = assignments[t]
        node = task_map[t]
        dur = float(node.estimated_duration_s or 60.0)
        cur_loc = tuple(node.location or (0.0, 0.0, 0.0))
        prev_loc = last_loc.get(ag, agent_init_pos.get(ag, (0.0, 0.0, 0.0)))
        speed = agent_speed.get(ag, default_speed_m_per_s) or default_speed_m_per_s
        travel = _euclid_3d(prev_loc, cur_loc) / speed if speed > 0 else 0.0
        pred_max = max((end[p] for p in preds.get(t, set()) if p in end), default=0.0)

        if travel_model == "eager":
            # paper Eq. (8) literal: avail(a) = free(a) + travel; agent pre-positions
            start = max(pred_max, free.get(ag, 0.0) + travel)
        else:  # "lazy"
            # coupled-style: agent only starts traveling once free AND pred done
            start = max(pred_max, free.get(ag, 0.0)) + travel

        e_t = start + dur
        end[t] = e_t
        free[ag] = e_t
        last_loc[ag] = cur_loc
        items.append(ScheduleItem(node_id=t, agent=ag, start_s=start, end_s=e_t))

    return Schedule(
        mission_id=mission_id,
        makespan_s=max((i.end_s for i in items), default=0.0),
        items=items,
    )


# ── simulation ────────────────────────────────────────────────────────────────
def simulate(tg: TaskGraph, sched: Schedule, agents: List[AgentState]) -> SimResult:
    terrain = TerrainGrid(width=400, height=400, cell_size=10.0)
    return LunarSimulator(terrain, agents, SimConfig()).evaluate(tg, sched)


# ── assignment extractors ─────────────────────────────────────────────────────
def random_assignments(
    tg: TaskGraph, agents: List[AgentState], seed: int = 42,
) -> Tuple[Dict[str, str], float]:
    """Type-compatible random assignment (deterministic with seed)."""
    rng = random.Random(seed)
    t0 = time.time()
    assigned: Dict[str, str] = {}
    for node in tg.nodes:
        compatible = [a for a in agents if not node.agent_type_reqs or a.agent_type in node.agent_type_reqs]
        if compatible:
            assigned[node.id] = rng.choice(compatible).id
        else:
            assigned[node.id] = rng.choice(agents).id
    return assigned, time.time() - t0


def greedy_assignments(tg: TaskGraph, agents: List[AgentState]) -> Tuple[Dict[str, str], float]:
    from core.scheduler.greedy_scheduler import AgentInfo, schedule as gs
    t0 = time.time()
    sched = gs(tg, [AgentInfo(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh) for a in agents])
    return {item.node_id: item.agent for item in sched.items}, time.time() - t0


def greedy_coupled_schedule(tg: TaskGraph, agents: List[AgentState]) -> Tuple[Schedule, float]:
    """Run the greedy scheduler with its own built-in timing (coupled)."""
    from core.scheduler.greedy_scheduler import AgentInfo, schedule as gs
    t0 = time.time()
    sched = gs(tg, [AgentInfo(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh) for a in agents])
    return sched, time.time() - t0


def auction_assignments(tg: TaskGraph, agents: List[AgentState]) -> Tuple[Dict[str, str], float]:
    from core.scheduler.auction_scheduler import AuctionAgent, schedule as aus
    t0 = time.time()
    sched = aus(tg, [AuctionAgent(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh) for a in agents])
    return {item.node_id: item.agent for item in sched.items}, time.time() - t0


def gnn_mlp_assignments(
    tg: TaskGraph, agents: List[AgentState], model_path: str = GNN_MLP_PATH,
) -> Tuple[Dict[str, str], float]:
    """Evaluate the MLP MaskablePPO policy.

    Note: the released MLP checkpoint was trained against a fixed action space
    of MLP_MAX_TASKS=128 x MLP_MAX_AGENTS=10 = 1280 dims. The current
    StateEncoder was later widened (MAX_TASKS=512 x MAX_AGENTS=16 = 8192). We
    therefore reshape the mask / observation to the policy's expected shape
    before calling .predict(). Scenarios with > 128 tasks or > 10 agents are
    not supported by this checkpoint and raise a clear error.
    """
    import numpy as np
    import torch
    torch.distributions.Distribution.set_default_validate_args(False)
    from sb3_contrib import MaskablePPO
    from core.scheduler.gnn.state_encoder import MAX_AGENTS, MAX_TASKS, StateEncoder

    # Frozen training-time action space of the released MLP checkpoint.
    MLP_MAX_TASKS = 128
    MLP_MAX_AGENTS = 10

    if len(tg.nodes) > MLP_MAX_TASKS:
        raise RuntimeError(
            f"Scenario has {len(tg.nodes)} tasks but the MLP checkpoint was "
            f"trained for at most {MLP_MAX_TASKS}. Use the HGT policy instead."
        )
    if len(agents) > MLP_MAX_AGENTS:
        raise RuntimeError(
            f"Scenario has {len(agents)} agents but the MLP checkpoint was "
            f"trained for at most {MLP_MAX_AGENTS}. Use the HGT policy instead."
        )

    t0 = time.time()
    model = MaskablePPO.load(model_path)
    enc = StateEncoder()
    assigned: Dict[str, str] = {}
    task_ids = [n.id for n in tg.nodes]
    agent_ids = [a.id for a in agents]

    # Probe what observation length the loaded policy expects (its observation
    # space is whatever the original training-time encoder pooled to). We
    # slice the current encoder's output down to that length.
    mlp_obs_dim = int(np.prod(model.observation_space.shape))

    for _ in range(len(task_ids) * 3):
        if len(assigned) >= len(task_ids):
            break

        # Current encoder produces a (MAX_TASKS x MAX_AGENTS) mask. Reshape
        # and slice the top-left (MLP_MAX_TASKS x MLP_MAX_AGENTS) block so
        # the indices line up with the MLP policy's flat action space.
        big_mask = enc.compute_action_mask(tg, agents, assigned).reshape(MAX_TASKS, MAX_AGENTS)
        mask = big_mask[:MLP_MAX_TASKS, :MLP_MAX_AGENTS].reshape(-1).astype(np.float32)
        if mask.sum() == 0:
            break

        hetero = enc.encode(tg, agents, assigned)
        big_obs = enc.get_pooled_observation(hetero)
        big_obs[-1] = len(assigned) / max(len(task_ids), 1)
        # Slice (or zero-pad) the observation to match the policy's frozen shape.
        if big_obs.shape[0] >= mlp_obs_dim:
            obs = big_obs[:mlp_obs_dim]
        else:
            obs = np.zeros(mlp_obs_dim, dtype=big_obs.dtype)
            obs[: big_obs.shape[0]] = big_obs

        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        ti, ai = int(action) // MLP_MAX_AGENTS, int(action) % MLP_MAX_AGENTS
        if ti < len(task_ids) and ai < len(agents):
            tid = task_ids[ti]
            if tid not in assigned:
                assigned[tid] = agent_ids[ai]
    return assigned, time.time() - t0


def gnn_custom_assignments(
    tg: TaskGraph, agents: List[AgentState], model_path: str = GNN_CUSTOM_PATH,
) -> Tuple[Dict[str, str], float]:
    import torch
    torch.distributions.Distribution.set_default_validate_args(False)
    from core.scheduler.gnn.policy_network import SchedulingPolicyNet
    from core.scheduler.gnn.state_encoder import (AGENT_FEAT_DIM, MAX_AGENTS,
                                                  TASK_FEAT_DIM, StateEncoder)

    pyg_meta = (
        ["task", "agent"],
        [("task", "depends_on", "task"), ("agent", "comm_with", "agent"), ("agent", "can_exec", "task")],
    )
    policy = SchedulingPolicyNet(
        task_feat_dim=TASK_FEAT_DIM, agent_feat_dim=AGENT_FEAT_DIM,
        hidden_dim=64, encoder_type="hgt", num_heads=4, num_layers=2,
        metadata=pyg_meta,
    )
    # Checkpoints may be either the full training-state format (policy + optimizer
    # + step) or the legacy bare-state_dict format; handle both.
    _ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    _state = (_ckpt["policy_state_dict"]
              if isinstance(_ckpt, dict) and "policy_state_dict" in _ckpt
              else _ckpt)
    policy.load_state_dict(_state)
    policy.eval()

    t0 = time.time()
    enc = StateEncoder()
    assigned: Dict[str, str] = {}
    task_ids = [n.id for n in tg.nodes]
    agent_ids = [a.id for a in agents]

    with torch.no_grad():
        for _ in range(len(task_ids) * 3):
            if len(assigned) >= len(task_ids):
                break
            mask_np = enc.compute_action_mask(tg, agents, assigned)
            if mask_np.sum() == 0:
                break
            hetero = enc.encode(tg, agents, assigned)
            mask_t = torch.tensor(mask_np, dtype=torch.float32)
            action, _, _ = policy.get_action(hetero, mask_t, deterministic=True)
            ti, ai = action // MAX_AGENTS, action % MAX_AGENTS
            if ti < len(task_ids) and ai < len(agents):
                tid = task_ids[ti]
                if tid not in assigned:
                    assigned[tid] = agent_ids[ai]
    return assigned, time.time() - t0


# Locations of the 5 HGT seeds shipped with the repo.
HGT_LS_SEEDS = [42, 137, 271, 419, 613]
HGT_LS_ROOT = PROJECT_ROOT / "models" / "checkpoints" / "gnn_hgt_ls"


def gnn_custom_5seed_makespan(
    tg, agents, travel_model: str = "eager",
    aggregate: str = "mean",
) -> Tuple[Dict[str, str], float, float, list]:
    """5-seed evaluation of the GNN-HGT-LS ensemble (Table 4 protocol).

    Loads each of the five shipped HGT checkpoints in turn, runs assignment,
    CPM-retimes under ``travel_model``, and simulates. The five resulting
    makespans are aggregated (``aggregate='mean'`` or ``'best'``).

    Returns
    -------
    canonical_asgn : dict
        Assignment dict from the seed whose makespan is closest to the chosen
        aggregate (so downstream visualisation / Gantt has a single concrete
        schedule that is representative of the reported number).
    wall_s : float
        Total wall time for all 5 seeds.
    aggregated_makespan : float
        Mean (or min) across the 5 seeds.
    per_seed : list[(seed, makespan)]
        Per-seed makespan, for diagnostic / CI reporting.
    """
    t0 = time.time()
    results = []
    for seed in HGT_LS_SEEDS:
        ckpt = HGT_LS_ROOT / f"seed_{seed}" / "gnn_hgt_final.pt"
        if not ckpt.exists():
            continue
        asgn, _ = gnn_custom_assignments(tg, agents, model_path=str(ckpt))
        sched = assignments_to_schedule(tg, asgn, agents, travel_model=travel_model)
        sr = simulate(tg, sched, agents)
        results.append({"seed": seed, "asgn": asgn, "makespan": sr.makespan,
                        "completion": sr.task_completion_rate,
                        "violations": len(sr.constraint_violations)})

    if not results:
        raise RuntimeError(f"No HGT checkpoints found under {HGT_LS_ROOT}")

    makespans = [r["makespan"] for r in results]
    mean_ms = sum(makespans) / len(makespans)

    if aggregate == "best":
        canonical = min(results, key=lambda r: r["makespan"])
        agg_ms = canonical["makespan"]
    else:  # 'mean'
        agg_ms = mean_ms
        # Pick the seed closest to mean as the canonical assignment.
        canonical = min(results, key=lambda r: abs(r["makespan"] - mean_ms))

    per_seed = [(r["seed"], r["makespan"]) for r in results]
    return canonical["asgn"], time.time() - t0, agg_ms, per_seed


def cpsat_schedule(tg: TaskGraph, agents: List[AgentState], timeout_s: int = 120) -> Tuple[Optional[Schedule], float]:
    from core.scheduler.cpsat_scheduler import CPSATAgent, schedule as cs
    t0 = time.time()
    sched = cs(tg, [CPSATAgent(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh,
                                position=a.position, speed_m_per_s=a.speed_m_per_s) for a in agents],
               timeout_s=timeout_s)
    return sched, time.time() - t0


def heft_schedule(tg: TaskGraph, agents: List[AgentState]) -> Tuple[Schedule, float]:
    """Run HEFT with its native EFT-based timing."""
    from core.scheduler.heft_scheduler import HEFTAgent, schedule as hs
    t0 = time.time()
    sched = hs(tg, [HEFTAgent(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh,
                               position=a.position, speed_m_per_s=a.speed_m_per_s)
                    for a in agents])
    return sched, time.time() - t0


def heft_assignments(tg: TaskGraph, agents: List[AgentState]) -> Tuple[Dict[str, str], float]:
    """Extract the HEFT assignment dict (for decoupled CPM timing comparison)."""
    sched, wall = heft_schedule(tg, agents)
    return {item.node_id: item.agent for item in sched.items}, wall


def heft_insertion_schedule(tg: TaskGraph, agents: List[AgentState]) -> Tuple[Schedule, float]:
    """Run HEFT-Insertion (Topcuoglu 2002 original variant) with travel."""
    from core.scheduler.heft_scheduler import HEFTAgent
    from core.scheduler.heft_insertion_scheduler import schedule as his
    t0 = time.time()
    sched = his(tg, [HEFTAgent(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh,
                                position=a.position, speed_m_per_s=a.speed_m_per_s)
                     for a in agents])
    return sched, time.time() - t0


def heft_insertion_assignments(tg: TaskGraph, agents: List[AgentState]) -> Tuple[Dict[str, str], float]:
    """Extract the HEFT-Insertion assignment dict for decoupled CPM comparison."""
    sched, wall = heft_insertion_schedule(tg, agents)
    return {item.node_id: item.agent for item in sched.items}, wall


# ── model availability checks ────────────────────────────────────────────────
def gnn_mlp_available() -> bool:
    return Path(GNN_MLP_PATH + ".zip").exists()


def gnn_custom_available() -> bool:
    return Path(GNN_CUSTOM_PATH).exists()


# ── matplotlib setup ──────────────────────────────────────────────────────────
def setup_matplotlib():
    """Configure matplotlib for publication-quality output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.figsize": (10, 6),
    })


# ── LaTeX table generation ───────────────────────────────────────────────────
def latex_booktabs(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
    col_fmt: str = "",
) -> str:
    """Generate a LaTeX booktabs table string."""
    if not col_fmt:
        col_fmt = "l" + "r" * (len(headers) - 1)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_tex(content: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  -> {path}")
