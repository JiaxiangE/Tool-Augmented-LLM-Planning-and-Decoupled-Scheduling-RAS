"""
HEFT (Heterogeneous Earliest Finish Time) baseline scheduler.

Algorithm (Topcuoglu et al., 2002):
  1. Compute upward rank for each task:
     rank_u(t) = dur(t) + max(rank_u(succ)) over ORDER+DATA successors.
     Sinks: rank_u = dur.
  2. Sort tasks by descending upward rank (critical path first).
  3. For each task in rank order, assign to the agent with minimum EFT:
     EFT(t, a) = max(agent_available[a], max(pred_finish_times)) + travel + dur
     Only type-compatible agents considered.

Key differences from Greedy:
  - Priority ordering: upward rank (backward from sinks) vs topological order
  - Agent selection: min EFT vs earliest-available

Interface:
  schedule(task_graph, agents) -> Schedule
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from core.schema.agent_types import AgentType
from core.schema.taskgraph import EdgeType, TaskGraph, TaskNode
from core.scheduler.types import Schedule, ScheduleItem


@dataclass
class HEFTAgent:
    """Static description of an agent available for HEFT scheduling."""
    id: str
    agent_type: AgentType
    energy_wh: float = 100.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_m_per_s: float = 0.5


def _build_adj(task_graph: TaskGraph):
    """Build successor and predecessor adjacency from ORDER+DATA edges."""
    succs: Dict[str, List[str]] = defaultdict(list)
    preds: Dict[str, List[str]] = defaultdict(list)
    for e in task_graph.edges:
        if e.type in (EdgeType.ORDER, EdgeType.DATA):
            succs[e.src].append(e.dst)
            preds[e.dst].append(e.src)
    return succs, preds


def compute_upward_rank(
    task_graph: TaskGraph,
    node_map: Dict[str, TaskNode],
) -> Dict[str, float]:
    """
    Compute upward rank for each task (backward BFS from sinks).

    rank_u(t) = dur(t) + max(rank_u(succ)) for all ORDER/DATA successors.
    Sinks have rank_u = dur.
    """
    succs, _ = _build_adj(task_graph)
    node_ids = [n.id for n in task_graph.nodes]

    rank_u: Dict[str, float] = {}

    # Reverse topological order via DFS post-order
    visited: Set[str] = set()
    post_order: List[str] = []

    def dfs(nid: str):
        if nid in visited:
            return
        visited.add(nid)
        for s in succs.get(nid, []):
            dfs(s)
        post_order.append(nid)

    for nid in node_ids:
        dfs(nid)

    # post_order is reverse topological → sinks come first
    for nid in post_order:
        dur = node_map[nid].estimated_duration_s or 60.0
        succ_ranks = [rank_u[s] for s in succs.get(nid, []) if s in rank_u]
        rank_u[nid] = dur + (max(succ_ranks) if succ_ranks else 0.0)

    return rank_u


def _travel_time(
    from_pos: Tuple[float, float, float],
    to_pos: Tuple[float, float, float],
    speed: float,
) -> float:
    dist = math.sqrt(
        (from_pos[0] - to_pos[0]) ** 2
        + (from_pos[1] - to_pos[1]) ** 2
        + (from_pos[2] - to_pos[2]) ** 2
    )
    return dist / speed if speed > 0 else 0.0


def schedule(
    task_graph: TaskGraph,
    agents: List[HEFTAgent],
) -> Schedule:
    """
    HEFT scheduler: produce a valid Schedule from a TaskGraph and agent list.

    Args:
        task_graph: TaskGraph to schedule.
        agents: List of available agents.

    Returns:
        A valid Schedule object with EFT-based timing.
    """
    if not agents:
        raise ValueError("Cannot schedule with zero agents")

    node_map = {n.id: n for n in task_graph.nodes}

    if not task_graph.nodes:
        return Schedule(mission_id=task_graph.mission_id, makespan_s=0, items=[])

    # 1. Compute upward rank
    rank_u = compute_upward_rank(task_graph, node_map)

    # 2. Sort by descending upward rank
    sorted_tasks = sorted(rank_u.keys(), key=lambda t: rank_u[t], reverse=True)

    # 3. Build predecessor map
    _, preds = _build_adj(task_graph)

    # 4. HEFT assignment: for each task, pick agent with minimum EFT
    agent_available: Dict[str, float] = {a.id: 0.0 for a in agents}
    agent_position: Dict[str, Tuple[float, float, float]] = {a.id: a.position for a in agents}
    task_end: Dict[str, float] = {}

    items: List[ScheduleItem] = []

    for nid in sorted_tasks:
        node = node_map[nid]
        dur = node.estimated_duration_s or 60.0
        task_loc = node.location or (0.0, 0.0, 0.0)
        required_types = set(node.agent_type_reqs) if node.agent_type_reqs else None

        # Earliest time all predecessors are done
        pred_finish = max(
            (task_end[p] for p in preds.get(nid, []) if p in task_end),
            default=0.0,
        )

        best_agent: Optional[str] = None
        best_eft: float = float("inf")
        best_start: float = 0.0

        for a in agents:
            # Type compatibility check
            if required_types and a.agent_type not in required_types:
                continue

            # EFT = max(agent_available, pred_finish) + travel + duration
            earliest = max(agent_available[a.id], pred_finish)
            travel = _travel_time(agent_position[a.id], task_loc, a.speed_m_per_s)
            start = earliest + travel
            eft = start + dur

            if eft < best_eft:
                best_eft = eft
                best_agent = a.id
                best_start = start

        if best_agent is None:
            # Fallback: assign to first agent (ignore type reqs)
            best_agent = agents[0].id
            earliest = max(agent_available[best_agent], pred_finish)
            best_start = earliest
            best_eft = best_start + dur

        # Update state
        agent_available[best_agent] = best_eft
        agent_position[best_agent] = task_loc
        task_end[nid] = best_eft

        # Build ScheduleItem with float times (matching greedy's int-ceil pattern)
        final_start = int(math.ceil(best_start))
        final_end = int(math.ceil(best_eft))
        if final_end < final_start:
            final_end = final_start

        items.append(ScheduleItem(
            node_id=nid,
            agent=best_agent,
            start_s=final_start,
            end_s=final_end,
        ))

    makespan = max(it.end_s for it in items) if items else 0

    return Schedule(
        mission_id=task_graph.mission_id,
        makespan_s=makespan,
        items=items,
    )
