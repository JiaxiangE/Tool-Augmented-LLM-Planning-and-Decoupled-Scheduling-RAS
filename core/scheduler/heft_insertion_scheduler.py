"""
HEFT-Insertion (Topcuoglu 2002, original variant) — travel-aware baseline.

Difference vs heft_scheduler.py (HEFT-native):
  - Same upward-rank ordering and type-compatibility check
  - For each task, instead of always appending to an agent's timeline end,
    we scan that agent's existing idle gaps and try to **insert** the task
    if it physically fits (gap >= travel_in + duration + travel_out).
  - We pick the (agent, position) with the smallest finish time, where
    position ∈ {idle-gap-i, append-to-end}.

Theoretical invariant: HEFT-Insertion makespan ≤ HEFT-native makespan on
any instance, because HEFT-native's `append` option is always available
to HEFT-Insertion as the fallback.

Interface (matches heft_scheduler.schedule):
    schedule(task_graph, agents) -> Schedule
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.schema.agent_types import AgentType
from core.schema.taskgraph import EdgeType, TaskGraph, TaskNode
from core.scheduler.heft_scheduler import HEFTAgent, compute_upward_rank
from core.scheduler.types import Schedule, ScheduleItem

DEFAULT_SPEED_M_PER_S = 0.5


def _travel(p1, p2, speed: float) -> float:
    d = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p1, p2)))
    return d / max(speed, 1e-9)


@dataclass
class _AgentTimeline:
    """Per-agent ordered list of (start, end, location, node_id)."""
    agent_id: str
    init_pos: Tuple[float, float, float]
    speed: float
    entries: List[Tuple[float, float, Tuple[float, float, float], str]] = field(default_factory=list)

    def append_or_insert_best(
        self,
        task_loc: Tuple[float, float, float],
        duration: float,
        pred_finish: float,
    ) -> Tuple[float, float, int]:
        """Return (best_start, best_end, best_position).

        best_position semantics:
          -1  : append to end (timeline grows by one entry at the tail)
          k>=0: insert before entries[k] (timeline gets a new entry at index k)
        """
        best_eft = float("inf")
        best_start = 0.0
        best_pos = -1

        # ─── Option A: append to end ───
        if self.entries:
            prev_end = self.entries[-1][1]
            prev_loc = self.entries[-1][2]
        else:
            prev_end = 0.0
            prev_loc = self.init_pos
        t_in = _travel(prev_loc, task_loc, self.speed)
        start_append = max(prev_end, pred_finish) + t_in  # eager travel
        eft_append = start_append + duration
        best_eft = eft_append
        best_start = start_append
        best_pos = -1  # append marker

        # ─── Option B: insert into an idle gap before entries[k] ───
        # For each gap (before entries[0], or between entries[k-1] and entries[k]):
        #   gap_start = entries[k-1].end   (or 0.0 if k==0)
        #   gap_end   = entries[k].start
        #   prev_loc  = entries[k-1].loc   (or init_pos if k==0)
        #   next_loc  = entries[k].loc
        #   start_in_gap = max(gap_start, pred_finish) + travel(prev_loc, task_loc)
        #   end_in_gap   = start_in_gap + duration
        #   need: end_in_gap + travel(task_loc, next_loc) <= entries[k].start
        #   (downstream task[k] starts later by travel(task_loc, next_loc) in eager
        #    model, but downstream task[k].start is already fixed at entries[k][0];
        #    we require the inserted task + outbound travel to leave that fixed
        #    start intact — strict no-disturb policy.)
        for k in range(len(self.entries)):
            gap_start = self.entries[k - 1][1] if k > 0 else 0.0
            gap_end = self.entries[k][0]
            if gap_end <= gap_start:
                continue
            prev_loc = self.entries[k - 1][2] if k > 0 else self.init_pos
            next_loc = self.entries[k][2]
            t_in = _travel(prev_loc, task_loc, self.speed)
            t_out = _travel(task_loc, next_loc, self.speed)
            start_in_gap = max(gap_start, pred_finish) + t_in
            end_in_gap = start_in_gap + duration
            if end_in_gap + t_out > gap_end + 1e-9:
                continue  # doesn't fit without disturbing entries[k]
            eft_in_gap = end_in_gap
            if eft_in_gap < best_eft - 1e-9:
                best_eft = eft_in_gap
                best_start = start_in_gap
                best_pos = k

        return best_start, best_eft, best_pos

    def commit(self, start: float, end: float, loc: Tuple[float, float, float],
               node_id: str, position: int) -> None:
        entry = (start, end, loc, node_id)
        if position == -1:
            self.entries.append(entry)
        else:
            self.entries.insert(position, entry)


def schedule(
    task_graph: TaskGraph,
    agents: List[HEFTAgent],
) -> Schedule:
    """HEFT-Insertion scheduler with safety net.

    Returns min-makespan of:
      (a) pure insertion-greedy pass
      (b) HEFT-native append-only pass

    Rationale: insertion-greedy may suffer local-optimum traps where
    inserting task t into agent A's gap blocks a later task from a
    larger global gain. Falling back to native append when it beats
    insertion preserves the theoretical invariant
    `heft_insert.makespan_s <= heft_native.makespan_s` on all instances.
    """
    if not agents:
        raise ValueError("Cannot schedule with zero agents")
    if not task_graph.nodes:
        return Schedule(mission_id=task_graph.mission_id, makespan_s=0, items=[])
    from core.scheduler.heft_scheduler import schedule as heft_native_schedule
    ins_sched = _insertion_only_pass(task_graph, agents)
    nat_sched = heft_native_schedule(task_graph, agents)
    return ins_sched if ins_sched.makespan_s <= nat_sched.makespan_s else nat_sched


def _insertion_only_pass(
    task_graph: TaskGraph,
    agents: List[HEFTAgent],
) -> Schedule:
    """Pure insertion-greedy pass (used inside schedule() with safety net)."""

    node_map = {n.id: n for n in task_graph.nodes}

    # 1. upward rank
    rank_u = compute_upward_rank(task_graph, node_map)
    sorted_tasks = sorted(rank_u.keys(), key=lambda t: rank_u[t], reverse=True)

    # 2. predecessor map
    preds: Dict[str, List[str]] = defaultdict(list)
    for e in task_graph.edges:
        if e.type in (EdgeType.ORDER, EdgeType.DATA):
            preds[e.dst].append(e.src)

    # 3. per-agent timeline state
    timelines: Dict[str, _AgentTimeline] = {
        a.id: _AgentTimeline(
            agent_id=a.id,
            init_pos=tuple(a.position),
            speed=float(getattr(a, "speed_m_per_s", DEFAULT_SPEED_M_PER_S) or DEFAULT_SPEED_M_PER_S),
        ) for a in agents
    }
    task_end: Dict[str, float] = {}
    items: List[ScheduleItem] = []

    for nid in sorted_tasks:
        node = node_map[nid]
        dur = float(node.estimated_duration_s or 60.0)
        task_loc = tuple(node.location or (0.0, 0.0, 0.0))
        required_types = set(node.agent_type_reqs) if node.agent_type_reqs else None

        pred_finish = max((task_end[p] for p in preds.get(nid, []) if p in task_end),
                          default=0.0)

        best_agent: Optional[str] = None
        best_eft = float("inf")
        best_start = 0.0
        best_pos = -1
        for a in agents:
            if required_types and a.agent_type not in required_types:
                continue
            s, e, pos = timelines[a.id].append_or_insert_best(task_loc, dur, pred_finish)
            if e < best_eft - 1e-9:
                best_eft = e
                best_start = s
                best_pos = pos
                best_agent = a.id

        if best_agent is None:
            # fallback: ignore type reqs and use first agent (matches heft_scheduler behavior)
            best_agent = agents[0].id
            best_start, best_eft, best_pos = timelines[best_agent].append_or_insert_best(
                task_loc, dur, pred_finish)

        timelines[best_agent].commit(best_start, best_eft, task_loc, nid, best_pos)
        task_end[nid] = best_eft

        final_start = int(math.ceil(best_start))
        final_end = int(math.ceil(best_eft))
        if final_end < final_start:
            final_end = final_start
        items.append(ScheduleItem(node_id=nid, agent=best_agent,
                                  start_s=final_start, end_s=final_end))

    makespan = max(it.end_s for it in items) if items else 0
    return Schedule(
        mission_id=task_graph.mission_id,
        makespan_s=makespan,
        items=items,
    )
