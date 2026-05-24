"""
Greedy baseline scheduler.

Algorithm:
  1. Topological sort (Kahn) on ORDER + DATA edges
  2. CPM forward pass → compute EST (Earliest Start Time) for each task
  3. MUTEX linearization based on EST ordering
  4. Greedy assignment: each ready task → earliest-idle agent whose AgentType
     satisfies agent_type_reqs

Interface:
  schedule(task_graph, agents) → Schedule

Works directly with TaskNode.location and TaskNode.estimated_duration_s
(no external world model required).
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.schema.agent_types import AgentType
from core.schema.taskgraph import Edge, EdgeType, TaskGraph, TaskNode
from core.scheduler.types import Schedule, ScheduleItem


# ---------------------------------------------------------------------------
# Agent descriptor
# ---------------------------------------------------------------------------

@dataclass
class AgentInfo:
    """Static description of an agent available for scheduling."""
    id: str
    agent_type: AgentType
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    energy_wh: float = 100.0
    speed_m_per_s: float = 0.5  # average rover speed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _topo_sort_kahn(
    node_ids: List[str],
    edges: List[Edge],
) -> List[str]:
    """
    Kahn's algorithm topological sort.

    Considers ORDER and DATA edges as precedence constraints.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    in_deg: Dict[str, int] = {nid: 0 for nid in node_ids}

    for e in edges:
        if e.type in (EdgeType.ORDER, EdgeType.DATA):
            adj[e.src].append(e.dst)
            in_deg[e.dst] = in_deg.get(e.dst, 0) + 1

    queue = deque(nid for nid in node_ids if in_deg[nid] == 0)
    result: List[str] = []

    while queue:
        n = queue.popleft()
        result.append(n)
        for succ in adj[n]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    if len(result) != len(node_ids):
        raise ValueError(
            f"Topological sort failed: graph has a cycle. "
            f"Sorted {len(result)}/{len(node_ids)} nodes."
        )

    return result


def _priority_topo_sort(
    node_ids: List[str],
    edges: List[Edge],
    priority: Dict[str, int],
) -> List[str]:
    """Topological sort honoring a priority hint.

    Like Kahn's algorithm, but the ready set is a min-heap keyed by
    `priority[task]` (lower rank = scheduled earlier when topology permits).
    Tasks absent from `priority` get the lowest preference. Precedence edges
    (ORDER/DATA) are always respected — the priority hint can only reorder
    tasks that are *simultaneously* ready.

    Used by the hybrid scheduler: the learned policy supplies a global task
    ordering as the priority hint; energy-aware agent assignment is delegated
    to `schedule()`.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    in_deg: Dict[str, int] = {nid: 0 for nid in node_ids}
    for e in edges:
        if e.type in (EdgeType.ORDER, EdgeType.DATA):
            adj[e.src].append(e.dst)
            in_deg[e.dst] = in_deg.get(e.dst, 0) + 1

    default_rank = len(node_ids) + 1
    # Heap entries (rank, task_id); task_id breaks ties deterministically.
    heap: List[Tuple[int, str]] = [
        (priority.get(nid, default_rank), nid)
        for nid in node_ids if in_deg[nid] == 0
    ]
    heapq.heapify(heap)

    result: List[str] = []
    while heap:
        _, n = heapq.heappop(heap)
        result.append(n)
        for succ in adj[n]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                heapq.heappush(heap, (priority.get(succ, default_rank), succ))

    if len(result) != len(node_ids):
        raise ValueError(
            f"Priority topological sort failed: graph has a cycle. "
            f"Sorted {len(result)}/{len(node_ids)} nodes."
        )
    return result


def _compute_est(
    topo_order: List[str],
    node_map: Dict[str, TaskNode],
    edges: List[Edge],
) -> Dict[str, float]:
    """
    CPM forward pass: compute Earliest Start Time for each task.

    EST[j] = max(EST[i] + duration[i])  for all i → j (ORDER/DATA edges)

    Duration comes from TaskNode.estimated_duration_s.
    If not set, defaults to 60s.
    """
    est: Dict[str, float] = {nid: 0.0 for nid in topo_order}

    # Build predecessor map
    preds: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e.type in (EdgeType.ORDER, EdgeType.DATA):
            preds[e.dst].append(e.src)

    for nid in topo_order:
        for pred_id in preds.get(nid, []):
            pred_dur = node_map[pred_id].estimated_duration_s or 60.0
            est[nid] = max(est[nid], est[pred_id] + pred_dur)

    return est


def _linearize_mutex(
    est: Dict[str, float],
    node_map: Dict[str, TaskNode],
    edges: List[Edge],
) -> Dict[str, float]:
    """
    Resolve MUTEX edges: force serial execution ordered by EST.

    For each MUTEX edge (Ti, Tj):
      - Whichever has lower EST goes first.
      - Second's EST is pushed to max(its EST, first's EST + first's duration).
    """
    for e in edges:
        if e.type != EdgeType.MUTEX:
            continue

        ti, tj = e.src, e.dst
        dur_i = node_map[ti].estimated_duration_s or 60.0
        dur_j = node_map[tj].estimated_duration_s or 60.0

        if est[ti] <= est[tj]:
            # Ti goes first
            est[tj] = max(est[tj], est[ti] + dur_i)
        else:
            # Tj goes first
            est[ti] = max(est[ti], est[tj] + dur_j)

    return est


def _travel_time(
    from_pos: Tuple[float, float, float],
    to_pos: Tuple[float, float, float],
    speed: float,
) -> float:
    """Compute travel time between two 3D positions at given speed."""
    dist = math.sqrt(
        (from_pos[0] - to_pos[0]) ** 2
        + (from_pos[1] - to_pos[1]) ** 2
        + (from_pos[2] - to_pos[2]) ** 2
    )
    return dist / speed if speed > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def schedule(
    task_graph: TaskGraph,
    agents: List[AgentInfo],
    task_order: Optional[List[str]] = None,
) -> Schedule:
    """
    Greedy scheduler: produce a valid Schedule from a TaskGraph and agent list.

    Algorithm:
      1. Topological sort
      2. CPM forward pass → EST
      3. MUTEX linearization
      4. Greedy assignment: for each task in topo order, assign to the
         earliest-available agent that satisfies agent_type_reqs.

    Args:
        task_graph: TaskGraph to schedule.
        agents: List of available agents.
        task_order: optional task-order hint (used by the hybrid scheduler).
            When given, step 1 uses a priority topological sort that follows
            this order among simultaneously-ready tasks (precedence edges are
            still enforced). When None, the default Kahn order is used.

    Returns:
        A valid Schedule object.
    """
    if not agents:
        raise ValueError("Cannot schedule with zero agents")

    node_ids = [n.id for n in task_graph.nodes]
    node_map = {n.id: n for n in task_graph.nodes}

    # 1. Topological sort (priority-hinted if task_order given)
    if task_order is not None:
        priority = {tid: i for i, tid in enumerate(task_order)}
        topo_order = _priority_topo_sort(node_ids, task_graph.edges, priority)
    else:
        topo_order = _topo_sort_kahn(node_ids, task_graph.edges)

    # 2. CPM forward pass
    est = _compute_est(topo_order, node_map, task_graph.edges)

    # 3. MUTEX linearization
    est = _linearize_mutex(est, node_map, task_graph.edges)

    # 4. Greedy assignment
    # Build predecessor map for dynamic dependency tracking
    preds: Dict[str, List[str]] = defaultdict(list)
    for e in task_graph.edges:
        if e.type in (EdgeType.ORDER, EdgeType.DATA):
            preds[e.dst].append(e.src)

    agent_available_at: Dict[str, float] = {a.id: 0.0 for a in agents}
    agent_position: Dict[str, Tuple[float, float, float]] = {
        a.id: a.position for a in agents
    }
    agent_info_map = {a.id: a for a in agents}
    end_times: Dict[str, float] = {}  # node_id → end_s

    items: List[ScheduleItem] = []

    for nid in topo_order:
        node = node_map[nid]
        duration = node.estimated_duration_s or 60.0
        task_loc = node.location or (0.0, 0.0, 0.0)
        required_types = set(node.agent_type_reqs) if node.agent_type_reqs else None

        # Dynamic predecessor constraint: all preds must have finished
        pred_finish = max(
            (end_times[p] for p in preds.get(nid, []) if p in end_times),
            default=0.0,
        )

        # Find the earliest-available compatible agent
        best_agent: Optional[str] = None
        best_start: float = float("inf")

        for a in agents:
            # Check type compatibility
            if required_types and a.agent_type not in required_types:
                continue

            # Earliest this agent can start = max(agent_available, pred_finish, EST)
            earliest = max(agent_available_at[a.id], pred_finish, est[nid])

            # Add travel time
            travel = _travel_time(agent_position[a.id], task_loc, a.speed_m_per_s)
            start_at = earliest + travel

            if start_at < best_start:
                best_start = start_at
                best_agent = a.id

        if best_agent is None:
            # Fallback: assign to first agent (ignore type reqs)
            best_agent = agents[0].id
            best_start = max(agent_available_at[best_agent], pred_finish, est[nid])

        end_time = best_start + duration
        agent_available_at[best_agent] = end_time
        agent_position[best_agent] = task_loc
        end_times[nid] = end_time

        # Int-cast for ScheduleItem. Ensure start_s >= prev_end_s.
        final_start_s = int(math.ceil(best_start))
        final_end_s = int(math.ceil(end_time))
        # Edge case: if duration was 0 or rounding caused collapse
        if final_end_s < final_start_s:
            final_end_s = final_start_s

        items.append(ScheduleItem(
            node_id=nid,
            agent=best_agent,
            start_s=final_start_s,
            end_s=final_end_s,
        ))

    makespan = max(it.end_s for it in items) if items else 0

    return Schedule(
        mission_id=task_graph.mission_id,
        makespan_s=makespan,
        items=items,
    )


if __name__ == "__main__":
    from core.schema.taskgraph import Edge, EdgeType, TaskNode, TaskGraph
    from core.schema.environment import AgentState
    from core.simulator.terrain import TerrainGrid
    from core.simulator.lunar_sim import LunarSimulator, SimConfig

    print("--- Running Greedy Scheduler Smoke Test ---")
    
    # 1. Create a minimal task graph
    tg = TaskGraph(
        mission_id="smoke_test",
        nodes=[
            TaskNode(id="T1", op_type="ExploreCell", location=(10.0, 10.0, 0.0), estimated_duration_s=100, agent_type_reqs=[AgentType.ROVER]),
            TaskNode(id="T2", op_type="ExploreCell", location=(20.0, 20.0, 0.0), estimated_duration_s=100, agent_type_reqs=[AgentType.ROVER])
        ],
        edges=[Edge(id="E1", type=EdgeType.ORDER, src="T1", dst="T2")]
    )

    # 2. Define agents
    agents_info = [AgentInfo(id="rover_1", agent_type=AgentType.ROVER, position=(0.0, 0.0, 0.0))]
    
    # 3. Schedule
    sched = schedule(tg, agents_info)
    print(f"\nSchedule Output (Makespan: {sched.makespan_s}s):")
    for item in sched.items:
        print(f"  {item.node_id} -> {item.agent} [{item.start_s}s - {item.end_s}s]")
        
    # 4. Simulate
    sim_agents = [AgentState(id=a.id, position=a.position, energy_wh=a.energy_wh, initial_energy_wh=a.energy_wh, agent_type=a.agent_type) for a in agents_info]
    sim = LunarSimulator(TerrainGrid(10, 10, 10.0), sim_agents, SimConfig())
    result = sim.evaluate(tg, sched)
    
    print("\nSimulation Result:")
    print(f"  Completion Rate: {result.task_completion_rate * 100}%")
    print(f"  Energy Consumed: {result.energy_consumption}")
    print(f"  Bottleneck Tasks: {result.bottleneck_tasks}")
    print("-----------------------------------------")
