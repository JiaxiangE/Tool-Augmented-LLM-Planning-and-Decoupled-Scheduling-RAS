"""
SSI Auction Scheduler — Baseline.

Sequential Single-Item auction where agents bid on tasks in topological order.
Each agent bids inversely proportional to travel time + execution duration.
Type-incompatible agents bid 0. Highest bidder wins.

After assignment, CPM-based timing is used (same as greedy scheduler).
Returns a Schedule object.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from core.schema.agent_types import AgentType
from core.schema.taskgraph import TaskGraph, TaskNode, Edge, EdgeType
from core.scheduler.types import Schedule, ScheduleItem


# ---------------------------------------------------------------------------
# Agent info (reuse the same dataclass pattern)
# ---------------------------------------------------------------------------

class AuctionAgent:
    """Agent state for auction bidding."""

    def __init__(
        self,
        id: str,
        agent_type: AgentType,
        energy_wh: float = 5000.0,
        position: Tuple[float, float] = (0.0, 0.0),
    ):
        self.id = id
        self.agent_type = agent_type
        self.energy_wh = energy_wh
        self.position = position
        self.assigned_tasks: List[str] = []
        self.busy_until: float = 0.0

    def bid(self, task: TaskNode) -> float:
        """
        Compute bid value for a task.

        Higher bid = more willing to execute.
        Bid = 1 / (travel_time + execution_duration + queue_wait)
        Type-incompatible → bid 0.
        """
        # Type compatibility check
        if task.agent_type_reqs and self.agent_type not in task.agent_type_reqs:
            return 0.0

        # Task location
        task_pos = task.location or (0.0, 0.0, 0.0)
        dx = task_pos[0] - self.position[0]
        dy = task_pos[1] - self.position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Travel time (assume ~0.5 m/s rover speed)
        travel_time = distance / 0.5

        # Execution duration
        exec_time = task.estimated_duration_s or 60.0

        # Queue wait (how long until this agent is free)
        queue_wait = max(self.busy_until, 0.0)

        total_cost = travel_time + exec_time + queue_wait
        if total_cost <= 0:
            total_cost = 1.0

        return 1.0 / total_cost

    def accept_task(self, task: TaskNode, start_time: float):
        """Assign a task to this agent."""
        exec_time = task.estimated_duration_s or 60.0
        task_pos = task.location or (0.0, 0.0, 0.0)

        # Travel time to task
        dx = task_pos[0] - self.position[0]
        dy = task_pos[1] - self.position[1]
        travel = math.sqrt(dx * dx + dy * dy) / 0.5

        self.busy_until = max(start_time, self.busy_until) + travel + exec_time
        self.position = (task_pos[0], task_pos[1])
        self.assigned_tasks.append(task.id)


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def _topo_sort(task_graph: TaskGraph) -> List[str]:
    """Kahn's algorithm for topological ordering."""
    adj: Dict[str, List[str]] = {n.id: [] for n in task_graph.nodes}
    in_degree: Dict[str, int] = {n.id: 0 for n in task_graph.nodes}

    for edge in task_graph.edges:
        if edge.type in (EdgeType.ORDER, EdgeType.DATA):
            if edge.src in adj and edge.dst in in_degree:
                adj[edge.src].append(edge.dst)
                in_degree[edge.dst] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    result = []

    while queue:
        nid = queue.popleft()
        result.append(nid)
        for child in adj[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return result


# ---------------------------------------------------------------------------
# SSI Auction Scheduler
# ---------------------------------------------------------------------------

def schedule(
    task_graph: TaskGraph,
    agents: List[AuctionAgent],
) -> Schedule:
    """
    Run SSI auction to assign tasks, then compute CPM-based timing.

    Args:
        task_graph: TaskGraph to schedule.
        agents: List of AuctionAgent instances.

    Returns:
        Schedule with assignments and timing.
    """
    node_map = {n.id: n for n in task_graph.nodes}
    topo_order = _topo_sort(task_graph)

    # Predecessor map for timing
    predecessors: Dict[str, Set[str]] = {n.id: set() for n in task_graph.nodes}
    for edge in task_graph.edges:
        if edge.type in (EdgeType.ORDER, EdgeType.DATA):
            if edge.dst in predecessors:
                predecessors[edge.dst].add(edge.src)

    # Phase 1: Auction — assign tasks in topological order
    assignments: Dict[str, str] = {}  # task_id → agent_id
    earliest_start: Dict[str, float] = {}

    for task_id in topo_order:
        task = node_map[task_id]

        # Compute earliest start (CPM)
        pred_end = 0.0
        for pred_id in predecessors[task_id]:
            if pred_id in earliest_start:
                pred_node = node_map[pred_id]
                pred_dur = pred_node.estimated_duration_s or 60.0
                pred_end = max(pred_end, earliest_start[pred_id] + pred_dur)

        # Collect bids from all agents
        bids = [(agent, agent.bid(task)) for agent in agents]
        bids.sort(key=lambda x: x[1], reverse=True)

        # Winner takes all (highest bid with bid > 0)
        winner = None
        for agent, bid_val in bids:
            if bid_val > 0:
                winner = agent
                break

        if winner is None:
            # No agent can execute — assign to first agent as fallback
            winner = agents[0]

        start_time = max(pred_end, winner.busy_until)
        earliest_start[task_id] = start_time
        winner.accept_task(task, start_time)
        assignments[task_id] = winner.id

    # Phase 2: Build Schedule with timing
    items = []
    for task_id in topo_order:
        task = node_map[task_id]
        agent_id = assignments[task_id]
        start = earliest_start[task_id]
        dur = task.estimated_duration_s or 60.0

        items.append(ScheduleItem(
            node_id=task_id,
            agent=agent_id,
            start_s=start,
            end_s=start + dur,
        ))

    mission_id = task_graph.metadata.get("mission_id", "auction_schedule")
    makespan = max(it.end_s for it in items) if items else 0.0

    return Schedule(
        mission_id=mission_id,
        items=items,
        makespan_s=makespan,
    )
