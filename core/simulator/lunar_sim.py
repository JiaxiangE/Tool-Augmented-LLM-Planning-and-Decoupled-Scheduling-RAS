"""
Discrete-event simulator for evaluating task schedules on the lunar surface.

Core contract:
  evaluate(task_graph, schedule) → SimResult

Deterministic: identical inputs always produce identical outputs.
This is the "referee" — all scheduling quality is measured through this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.schema.environment import AgentState
from core.schema.taskgraph import EdgeType, TaskGraph, TaskNode
from core.scheduler.types import Schedule, ScheduleItem

from .comm_model import CommConfig, CommModel
from .energy_model import (EnergyConfig, EnergyModel, RechargeConfig,
                           RechargeEvent)
from .terrain import TerrainGrid


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """
    Immutable output of a single simulation run.

    This is the ONLY data structure that flows from the lower layer
    (scheduling + simulation) back to the upper layer (LLM planner).
    """
    makespan: float                                 # total completion time (s)
    task_completion_rate: float                      # fraction of tasks completed
    energy_consumption: Dict[str, float]             # agent_id → Wh consumed
    constraint_violations: List[str]                 # human-readable violation descriptions
    comm_connectivity_rate: float                    # fraction of connected time-steps
    bottleneck_tasks: List[str]                      # task IDs on the critical path
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    # Return-to-base recharge bookkeeping (empty when recharge disabled)
    recharge_events: List[RechargeEvent] = field(default_factory=list)
    total_recharge_time_s: float = 0.0
    n_recharges_per_agent: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulator configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Top-level configuration bundle for the simulator."""
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    recharge: RechargeConfig = field(default_factory=RechargeConfig)


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class LunarSimulator:
    """
    Deterministic discrete-event simulator.

    Walk through a schedule in chronological order, simulate energy use and
    communication, and emit a SimResult.
    """

    def __init__(
        self,
        terrain: TerrainGrid,
        agents: List[AgentState],
        config: Optional[SimConfig] = None,
    ):
        self.terrain = terrain
        self._initial_agents = {a.id: a for a in agents}
        self.cfg = config or SimConfig()
        self._energy = EnergyModel(self.cfg.energy)
        self._comm = CommModel(self.cfg.comm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, task_graph: TaskGraph, schedule: Schedule) -> SimResult:
        """
        Run a deterministic simulation and return the result.

        Steps:
          1. Sort schedule items by start_s.
          2. For each item, simulate travel → task execution.
          3. Track energy, connectivity, violations.
          4. Compute summary metrics.

        Args:
            task_graph: The task graph being scheduled.
            schedule: A valid Schedule that references nodes in task_graph.

        Returns:
            SimResult with all metrics.
        """
        # Build lookup maps
        node_map: Dict[str, TaskNode] = {n.id: n for n in task_graph.nodes}
        items_sorted = sorted(schedule.items, key=lambda it: (it.start_s, it.end_s))

        # Deep-copy agent states so the simulator is stateless across calls
        agents: Dict[str, AgentState] = {
            aid: AgentState(
                id=a.id,
                position=a.position,
                energy_wh=a.energy_wh,
                initial_energy_wh=a.initial_energy_wh,
                agent_type=a.agent_type,
            )
            for aid, a in self._initial_agents.items()
        }

        completed: List[str] = []
        failed: List[str] = []
        violations: List[str] = []
        connectivity_checks: List[bool] = []

        # --- Return-to-base recharge bookkeeping ---
        rc = self.cfg.recharge
        recharge_events: List[RechargeEvent] = []
        recharge_delay: Dict[str, float] = {aid: 0.0 for aid in agents}
        n_recharges: Dict[str, int] = {aid: 0 for aid in agents}

        # Track relay positions (deployed during simulation)
        relay_positions: List[Tuple[float, float, float]] = []

        # Identify relay agents that are static from the start
        for a in agents.values():
            if a.agent_type == "relay":
                relay_positions.append(a.position)

        # Walk the schedule
        for item in items_sorted:
            node = node_map.get(item.node_id)
            if node is None:
                violations.append(f"ScheduleItem references unknown node: {item.node_id}")
                failed.append(item.node_id)
                continue

            agent = agents.get(item.agent)
            if agent is None:
                violations.append(f"ScheduleItem references unknown agent: {item.agent}")
                failed.append(item.node_id)
                continue

            # --- Return-to-base recharge pre-check ---
            # If recharge is enabled and the agent's energy has dropped below
            # threshold_frac of its initial budget, insert a recharge cycle
            # (travel to base -> recharge -> travel back). The cycle's wall-clock
            # cost is accumulated into recharge_delay[agent] and reflected in the
            # final makespan. Recharge does not reorder tasks or add new tasks.
            if rc.enabled:
                initial_e = agent.initial_energy_wh
                if initial_e > 0 and agent.energy_wh < rc.threshold_frac * initial_e:
                    task_loc_rc = node.location or agent.position
                    spd = rc.speed_during_recharge_return_m_per_s or 0.5
                    d_to_base = self._energy.euclidean_distance(agent.position, rc.base_position)
                    t_to_base = d_to_base / spd if spd > 0 else 0.0
                    d_back = self._energy.euclidean_distance(rc.base_position, task_loc_rc)
                    t_back = d_back / spd if spd > 0 else 0.0
                    cycle_s = t_to_base + rc.recharge_duration_s + t_back
                    recharge_delay[agent.id] = recharge_delay.get(agent.id, 0.0) + cycle_s
                    n_recharges[agent.id] = n_recharges.get(agent.id, 0) + 1
                    recharge_events.append(RechargeEvent(
                        agent_id=agent.id,
                        before_task=item.node_id,
                        energy_before_wh=round(agent.energy_wh, 2),
                        travel_to_base_s=round(t_to_base, 2),
                        recharge_duration_s=rc.recharge_duration_s,
                        travel_back_s=round(t_back, 2),
                        total_cycle_s=round(cycle_s, 2),
                    ))
                    agent.energy_wh = initial_e          # recharged to full
                    agent.position = rc.base_position    # now at base; Travel phase below moves it to task

            # --- 1. Travel phase ---
            task_location = node.location or agent.position  # no movement if location unset
            travel_dist = self._energy.euclidean_distance(agent.position, task_location)
            slope = self.terrain.get_slope(task_location[0], task_location[1])
            travel_energy = self._energy.compute_travel_cost(travel_dist, slope)

            # Check terrain traversability
            if not self.terrain.is_traversable(task_location[0], task_location[1]):
                violations.append(
                    f"Task {node.id}: location ({task_location[0]:.0f}, {task_location[1]:.0f}) "
                    f"is not traversable (slope={slope:.1f}° or obstacle)"
                )
                failed.append(item.node_id)
                continue

            # --- 2. Energy feasibility ---
            task_energy = self._energy.compute_task_cost(node.op_type)
            total_energy = travel_energy + task_energy

            if not self._energy.check_energy_feasible(agent.energy_wh, total_energy):
                violations.append(
                    f"Task {node.id}: agent {agent.id} has insufficient energy "
                    f"({agent.energy_wh:.1f} Wh < {total_energy:.1f} Wh needed)"
                )
                failed.append(item.node_id)
                continue

            # Deduct energy
            agent.energy_wh -= total_energy
            agent.position = task_location

            # Low-energy warning (not a hard failure, but recorded)
            if self._energy.is_low_energy(agent.energy_wh):
                violations.append(
                    f"Task {node.id}: agent {agent.id} energy critically low "
                    f"({agent.energy_wh:.1f} Wh) — forced return recommended"
                )

            # --- 3. Communication check ---
            if node.requires_connected:
                is_conn = self._comm.is_connected(
                    agent.position, relay_positions, self.cfg.base_position
                )
                connectivity_checks.append(is_conn)
                if not is_conn:
                    violations.append(
                        f"Task {node.id}: agent {agent.id} has no comm connectivity "
                        f"at ({agent.position[0]:.0f}, {agent.position[1]:.0f})"
                    )
            else:
                # Still track connectivity for overall rate
                is_conn = self._comm.is_connected(
                    agent.position, relay_positions, self.cfg.base_position
                )
                connectivity_checks.append(is_conn)

            # --- 4. Task execution succeeds ---
            completed.append(item.node_id)

            # If this was a relay deployment, register the new relay
            if node.op_type in ("DeployRelay", "deploy_relay"):
                relay_positions.append(task_location)

        # --- Compute summary metrics ---
        total_tasks = len(task_graph.nodes)
        completion_rate = len(completed) / total_tasks if total_tasks > 0 else 0.0

        # When recharge inserted cycles, makespan is the schedule makespan
        # shifted by each agent's accumulated recharge delay. For an agent A, all
        # of A's tasks are pushed by A's total recharge delay; the agent's final
        # task therefore carries the full delay, and taking the max over all items
        # of (end_s + total_delay[agent]) yields the correct recharge-aware makespan.
        # When recharge is disabled (default), this reduces exactly to
        # schedule.makespan_s, preserving backward compatibility.
        if rc.enabled and recharge_events:
            makespan = max(
                (float(it.end_s) + recharge_delay.get(it.agent, 0.0)
                 for it in schedule.items),
                default=float(schedule.makespan_s),
            )
        else:
            makespan = float(schedule.makespan_s)

        energy_consumption = {
            aid: a.energy_consumed for aid, a in agents.items()
        }

        conn_rate = self._comm.connectivity_rate(connectivity_checks)

        # Bottleneck: tasks on the critical path (longest path in the ORDER DAG)
        bottleneck = self._find_critical_path(task_graph, schedule)

        return SimResult(
            makespan=makespan,
            task_completion_rate=completion_rate,
            energy_consumption=energy_consumption,
            constraint_violations=violations,
            comm_connectivity_rate=conn_rate,
            bottleneck_tasks=bottleneck,
            completed_tasks=completed,
            failed_tasks=failed,
            recharge_events=recharge_events,
            total_recharge_time_s=round(sum(e.total_cycle_s for e in recharge_events), 2),
            n_recharges_per_agent={k: v for k, v in n_recharges.items() if v > 0},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_critical_path(
        self, task_graph: TaskGraph, schedule: Schedule
    ) -> List[str]:
        """
        Identify the critical path — the longest chain of ORDER-dependent tasks.

        Uses a simple DP on topo-sorted ORDER edges with durations from the schedule.
        """
        # Build duration map from schedule
        duration_map: Dict[str, float] = {}
        for item in schedule.items:
            duration_map[item.node_id] = float(item.end_s - item.start_s)

        # Build adjacency from ORDER edges
        order_edges: Dict[str, List[str]] = {}
        all_nodes = {n.id for n in task_graph.nodes}
        in_degree: Dict[str, int] = {nid: 0 for nid in all_nodes}

        for e in task_graph.edges:
            if e.type == EdgeType.ORDER:
                order_edges.setdefault(e.src, []).append(e.dst)
                in_degree[e.dst] = in_degree.get(e.dst, 0) + 1

        # Topo sort (Kahn)
        queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
        topo_order: List[str] = []
        while queue:
            n = queue.pop(0)
            topo_order.append(n)
            for succ in order_edges.get(n, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        # DP: longest path
        dist: Dict[str, float] = {nid: 0.0 for nid in all_nodes}
        parent: Dict[str, Optional[str]] = {nid: None for nid in all_nodes}

        for n in topo_order:
            dur = duration_map.get(n, 0.0)
            for succ in order_edges.get(n, []):
                new_dist = dist[n] + dur
                if new_dist > dist[succ]:
                    dist[succ] = new_dist
                    parent[succ] = n

        # Trace back from the node with maximum dist + its own duration
        if not dist:
            return []

        end_node = max(dist, key=lambda nid: dist[nid] + duration_map.get(nid, 0.0))

        path: List[str] = [end_node]
        current = end_node
        while parent.get(current) is not None:
            current = parent[current]  # type: ignore
            path.append(current)

        path.reverse()
        return path
