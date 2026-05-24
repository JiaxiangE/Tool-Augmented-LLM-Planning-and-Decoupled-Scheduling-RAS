"""
CP-SAT Exact Scheduler — Performance Upper Bound.

Uses Google OR-Tools CP-SAT solver for optimal task-agent assignment
with precedence, type-compatibility, non-overlap, inter-task travel,
and makespan minimization constraints. Travel time between consecutive
tasks on the same agent is modelled with per-agent AddCircuit constraints,
so the agent traverses to the next task immediately after completing
the previous one.

Only practical for small scenarios (<= 8 agents, <= 30 tasks).
Default timeout: 120s (returns best feasible solution found).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState
from core.schema.taskgraph import TaskGraph, TaskNode, Edge, EdgeType
from core.scheduler.types import Schedule, ScheduleItem

logger = logging.getLogger(__name__)

# Maximum problem size limits
MAX_CPSAT_TASKS = 30
MAX_CPSAT_AGENTS = 8
DEFAULT_TIMEOUT_S = 120
DEFAULT_SPEED_M_PER_S = 0.5


class CPSATAgent:
    """Agent descriptor for CP-SAT scheduling."""

    def __init__(
        self,
        id: str,
        agent_type: AgentType,
        energy_wh: float = 5000.0,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        speed_m_per_s: float = DEFAULT_SPEED_M_PER_S,
    ):
        self.id = id
        self.agent_type = agent_type
        self.energy_wh = energy_wh
        self.position = position
        self.speed_m_per_s = speed_m_per_s


def _travel_int(p1, p2, speed: float) -> int:
    """Ceiling-int travel seconds (CP-SAT needs integer constants)."""
    dist = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p1, p2)))
    return int(math.ceil(dist / max(speed, 1e-9)))


def schedule(
    task_graph: TaskGraph,
    agents: List[CPSATAgent],
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Optional[Schedule]:
    """
    Solve the scheduling problem exactly using CP-SAT, with travel.

    Model:
      Variables
        x[t, a]    : Boolean, task t assigned to agent a
        start[t]   : IntVar, start time of task t (seconds)
        end[t]     : IntVar, end[t] = start[t] + duration[t]
        makespan   : IntVar = max(end[t])
        per agent: a directed graph of {task indices + dummy depot}, with
                   arc literals; AddCircuit forces a Hamiltonian circuit,
                   with skip-self-loops for tasks not assigned to this agent.
      Constraints
        - Exactly one agent per task
        - Type compatibility (x[t, a] = 0 if agent type not in t.agent_type_reqs)
        - Precedence: ORDER/DATA edge (src, dst) ⇒ start[dst] >= end[src]
        - Per-agent NoOverlap implied by Hamiltonian circuit on intervals
        - Per-agent transition (travel) baked into arc constraints:
            depot→t   ⇒ start[t]   >= travel(agent.init_pos, t.loc)
            i→j      ⇒ start[j]   >= end[i] + travel(t_i.loc, t_j.loc)
      Objective: minimize makespan

    Args:
        task_graph: TaskGraph to schedule.
        agents: List of agents (must carry position + speed for travel).
        timeout_s: Solver timeout in seconds.

    Returns:
        Optimal (or best feasible) Schedule, or None if infeasible/timeout.
    """
    num_tasks = len(task_graph.nodes)
    num_agents = len(agents)

    if num_tasks > MAX_CPSAT_TASKS or num_agents > MAX_CPSAT_AGENTS:
        logger.warning(
            f"Problem too large for CP-SAT: {num_tasks} tasks, {num_agents} agents. "
            f"Limit: {MAX_CPSAT_TASKS} tasks, {MAX_CPSAT_AGENTS} agents."
        )
        return None

    model = cp_model.CpModel()
    nodes = task_graph.nodes
    node_to_idx = {n.id: i for i, n in enumerate(nodes)}
    durations = [int(n.estimated_duration_s or 60) for n in nodes]

    # Generous horizon — travel can add a lot relative to plain duration
    # (e.g. large_c chain has 1200m diagonal with 0.5 m/s = 2400s/edge)
    horizon = sum(durations) * 3 + 50000

    # ─── Decision variables ──────────────────────────────────────────
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for t in range(num_tasks):
        for a in range(num_agents):
            x[t, a] = model.NewBoolVar(f"x_t{t}_a{a}")

    start = {t: model.NewIntVar(0, horizon, f"start_t{t}") for t in range(num_tasks)}
    end = {t: model.NewIntVar(0, horizon, f"end_t{t}") for t in range(num_tasks)}
    for t in range(num_tasks):
        model.Add(end[t] == start[t] + durations[t])

    # ─── Each task to exactly one agent ──────────────────────────────
    for t in range(num_tasks):
        model.AddExactlyOne(x[t, a] for a in range(num_agents))

    # ─── Type compatibility ─────────────────────────────────────────
    for t, node in enumerate(nodes):
        for a, agent in enumerate(agents):
            if node.agent_type_reqs and agent.agent_type not in node.agent_type_reqs:
                model.Add(x[t, a] == 0)

    # ─── Precedence (ORDER/DATA edges) ──────────────────────────────
    for edge in task_graph.edges:
        if edge.type in (EdgeType.ORDER, EdgeType.DATA):
            si = node_to_idx.get(edge.src)
            di = node_to_idx.get(edge.dst)
            if si is not None and di is not None:
                model.Add(start[di] >= end[si])

    # ─── Per-agent travel-aware NoOverlap via AddCircuit ─────────────
    # For each agent a, define a directed graph with nodes:
    #     [0..num_tasks-1]: type-compatible task indices
    #     num_tasks       : dummy depot (agent's initial position)
    # Arcs:
    #     depot→task  : "task is first on agent a"  (literal: first_a_t)
    #     task→depot  : "task is last on agent a"   (literal: last_a_t)
    #     i→j (i≠j)   : "task j immediately follows task i on agent a"
    #     i→i         : skip self-loop, literal = NOT x[i, a]
    #     depot→depot : taken iff NO task on agent a (literal: NOT any_on_a)
    # AddCircuit enforces a single Hamiltonian circuit, so each task is either
    # part of the chain (visited via in-arc + out-arc) or skipped (self-loop).
    for a, agent in enumerate(agents):
        speed = float(getattr(agent, "speed_m_per_s", DEFAULT_SPEED_M_PER_S)
                      or DEFAULT_SPEED_M_PER_S)
        init_pos = tuple(getattr(agent, "position", (0.0, 0.0, 0.0)))

        cands = [t for t, node in enumerate(nodes) if not node.agent_type_reqs
                 or agent.agent_type in node.agent_type_reqs]
        if not cands:
            continue

        depot = num_tasks
        arcs: List[Tuple[int, int, cp_model.IntVar]] = []

        # any_on_a = OR over x[t, a] for t in cands
        any_on_a = model.NewBoolVar(f"any_on_a{a}")
        model.AddBoolOr([x[t, a] for t in cands]).OnlyEnforceIf(any_on_a)
        model.AddBoolAnd([x[t, a].Not() for t in cands]).OnlyEnforceIf(any_on_a.Not())

        # Depot self-loop = NOT any_on_a (taken if no task on this agent)
        arcs.append((depot, depot, any_on_a.Not()))

        # Per-task self-loop = NOT x[t, a] (taken if t not on this agent)
        for t in cands:
            arcs.append((t, t, x[t, a].Not()))

        # depot → task : "task t is first"
        first_lits = {}
        for t in cands:
            ft = model.NewBoolVar(f"first_a{a}_t{t}")
            first_lits[t] = ft
            model.AddImplication(ft, x[t, a])
            tt = _travel_int(init_pos, nodes[t].location or (0, 0, 0), speed)
            if tt > 0:
                model.Add(start[t] >= tt).OnlyEnforceIf(ft)
            arcs.append((depot, t, ft))

        # task → depot : "task t is last"
        for t in cands:
            lt = model.NewBoolVar(f"last_a{a}_t{t}")
            model.AddImplication(lt, x[t, a])
            arcs.append((t, depot, lt))

        # task i → task j  (i ≠ j) : "j immediately follows i"
        for i in cands:
            for j in cands:
                if i == j:
                    continue
                aij = model.NewBoolVar(f"arc_a{a}_t{i}_t{j}")
                model.AddImplication(aij, x[i, a])
                model.AddImplication(aij, x[j, a])
                tt = _travel_int(
                    nodes[i].location or (0, 0, 0),
                    nodes[j].location or (0, 0, 0),
                    speed,
                )
                if tt > 0:
                    model.Add(start[j] >= end[i] + tt).OnlyEnforceIf(aij)
                else:
                    model.Add(start[j] >= end[i]).OnlyEnforceIf(aij)
                arcs.append((i, j, aij))

        # Hamiltonian circuit constraint
        model.AddCircuit(arcs)

    # ─── Makespan + objective ────────────────────────────────────────
    makespan = model.NewIntVar(0, horizon, "makespan")
    for t in range(num_tasks):
        model.Add(makespan >= end[t])
    model.Minimize(makespan)

    # ─── Solve ───────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = 4

    logger.info(
        f"CP-SAT solving (travel-aware): {num_tasks} tasks, {num_agents} agents, "
        f"timeout={timeout_s}s"
    )

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        logger.info(f"CP-SAT: OPTIMAL, makespan={solver.Value(makespan)}s, "
                    f"wall={solver.WallTime():.2f}s")
    elif status == cp_model.FEASIBLE:
        logger.info(f"CP-SAT: FEASIBLE (best found), makespan={solver.Value(makespan)}s, "
                    f"wall={solver.WallTime():.2f}s")
    else:
        logger.warning(f"CP-SAT: No feasible solution (status={solver.StatusName(status)}, "
                       f"wall={solver.WallTime():.2f}s)")
        return None

    # ─── Extract solution ────────────────────────────────────────────
    items = []
    for t in range(num_tasks):
        assigned_agent = None
        for a in range(num_agents):
            if solver.Value(x[t, a]) == 1:
                assigned_agent = agents[a].id
                break
        items.append(ScheduleItem(
            node_id=nodes[t].id,
            agent=assigned_agent or agents[0].id,
            start_s=float(solver.Value(start[t])),
            end_s=float(solver.Value(end[t])),
        ))

    optimal_makespan = float(solver.Value(makespan))
    sched = Schedule(
        mission_id=getattr(task_graph, "mission_id", "cpsat_schedule"),
        items=items,
        makespan_s=optimal_makespan,
    )
    return sched
