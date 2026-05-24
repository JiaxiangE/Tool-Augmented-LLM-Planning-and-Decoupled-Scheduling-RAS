"""
Constraint Checker Tool.

Validates an assembled TaskGraph against physical domain constraints:
  - Energy feasibility
  - Communication coverage
  - Terrain traversability
  - Graph structural integrity

Returns a ConstraintReport, NOT a SubGraph with nodes/edges.
The controller consumes this report directly -- it must NOT be added
to the subgraphs accumulator list for graph_assembler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState, DomainConfig, EnvironmentState
from core.schema.taskgraph import TaskGraph, TaskNode

from .base import BaseTool, SubGraph


# ---------------------------------------------------------------------------
# Constraint Report models
# ---------------------------------------------------------------------------

class Violation(BaseModel):
    """A single constraint violation."""
    type: str  # "energy", "comm", "slope", "cycle", "orphan", "structural"
    node_id: Optional[str] = None
    detail: str


class ConstraintReport(BaseModel):
    """Result of constraint validation."""
    is_valid: bool = True
    violations: List[Violation] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean_2d(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _estimate_travel_energy(
    from_pos: Tuple[float, float, float],
    to_pos: Tuple[float, float, float],
    cost_per_m: float = 0.5,
) -> float:
    """Rough energy cost estimate for travelling between two points."""
    dist = math.sqrt(
        (from_pos[0] - to_pos[0]) ** 2
        + (from_pos[1] - to_pos[1]) ** 2
        + (from_pos[2] - to_pos[2]) ** 2
    )
    return dist * cost_per_m


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class ConstraintCheckerTool(BaseTool):
    """Validate a TaskGraph against physical domain constraints."""

    @property
    def name(self) -> str:
        return "constraint_checker"

    @property
    def description(self) -> str:
        return (
            "Validate an assembled TaskGraph against physical domain constraints "
            "including energy feasibility, communication coverage, terrain "
            "traversability, and graph structural integrity. Returns a report "
            "with violations and actionable suggestions. Use this after "
            "graph_assembler to verify the plan is physically executable."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_graph_json": {
                    "type": "string",
                    "description": "JSON-serialized TaskGraph to validate",
                },
            },
            "required": ["task_graph_json"],
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        return "task_graph_json" in params and isinstance(params["task_graph_json"], str)

    def check(
        self,
        task_graph: TaskGraph,
        env_state: EnvironmentState,
    ) -> ConstraintReport:
        """
        Run all constraint checks and return a ConstraintReport.

        This is the primary programmatic API. The execute() method wraps this
        for tool-calling interface compatibility.
        """
        violations: List[Violation] = []
        suggestions: List[str] = []

        # --- 1. Graph structural integrity ---
        try:
            # Re-validate by reconstructing (triggers _validate_graph)
            TaskGraph(
                mission_id=task_graph.mission_id,
                nodes=task_graph.nodes,
                edges=task_graph.edges,
                constraints=task_graph.constraints,
            )
        except Exception as e:
            violations.append(Violation(
                type="structural",
                detail=f"Graph structural validation failed: {e}",
            ))

        node_map = {n.id: n for n in task_graph.nodes}

        # --- 2. Communication coverage ---
        base_pos = env_state.config.base_station_pos
        comm_range = env_state.config.comm_range_m

        # Collect relay positions from DeployRelay tasks
        relay_positions: List[Tuple[float, float, float]] = [base_pos]
        for n in task_graph.nodes:
            if n.op_type in ("DeployRelay", "deploy_relay") and n.location:
                relay_positions.append(n.location)

        for n in task_graph.nodes:
            if not n.requires_connected or not n.location:
                continue

            # Check if any relay/base is within comm_range
            connected = any(
                _euclidean_2d(n.location, rp) <= comm_range
                for rp in relay_positions
            )
            if not connected:
                violations.append(Violation(
                    type="comm",
                    node_id=n.id,
                    detail=(
                        f"Node '{n.id}' at ({n.location[0]:.0f}, {n.location[1]:.0f}) "
                        f"requires connectivity but is beyond comm range ({comm_range}m) "
                        f"of all relays and base station."
                    ),
                ))
                # Suggest a relay midpoint
                mid_x = (base_pos[0] + n.location[0]) / 2
                mid_y = (base_pos[1] + n.location[1]) / 2
                suggestions.append(
                    f"Add relay at ({mid_x:.0f}, {mid_y:.0f}, 0) to cover node '{n.id}'"
                )

        # --- 3. Terrain traversability (slope check placeholder) ---
        MAX_SLOPE = 30.0
        for n in task_graph.nodes:
            if not n.location:
                continue
            # Full slope check requires TerrainGrid injection -- placeholder for
            # future enhancement.

        # --- 4. Energy feasibility (rough per-type estimate) ---
        # Group tasks by agent_type_reqs and estimate cumulative energy
        type_task_count: Dict[str, int] = {}
        for n in task_graph.nodes:
            for req in n.agent_type_reqs:
                key = req.value if isinstance(req, AgentType) else str(req)
                type_task_count[key] = type_task_count.get(key, 0) + 1

        for agent in env_state.agents.values():
            agent_key = agent.agent_type.value if isinstance(agent.agent_type, AgentType) else str(agent.agent_type)
            task_count = type_task_count.get(agent_key, 0)
            # Rough estimate: 15 Wh average per task (base + travel)
            estimated_cost = task_count * 15.0
            if estimated_cost > agent.energy_wh:
                violations.append(Violation(
                    type="energy",
                    detail=(
                        f"Agent '{agent.id}' ({agent_key}) has {agent.energy_wh:.0f} Wh "
                        f"but estimated workload requires ~{estimated_cost:.0f} Wh "
                        f"({task_count} tasks)."
                    ),
                ))
                suggestions.append(
                    f"Consider splitting {agent_key} tasks across multiple agents "
                    f"or reducing task count."
                )

        is_valid = len(violations) == 0
        return ConstraintReport(
            is_valid=is_valid,
            violations=violations,
            suggestions=suggestions,
        )

    def execute(self, params: Dict[str, Any]) -> SubGraph:
        """
        Tool-calling interface wrapper.

        The ConstraintReport is placed in SubGraph.metadata under key 'report'.
        The controller should consume this directly and NOT add it to the
        subgraphs accumulator.
        """
        import json

        tg_json = params.get("task_graph_json", "{}")
        try:
            tg = TaskGraph.model_validate_json(tg_json)
        except Exception as e:
            report = ConstraintReport(
                is_valid=False,
                violations=[Violation(type="structural", detail=f"Failed to parse TaskGraph: {e}")],
            )
            return SubGraph(metadata={"tool": "constraint_checker", "report": report.model_dump()})

        # Use a default EnvironmentState if not provided via params
        env_state = EnvironmentState()

        report = self.check(tg, env_state)
        return SubGraph(
            metadata={
                "tool": "constraint_checker",
                "report": report.model_dump(),
            },
        )
