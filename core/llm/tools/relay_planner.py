"""
Relay Planner Tool.

Calculates optimal relay deployment positions along the line-of-sight
path between a base station and a remote target, ensuring continuous
communication coverage.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from core.schema.agent_types import AgentType
from core.schema.taskgraph import Edge, EdgeType, TaskNode

from .base import BaseTool, SubGraph


def _euclidean_2d(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """2D Euclidean distance (ignoring z)."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class RelayPlannerTool(BaseTool):
    """Plan communication relay deployments to cover a remote target."""

    @property
    def name(self) -> str:
        return "relay_planner"

    @property
    def description(self) -> str:
        return (
            "Calculate relay deployment positions along the path from the base "
            "station to a remote target location. Generates DeployRelay tasks "
            "spaced within comm_range to ensure continuous communication coverage. "
            "Use this when tasks are planned beyond the base station's direct "
            "communication range (default 500m)."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_pos": {
                    "type": "array",
                    "description": "Target position [x, y, z] that needs comm coverage",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "base_pos": {
                    "type": "array",
                    "description": "Base station position [x, y, z] (default [0,0,0])",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0.0, 0.0, 0.0],
                },
                "comm_range": {
                    "type": "number",
                    "description": "Maximum single-hop comm range in meters (default 500.0)",
                    "default": 500.0,
                },
                "prefix": {
                    "type": "string",
                    "description": "Node ID prefix (default 'relay')",
                    "default": "relay",
                },
            },
            "required": ["target_pos"],
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        target = params.get("target_pos")
        if not target or len(target) < 3:
            return False
        comm_range = params.get("comm_range", 500.0)
        if not isinstance(comm_range, (int, float)) or comm_range <= 0:
            return False
        return True

    def execute(self, params: Dict[str, Any]) -> SubGraph:
        """
        Generate DeployRelay nodes along the base->target path.

        Relays are placed at intervals <= comm_range along the straight line.
        If the target is already within comm_range of the base, an empty
        SubGraph is returned.
        """
        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for relay_planner: {params}")

        target = tuple(float(v) for v in params["target_pos"])
        base = tuple(float(v) for v in params.get("base_pos", [0.0, 0.0, 0.0]))
        comm_range = float(params.get("comm_range", 500.0))
        prefix = params.get("prefix", "relay")

        dist = _euclidean_2d(base, target)

        # If target is within range of base, no relays needed
        if dist <= comm_range:
            return SubGraph(
                nodes=[],
                edges=[],
                metadata={
                    "tool": "relay_planner",
                    "info": "Target within direct comm range, no relays needed",
                    "distance": dist,
                },
            )

        # Calculate number of relay hops needed
        # We want each hop <= comm_range, including base->first_relay and last_relay->target
        # So we need ceil(dist / comm_range) - 1 relays
        num_relays = max(1, math.ceil(dist / comm_range) - 1)

        # Place relays evenly along the path
        nodes: List[TaskNode] = []
        edges: List[Edge] = []

        for i in range(num_relays):
            # Interpolation factor (evenly spaced, excluding base and target)
            t = (i + 1) / (num_relays + 1)
            rx = base[0] + t * (target[0] - base[0])
            ry = base[1] + t * (target[1] - base[1])
            rz = base[2] + t * (target[2] - base[2])

            node_id = f"{prefix}_deploy_{i}"
            node = TaskNode(
                id=node_id,
                op_type="DeployRelay",
                location=(rx, ry, rz),
                estimated_duration_s=300.0,  # 5 minutes for relay deployment
                agent_type_reqs=[AgentType.RELAY],
                generated_by="relay_planner",
                params={"hop_index": i, "total_hops": num_relays},
            )
            nodes.append(node)

            # Chain deployments: must deploy in order along path
            if i > 0:
                prev_id = f"{prefix}_deploy_{i - 1}"
                edges.append(Edge(
                    id=f"{prefix}_order_{i - 1}_{i}",
                    type=EdgeType.ORDER,
                    src=prev_id,
                    dst=node_id,
                ))

        return SubGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "tool": "relay_planner",
                "total_distance": dist,
                "num_relays": num_relays,
                "comm_range": comm_range,
            },
        )
