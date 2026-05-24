"""
Coverage Planner Tool.

Generates ExploreCell task sequences for rectangular region grid coverage.
Each cell becomes one TaskNode producing a map chunk data artifact.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from core.schema.agent_types import AgentType
from core.schema.taskgraph import Edge, EdgeType, TaskNode

from .base import BaseTool, SubGraph


class CoveragePlannerTool(BaseTool):
    """Plan grid-based exploration coverage for a rectangular region."""

    @property
    def name(self) -> str:
        return "coverage_planner"

    @property
    def description(self) -> str:
        return (
            "Generate a sequence of ExploreCell tasks that cover a rectangular region "
            "in a grid pattern. Each cell produces a map_chunk data artifact that "
            "downstream tools (e.g. sampling_template) can depend on. "
            "Use this when the mission requires surveying or mapping an area."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "region_bounds": {
                    "type": "array",
                    "description": "Bounding box as [[x1,y1],[x2,y2]] in meters",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "minItems": 2,
                    "maxItems": 2,
                },
                "cell_size": {
                    "type": "number",
                    "description": "Grid cell size in meters (default 10.0)",
                    "default": 10.0,
                },
                "prefix": {
                    "type": "string",
                    "description": "Node ID prefix for generated tasks (e.g. 'scan_sector_A')",
                },
            },
            "required": ["region_bounds", "prefix"],
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        bounds = params.get("region_bounds")
        if not bounds or len(bounds) != 2:
            return False
        if len(bounds[0]) < 2 or len(bounds[1]) < 2:
            return False
        prefix = params.get("prefix")
        if not prefix or not isinstance(prefix, str):
            return False
        cell_size = params.get("cell_size", 10.0)
        if not isinstance(cell_size, (int, float)) or cell_size <= 0:
            return False
        return True

    def execute(self, params: Dict[str, Any]) -> SubGraph:
        """
        Generate ExploreCell nodes covering the grid.

        Grid is laid out row-major. Each node produces a unique map_chunk key
        and requires communication connectivity for data upload.
        """
        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for coverage_planner: {params}")

        bounds = params["region_bounds"]
        cell_size = float(params.get("cell_size", 10.0))
        prefix = params["prefix"]

        x1, y1 = float(bounds[0][0]), float(bounds[0][1])
        x2, y2 = float(bounds[1][0]), float(bounds[1][1])

        # Ensure x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Calculate grid dimensions
        cols = max(1, math.ceil((x2 - x1) / cell_size))
        rows = max(1, math.ceil((y2 - y1) / cell_size))

        nodes: List[TaskNode] = []
        edges: List[Edge] = []

        cell_idx = 0
        for row in range(rows):
            for col in range(cols):
                cx = x1 + (col + 0.5) * cell_size
                cy = y1 + (row + 0.5) * cell_size

                node_id = f"{prefix}_cell_{cell_idx}"
                data_key = f"map_chunk_{prefix}_{cell_idx}"

                node = TaskNode(
                    id=node_id,
                    op_type="ExploreCell",
                    location=(cx, cy, 0.0),
                    estimated_duration_s=max(60.0, cell_size * 6.0),  # ~6s per meter
                    agent_type_reqs=[AgentType.ROVER],
                    produces=[data_key],
                    requires_connected=True,
                    generated_by="coverage_planner",
                    params={"cell_row": row, "cell_col": col},
                )
                nodes.append(node)

                # Chain consecutive cells with ORDER edges
                if cell_idx > 0:
                    prev_id = f"{prefix}_cell_{cell_idx - 1}"
                    edges.append(Edge(
                        id=f"{prefix}_order_{cell_idx - 1}_{cell_idx}",
                        type=EdgeType.ORDER,
                        src=prev_id,
                        dst=node_id,
                    ))

                cell_idx += 1

        return SubGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "tool": "coverage_planner",
                "region_bounds": bounds,
                "cell_size": cell_size,
                "total_cells": cell_idx,
            },
        )
