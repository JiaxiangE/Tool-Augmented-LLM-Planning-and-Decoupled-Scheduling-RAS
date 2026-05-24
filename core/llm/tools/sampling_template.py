"""
Sampling Template Tool.

Generates the standard 4-step sampling pipeline:
  Approach -> DeployArm -> Extract -> Store

This tool does NOT include Analyze -- analysis is a separate task
decided at a higher level by the LLM controller.
"""
from __future__ import annotations

from typing import Any, Dict, List

from core.schema.agent_types import AgentType
from core.schema.taskgraph import Edge, EdgeType, TaskNode

from .base import BaseTool, SubGraph

# The 4-step pipeline definition
_PIPELINE_STEPS = [
    {
        "suffix": "approach",
        "op_type": "Approach",
        "duration_s": 120.0,
        "description": "Navigate to sampling site",
    },
    {
        "suffix": "deploy_arm",
        "op_type": "DeployArm",
        "duration_s": 60.0,
        "description": "Extend sampling arm/drill mechanism",
    },
    {
        "suffix": "extract",
        "op_type": "Extract",
        "duration_s": 600.0,
        "description": "Drill and collect sample material",
    },
    {
        "suffix": "store",
        "op_type": "Store",
        "duration_s": 120.0,
        "description": "Seal and stow collected sample",
    },
]


class SamplingTemplateTool(BaseTool):
    """Generate a 4-step sampling pipeline for an identified site."""

    @property
    def name(self) -> str:
        return "sampling_template"

    @property
    def description(self) -> str:
        return (
            "Generate a standard 4-step sampling pipeline (Approach -> DeployArm -> "
            "Extract -> Store) for a specific site. The Approach step carries data "
            "prerequisites so the graph_assembler can auto-inject DATA edges from "
            "upstream exploration tasks. All steps require a SAMPLER agent. "
            "Use this when samples need to be collected from a surveyed location."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "Unique identifier for the sampling site (e.g. 'anomaly_1')",
                },
                "location": {
                    "type": "array",
                    "description": "Sampling coordinates [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "requires_data": {
                    "type": "array",
                    "description": "Data prerequisites from upstream tasks (e.g. map_chunk keys)",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": ["site_id", "location"],
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        site_id = params.get("site_id")
        if not site_id or not isinstance(site_id, str):
            return False
        location = params.get("location")
        if not location or len(location) < 3:
            return False
        return True

    def execute(self, params: Dict[str, Any]) -> SubGraph:
        """
        Generate the 4-step sampling pipeline.

        The Approach node inherits `requires_data` from the parameter so
        graph_assembler can auto-wire DATA edges from exploration producers.
        The Store node produces a sample artifact keyed by site_id.
        """
        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for sampling_template: {params}")

        site_id = params["site_id"]
        location = tuple(float(v) for v in params["location"])
        data_deps: List[str] = params.get("requires_data", [])

        nodes: List[TaskNode] = []
        edges: List[Edge] = []

        for i, step in enumerate(_PIPELINE_STEPS):
            node_id = f"sample_{site_id}_{step['suffix']}"

            # Build node
            node_kwargs = {
                "id": node_id,
                "op_type": step["op_type"],
                "location": location,
                "estimated_duration_s": step["duration_s"],
                "agent_type_reqs": [AgentType.SAMPLER],
                "generated_by": "sampling_template",
                "params": {"site_id": site_id, "step_index": i},
            }

            # Approach node carries upstream data dependencies
            if i == 0 and data_deps:
                node_kwargs["requires_data"] = data_deps

            # Store node produces the sample artifact
            if step["suffix"] == "store":
                node_kwargs["produces"] = [f"sample_{site_id}"]

            nodes.append(TaskNode(**node_kwargs))

            # Chain steps with ORDER edges
            if i > 0:
                prev_id = f"sample_{site_id}_{_PIPELINE_STEPS[i - 1]['suffix']}"
                edges.append(Edge(
                    id=f"sample_{site_id}_order_{i - 1}_{i}",
                    type=EdgeType.ORDER,
                    src=prev_id,
                    dst=node_id,
                ))

        return SubGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "tool": "sampling_template",
                "site_id": site_id,
                "location": list(location),
                "pipeline_steps": [s["op_type"] for s in _PIPELINE_STEPS],
            },
        )
