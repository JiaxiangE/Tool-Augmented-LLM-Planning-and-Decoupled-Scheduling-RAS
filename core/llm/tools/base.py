"""
Tool framework base classes for LLM-driven task decomposition.

Defines:
  - SubGraph: a fragment of a TaskGraph produced by a single tool invocation.
  - BaseTool: abstract base class that all domain tools must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.schema.taskgraph import Edge, TaskNode


class SubGraph(BaseModel):
    """
    A fragment of a TaskGraph produced by a single tool invocation.

    Multiple SubGraphs are merged by the graph_assembler into a complete
    TaskGraph. Each SubGraph carries metadata about its provenance.
    """
    nodes: List[TaskNode] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]


class BaseTool(ABC):
    """
    Abstract base class for all domain tools in the LLM planning layer.

    Each tool knows how to produce a SubGraph given a set of parameters.
    The LLM controller calls tools via the ReAct loop, inspects their
    SubGraph output, and eventually assembles the full TaskGraph.

    Implementors must provide:
      - name: unique string identifier
      - description: natural-language description for the LLM prompt
      - parameters_schema: JSON Schema dict for LLM function calling
      - execute(params) -> SubGraph
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this tool (e.g. 'coverage_planner')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description included in the LLM system prompt."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """
        JSON Schema describing the parameters this tool accepts.

        Used by the LLM for function-calling argument generation.
        Example:
            {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Target region name"},
                    "resolution": {"type": "number", "description": "Grid resolution in meters"}
                },
                "required": ["region"]
            }
        """
        ...

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> SubGraph:
        """
        Execute the tool with the given parameters and return a SubGraph.

        This method must be deterministic and side-effect-free.
        It should NOT depend on LLM state or other tools' outputs.

        Args:
            params: Dictionary of parameters matching parameters_schema.

        Returns:
            SubGraph containing generated TaskNodes and Edges.

        Raises:
            ValueError: If params are invalid.
        """
        ...

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Optional parameter validation before execute().

        Default implementation returns True (no validation).
        Override to add custom checks.
        """
        return True

    def to_function_spec(self) -> Dict[str, Any]:
        """
        Generate an OpenAI-compatible function specification for this tool.

        Used by the LLM controller to build the tools list in the system prompt.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
