"""
Graph Assembler Tool.

Merges multiple SubGraphs into a complete TaskGraph via a 3-step process:
  1. SubGraph merge with ID conflict resolution (cascading rename)
  2. Automatic DATA edge injection from produces/requires_data
  3. Cross-edge insertion + validation via TaskGraph._validate_graph()
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from core.schema.taskgraph import Edge, EdgeType, TaskGraph, TaskNode

from .base import BaseTool, SubGraph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_id_conflicts(
    all_nodes: List[TaskNode],
    all_edges: List[Edge],
    node_origins: List[int],
    edge_origins: List[int],
) -> List[str]:
    """
    Detect and resolve node ID conflicts across merged SubGraphs.

    Strategy: First occurrence keeps its ID. Subsequent duplicates get
    renamed to {id}_dup{N}. Only edges from the SAME SubGraph origin as
    the duplicate node are updated.

    Args:
        all_nodes: Flat list of all nodes from all SubGraphs.
        all_edges: Flat list of all edges from all SubGraphs.
        node_origins: SubGraph index for each node (parallel to all_nodes).
        edge_origins: SubGraph index for each edge (parallel to all_edges).

    Returns a list of warnings about renames that occurred.
    """
    warnings: List[str] = []

    # First pass: identify first-occurrence vs duplicates
    first_occurrence: Dict[str, int] = {}  # id -> node index
    duplicates: List[Tuple[int, str]] = []  # (node_index, original_id)

    for i, node in enumerate(all_nodes):
        if node.id not in first_occurrence:
            first_occurrence[node.id] = i
        else:
            duplicates.append((i, node.id))

    # Second pass: rename each duplicate
    all_used_ids = set(n.id for n in all_nodes)

    for node_idx, old_id in duplicates:
        dup_count = 1
        while True:
            new_id = f"{old_id}_dup{dup_count}"
            if new_id not in all_used_ids:
                break
            dup_count += 1

        # Origin SubGraph of this duplicate node
        dup_origin = node_origins[node_idx]

        # 1. Rename the node itself
        all_nodes[node_idx].id = new_id
        all_used_ids.add(new_id)

        # 2. Update edges ONLY from the same SubGraph origin
        for ei, e in enumerate(all_edges):
            if edge_origins[ei] != dup_origin:
                continue
            if e.src == old_id:
                e.src = new_id
            if e.dst == old_id:
                e.dst = new_id

        # 3. Update requires_data / produces in nodes from the same origin
        for ni, n in enumerate(all_nodes):
            if node_origins[ni] != dup_origin:
                continue
            if old_id in n.requires_data:
                n.requires_data = [new_id if d == old_id else d for d in n.requires_data]
            if old_id in n.produces:
                n.produces = [new_id if p == old_id else p for p in n.produces]

        warnings.append(f"Renamed duplicate node '{old_id}' -> '{new_id}'")

    # Resolve edge ID conflicts
    seen_edge_ids: Set[str] = set()
    for edge in all_edges:
        if edge.id not in seen_edge_ids:
            seen_edge_ids.add(edge.id)
            continue
        dup_count = 1
        while True:
            new_eid = f"{edge.id}_dup{dup_count}"
            if new_eid not in seen_edge_ids:
                break
            dup_count += 1
        edge.id = new_eid
        seen_edge_ids.add(new_eid)

    return warnings


def _inject_data_edges(
    nodes: List[TaskNode],
    existing_edges: List[Edge],
) -> Tuple[List[Edge], List[str]]:
    """
    Build DATA edges by matching produces -> requires_data across all nodes.

    Returns (new_data_edges, warnings_for_unresolved_deps).
    """
    # Build producer index: data_key -> producer_node_id
    producer_index: Dict[str, str] = {}
    for n in nodes:
        for key in n.produces:
            producer_index[key] = n.id

    # Build set of existing DATA edges for dedup
    existing_data_pairs: Set[Tuple[str, str]] = set()
    for e in existing_edges:
        if e.type == EdgeType.DATA:
            existing_data_pairs.add((e.src, e.dst))

    new_edges: List[Edge] = []
    warnings: List[str] = []
    edge_counter = 0

    for n in nodes:
        for key in n.requires_data:
            producer = producer_index.get(key)
            if producer is None:
                warnings.append(
                    f"Node '{n.id}' requires data '{key}' but no producer found"
                )
                continue

            if producer == n.id:
                continue  # Self-dependency, skip

            if (producer, n.id) in existing_data_pairs:
                continue  # Already exists

            new_edges.append(Edge(
                id=f"auto_data_{edge_counter}",
                type=EdgeType.DATA,
                src=producer,
                dst=n.id,
            ))
            existing_data_pairs.add((producer, n.id))
            edge_counter += 1

    return new_edges, warnings


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class GraphAssemblerTool(BaseTool):
    """Merge SubGraphs into a complete TaskGraph."""

    @property
    def name(self) -> str:
        return "graph_assembler"

    @property
    def description(self) -> str:
        return (
            "Merge previously generated SubGraphs into a complete TaskGraph. "
            "Performs three steps: (1) merge all nodes/edges with ID conflict "
            "resolution, (2) automatically inject DATA edges by matching produces "
            "and requires_data fields, (3) add cross-SubGraph ORDER edges specified "
            "by the caller and validate the final graph. This should be the last "
            "tool called in the planning sequence."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mission_id": {
                    "type": "string",
                    "description": "Mission identifier for the TaskGraph",
                },
                "cross_edges": {
                    "type": "array",
                    "description": (
                        "Cross-SubGraph ORDER edges to add. Each item has "
                        "'src' (source node ID) and 'dst' (destination node ID). "
                        "These represent ordering constraints between SubGraphs."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "src": {"type": "string"},
                            "dst": {"type": "string"},
                        },
                        "required": ["src", "dst"],
                    },
                    "default": [],
                },
            },
            "required": ["mission_id"],
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        mission_id = params.get("mission_id")
        if not mission_id or not isinstance(mission_id, str):
            return False
        cross_edges = params.get("cross_edges", [])
        if not isinstance(cross_edges, list):
            return False
        return True

    def assemble(
        self,
        subgraphs: List[SubGraph],
        mission_id: str,
        cross_edges: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Optional[TaskGraph], Dict[str, Any]]:
        """
        Core assembly logic. Used programmatically by the controller.

        Returns:
            (task_graph, metadata) where task_graph is None if validation failed.
            metadata contains warnings, errors, and assembly stats.
        """
        cross_edges = cross_edges or []

        # --- Step 1: Merge ---
        # Deep-copy to avoid mutating originals, track origin SubGraph index
        all_nodes: List[TaskNode] = []
        all_edges: List[Edge] = []
        node_origins: List[int] = []  # SubGraph index for each node
        edge_origins: List[int] = []  # SubGraph index for each edge

        for sg_idx, sg in enumerate(subgraphs):
            for n in sg.nodes:
                all_nodes.append(n.model_copy(deep=True))
                node_origins.append(sg_idx)
            for e in sg.edges:
                all_edges.append(e.model_copy(deep=True))
                edge_origins.append(sg_idx)

        rename_warnings = _resolve_id_conflicts(all_nodes, all_edges, node_origins, edge_origins)

        # --- Step 2: Automatic DATA edge injection ---
        data_edges, data_warnings = _inject_data_edges(all_nodes, all_edges)
        all_edges.extend(data_edges)

        # --- Step 3: Cross-edges + validation ---
        for i, ce in enumerate(cross_edges):
            all_edges.append(Edge(
                id=f"cross_order_{i}",
                type=EdgeType.ORDER,
                src=ce["src"],
                dst=ce["dst"],
            ))

        metadata: Dict[str, Any] = {
            "tool": "graph_assembler",
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "data_edges_injected": len(data_edges),
            "cross_edges_added": len(cross_edges),
            "rename_warnings": rename_warnings,
            "data_warnings": data_warnings,
        }

        try:
            tg = TaskGraph(
                mission_id=mission_id,
                nodes=all_nodes,
                edges=all_edges,
            )
            metadata["validation"] = "passed"
            return tg, metadata
        except Exception as e:
            metadata["validation"] = "failed"
            metadata["validation_error"] = str(e)
            return None, metadata

    def execute(self, params: Dict[str, Any]) -> SubGraph:
        """
        Tool-calling interface wrapper.

        Note: The actual SubGraphs to merge are injected by the controller
        (they are maintained in the controller's accumulator, not passed as params).
        This execute() is primarily for interface compliance. The controller
        should call self.assemble() directly.
        """
        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for graph_assembler: {params}")

        # When called through the tool interface, subgraphs should have been
        # injected into params by the controller
        subgraphs = params.get("_subgraphs", [])
        mission_id = params["mission_id"]
        cross_edges = params.get("cross_edges", [])

        tg, metadata = self.assemble(subgraphs, mission_id, cross_edges)

        if tg is not None:
            metadata["task_graph_json"] = tg.model_dump_json()

        return SubGraph(metadata=metadata)
