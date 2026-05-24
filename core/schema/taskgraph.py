from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from .agent_types import AgentType


# -----------------------------
# Enums
# -----------------------------

class EdgeType(str, Enum):
    """Semantics of edges between tasks."""
    ORDER = "order"        # precedence / partial order constraint
    MUTEX = "mutex"        # cannot overlap in time (typically shared resource)
    SYNC = "sync"          # rendezvous / barrier / handover synchronization
    DATA = "data"          # information dependency (e.g., map chunk needed)


class NodeStatus(str, Enum):
    """Runtime status (optional field in plans/execution traces)."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class FailureCode(str, Enum):
    """Standardized failure codes for BT/action execution feedback."""
    TIMEOUT = "timeout"
    PRECONDITION = "precondition_failed"
    RESOURCE = "resource_unavailable"
    COMM = "comm_lost"
    LOW_BATTERY = "low_battery"
    PATH_BLOCKED = "path_blocked"
    SENSOR_FAIL = "sensor_fail"
    COLLISION_RISK = "collision_risk"
    UNKNOWN = "unknown"


# -----------------------------
# Typed parameters / references
# -----------------------------

class Ref(BaseModel):
    """A typed reference to an entity in the world model."""
    kind: Literal["agent", "region", "cell", "worksite", "item", "tool", "relay", "formation", "milestone"]
    name: str = Field(min_length=1)


class Quantity(BaseModel):
    """Simple numeric value with unit."""
    value: float
    unit: Literal["percent", "seconds", "meters", "cells", "count"]


# -----------------------------
# Resource specs & constraints
# -----------------------------

class ResourceMode(str, Enum):
    """How a task interacts with a resource."""
    ACQUIRE = "acquire"        # acquire token / lock
    CONSUME = "consume"        # consume budget (battery, bandwidth, time budget)
    REQUIRE = "require"        # requires availability but doesn't lock (soft)
    RELEASE = "release"        # explicit release (optional)


class ResourceSpec(BaseModel):
    """A resource requirement/usage declaration attached to a node."""
    name: str = Field(min_length=1)  # e.g., "uplink", "crane", "tool:drill", "token:comm"
    mode: ResourceMode
    amount: Optional[Quantity] = None  # e.g. bandwidth seconds, battery percent, etc.
    exclusive: bool = False            # if True, no overlap with other tasks requiring same resource


class TimeWindow(BaseModel):
    """Hard window in mission time (seconds since t0)."""
    earliest_start_s: Optional[int] = Field(default=None, ge=0)
    latest_end_s: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_window(self) -> "TimeWindow":
        if self.earliest_start_s is not None and self.latest_end_s is not None:
            if self.latest_end_s < self.earliest_start_s:
                raise ValueError("TimeWindow.latest_end_s must be >= earliest_start_s")
        return self


class PeriodicConstraint(BaseModel):
    """
    A global periodic requirement (e.g., upload at least once every 120s).
    This is a constraint, not a task; the scheduler may insert tasks to satisfy it.
    """
    name: str = Field(min_length=1)
    period_s: int = Field(ge=1)
    target: Ref  # what entity/region this constraint refers to
    must_hold: bool = True
    description: Optional[str] = None


class GlobalConstraints(BaseModel):
    """Global mission-level constraints."""
    min_battery_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    max_concurrent_tasks_per_agent: Optional[int] = Field(default=1, ge=1)
    periodic: List[PeriodicConstraint] = Field(default_factory=list)


# -----------------------------
# Task node & edge
# -----------------------------

class TaskNode(BaseModel):
    """
    One node in the task graph.

    op_type should be from our meta-operator white-list. The scheduler and BT compiler
    should only accept known op_type values.
    """
    id: str = Field(min_length=1)
    op_type: str = Field(min_length=1)  # e.g. "ExploreCell", "DeployRelay", "UploadMapChunk", ...
    params: Dict[str, Union[str, int, float, bool, Ref, Quantity, List[Ref]]] = Field(default_factory=dict)

    # Optional but useful fields
    assigned_agent: Optional[str] = None   # scheduler output, or planner suggestion
    capability_reqs: List[str] = Field(default_factory=list) # e.g. ["mobility", "sampling"]
    produces: List[str] = Field(default_factory=list)        # e.g. ["map_chunk_north"]
    requires_data: List[str] = Field(default_factory=list)   # e.g. ["map_chunk_north"]
    requires_connected: bool = False                         # if True, scheduler must ensure connectivity
    
    time_window: Optional[TimeWindow] = None
    resources: List[ResourceSpec] = Field(default_factory=list)

    # Runtime/execution feedback (optional; can be filled during execution)
    status: Optional[NodeStatus] = None
    failure_code: Optional[FailureCode] = None
    failure_detail: Optional[str] = None
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Hints
    timeout_s: Optional[int] = Field(default=None, ge=1)
    retry_limit: int = Field(default=0, ge=0)

    # --- Extended fields (all Optional for backward compatibility) ---
    location: Optional[Tuple[float, float, float]] = None        # (x, y, z) lunar surface coords
    estimated_duration_s: Optional[float] = Field(default=None, gt=0.0)  # direct duration estimate (>0)
    agent_type_reqs: List[AgentType] = Field(default_factory=list)       # required agent types
    generated_by: Optional[str] = None                                    # provenance: which tool created this node

    @model_validator(mode="after")
    def _check_resources(self) -> "TaskNode":
        # If a resource is exclusive, it should usually be acquire/require (not consume)
        for r in self.resources:
            if r.exclusive and r.mode == ResourceMode.CONSUME:
                raise ValueError(f"Node {self.id}: exclusive resource should not use mode=consume: {r.name}")
        return self


class Edge(BaseModel):
    """A typed edge between nodes."""
    id: str = Field(min_length=1)
    type: EdgeType
    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)

    # Optional edge payload (e.g., shared resource name for mutex)
    resource: Optional[str] = None          # for MUTEX edges
    milestone: Optional[Ref] = None         # for SYNC edges (barrier/handover)
    note: Optional[str] = None


# -----------------------------
# TaskGraph root
# -----------------------------

class TaskGraph(BaseModel):
    """
    TaskGraph v0

    - nodes + edges define a partially ordered set of tasks.
    - constraints capture global requirements that schedulers must satisfy.
    - metadata helps with versioning and traceability.
    """
    version: Literal["0.1"] = "0.1"
    mission_id: str = Field(min_length=1)

    nodes: List[TaskNode]
    edges: List[Edge] = Field(default_factory=list)
    constraints: GlobalConstraints = Field(default_factory=GlobalConstraints)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_graph(self) -> "TaskGraph":
        node_ids = [n.id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node ids found in TaskGraph.nodes")

        node_set = set(node_ids)

        # Check edges reference valid nodes
        for e in self.edges:
            if e.src not in node_set or e.dst not in node_set:
                raise ValueError(f"Edge {e.id} references missing node(s): {e.src} -> {e.dst}")

            if e.type == EdgeType.MUTEX and not e.resource:
                raise ValueError(f"Edge {e.id} is mutex but resource is not set")

        # ORDER edges must form a DAG
        order_edges: List[Tuple[str, str]] = [(e.src, e.dst) for e in self.edges if e.type == EdgeType.ORDER]
        if _has_cycle(order_edges):
            raise ValueError("ORDER edges contain a cycle (TaskGraph must be acyclic for scheduling)")

        return self


def _has_cycle(edges: List[Tuple[str, str]]) -> bool:
    """
    Simple cycle detection on directed graph given as edge list.
    Kahn's algorithm on adjacency.
    """
    if not edges:
        return False

    # Build adjacency and indegree
    adj: Dict[str, Set[str]] = {}
    indeg: Dict[str, int] = {}

    nodes: Set[str] = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
        adj.setdefault(u, set()).add(v)

    for n in nodes:
        indeg[n] = 0
    for u, vs in adj.items():
        for v in vs:
            indeg[v] += 1

    # Kahn
    queue = [n for n in nodes if indeg[n] == 0]
    visited = 0
    while queue:
        n = queue.pop()
        visited += 1
        for v in adj.get(n, set()):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    return visited != len(nodes)
