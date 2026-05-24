"""
Heterogeneous Graph State Encoder.

Converts TaskGraph + agent states into a PyG HeteroData object for GNN
processing, and provides a fixed-dimension pooled observation vector
compatible with Stable-Baselines3.

Node types:
  - task: [remaining_duration_norm, dep_satisfaction, priority, op_type_onehot(6)]
  - agent: [pos_x_norm, pos_y_norm, energy_frac, current_load_norm, type_onehot(5)]

Edge types:
  - ("task", "depends_on", "task"): ORDER/DATA edges from TaskGraph
  - ("agent", "comm_with", "agent"): within comm range (bidirectional)
  - ("agent", "can_exec", "task"): agent.type in task.agent_type_reqs
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Set, Tuple

from torch_geometric.data import HeteroData

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState, EnvironmentState
from core.schema.taskgraph import TaskGraph, TaskNode, Edge, EdgeType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TASKS = 512    # Upper bound on tasks per scenario. Chosen to fit the
                   # largest training scenarios (n_tasks <= 480). Only
                   # controls padding amount + output reshape; no learned
                   # weight tensor depends on it, so checkpoints trained
                   # with a smaller value remain compatible.
MAX_AGENTS = 16    # Upper bound on agents per scenario (n_agents <= 15 in
                   # the training corpus). Same backward-compat guarantee.

# Op-type one-hot encoding (6 types)
OP_TYPE_INDEX = {
    "ExploreCell": 0,
    "DeployRelay": 1,
    "Approach": 2,
    "DeployArm": 3,
    "Extract": 4,
    "Store": 5,
}
NUM_OP_TYPES = 6

# Agent-type one-hot encoding (5 types)
AGENT_TYPE_INDEX = {
    AgentType.ROVER: 0,
    AgentType.RELAY: 1,
    AgentType.SAMPLER: 2,
    AgentType.ANALYZER: 3,
}
NUM_AGENT_TYPES = 5  # 4 known + 1 "other"

# Feature dimensions (must match encode() output)
TASK_FEAT_DIM = 3 + NUM_OP_TYPES   # duration_norm + dep_sat + priority + op_type_onehot = 9
AGENT_FEAT_DIM = 4 + NUM_AGENT_TYPES  # x + y + energy + load + type_onehot = 9

# Normalization constants
MAX_DURATION_S = 300.0   # ~5 min per task
MAX_COORD = 2000.0       # max expected coordinate
MAX_LOAD = 20            # max tasks per agent


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _op_type_onehot(op_type: str) -> List[float]:
    """One-hot encode an op_type string."""
    vec = [0.0] * NUM_OP_TYPES
    idx = OP_TYPE_INDEX.get(op_type, NUM_OP_TYPES - 1)  # last slot = "other"
    vec[idx] = 1.0
    return vec


def _agent_type_onehot(agent_type: AgentType) -> List[float]:
    """One-hot encode an agent type."""
    vec = [0.0] * NUM_AGENT_TYPES
    idx = AGENT_TYPE_INDEX.get(agent_type, NUM_AGENT_TYPES - 1)
    vec[idx] = 1.0
    return vec


def _euclidean_2d(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# State Encoder
# ---------------------------------------------------------------------------

class StateEncoder:
    """
    Encodes a TaskGraph + agent states into a PyG HeteroData object.

    Also computes:
      - action_mask: valid (task, agent) assignment pairs
      - pooled fixed-dim observation for SB3 MaskablePPO
    """

    TASK_FEAT_DIM = 3 + NUM_OP_TYPES   # duration_norm + dep_sat + priority + onehot
    AGENT_FEAT_DIM = 4 + NUM_AGENT_TYPES  # x + y + energy + load + onehot

    def __init__(self, comm_range: float = 500.0):
        self.comm_range = comm_range

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        task_graph: TaskGraph,
        agents: List[AgentState],
        assigned: Dict[str, str],
    ) -> HeteroData:
        """
        Build a HeteroData graph from the current state.

        Args:
            task_graph: The full TaskGraph to schedule.
            agents: List of available agents.
            assigned: Dict[task_id → agent_id] of already-assigned tasks.

        Returns:
            HeteroData with node features and edge indices.
        """
        data = HeteroData()
        num_tasks = len(task_graph.nodes)
        num_agents = len(agents)

        # --- Build predecessor map + dependent-count map ---
        # We precompute dependent_count once per encode() call. The inner
        # `sum(1 for e in edges if e.src == node.id)` would be O(E) per node,
        # i.e. O(N*E) per encode; precomputing brings this to O(N+E).
        predecessors: Dict[str, Set[str]] = {n.id: set() for n in task_graph.nodes}
        dependent_count: Dict[str, int] = {n.id: 0 for n in task_graph.nodes}
        for edge in task_graph.edges:
            if edge.type in (EdgeType.ORDER, EdgeType.DATA):
                if edge.dst in predecessors:
                    predecessors[edge.dst].add(edge.src)
                if edge.src in dependent_count:
                    dependent_count[edge.src] += 1

        # --- Task node features ---
        task_feats = []
        task_id_to_idx = {}

        for i, node in enumerate(task_graph.nodes):
            task_id_to_idx[node.id] = i

            # Feature 1: remaining duration (normalized)
            dur = (node.estimated_duration_s or 60.0) / MAX_DURATION_S

            # Feature 2: dependency satisfaction
            preds = predecessors[node.id]
            if len(preds) == 0:
                dep_sat = 1.0
            else:
                assigned_count = sum(1 for p in preds if p in assigned)
                dep_sat = assigned_count / len(preds)

            # Feature 3: priority (more dependents = higher); O(1) via precomputed map
            priority = min(dependent_count[node.id] / 5.0, 1.0)  # normalize to [0, 1]

            # Feature 4-9: op_type one-hot
            onehot = _op_type_onehot(node.op_type)

            task_feats.append([dur, dep_sat, priority] + onehot)

        data["task"].x = torch.tensor(task_feats, dtype=torch.float32)

        # --- Agent node features ---
        agent_feats = []
        agent_id_to_idx = {}

        for j, agent in enumerate(agents):
            agent_id_to_idx[agent.id] = j
            pos = agent.position or (0.0, 0.0, 0.0)

            # Normalized position
            px = pos[0] / MAX_COORD
            py = pos[1] / MAX_COORD

            # Energy fraction
            energy_frac = agent.energy_wh / max(agent.initial_energy_wh, 1.0)

            # Current load (how many tasks assigned to this agent)
            load = sum(1 for a in assigned.values() if a == agent.id)
            load_norm = min(load / MAX_LOAD, 1.0)

            # Type one-hot
            onehot = _agent_type_onehot(agent.agent_type)

            agent_feats.append([px, py, energy_frac, load_norm] + onehot)

        data["agent"].x = torch.tensor(agent_feats, dtype=torch.float32)

        # --- Edge type 1: task depends_on task ---
        dep_src, dep_dst = [], []
        for edge in task_graph.edges:
            if edge.type in (EdgeType.ORDER, EdgeType.DATA):
                src_idx = task_id_to_idx.get(edge.src)
                dst_idx = task_id_to_idx.get(edge.dst)
                if src_idx is not None and dst_idx is not None:
                    dep_src.append(src_idx)
                    dep_dst.append(dst_idx)

        if dep_src:
            data["task", "depends_on", "task"].edge_index = torch.tensor(
                [dep_src, dep_dst], dtype=torch.long
            )
        else:
            data["task", "depends_on", "task"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

        # --- Edge type 2: agent comm_with agent ---
        comm_src, comm_dst = [], []
        for i, a1 in enumerate(agents):
            for j, a2 in enumerate(agents):
                if i >= j:
                    continue
                p1 = a1.position or (0, 0, 0)
                p2 = a2.position or (0, 0, 0)
                if _euclidean_2d(p1, p2) <= self.comm_range:
                    comm_src.extend([i, j])
                    comm_dst.extend([j, i])

        if comm_src:
            data["agent", "comm_with", "agent"].edge_index = torch.tensor(
                [comm_src, comm_dst], dtype=torch.long
            )
        else:
            data["agent", "comm_with", "agent"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

        # --- Edge type 3: agent can_exec task ---
        exec_src, exec_dst = [], []
        for j, agent in enumerate(agents):
            for i, node in enumerate(task_graph.nodes):
                if _agent_can_execute(agent, node):
                    exec_src.append(j)
                    exec_dst.append(i)

        if exec_src:
            data["agent", "can_exec", "task"].edge_index = torch.tensor(
                [exec_src, exec_dst], dtype=torch.long
            )
        else:
            data["agent", "can_exec", "task"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

        # Store metadata for downstream use
        data.num_real_tasks = num_tasks
        data.num_real_agents = num_agents
        data.task_ids = [n.id for n in task_graph.nodes]
        data.agent_ids = [a.id for a in agents]

        return data

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def compute_action_mask(
        self,
        task_graph: TaskGraph,
        agents: List[AgentState],
        assigned: Dict[str, str],
    ) -> np.ndarray:
        """
        Compute a flat binary mask of shape (MAX_TASKS * MAX_AGENTS,).

        A position (t * MAX_AGENTS + a) is 1 iff:
          1. Task t is not yet assigned
          2. Agent a is type-compatible with task t
          3. All predecessors of task t are already assigned

        Positions beyond real tasks/agents are always 0.
        """
        mask = np.zeros(MAX_TASKS * MAX_AGENTS, dtype=np.float32)

        # Build predecessor map
        predecessors: Dict[str, Set[str]] = {n.id: set() for n in task_graph.nodes}
        for edge in task_graph.edges:
            if edge.type in (EdgeType.ORDER, EdgeType.DATA):
                if edge.dst in predecessors:
                    predecessors[edge.dst].add(edge.src)

        for t_idx, node in enumerate(task_graph.nodes):
            if t_idx >= MAX_TASKS:
                break

            # Condition 1: not yet assigned
            if node.id in assigned:
                continue

            # Condition 3: all predecessors assigned
            preds = predecessors[node.id]
            if not all(p in assigned for p in preds):
                continue

            # Condition 2: type compatibility
            for a_idx, agent in enumerate(agents):
                if a_idx >= MAX_AGENTS:
                    break
                if _agent_can_execute(agent, node):
                    mask[t_idx * MAX_AGENTS + a_idx] = 1.0

        # Safety: MaskablePPO crashes on all-zero masks
        if mask.sum() == 0:
            mask[0] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Pooled observation for SB3
    # ------------------------------------------------------------------

    def get_pooled_observation(
        self,
        hetero_data: HeteroData,
        hidden_dim: int = 64,
    ) -> np.ndarray:
        """
        Produce a fixed-dimension observation vector by mean-pooling
        node features (before GNN encoding — raw features).

        For use before training starts or as a simple baseline.
        After GNN training, the policy network does its own encoding.

        Returns: np.ndarray of shape (TASK_FEAT_DIM + AGENT_FEAT_DIM + 3,)
        """
        # Mean-pool task features
        task_pool = hetero_data["task"].x.mean(dim=0).numpy()

        # Mean-pool agent features
        agent_pool = hetero_data["agent"].x.mean(dim=0).numpy()

        # Global features
        num_tasks = hetero_data.num_real_tasks
        num_agents = hetero_data.num_real_agents
        progress = 0.0  # Will be set by env based on assignment progress

        global_feats = np.array([
            num_tasks / MAX_TASKS,
            num_agents / MAX_AGENTS,
            progress,
        ], dtype=np.float32)

        return np.concatenate([task_pool, agent_pool, global_feats])

    @property
    def raw_obs_dim(self) -> int:
        """Dimension of the raw pooled observation (before GNN encoding)."""
        return self.TASK_FEAT_DIM + self.AGENT_FEAT_DIM + 3


# ---------------------------------------------------------------------------
# Helper: type compatibility check
# ---------------------------------------------------------------------------

def _agent_can_execute(agent: AgentState, task: TaskNode) -> bool:
    """Check if an agent's type matches the task's requirements."""
    if not task.agent_type_reqs:
        return True  # No requirement = any agent can do it
    return agent.agent_type in task.agent_type_reqs
