"""
Policy Network for GNN+DRL Scheduling.

Two encoder variants:
  - HGT (Heterogeneous Graph Transformer) — PyG HGTConv
  - RGCN (Relational Graph Convolutional Network) — PyG RGCNConv

Both share a cross-attention policy head that scores (task, agent) pairs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, RGCNConv, Linear

from core.scheduler.gnn.state_encoder import MAX_AGENTS, MAX_TASKS


# ---------------------------------------------------------------------------
# Encoder Variant A: Heterogeneous Graph Transformer
# ---------------------------------------------------------------------------

class HGTEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer encoder.

    Uses PyG's HGTConv to produce per-node embeddings for each node type.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        metadata: Optional[Tuple] = None,
    ):
        """
        Args:
            in_channels_dict: {"task": task_feat_dim, "agent": agent_feat_dim}
            hidden_dim: Hidden dimension for all layers.
            num_heads: Number of attention heads in HGTConv.
            num_layers: Number of HGT layers.
            metadata: PyG metadata tuple (node_types, edge_types).
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project each node type to hidden_dim
        self.input_projections = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.input_projections[ntype] = Linear(in_dim, hidden_dim)

        # HGT convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
            )
            # Layer norm per node type
            norm_dict = nn.ModuleDict()
            for ntype in in_channels_dict:
                norm_dict[ntype] = nn.LayerNorm(hidden_dim)
            self.norms.append(norm_dict)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x_dict: {node_type: feature_tensor}
            edge_index_dict: {edge_type_tuple: edge_index}

        Returns:
            {node_type: embedding_tensor} of shape (num_nodes, hidden_dim)
        """
        # Input projection
        h_dict = {}
        for ntype, proj in self.input_projections.items():
            if ntype in x_dict:
                h_dict[ntype] = proj(x_dict[ntype])

        # HGT layers
        for conv, norm_dict in zip(self.convs, self.norms):
            h_out = conv(h_dict, edge_index_dict)
            # Residual + LayerNorm
            for ntype in h_dict:
                if ntype in h_out:
                    h_dict[ntype] = norm_dict[ntype](
                        h_dict[ntype] + h_out[ntype]
                    )

        return h_dict


# ---------------------------------------------------------------------------
# Encoder Variant B: Relational GCN
# ---------------------------------------------------------------------------

class RGCNEncoder(nn.Module):
    """
    Relational GCN encoder for heterogeneous graphs.

    Converts the heterogeneous graph to homogeneous (with relation types),
    applies RGCN layers, then splits back by node type.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_dim: int = 64,
        num_relations: int = 3,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations

        # Project each node type to hidden_dim
        self.input_projections = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.input_projections[ntype] = Linear(in_dim, hidden_dim)

        # RGCN layers (operate on homogeneous graph with edge types)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """Forward pass with heterogeneous → homogeneous conversion."""

        # Project inputs
        h_dict = {}
        for ntype, proj in self.input_projections.items():
            if ntype in x_dict:
                h_dict[ntype] = proj(x_dict[ntype])

        # Merge into a single homogeneous graph
        node_types_order = sorted(h_dict.keys())
        offsets = {}
        all_h = []
        offset = 0
        for ntype in node_types_order:
            offsets[ntype] = offset
            all_h.append(h_dict[ntype])
            offset += h_dict[ntype].size(0)
        x_homo = torch.cat(all_h, dim=0)

        # Map edge types to integer relation indices
        edge_type_to_idx = {}
        all_edges = []
        all_edge_types = []

        for idx, (etype, edge_index) in enumerate(edge_index_dict.items()):
            if edge_index.numel() == 0:
                continue
            src_type, _, dst_type = etype
            rel_idx = idx % self.num_relations

            # Offset node indices
            src_offset = offsets.get(src_type, 0)
            dst_offset = offsets.get(dst_type, 0)
            shifted = edge_index.clone()
            shifted[0] += src_offset
            shifted[1] += dst_offset

            all_edges.append(shifted)
            all_edge_types.append(
                torch.full((shifted.size(1),), rel_idx, dtype=torch.long)
            )

        if all_edges:
            edge_index_homo = torch.cat(all_edges, dim=1)
            edge_type_homo = torch.cat(all_edge_types)
        else:
            edge_index_homo = torch.zeros((2, 0), dtype=torch.long)
            edge_type_homo = torch.zeros(0, dtype=torch.long)

        # RGCN layers
        h = x_homo
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_homo, edge_type_homo)
            h = norm(h + F.relu(h_new))

        # Split back by node type
        out_dict = {}
        for ntype in node_types_order:
            start = offsets[ntype]
            size = h_dict[ntype].size(0)
            out_dict[ntype] = h[start: start + size]

        return out_dict


# ---------------------------------------------------------------------------
# Cross-Attention Policy Head
# ---------------------------------------------------------------------------

class CrossAttentionHead(nn.Module):
    """
    Scores (task, agent) pairs using cross-attention.

    task embeddings = queries, agent embeddings = keys/values.
    Output: (MAX_TASKS, MAX_AGENTS) score matrix, flattened.
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

        # Value head for PPO critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        task_emb: Tensor,
        agent_emb: Tensor,
        num_real_tasks: int,
        num_real_agents: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute action logits and state value.

        Args:
            task_emb: (num_tasks, hidden_dim) — GNN-encoded task embeddings
            agent_emb: (num_agents, hidden_dim) — GNN-encoded agent embeddings
            num_real_tasks: actual number of tasks (before padding)
            num_real_agents: actual number of agents (before padding)

        Returns:
            action_logits: (MAX_TASKS * MAX_AGENTS,) — padded and flattened
            value: (1,) — state value estimate
        """
        # Pad to MAX_TASKS × MAX_AGENTS
        task_padded = F.pad(
            task_emb,
            (0, 0, 0, MAX_TASKS - task_emb.size(0)),
            value=0.0,
        )  # (MAX_TASKS, hidden_dim)

        agent_padded = F.pad(
            agent_emb,
            (0, 0, 0, MAX_AGENTS - agent_emb.size(0)),
            value=0.0,
        )  # (MAX_AGENTS, hidden_dim)

        # Cross-attention: Q=tasks, K=agents
        Q = self.query_proj(task_padded)   # (MAX_TASKS, hidden_dim)
        K = self.key_proj(agent_padded)    # (MAX_AGENTS, hidden_dim)

        # Score each (task, agent) pair
        # Expand: Q -> (MAX_TASKS, 1, dim), K -> (1, MAX_AGENTS, dim)
        Q_exp = Q.unsqueeze(1).expand(-1, MAX_AGENTS, -1)
        K_exp = K.unsqueeze(0).expand(MAX_TASKS, -1, -1)

        # Element-wise product + project to scalar
        combined = Q_exp * K_exp  # (MAX_TASKS, MAX_AGENTS, hidden_dim)
        scores = self.score_proj(combined).squeeze(-1)  # (MAX_TASKS, MAX_AGENTS)

        action_logits = scores.reshape(-1)  # (MAX_TASKS * MAX_AGENTS,)

        # Value estimate: pool task and agent embeddings
        task_pool = task_emb.mean(dim=0)    # (hidden_dim,)
        agent_pool = agent_emb.mean(dim=0)  # (hidden_dim,)
        value_input = torch.cat([task_pool, agent_pool])  # (2 * hidden_dim,)
        value = self.value_head(value_input)  # (1,)

        return action_logits, value


# ---------------------------------------------------------------------------
# Full Policy Network
# ---------------------------------------------------------------------------

class SchedulingPolicyNet(nn.Module):
    """
    Complete policy: GNN encoder → cross-attention → action logits + value.

    Wraps either HGTEncoder or RGCNEncoder with the CrossAttentionHead.
    """

    def __init__(
        self,
        task_feat_dim: int,
        agent_feat_dim: int,
        hidden_dim: int = 64,
        encoder_type: str = "hgt",
        num_heads: int = 4,
        num_layers: int = 2,
        metadata: Optional[Tuple] = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim

        in_channels = {"task": task_feat_dim, "agent": agent_feat_dim}

        if encoder_type == "hgt":
            self.encoder = HGTEncoder(
                in_channels_dict=in_channels,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                metadata=metadata,
            )
        elif encoder_type == "rgcn":
            self.encoder = RGCNEncoder(
                in_channels_dict=in_channels,
                hidden_dim=hidden_dim,
                num_relations=3,  # depends_on, comm_with, can_exec
                num_layers=num_layers,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.head = CrossAttentionHead(hidden_dim=hidden_dim, num_heads=num_heads)

    def forward(self, hetero_data: HeteroData) -> Tuple[Tensor, Tensor]:
        """
        Full forward pass.

        Args:
            hetero_data: PyG HeteroData from StateEncoder.

        Returns:
            action_logits: (MAX_TASKS * MAX_AGENTS,) — padded, pre-mask
            value: (1,) — state value estimate
        """
        x_dict = {ntype: hetero_data[ntype].x for ntype in hetero_data.node_types}
        edge_index_dict = {
            etype: hetero_data[etype].edge_index
            for etype in hetero_data.edge_types
        }

        # GNN encode
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Extract embeddings
        task_emb = h_dict["task"]     # (num_real_tasks, hidden_dim)
        agent_emb = h_dict["agent"]   # (num_real_agents, hidden_dim)

        num_real_tasks = hetero_data.num_real_tasks
        num_real_agents = hetero_data.num_real_agents

        # Policy head
        action_logits, value = self.head(
            task_emb, agent_emb, num_real_tasks, num_real_agents
        )

        return action_logits, value

    def get_action(
        self,
        hetero_data: HeteroData,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, Tensor, Tensor]:
        """
        Sample or greedily select an action.

        Args:
            hetero_data: Current state graph.
            action_mask: Binary mask (MAX_TASKS * MAX_AGENTS,).
            deterministic: If True, pick argmax. If False, sample.

        Returns:
            action: Selected action index.
            log_prob: Log probability of the action.
            value: State value estimate.
        """
        logits, value = self.forward(hetero_data)

        # Mask invalid actions with large negative value
        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = float("-inf")

        probs = F.softmax(masked_logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()

        # Log prob
        log_prob = torch.log(probs[action] + 1e-10)

        return action, log_prob, value
