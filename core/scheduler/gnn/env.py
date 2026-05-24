"""
Gymnasium Environment for GNN+DRL Task Scheduling.

Wraps TaskGraph + LunarSimulator as a Gymnasium environment compatible
with sb3-contrib's MaskablePPO.

Observation: fixed-dim pooled vector (raw task/agent feature means + globals)
Action: Discrete(MAX_TASKS * MAX_AGENTS) with action masking
Episode: assign tasks one-by-one → when all assigned → simulate → terminal reward
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState
from core.schema.taskgraph import TaskGraph, EdgeType
from core.scheduler.greedy_scheduler import AgentInfo, schedule as greedy_schedule
from core.scheduler.gnn.reward import RewardConfig, compute_reward, compute_step_reward
from core.scheduler.gnn.state_encoder import (
    MAX_AGENTS,
    MAX_TASKS,
    StateEncoder,
)
from core.simulator.terrain import TerrainGrid
from core.simulator.lunar_sim import LunarSimulator, SimConfig


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir) -> List[Dict[str, Any]]:
    """Load all training corpus JSON files.

    `corpus_dir` may be a single path OR a list of paths (e.g. an in-distribution
    directory plus an out-of-distribution one). Files are loaded in directory
    order, then filename order, so the resulting corpus index ordering is
    deterministic.

    Skips sidecar files whose name starts with `_`, which are metadata
    artifacts and not TaskGraph scenarios.
    """
    if isinstance(corpus_dir, (list, tuple)):
        dirs = [Path(d) for d in corpus_dir]
    else:
        dirs = [Path(corpus_dir)]

    corpus = []
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("_"):
                continue
            with open(f, "r", encoding="utf-8") as fp:
                entry = json.load(fp)
            if "taskgraph" not in entry:
                continue  # defensive: skip anything that isn't a scenario
            corpus.append(entry)
    return corpus


def _agents_from_metadata(meta: Dict) -> List[AgentState]:
    """Reconstruct AgentState list from corpus metadata scale."""
    scale = meta.get("scale", "small")
    agent_count = meta.get("agent_count", 3)

    # Build a reasonable agent set based on scale
    agents: List[AgentState] = []
    if scale == "small":
        agents = [
            AgentState(id="rover_1", agent_type=AgentType.ROVER,
                       energy_wh=3000.0, initial_energy_wh=3000.0),
            AgentState(id="relay_1", agent_type=AgentType.RELAY,
                       energy_wh=3000.0, initial_energy_wh=3000.0),
            AgentState(id="sampler_1", agent_type=AgentType.SAMPLER,
                       energy_wh=3000.0, initial_energy_wh=3000.0),
        ]
    elif scale == "medium":
        agents = [
            AgentState(id="rover_1", agent_type=AgentType.ROVER,
                       energy_wh=5000.0, initial_energy_wh=5000.0),
            AgentState(id="rover_2", agent_type=AgentType.ROVER,
                       energy_wh=5000.0, initial_energy_wh=5000.0),
            AgentState(id="relay_1", agent_type=AgentType.RELAY,
                       energy_wh=5000.0, initial_energy_wh=5000.0),
            AgentState(id="relay_2", agent_type=AgentType.RELAY,
                       energy_wh=5000.0, initial_energy_wh=5000.0),
            AgentState(id="sampler_1", agent_type=AgentType.SAMPLER,
                       energy_wh=5000.0, initial_energy_wh=5000.0),
        ]
    else:  # large
        agents = [
            AgentState(id="rover_1", agent_type=AgentType.ROVER,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
            AgentState(id="rover_2", agent_type=AgentType.ROVER,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
            AgentState(id="rover_3", agent_type=AgentType.ROVER,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
            AgentState(id="relay_1", agent_type=AgentType.RELAY,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
            AgentState(id="relay_2", agent_type=AgentType.RELAY,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
            AgentState(id="sampler_1", agent_type=AgentType.SAMPLER,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
            AgentState(id="sampler_2", agent_type=AgentType.SAMPLER,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
            AgentState(id="sampler_3", agent_type=AgentType.SAMPLER,
                       energy_wh=8000.0, initial_energy_wh=8000.0),
        ]
    return agents


# ---------------------------------------------------------------------------
# Scheduling Environment
# ---------------------------------------------------------------------------

class SchedulingEnv(gym.Env):
    """
    Gymnasium environment for task-agent assignment.

    Each step: agent picks one (task, agent) pair to assign.
    When all tasks are assigned, the greedy scheduler computes timing,
    the simulator runs, and terminal reward is computed.

    Compatible with sb3-contrib MaskablePPO via action_masks() method.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        corpus_dir,                       # str | Path | list[str|Path]
        reward_config: Optional[RewardConfig] = None,
        comm_range: float = 500.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.corpus = load_corpus(corpus_dir)
        if not self.corpus:
            raise ValueError(f"No corpus files found in {corpus_dir}")

        self.reward_config = reward_config or RewardConfig()
        self.encoder = StateEncoder(comm_range=comm_range)
        self.render_mode = render_mode

        # Action space: flat index into (task_idx, agent_idx) grid
        self.action_space = spaces.Discrete(MAX_TASKS * MAX_AGENTS)

        # Observation space: fixed-dim pooled vector
        obs_dim = self.encoder.raw_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State (set in reset())
        self._task_graph: Optional[TaskGraph] = None
        self._agents: List[AgentState] = []
        self._assigned: Dict[str, str] = {}
        self._greedy_makespan: float = 1.0
        self._corpus_entry: Dict[str, Any] = {}
        self._step_count: int = 0
        self._corpus_subset: Optional[List[int]] = None  # Curriculum learning
        self._agent_load: Dict[str, float] = {}  # Cumulative assigned duration per agent
        self._predecessor_map: Optional[Dict[str, set]] = None  # per-episode cache for parallelism bonus

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pick a random TaskGraph from corpus and reset state."""
        super().reset(seed=seed)

        # Select a corpus entry (respecting curriculum subset)
        if options and "corpus_index" in options:
            idx = options["corpus_index"]
        elif self._corpus_subset is not None:
            idx = random.choice(self._corpus_subset)
        else:
            idx = random.randint(0, len(self.corpus) - 1)

        entry = self.corpus[idx]
        self._corpus_entry = entry

        # Reconstruct TaskGraph from JSON
        self._task_graph = TaskGraph(**entry["taskgraph"])
        self._agents = _agents_from_metadata(entry["metadata"])
        self._assigned = {}
        self._step_count = 0
        self._agent_load = {a.id: 0.0 for a in self._agents}
        self._predecessor_map = None  # invalidate cache; rebuilt lazily on first reward call

        # Greedy baseline makespan for reward normalization
        self._greedy_makespan = entry.get("baseline_sim_result", {}).get(
            "makespan", 1.0
        )
        if self._greedy_makespan <= 0:
            self._greedy_makespan = 1.0

        obs = self._get_obs()
        info = {
            "corpus_file": entry.get("metadata", {}).get("variant_description", ""),
            "num_tasks": len(self._task_graph.nodes),
            "num_agents": len(self._agents),
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one assignment: decode action → (task_id, agent_id).

        Returns:
            obs, reward, terminated, truncated, info
        """
        task_idx = action // MAX_AGENTS
        agent_idx = action % MAX_AGENTS

        reward = 0.0
        info: Dict[str, Any] = {}

        # Validate action
        num_tasks = len(self._task_graph.nodes)
        num_agents = len(self._agents)

        if task_idx >= num_tasks or agent_idx >= num_agents:
            # Invalid action (beyond real nodes) — should be masked out
            # Give small penalty and skip
            reward = -0.1
            info["error"] = "action_out_of_range"
        else:
            task_node = self._task_graph.nodes[task_idx]
            agent = self._agents[agent_idx]

            if task_node.id in self._assigned:
                # Already assigned — should be masked
                reward = -0.1
                info["error"] = "already_assigned"
            else:
                # Valid assignment
                self._assigned[task_node.id] = agent.id
                self._step_count += 1

                dur = task_node.estimated_duration_s or 60.0
                self._agent_load[agent.id] = self._agent_load.get(agent.id, 0.0) + dur

                # Step-level shaping reward with load-balance + parallelism
                reward = self._compute_shaping_reward(task_node, agent, dur)

        # Check if all tasks are assigned
        all_assigned = len(self._assigned) >= len(self._task_graph.nodes)

        # Also terminate if no valid actions remain (prevent empty mask)
        if not all_assigned:
            future_mask = self.encoder.compute_action_mask(
                self._task_graph, self._agents, self._assigned
            )
            if future_mask.sum() == 0:
                all_assigned = True  # Force termination

        if all_assigned:
            # Terminal step: run simulation and compute full reward
            terminal_reward, sim_result = self._run_simulation()
            reward = terminal_reward
            info["terminal"] = True
            info["assignments"] = dict(self._assigned)
            if sim_result:
                info["sim_result"] = sim_result

        obs = self._get_obs()
        terminated = all_assigned
        truncated = self._step_count > MAX_TASKS * 2  # safety cutoff

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return binary mask for MaskablePPO.

        Shape: (MAX_TASKS * MAX_AGENTS,)

        SAFETY: MaskablePPO crashes if all actions are masked (Simplex violation).
        If the real mask is all-zero (all tasks assigned or no valid pairs),
        we set a single fallback bit to prevent the crash. The step() method
        handles this gracefully by terminating the episode.
        """
        if self._task_graph is None:
            # Fallback: allow action 0 to prevent empty mask
            mask = np.zeros(MAX_TASKS * MAX_AGENTS, dtype=np.float32)
            mask[0] = 1.0
            return mask

        mask = self.encoder.compute_action_mask(
            self._task_graph, self._agents, self._assigned
        )

        # Safety: if mask is all-zero, set a fallback bit
        if mask.sum() == 0:
            mask[0] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Custom PPO support methods
    # ------------------------------------------------------------------

    def get_hetero_state(self):
        """
        Return raw HeteroData graph state + action mask for custom PPO.

        Returns:
            (HeteroData, np.ndarray): Graph-structured observation and action mask.
        """
        if self._task_graph is None:
            from torch_geometric.data import HeteroData
            import torch
            hetero = HeteroData()
            hetero['task'].x = torch.zeros(1, 4)
            hetero['agent'].x = torch.zeros(1, 8)
            hetero.num_real_tasks = 0
            hetero.num_real_agents = 0
            mask = np.zeros(MAX_TASKS * MAX_AGENTS, dtype=np.float32)
            mask[0] = 1.0
            return hetero, mask

        hetero = self.encoder.encode(self._task_graph, self._agents, self._assigned)
        mask = self.action_masks()
        return hetero, mask

    def set_corpus_subset(self, indices: list):
        """
        Restrict episode sampling to a subset of corpus files (for curriculum learning).

        Args:
            indices: List of corpus indices to use, or None for all.
        """
        if indices is None:
            self._corpus_subset = None
        else:
            # Clamp to valid range
            valid = [i for i in indices if 0 <= i < len(self.corpus)]
            self._corpus_subset = valid if valid else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Compute current observation (pooled, for SB3 compatibility)."""
        if self._task_graph is None:
            return np.zeros(self.encoder.raw_obs_dim, dtype=np.float32)

        hetero = self.encoder.encode(self._task_graph, self._agents, self._assigned)

        obs = self.encoder.get_pooled_observation(hetero)

        # Update progress fraction (global feature #3)
        total = len(self._task_graph.nodes)
        progress = len(self._assigned) / max(total, 1)
        obs[-1] = progress

        return obs

    def _compute_shaping_reward(
        self, task_node, agent, dur: float
    ) -> float:
        """
        Step-level shaping reward with three components:
        1. Base: small negative proportional to duration (encourage efficiency)
        2. Load balance: penalty when agent load exceeds 1.5× average
        3. Parallelism: bonus for unlocking new ready tasks
        """
        # --- 1. Base shaping ---
        base = -0.01 * (dur / max(self._greedy_makespan, 1.0))

        # --- 2. Load balance penalty ---
        loads = list(self._agent_load.values())
        avg_load = sum(loads) / max(len(loads), 1)
        agent_load = self._agent_load.get(agent.id, 0.0)
        balance_penalty = 0.0
        if avg_load > 0 and agent_load > 1.5 * avg_load:
            balance_penalty = -0.02 * (agent_load / avg_load - 1.0)

        # --- 3. Parallelism bonus ---
        # Count how many new tasks become "ready" after this assignment.
        # Caching a predecessor map once per episode reduces this step from
        # O(|E|^2) to O(out-degree of task_node), which is essential on dense
        # graphs (n_edges in the thousands).
        if self._predecessor_map is None:
            self._predecessor_map = {n.id: set() for n in self._task_graph.nodes}
            for e in self._task_graph.edges:
                if e.type in (EdgeType.ORDER, EdgeType.DATA):
                    if e.dst in self._predecessor_map:
                        self._predecessor_map[e.dst].add(e.src)
        parallelism_bonus = 0.0
        for edge in self._task_graph.edges:
            if edge.type in (EdgeType.ORDER, EdgeType.DATA):
                if edge.src == task_node.id and edge.dst not in self._assigned:
                    preds_of_dst = self._predecessor_map.get(edge.dst, ())
                    if all(p in self._assigned for p in preds_of_dst):
                        parallelism_bonus += 0.01  # Newly unlocked task

        return base + balance_penalty + parallelism_bonus

    def _run_simulation(self) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Run greedy time-slot scheduling + lunar simulation.

        Returns terminal reward and the simulation result dict (if successful).
        """
        # Convert assigned dict to greedy scheduler format
        # We need AgentInfo objects
        agent_infos = [
            AgentInfo(
                id=a.id,
                agent_type=a.agent_type,
                energy_wh=a.energy_wh,
            )
            for a in self._agents
        ]

        # Run greedy scheduler (it will use our assignments for timing)
        try:
            sched = greedy_schedule(self._task_graph, agent_infos)
        except Exception:
            # If scheduling fails, maximum penalty
            return -10.0, None

        # Run simulation
        terrain = TerrainGrid(width=400, height=400, cell_size=10.0)
        sim = LunarSimulator(terrain, self._agents, SimConfig())

        try:
            result = sim.evaluate(self._task_graph, sched)
        except Exception:
            return -10.0, None

        reward = compute_reward(result, self._greedy_makespan, self.reward_config)
        return reward, result
