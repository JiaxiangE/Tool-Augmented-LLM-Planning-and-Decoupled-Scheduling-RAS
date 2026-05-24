"""
PPO Trainer for GNN+DRL Scheduling.

Uses sb3-contrib MaskablePPO with a custom feature extractor that wraps
the GNN state encoder and a pooling layer.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure as sb3_configure_logger

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState
from core.schema.taskgraph import TaskGraph
from core.scheduler.gnn.env import SchedulingEnv
from core.scheduler.gnn.reward import RewardConfig
from core.scheduler.gnn.state_encoder import StateEncoder
from core.scheduler.greedy_scheduler import AgentInfo, schedule as greedy_schedule
from core.simulator.terrain import TerrainGrid
from core.simulator.lunar_sim import LunarSimulator, SimConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action mask wrapper for MaskablePPO
# ---------------------------------------------------------------------------

def _mask_fn(env: gym.Env) -> np.ndarray:
    """Extract action mask from our SchedulingEnv."""
    return env.action_masks()


# ---------------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------------

class ConvergenceCheckCallback(BaseCallback):
    """
    Check that reward improves beyond random baseline at a specified step.
    Logs a warning if not.
    """

    def __init__(self, check_at_step: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.check_at_step = check_at_step
        self._checked = False

    def _on_step(self) -> bool:
        if not self._checked and self.num_timesteps >= self.check_at_step:
            self._checked = True
            # Check recent mean reward
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean(
                    [ep["r"] for ep in self.model.ep_info_buffer]
                )
                logger.info(
                    f"[ConvergenceCheck @{self.num_timesteps} steps] "
                    f"Mean episode reward: {mean_reward:.4f}"
                )
                if mean_reward < -5.0:
                    logger.warning(
                        "⚠ Reward is very low at convergence checkpoint. "
                        "Check reward function or state encoding."
                    )
            else:
                logger.warning(
                    "⚠ No episodes completed by convergence checkpoint."
                )
        return True


class TrainingProgressCallback(BaseCallback):
    """Log training progress periodically."""

    def __init__(self, log_interval: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            self._last_log = self.num_timesteps
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                logger.info(
                    f"[Step {self.num_timesteps}] "
                    f"mean_reward={mean_r:.4f}, mean_ep_len={mean_len:.1f}"
                )
        return True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    High-level trainer for the GNN+DRL scheduling policy.

    Wraps MaskablePPO with our SchedulingEnv and handles the
    multi-phase training protocol.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        log_dir: str | Path = "experiments/gnn_drl/tensorboard",
        checkpoint_dir: str | Path = "experiments/gnn_drl/checkpoints",
        reward_config: Optional[RewardConfig] = None,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.corpus_dir = Path(corpus_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.reward_config = reward_config or RewardConfig()
        self.device = device

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create environment with action masking wrapper
        base_env = SchedulingEnv(
            corpus_dir=self.corpus_dir,
            reward_config=self.reward_config,
        )
        self.env = ActionMasker(base_env, _mask_fn)

        # Create MaskablePPO model
        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=1,
            device=device,
            tensorboard_log=str(self.log_dir),
        )

        logger.info(
            f"Trainer initialized: corpus={self.corpus_dir}, "
            f"device={device}, lr={learning_rate}"
        )

    def train(
        self,
        total_timesteps: int = 200_000,
        checkpoint_interval: int = 50_000,
        convergence_check_at: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Run training.

        Args:
            total_timesteps: Total environment steps.
            checkpoint_interval: Save model every N steps.
            convergence_check_at: Check convergence at this step count.

        Returns:
            Dict with training summary.
        """
        logger.info(f"Starting training: {total_timesteps} timesteps")
        start_time = time.time()

        # Callbacks
        callbacks = [
            ConvergenceCheckCallback(check_at_step=convergence_check_at),
            TrainingProgressCallback(log_interval=5000),
            CheckpointCallback(
                save_freq=checkpoint_interval,
                save_path=str(self.checkpoint_dir),
                name_prefix="scheduling_policy",
            ),
        ]

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        elapsed = time.time() - start_time

        # Save final model
        final_path = self.checkpoint_dir / "scheduling_policy_final"
        self.model.save(str(final_path))

        summary = {
            "total_timesteps": total_timesteps,
            "elapsed_seconds": elapsed,
            "final_model_path": str(final_path),
        }

        if len(self.model.ep_info_buffer) > 0:
            summary["final_mean_reward"] = float(
                np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            )
            summary["final_mean_ep_len"] = float(
                np.mean([ep["l"] for ep in self.model.ep_info_buffer])
            )

        logger.info(f"Training complete: {summary}")
        return summary

    def load(self, path: str | Path):
        """Load a saved model."""
        self.model = MaskablePPO.load(str(path), env=self.env)
        logger.info(f"Loaded model from {path}")

    def evaluate(
        self,
        task_graph: TaskGraph,
        agents: List[AgentState],
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Use the trained policy to schedule a TaskGraph.

        Args:
            task_graph: TaskGraph to schedule.
            agents: Available agents.
            deterministic: Use greedy (argmax) policy.

        Returns:
            Dict with schedule, sim_result, and timing info.
        """
        start_time = time.time()

        # Create a temporary env for this graph
        # We need to run through the episode manually
        encoder = StateEncoder()
        assigned: Dict[str, str] = {}
        task_ids = [n.id for n in task_graph.nodes]
        agent_ids = [a.id for a in agents]

        max_steps = len(task_ids) * 2  # safety limit
        step = 0

        while len(assigned) < len(task_ids) and step < max_steps:
            # Compute action mask
            mask = encoder.compute_action_mask(task_graph, agents, assigned)

            if mask.sum() == 0:
                logger.warning("No valid actions available — breaking")
                break

            # Encode state using env's observation logic
            hetero = encoder.encode(task_graph, agents, assigned)
            obs = encoder.get_pooled_observation(hetero)
            progress = len(assigned) / max(len(task_ids), 1)
            obs[-1] = progress

            # Get action from policy
            action, _ = self.model.predict(
                obs,
                action_masks=mask,
                deterministic=deterministic,
            )

            # Decode action
            from core.scheduler.gnn.state_encoder import MAX_AGENTS
            task_idx = int(action) // MAX_AGENTS
            agent_idx = int(action) % MAX_AGENTS

            if task_idx < len(task_ids) and agent_idx < len(agents):
                tid = task_ids[task_idx]
                aid = agent_ids[agent_idx]
                if tid not in assigned:
                    assigned[tid] = aid

            step += 1

        inference_time = time.time() - start_time

        # Run greedy scheduler for timing
        agent_infos = [
            AgentInfo(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh)
            for a in agents
        ]
        schedule = greedy_schedule(task_graph, agent_infos)

        # Simulate
        terrain = TerrainGrid(width=400, height=400, cell_size=10.0)
        sim = LunarSimulator(terrain, agents, SimConfig())
        sim_result = sim.evaluate(task_graph, schedule)

        return {
            "schedule": schedule,
            "sim_result": sim_result,
            "assignments": assigned,
            "inference_time_s": inference_time,
            "steps": step,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "data/training_corpus"
    timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000

    trainer = Trainer(corpus_dir=corpus_dir)
    summary = trainer.train(total_timesteps=timesteps)

    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
