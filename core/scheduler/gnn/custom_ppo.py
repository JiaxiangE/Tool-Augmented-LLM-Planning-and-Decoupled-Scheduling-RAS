"""
Custom PPO Trainer for GNN+DRL Scheduling.

Replaces SB3 MaskablePPO to feed HeteroData directly to SchedulingPolicyNet,
preserving node-level graph structure (no information bottleneck from pooling).

Core components:
  - RolloutBuffer: stores (hetero_data, action, reward, done, log_prob, value, mask)
  - compute_gae(): Generalized Advantage Estimation
  - update(): clipped surrogate loss + value loss + entropy bonus
  - train(): main loop with curriculum learning schedule
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import HeteroData

from core.scheduler.gnn.policy_network import SchedulingPolicyNet
from core.scheduler.gnn.state_encoder import StateEncoder, MAX_TASKS, MAX_AGENTS

logger = logging.getLogger(__name__)

# Disable simplex validation for large masked softmax
torch.distributions.Distribution.set_default_validate_args(False)


# ---------------------------------------------------------------------------
# Reward Normalization (SB3 VecNormalize style)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Online Welford-style running mean/variance tracker."""

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = int(x.size)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RewardNormalizer:
    """Normalize rewards by running std of *discounted returns*.

    Mirrors SB3 VecNormalize: divides reward by std(running discounted return),
    *without* mean subtraction (mean subtraction would shift the optimal policy
    on infinite-horizon problems). The running stats are updated online from
    each rollout's discounted-return sequence.

    Use when raw rewards have large per-step magnitude (e.g. large-scale
    scenarios where untrained PPO returns are an order of magnitude worse
    than the greedy baseline, so value loss explodes and PPO advantage
    estimates become noisy).
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.returns_rms = RunningMeanStd()
        self.gamma = gamma
        self.epsilon = epsilon
        self._running_return = 0.0  # online discounted return across rollouts

    def normalize(self, rewards: list, dones: list) -> list:
        """Update running stats from this rollout and return normalized rewards.

        Matches SB3 VecNormalize.normalize_reward exactly:
          (1) accumulate first  :  r_t = gamma * r_{t-1} + reward_t   (no done mask)
          (2) log the result    :  returns_rms.update([r_0, r_1, ..., r_T])
          (3) reset AFTER log   :  if dones[t]: r = 0   (only AFTER logging r_t)

        The accumulation-then-log-then-reset order is important on long
        episodes (e.g. n_tasks ~ 480, gamma = 0.99): the discounted
        accumulator at the terminal step can be two orders of magnitude
        larger than the terminal reward itself, so a misplaced done mask
        biases the running variance noticeably.
        """
        discounted = []
        r = self._running_return
        for reward, done in zip(rewards, dones):
            r = self.gamma * r + float(reward)   # (1) accumulate, no done mask
            discounted.append(r)                  # (2) log this value
            if done:
                r = 0.0                           # (3) reset only after logging
        self._running_return = r  # persist for next rollout boundary
        self.returns_rms.update(np.asarray(discounted))
        std = float(np.sqrt(self.returns_rms.var))
        return [float(rwd) / (std + self.epsilon) for rwd in rewards]


# ---------------------------------------------------------------------------
# Rollout Storage
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A single environment transition."""
    hetero_data: HeteroData
    action: int
    reward: float
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor
    action_mask: np.ndarray


class RolloutBuffer:
    """Stores transitions for one rollout collection phase."""

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)


# ---------------------------------------------------------------------------
# GAE Computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation.

    Returns:
        advantages: Per-step advantages.
        returns: Per-step discounted returns (advantage + value).
    """
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n

    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = last_value
            next_done = False
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]

        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


# ---------------------------------------------------------------------------
# Custom PPO Trainer
# ---------------------------------------------------------------------------

class CustomPPOTrainer:
    """
    PPO trainer that feeds HeteroData directly to SchedulingPolicyNet.

    No SB3 dependency — handles rollout collection, GAE, and policy updates
    with raw graph-structured observations.
    """

    def __init__(
        self,
        policy: SchedulingPolicyNet,
        env,  # SchedulingEnv
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        log_dir: str = "experiments/gnn_drl/tensorboard/custom",
        lr_schedule: str = "constant",   # "constant" | "reduce_on_plateau" | "cosine"
        device: str = "cpu",             # "cpu" | "cuda" | "cuda:N"
        reward_normalization: bool = False,   # VecNormalize-style reward rescaling
        lr_min: float = 1e-5,            # eta_min for cosine schedule
    ):
        # Device handling. Default "cpu" gives deterministic, hardware-independent
        # results; "cuda" / "cuda:N" enables GPU training for the large-scale
        # corpora where CPU-only training is impractical.
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._init_lr = lr  # remember initial LR for cosine T_max calc
        # Optional LR schedules:
        #   "constant"           : no schedule (default)
        #   "reduce_on_plateau"  : halve lr when rollout mean_reward plateaus
        #   "cosine"             : cosine anneal lr -> lr_min over remaining iterations
        # For cosine the scheduler is built lazily in train() once total_steps is known.
        self.lr_schedule = lr_schedule
        self.lr_min = lr_min
        if lr_schedule == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
            )
        elif lr_schedule == "constant":
            self.scheduler = None
        elif lr_schedule == "cosine":
            # Built in train() once we know total_steps; placeholder here.
            self.scheduler = None
        else:
            raise ValueError(f"unknown lr_schedule: {lr_schedule!r} "
                             f"(expected 'constant' | 'reduce_on_plateau' | 'cosine')")

        # Reward normalization (SB3 VecNormalize style; off by default).
        self.reward_normalization = reward_normalization
        self.reward_normalizer = (
            RewardNormalizer(gamma=gamma) if reward_normalization else None
        )

        self.buffer = RolloutBuffer()

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.episode_count = 0

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path) -> None:
        """Save full trainer state: policy + optimizer + step counters.

        Persisting the optimizer state lets `--resume-from` fine-tunes
        continue with the existing Adam first/second-moment estimates,
        avoiding the few-hundred-step optimizer re-warm-up that occurs
        when only the policy weights are reloaded.
        """
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load_checkpoint(self, path, map_location: str = "cpu") -> Dict[str, Any]:
        """Load a checkpoint produced by `save_checkpoint`.

        Backward compatible with the legacy bare-state_dict format (policy
        weights only): when resuming from such a checkpoint only the policy
        is restored and the optimizer starts fresh (the caller is expected
        to set `global_step` via `--start-step`).
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
            self.policy.load_state_dict(ckpt["policy_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.global_step = int(ckpt.get("global_step", 0))
            self.episode_count = int(ckpt.get("episode_count", 0))
            return {"format": "full", "optimizer_restored": True,
                    "global_step": self.global_step}
        self.policy.load_state_dict(ckpt)
        return {"format": "legacy_policy_only", "optimizer_restored": False,
                "global_step": None}

    # ------------------------------------------------------------------
    # Rollout Collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> Dict[str, float]:
        """
        Collect n_steps of experience using the current policy.

        Returns dict with rollout stats.
        """
        self.buffer.clear()
        self.policy.eval()

        ep_rewards = []
        ep_lengths = []
        current_ep_reward = 0.0
        current_ep_len = 0

        # Reset env if needed
        obs, info = self.env.reset()
        hetero, mask = self.env.get_hetero_state()

        with torch.no_grad():
            for _ in range(self.n_steps):
                # Move per-step state onto trainer device (no-op when device='cpu').
                # HeteroData supports .to(device) (PyG built-in); buffer stores the
                # device-resident copy so update() re-uses it without re-transfer.
                hetero_dev = hetero.to(self.device) if str(self.device) != "cpu" else hetero
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
                action, log_prob, value = self.policy.get_action(
                    hetero_dev, mask_tensor, deterministic=False
                )

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(Transition(
                    hetero_data=hetero_dev,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob.detach(),
                    value=value.detach().squeeze(),
                    action_mask=mask.copy(),
                ))

                current_ep_reward += reward
                current_ep_len += 1
                self.global_step += 1

                if done:
                    ep_rewards.append(current_ep_reward)
                    ep_lengths.append(current_ep_len)
                    self.episode_count += 1
                    current_ep_reward = 0.0
                    current_ep_len = 0
                    obs, info = self.env.reset()

                hetero, mask = self.env.get_hetero_state()

        stats = {
            "episodes": len(ep_rewards),
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_ep_len": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "std_reward": float(np.std(ep_rewards)) if ep_rewards else 0.0,
        }
        return stats

    # ------------------------------------------------------------------
    # Policy Update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        Run PPO update on collected rollout.

        Returns dict with training stats.
        """
        self.policy.train()

        transitions = self.buffer.transitions
        n = len(transitions)
        if n == 0:
            return {}

        # Extract arrays
        rewards = [t.reward for t in transitions]
        values = [t.value.item() for t in transitions]
        dones = [t.done for t in transitions]

        # Optional VecNormalize-style reward rescale.
        # Divides each reward by running std(discounted return); keeps sign and
        # relative magnitudes intact but bounds value-loss / advantage scale.
        # Important when reward magnitudes are large per episode, where
        # otherwise the value loss saturates and PPO advantage estimates
        # become very noisy.
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.normalize(rewards, dones)

        # GAE
        last_value = 0.0  # terminal
        advantages, returns = compute_gae(
            rewards, values, dones, self.gamma, self.lam, last_value
        )

        # Convert to tensors on the trainer device
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.stack([t.log_prob.to(self.device) for t in transitions])

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0

        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n)

            for idx in indices:
                t = transitions[idx]
                adv = advantages_t[idx]
                ret = returns_t[idx]
                old_lp = old_log_probs_t[idx]

                # Forward pass — mask + hetero on trainer device;
                # HeteroData.to(device) is a no-op when already on that device.
                mask_tensor = torch.tensor(t.action_mask, dtype=torch.float32,
                                            device=self.device)
                hetero_dev = (t.hetero_data.to(self.device)
                              if str(self.device) != "cpu" else t.hetero_data)
                logits, value = self.policy.forward(hetero_dev)

                # Masked softmax
                masked_logits = logits.clone()
                masked_logits[mask_tensor == 0] = float("-inf")
                log_probs_all = F.log_softmax(masked_logits, dim=-1)
                probs_all = torch.exp(log_probs_all)

                new_log_prob = log_probs_all[t.action]

                # Entropy (over valid actions only)
                valid_probs = probs_all[mask_tensor == 1]
                if valid_probs.numel() > 0:
                    entropy = -(valid_probs * torch.log(valid_probs + 1e-10)).sum()
                else:
                    entropy = torch.tensor(0.0, device=self.device)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_prob - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2)

                # Value loss
                value_loss = F.mse_loss(value.squeeze(), ret)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_frac += (torch.abs(ratio - 1.0) > self.clip_eps).float().item()

        num_updates = self.n_epochs * n
        stats = {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "clip_fraction": total_clip_frac / max(num_updates, 1),
        }
        return stats

    # ------------------------------------------------------------------
    # Main Training Loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_steps: int = 500_000,
        curriculum_schedule: Optional[Dict[int, List[int]]] = None,
        checkpoint_dir: str = "experiments/gnn_drl/checkpoints_custom",
        checkpoint_interval: int = 50_000,
        log_interval: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the policy with PPO.

        Args:
            total_steps: Total environment steps.
            curriculum_schedule: {step_threshold: [corpus_indices]}.
                Example: {0: [0,1,2], 100000: [0,1,2,3,4,5], 300000: None}
                None = use all corpus files.
            checkpoint_dir: Where to save model checkpoints.
            checkpoint_interval: Steps between checkpoints.
            log_interval: Rollouts between logging.

        Returns:
            Training summary dict.
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting custom PPO training: {total_steps} steps")
        start_time = time.time()

        # Lazily build cosine scheduler now that we know total_steps.
        # T_max is in optimizer.step() units; for simplicity we step once per
        # rollout iteration, so n_iters ~ (remaining_steps / n_steps).
        if self.lr_schedule == "cosine" and self.scheduler is None:
            remaining_steps = max(1, total_steps - self.global_step)
            n_iters = max(1, remaining_steps // self.n_steps)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_iters, eta_min=self.lr_min,
            )
            logger.info(f"cosine LR: T_max={n_iters} iterations, "
                        f"eta_min={self.lr_min:.2e} (init lr={self._init_lr:.2e})")

        iteration = 0
        all_rewards = []
        last_checkpoint_step = 0
        prev_corpus_subset_key = None  # detect curriculum stage transitions

        while self.global_step < total_steps:
            # Apply curriculum
            if curriculum_schedule:
                corpus_subset = None
                for threshold in sorted(curriculum_schedule.keys()):
                    if self.global_step >= threshold:
                        corpus_subset = curriculum_schedule[threshold]
                if corpus_subset is not None:
                    self.env.set_corpus_subset(corpus_subset)
                # At every curriculum stage boundary, reset the reward
                # normalizer. The running std built up on the previous stage's
                # reward magnitude lags badly when the new stage's reward
                # magnitudes are much larger; resetting gives the normalizer
                # a clean start for the new reward regime and avoids several
                # thousand steps of degraded advantage estimates per transition.
                subset_key = (tuple(corpus_subset)
                              if corpus_subset is not None else "ALL")
                if (subset_key != prev_corpus_subset_key
                        and prev_corpus_subset_key is not None
                        and self.reward_normalizer is not None):
                    logger.info(
                        f"curriculum transition at step={self.global_step}: "
                        f"resetting reward_normalizer "
                        f"(was tracking returns from prev stage)")
                    self.reward_normalizer = RewardNormalizer(gamma=self.gamma)
                prev_corpus_subset_key = subset_key

            # Collect rollout
            rollout_stats = self.collect_rollout()

            # Update policy
            update_stats = self.update()

            iteration += 1

            # Logging
            if rollout_stats["episodes"] > 0:
                all_rewards.append(rollout_stats["mean_reward"])

            # LR scheduler step.
            # ReduceLROnPlateau takes the rollout reward; CosineAnnealingLR
            # takes nothing and steps once per iteration regardless of episodes.
            if self.scheduler is not None:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if rollout_stats["episodes"] > 0:
                        self.scheduler.step(rollout_stats["mean_reward"])
                else:
                    # Cosine (or any future epoch-driven scheduler) steps each iter.
                    self.scheduler.step()

            if iteration % log_interval == 0:
                cur_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("rollout/mean_reward", rollout_stats["mean_reward"], self.global_step)
                self.writer.add_scalar("rollout/mean_ep_len", rollout_stats["mean_ep_len"], self.global_step)
                self.writer.add_scalar("rollout/episodes", rollout_stats["episodes"], self.global_step)
                self.writer.add_scalar("train/lr", cur_lr, self.global_step)

                if update_stats:
                    self.writer.add_scalar("train/policy_loss", update_stats["policy_loss"], self.global_step)
                    self.writer.add_scalar("train/value_loss", update_stats["value_loss"], self.global_step)
                    self.writer.add_scalar("train/entropy", update_stats["entropy"], self.global_step)
                    self.writer.add_scalar("train/clip_fraction", update_stats["clip_fraction"], self.global_step)

                logger.info(
                    f"[Step {self.global_step}/{total_steps}] "
                    f"reward={rollout_stats['mean_reward']:.4f} "
                    f"ep_len={rollout_stats['mean_ep_len']:.1f} "
                    f"pl={update_stats.get('policy_loss', 0):.4f} "
                    f"vl={update_stats.get('value_loss', 0):.4f} "
                    f"ent={update_stats.get('entropy', 0):.3f} "
                    f"lr={cur_lr:.2e}"
                )

            # Checkpoint
            if self.global_step - last_checkpoint_step >= checkpoint_interval:
                ckpt_path = Path(checkpoint_dir) / f"gnn_drl_step_{self.global_step}.pt"
                self.save_checkpoint(ckpt_path)
                last_checkpoint_step = self.global_step
                logger.info(f"Checkpoint saved: {ckpt_path}")

        elapsed = time.time() - start_time

        # Save final model
        final_path = Path(checkpoint_dir) / "gnn_drl_custom_final.pt"
        self.save_checkpoint(final_path)

        # Summary
        summary = {
            "total_steps": self.global_step,
            "elapsed_seconds": round(elapsed, 1),
            "iterations": iteration,
            "episodes": self.episode_count,
            "final_model_path": str(final_path),
        }

        if all_rewards:
            summary["final_mean_reward"] = round(float(np.mean(all_rewards[-10:])), 4)
            n = len(all_rewards)
            if n >= 4:
                early = np.mean(all_rewards[:n // 4])
                late = np.mean(all_rewards[-n // 4:])
                summary["early_mean_reward"] = round(float(early), 4)
                summary["late_mean_reward"] = round(float(late), 4)
                summary["reward_trend"] = "improving" if late > early + 0.001 else "flat/declining"

        self.writer.close()
        logger.info(f"Training complete: {summary}")
        return summary
