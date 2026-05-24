"""
Multi-Objective Reward Function for GNN+DRL Scheduling.

R = -α·(makespan / greedy_cpm_makespan) + λ1·completion + λ2·energy_pen + λ3·viol_pen

The makespan is normalized by the greedy+CPM baseline stored in the training
corpus JSON. α=5.0 is deliberately large: the greedy+CPM baseline is already
close to optimal, so makespan differences between strategies are small (2-10%),
and a high α amplifies that signal for stronger policy gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.simulator.lunar_sim import SimResult


@dataclass
class RewardConfig:
    """Hyperparameters for the reward function."""
    alpha: float = 0.1       # makespan weight (kept small so it does not drown other signals)
    lambda_1: float = 2.0    # completion-rate weight (primary driver in early training)
    lambda_2: float = 0.3    # energy penalty weight (secondary objective)
    lambda_3: float = 1.0    # constraint-violation penalty weight


def compute_reward(
    sim_result: SimResult,
    greedy_makespan: float,
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Compute the terminal reward after full simulation.

    Args:
        sim_result: Result from LunarSimulator.evaluate().
        greedy_makespan: Baseline makespan from greedy scheduler (from corpus JSON).
        config: Reward hyperparameters.

    Returns:
        Scalar reward value. Higher is better.
        A value of ~0 means "roughly same as greedy baseline".
        Positive = better than greedy, negative = worse.
    """
    if config is None:
        config = RewardConfig()

    # Guard against zero/invalid greedy baseline
    if greedy_makespan <= 0:
        greedy_makespan = max(sim_result.makespan, 1.0)

    # 1. Normalized makespan (lower is better → negate)
    makespan_norm = sim_result.makespan / greedy_makespan
    makespan_term = -config.alpha * makespan_norm

    # 2. Completion rate (higher is better)
    completion_term = config.lambda_1 * sim_result.task_completion_rate

    # 3. Energy penalty (mean energy consumption fraction, lower is better)
    if sim_result.energy_consumption:
        # energy_consumption is Dict[agent_id, Wh consumed]
        # We want fraction consumed — need initial energy which isn't in SimResult
        # So we use a proxy: penalize total energy consumed relative to max capacity
        total_consumed = sum(sim_result.energy_consumption.values())
        num_agents = max(len(sim_result.energy_consumption), 1)
        # Assume typical agent has ~5000 Wh → normalize
        energy_frac = total_consumed / (num_agents * 5000.0)
        energy_term = -config.lambda_2 * min(energy_frac, 2.0)  # cap at 2x
    else:
        energy_term = 0.0

    # 4. Constraint violation penalty (count of violations)
    num_violations = len(sim_result.constraint_violations)
    violation_term = -config.lambda_3 * num_violations

    return makespan_term + completion_term + energy_term + violation_term


def compute_step_reward(
    estimated_duration: float,
    greedy_makespan: float,
) -> float:
    """
    Small shaping reward for intermediate steps (before episode end).

    Gives a tiny negative reward proportional to the task duration
    to encourage efficient assignment ordering.

    Args:
        estimated_duration: Duration of the task just assigned (seconds).
        greedy_makespan: Baseline makespan for normalization.

    Returns:
        Small negative shaping reward.
    """
    if greedy_makespan <= 0:
        greedy_makespan = 1.0
    return -0.01 * (estimated_duration / greedy_makespan)
