"""
Communication connectivity model for lunar surface agents.

Simple distance-based check: an agent is connected if it is within
comm_range of any relay node or the base station.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


_DEFAULT_COMM_RANGE = 500.0  # meters


@dataclass
class CommConfig:
    """Tunable communication model parameters."""
    comm_range: float = _DEFAULT_COMM_RANGE


class CommModel:
    """
    Deterministic communication connectivity model.

    An agent is considered connected if its distance to at least one relay
    or the base station is within the configured comm_range.
    """

    def __init__(self, config: Optional[CommConfig] = None):
        self.cfg = config or CommConfig()

    def is_connected(
        self,
        agent_pos: Tuple[float, float, float],
        relay_positions: List[Tuple[float, float, float]],
        base_pos: Tuple[float, float, float],
    ) -> bool:
        """
        Check if an agent at `agent_pos` has communication connectivity.

        Args:
            agent_pos: Agent's current 3D position.
            relay_positions: List of active relay node positions.
            base_pos: Base station position.

        Returns:
            True if connected to at least one relay or base.
        """
        # Check distance to base
        if self._distance(agent_pos, base_pos) <= self.cfg.comm_range:
            return True

        # Check distance to any relay
        for rp in relay_positions:
            if self._distance(agent_pos, rp) <= self.cfg.comm_range:
                return True

        return False

    def connectivity_rate(
        self,
        checks: List[bool],
    ) -> float:
        """
        Compute fraction of time steps where connectivity was maintained.

        Args:
            checks: List of per-step connectivity booleans.

        Returns:
            Fraction in [0.0, 1.0].
        """
        if not checks:
            return 1.0
        return sum(1 for c in checks if c) / len(checks)

    @staticmethod
    def _distance(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        """3D Euclidean distance."""
        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2
        )
