"""Agent type taxonomy for the multi-agent lunar exploration system."""

from __future__ import annotations

from enum import Enum


class AgentType(str, Enum):
    """Type of agent in the lunar exploration fleet."""
    ROVER = "rover"        # mobile ground robot for exploration and traversal
    RELAY = "relay"        # communication relay node (static or mobile)
    SAMPLER = "sampler"    # sample collection specialist
    ANALYZER = "analyzer"  # on-site analysis instrument carrier
