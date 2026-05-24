"""
Base environment definitions.

Provides a unified environment state object connecting the Simulation,
Planning, and Scheduling layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.schema.agent_types import AgentType


@dataclass
class AgentState:
    """Represents the real-time or configured state of an agent."""
    id: str
    agent_type: AgentType
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    energy_wh: float = 100.0
    initial_energy_wh: float = 100.0
    speed_m_per_s: float = 0.5
    
    @property
    def energy_consumed(self) -> float:
        return self.initial_energy_wh - self.energy_wh


@dataclass
class DomainConfig:
    """Mission parameters encompassing all physical constraints configuration."""
    base_station_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    comm_range_m: float = 500.0
    low_energy_threshold_wh: float = 10.0


@dataclass
class EnvironmentState:
    """
    Unified environment state passed between all system layers.
    Includes current agent states, world domain configuration, and map references.
    """
    agents: Dict[str, AgentState] = field(default_factory=dict)
    config: DomainConfig = field(default_factory=DomainConfig)
    
    # Path to the terrain YAML configuration if applicable
    terrain_config_path: str = ""

    def get_available_agents(self) -> List[AgentState]:
        return list(self.agents.values())
