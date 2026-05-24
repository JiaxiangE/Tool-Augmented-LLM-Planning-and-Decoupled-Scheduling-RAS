"""
Common scheduling types and primitives.

Includes the central Schedule specification data objects mapping tasks to agents on a timeline.
"""
from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field, model_validator


class ScheduleItem(BaseModel):
    """A single scheduled task assignment."""
    node_id: str
    agent: str
    start_s: float
    end_s: float


class Schedule(BaseModel):
    """A complete schedule mapping tasks to agents over time."""
    mission_id: str
    makespan_s: float
    items: List[ScheduleItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_overlaps(self) -> Schedule:
        """Ensure no agent is scheduled for overlapping tasks."""
        by_ag = self.by_agent()
        for agent, ag_items in by_ag.items():
            for i in range(1, len(ag_items)):
                if ag_items[i].start_s < ag_items[i-1].end_s:
                    raise ValueError(
                        f"Overlap detected for agent {agent}: "
                        f"task {ag_items[i-1].node_id} (ends {ag_items[i-1].end_s}) "
                        f"and task {ag_items[i].node_id} (starts {ag_items[i].start_s})"
                    )
        return self

    def by_agent(self) -> Dict[str, List[ScheduleItem]]:
        """Returns the schedule grouped by agent, sorted by start time."""
        result: Dict[str, List[ScheduleItem]] = {}
        for item in self.items:
            result.setdefault(item.agent, []).append(item)
        for agent_items in result.values():
            agent_items.sort(key=lambda x: x.start_s)
        return result

__all__ = ["Schedule", "ScheduleItem"]
