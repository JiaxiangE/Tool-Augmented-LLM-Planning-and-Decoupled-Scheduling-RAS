"""Simulation environment for evaluating task schedules.

Exports the four primary modules used by experiments and schedulers:
- lunar_sim: discrete-event simulator (the "referee" that scores schedules)
- terrain:   2D grid terrain model with slope and traversability
- energy_model: linear energy consumption model (travel + per-task cost)
- comm_model:   distance-based communication connectivity model
"""

from . import comm_model, energy_model, lunar_sim, terrain

__all__ = ["lunar_sim", "terrain", "energy_model", "comm_model"]
