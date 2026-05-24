"""
Terrain model for lunar surface simulation.

A 2D grid with per-cell attributes (height, slope, PSR flag, obstacle flag).
Loads from YAML configuration; for synthetic experiments no real DEM is required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class CellAttributes:
    """Physical attributes of a single terrain grid cell."""
    height: float = 0.0        # elevation in meters
    slope: float = 0.0         # slope angle in degrees
    is_psr: bool = False       # permanently shadowed region
    is_obstacle: bool = False  # impassable terrain


# Default terrain parameters
_DEFAULT_GRID_SIZE = 100
_MAX_TRAVERSABLE_SLOPE = 30.0  # degrees


class TerrainGrid:
    """
    2D grid-based terrain representation.

    Coordinate convention:
      - (0, 0) is top-left corner
      - x increases rightward (column index)
      - y increases downward (row index)
      - cell_size is in meters (default 10m per cell)
    """

    def __init__(
        self,
        width: int = _DEFAULT_GRID_SIZE,
        height: int = _DEFAULT_GRID_SIZE,
        cell_size: float = 10.0,
    ):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Internal arrays (row, col) = (y, x)
        self._elevation = np.zeros((height, width), dtype=np.float32)
        self._slope = np.zeros((height, width), dtype=np.float32)
        self._is_psr = np.zeros((height, width), dtype=bool)
        self._is_obstacle = np.zeros((height, width), dtype=bool)

    # ---- Construction ----

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TerrainGrid":
        """Load terrain from a YAML configuration file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        terrain_cfg = data.get("terrain", data)
        width = terrain_cfg.get("width", _DEFAULT_GRID_SIZE)
        height = terrain_cfg.get("height", _DEFAULT_GRID_SIZE)
        cell_size = terrain_cfg.get("cell_size", 10.0)

        grid = cls(width=width, height=height, cell_size=cell_size)

        # Apply PSR regions
        for region in terrain_cfg.get("psr_regions", []):
            x0, y0 = region["x_start"], region["y_start"]
            x1, y1 = region["x_end"], region["y_end"]
            grid._is_psr[y0:y1, x0:x1] = True

        # Apply obstacles
        for obs in terrain_cfg.get("obstacles", []):
            x0, y0 = obs["x_start"], obs["y_start"]
            x1, y1 = obs["x_end"], obs["y_end"]
            grid._is_obstacle[y0:y1, x0:x1] = True
            grid._slope[y0:y1, x0:x1] = obs.get("slope", 45.0)

        # Apply elevation patches
        for patch in terrain_cfg.get("elevation_patches", []):
            x0, y0 = patch["x_start"], patch["y_start"]
            x1, y1 = patch["x_end"], patch["y_end"]
            grid._elevation[y0:y1, x0:x1] = patch["height"]
            if "slope" in patch:
                grid._slope[y0:y1, x0:x1] = patch["slope"]

        return grid

    @classmethod
    def from_npz(cls, path: str | Path) -> "TerrainGrid":
        """Load terrain from a pre-processed NPZ file (e.g., real DEM data)."""
        data = np.load(str(path), allow_pickle=True)
        elev = data["elevation"]
        h, w = elev.shape
        cell_size = float(data["cell_size"]) if "cell_size" in data else 10.0

        grid = cls(width=w, height=h, cell_size=cell_size)
        grid._elevation = elev.astype(np.float32)
        grid._slope = data["slope"].astype(np.float32)
        grid._is_psr = data["is_psr"].astype(bool)
        grid._is_obstacle = data["is_obstacle"].astype(bool)
        return grid

    # ---- Queries ----

    def _to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices (col, row)."""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        col = max(0, min(col, self.width - 1))
        row = max(0, min(row, self.height - 1))
        return col, row

    def get_cell(self, x: float, y: float) -> CellAttributes:
        """Get attributes of the cell at world coordinates (x, y)."""
        col, row = self._to_grid(x, y)
        return CellAttributes(
            height=float(self._elevation[row, col]),
            slope=float(self._slope[row, col]),
            is_psr=bool(self._is_psr[row, col]),
            is_obstacle=bool(self._is_obstacle[row, col]),
        )

    def is_traversable(self, x: float, y: float) -> bool:
        """Check if the cell at (x, y) can be traversed by a rover."""
        cell = self.get_cell(x, y)
        return (not cell.is_obstacle) and (cell.slope <= _MAX_TRAVERSABLE_SLOPE)

    def get_slope(self, x: float, y: float) -> float:
        """Get slope in degrees at world coordinates."""
        col, row = self._to_grid(x, y)
        return float(self._slope[row, col])

    def get_elevation(self, x: float, y: float) -> float:
        """Get elevation in meters at world coordinates."""
        col, row = self._to_grid(x, y)
        return float(self._elevation[row, col])

    @property
    def world_width(self) -> float:
        """Total terrain width in meters."""
        return self.width * self.cell_size

    @property
    def world_height(self) -> float:
        """Total terrain height in meters."""
        return self.height * self.cell_size
