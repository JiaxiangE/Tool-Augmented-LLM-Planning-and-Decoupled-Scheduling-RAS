"""Hybrid scheduler: learned task ordering + greedy agent assignment + portfolio selection.

The learned scheduling policy produces, for each scheduling step, both
(a) a task ordering and (b) an agent assignment. Empirically the policy's
full output underperforms a classical greedy+CPM baseline on hold-out
scenarios; the bottleneck is the policy's agent-assignment decisions, not
its task ordering.

This hybrid therefore keeps only the policy's *task ordering* and delegates
the energy-aware agent assignment to the greedy scheduler. A portfolio
wrapper selects whichever of {pure greedy, learned-ordered greedy} yields
the lower makespan, guaranteeing the hybrid never regresses below the
greedy baseline.

The module is intentionally torch-free: extracting the task ordering from
a trained model is the responsibility of the evaluation driver. This file
contains only the pure scheduling/selection logic so it is unit-testable
without loading a model.
"""
from __future__ import annotations

from typing import Dict, Tuple


def portfolio_select(candidates: Dict[str, float]) -> Tuple[str, float]:
    """Return (name, makespan) of the candidate with the smallest makespan.

    `candidates` maps a candidate name (e.g. "greedy", "hybrid") to its
    makespan. Including the pure "greedy" candidate guarantees the selected
    makespan is never worse than greedy's — the portfolio's no-regression
    property.

    Ties are broken by candidate name for determinism.
    """
    if not candidates:
        raise ValueError("portfolio_select: no candidates given")
    name = min(candidates, key=lambda k: (candidates[k], k))
    return name, candidates[name]
