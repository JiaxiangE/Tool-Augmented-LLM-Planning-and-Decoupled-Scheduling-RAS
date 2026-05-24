"""
Driver E — Hybrid scheduler decomposition evaluation.

For each (trained seed, held-out scenario):

  1. greedy            : pure greedy+CPM assignment  → makespan_greedy
  2. extract GNN order : the sequence in which the trained policy picks tasks
  3. hybrid            : greedy assignment driven by the GNN task order
                         (greedy_scheduler.schedule(task_order=gnn_order))
                         → makespan_hybrid
  4. portfolio         : min(makespan_greedy, makespan_hybrid) — never worse
                         than greedy by construction (portfolio_select)

All makespans are measured the same way as the held-out evaluation:
``assignment dict → assignments_to_schedule(eager) → simulate``.

Research question: does the trained policy's learned *global task ordering*
help greedy, when energy-aware agent assignment is delegated to greedy?

  - mean hybrid gap_pct < 0  → GNN ordering helps
  - mean hybrid gap_pct > 0  → GNN ordering does not help (portfolio still
                                guarantees the final result ≥ greedy)

Pure CPU, existing checkpoints, no retraining.

Usage:
    python experiments/E_hybrid_decomposition.py \
        --checkpoint-root models/checkpoints/gnn_hgt_ls \
        --holdout-dir data/test_corpus_holdout \
        --output results/E_hybrid
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

import torch
torch.distributions.Distribution.set_default_validate_args(False)

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState
from core.schema.taskgraph import TaskGraph
from common import assignments_to_schedule, simulate
from core.scheduler.greedy_scheduler import AgentInfo, schedule
from core.scheduler.hybrid_scheduler import portfolio_select
from core.scheduler.gnn.policy_network import SchedulingPolicyNet
from core.scheduler.gnn.state_encoder import (AGENT_FEAT_DIM, MAX_AGENTS,
                                               TASK_FEAT_DIM, StateEncoder)

DEFAULT_CKPT_ROOT = PROJECT_ROOT / "models" / "checkpoints" / "gnn_hgt_ls"
DEFAULT_HOLDOUT_DIR = PROJECT_ROOT / "data" / "test_corpus_holdout"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "E_hybrid"
SEEDS = [42, 137, 271, 419, 613]


def _build_agents(n_agents: int) -> list[AgentState]:
    """Mixed ROVER/RELAY/SAMPLER/ANALYZER cycle. Energy budget 20000 Wh so
    greedy can complete on all evaluation scenarios."""
    type_cycle = [AgentType.ROVER, AgentType.RELAY, AgentType.SAMPLER, AgentType.ANALYZER]
    agents = []
    for i in range(n_agents):
        agents.append(AgentState(
            id=f"{type_cycle[i % 4].value}_{i}", agent_type=type_cycle[i % 4],
            position=(0.0, 0.0, 0.0), energy_wh=20000.0, initial_energy_wh=20000.0))
    return agents


def load_policy(ckpt_root: Path, seed: int):
    pyg_meta = (["task", "agent"],
                [("task", "depends_on", "task"),
                 ("agent", "comm_with", "agent"),
                 ("agent", "can_exec", "task")])
    policy = SchedulingPolicyNet(
        task_feat_dim=TASK_FEAT_DIM, agent_feat_dim=AGENT_FEAT_DIM,
        hidden_dim=64, encoder_type="hgt", num_heads=4, num_layers=2,
        metadata=pyg_meta)
    cp = None
    for sub in (f"seed_{seed}_custom", f"seed_{seed}"):
        cand = ckpt_root / sub / "gnn_hgt_final.pt"
        if cand.exists():
            cp = cand
            break
    if cp is None:
        raise FileNotFoundError(
            f"No checkpoint found for seed {seed} under {ckpt_root}")
    ckpt = torch.load(cp, map_location="cpu", weights_only=True)
    state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
    policy.load_state_dict(state)
    policy.eval()
    return policy


def extract_gnn_order(policy, tg, agents):
    """Run the policy's assignment loop and return task ids in the order picked.

    Only the *ordering* the GNN produces is kept; the agent it picked is
    discarded. Any task never picked is appended at the end (defensive).
    """
    enc = StateEncoder()
    assigned = {}
    order = []
    task_ids = [n.id for n in tg.nodes]
    agent_ids = [a.id for a in agents]
    with torch.no_grad():
        for _ in range(len(task_ids) * 3):
            if len(assigned) >= len(task_ids):
                break
            mask_np = enc.compute_action_mask(tg, agents, assigned)
            if mask_np.sum() == 0:
                break
            hetero = enc.encode(tg, agents, assigned)
            mask_t = torch.tensor(mask_np, dtype=torch.float32)
            action, _, _ = policy.get_action(hetero, mask_t, deterministic=True)
            ti, ai = int(action) // MAX_AGENTS, int(action) % MAX_AGENTS
            if ti < len(task_ids) and ai < len(agents):
                tid = task_ids[ti]
                if tid not in assigned:
                    assigned[tid] = agent_ids[ai]
                    order.append(tid)
    order.extend(t for t in task_ids if t not in assigned)
    assert sorted(order) == sorted(task_ids), "extract_gnn_order: not a permutation"
    return order


def assignment_dict(sched):
    return {it.node_id: it.agent for it in sched.items}


def measure(tg, asgn, agents_state):
    """assignment dict → eager schedule → simulate → (makespan, completion, violations)."""
    sched = assignments_to_schedule(tg, asgn, agents_state, travel_model="eager")
    sim = simulate(tg, sched, agents_state)
    return (float(sim.makespan), float(sim.task_completion_rate),
            len(sim.constraint_violations))


def main():
    ap = argparse.ArgumentParser(description="Hybrid scheduler decomposition evaluation")
    ap.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT,
                    help="Directory containing seed_{N}/gnn_hgt_final.pt")
    ap.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR,
                    help="Directory of held-out scenario JSONs")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory")
    ap.add_argument("--scenario-glob", default="*.json",
                    help="Glob for scenario JSONs (default: *.json)")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(p for p in args.holdout_dir.glob(args.scenario_glob)
                       if not p.stem.startswith("_"))
    print(f"hold-out scenarios: {len(scenarios)}")
    rows = []
    t0 = time.time()

    for seed in SEEDS:
        try:
            policy = load_policy(args.checkpoint_root, seed)
        except FileNotFoundError as e:
            print(f"  skip seed {seed}: {e}")
            continue
        for fp in scenarios:
            d = json.load(open(fp, encoding="utf-8"))
            n_agents = d.get("metadata", {}).get("n_agents_recommended", 8)
            agents_state = _build_agents(n_agents)
            tg = TaskGraph(**d["taskgraph"])
            agent_infos = [AgentInfo(id=a.id, agent_type=a.agent_type, energy_wh=a.energy_wh)
                           for a in agents_state]
            try:
                # 1. pure greedy
                g_sched = schedule(tg, agent_infos)
                g_ms, g_comp, g_viol = measure(tg, assignment_dict(g_sched), agents_state)
                # 2-3. hybrid = greedy assignment driven by GNN task order
                gnn_order = extract_gnn_order(policy, tg, agents_state)
                a_sched = schedule(tg, agent_infos, task_order=gnn_order)
                a_ms, a_comp, a_viol = measure(tg, assignment_dict(a_sched), agents_state)
                # 4. portfolio
                pick, port_ms = portfolio_select({"greedy": g_ms, "hybrid": a_ms})
                rows.append({
                    "seed": seed, "scenario": fp.stem, "n_tasks": len(tg.nodes),
                    "greedy_makespan": round(g_ms, 1),
                    "hybrid_makespan": round(a_ms, 1),
                    "hybrid_gap_pct": round(100*(a_ms-g_ms)/g_ms, 1) if g_ms > 0 else None,
                    "portfolio_makespan": round(port_ms, 1),
                    "portfolio_pick": pick,
                    "portfolio_gap_pct": round(100*(port_ms-g_ms)/g_ms, 1) if g_ms > 0 else None,
                    "greedy_completion": round(g_comp, 3),
                    "hybrid_completion": round(a_comp, 3),
                    "greedy_violations": g_viol,
                    "hybrid_violations": a_viol,
                })
            except Exception as e:
                rows.append({"seed": seed, "scenario": fp.stem, "error": str(e)[:200]})
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    valid = [r for r in rows if "error" not in r]

    def m(key):
        vals = [r[key] for r in valid if r.get(key) is not None]
        return round(statistics.fmean(vals), 2) if vals else float("nan")

    n_hybrid_wins = sum(1 for r in valid if r.get("hybrid_gap_pct", 1) < 0)
    n_portfolio_used_hybrid = sum(1 for r in valid if r.get("portfolio_pick") == "hybrid")

    summary = {
        "n_cells": len(valid),
        "mean_hybrid_gap_pct": m("hybrid_gap_pct"),
        "mean_portfolio_gap_pct": m("portfolio_gap_pct"),
        "hybrid_wins_vs_greedy": f"{n_hybrid_wins}/{len(valid)}",
        "portfolio_picked_hybrid": f"{n_portfolio_used_hybrid}/{len(valid)}",
        "mean_greedy_completion": m("greedy_completion"),
        "mean_hybrid_completion": m("hybrid_completion"),
        "mean_greedy_violations": m("greedy_violations"),
        "mean_hybrid_violations": m("hybrid_violations"),
    }
    json.dump({"summary": summary, "raw": rows},
              open(args.output / "hybrid_decomposition.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("\n" + "=" * 66)
    print("HYBRID DECOMPOSITION — held-out, makespan gap vs greedy")
    print("=" * 66)
    print(f"  cells                          : {summary['n_cells']}")
    print(f"  hybrid (GNN-order + greedy)    : {summary['mean_hybrid_gap_pct']:+.1f}%")
    print(f"  portfolio (min of the two)     : {summary['mean_portfolio_gap_pct']:+.1f}%")
    print(f"  hybrid beats greedy on         : {summary['hybrid_wins_vs_greedy']} cells")
    print(f"  completion  greedy / hybrid    : {summary['mean_greedy_completion']:.3f}"
          f" / {summary['mean_hybrid_completion']:.3f}")
    print(f"  violations  greedy / hybrid    : {summary['mean_greedy_violations']:.1f}"
          f" / {summary['mean_hybrid_violations']:.1f}")
    print("=" * 66)
    print(f"\n  {time.time()-t0:.0f}s total  -> {args.output/'hybrid_decomposition.json'}")


if __name__ == "__main__":
    main()
