"""
Driver B — Out-of-Distribution (OOD) Benchmark.

One CLI entrypoint exposing three subcommands that together reproduce the
paper's OOD analysis tables:

  run    — sweeps 40 OOD scenarios × N seeds × R rounds × 2 SS variants and
           writes per-cell raw metrics + aggregated summary.
  stats  — re-aggregates the raw JSON into headline tables, paired Δ between
           the two SS variants, per-topology breakdowns, and a Result A/B/C
           generalization verdict.
  probe  — quick diagnostic sweep on a small held-out probe set (e.g. 10
           scenarios) for fast iteration during checkpoint development.

For each (variant, seed, OOD scenario, round):

  GNN assignment -> travel-aware CPM(eager) -> simulate -> makespan
  Greedy+CPM(eager) reference on same scenario (computed once per scenario)
  gap_pct = (gnn - greedy) / greedy * 100 %

Usage examples
--------------
    # Full benchmark
    python experiments/B_ood_benchmark.py run \
        --output results/B_ood_raw.json \
        --rounds 5

    # Aggregate to paper tables
    python experiments/B_ood_benchmark.py stats \
        --input  results/B_ood_raw.json \
        --output results/B_ood_summary.csv

    # Quick diagnostic on a 10-scenario probe set
    python experiments/B_ood_benchmark.py probe \
        --probe-set data/training_corpus_ood_probe_10.json \
        --output    results/B_ood_probe.json

No LLM calls. Pure CPU inference + CPM timing.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from core.schema.agent_types import AgentType
from core.schema.environment import AgentState
from core.schema.taskgraph import TaskGraph
from common import (assignments_to_schedule, greedy_assignments, make_agents,
                    simulate)

SEEDS = [42, 137, 271, 419, 613]
MODES = ["custom", "sb3"]
OOD_DIR = PROJECT_ROOT / "data" / "training_corpus_ood"

# (variant_id, mode, training_label)
VARIANTS = {
    "gnn_hgt_ss": ("custom", "SS"),
    "gnn_mlp_ss": ("sb3",    "SS"),
}
TOPOLOGY_CLASSES = ["deep_chain", "mixed", "mutex_dense", "parallel_pure", "sampling_burst"]


# ────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────────────────
def _build_ood_agents(n_agents: int) -> list[AgentState]:
    """Mixed ROVER/RELAY/SAMPLER/ANALYZER cycle for OOD scenarios.

    Energy budget 20000 Wh so greedy can complete all 40 OOD scenarios. This
    mirrors the agent builder used to pre-compute the OOD greedy baselines.
    """
    type_cycle = [AgentType.ROVER, AgentType.RELAY, AgentType.SAMPLER, AgentType.ANALYZER]
    agents = []
    for i in range(n_agents):
        agents.append(AgentState(
            id=f"{type_cycle[i % 4].value}_{i}", agent_type=type_cycle[i % 4],
            position=(0.0, 0.0, 0.0), energy_wh=20000.0, initial_energy_wh=20000.0))
    return agents


def _ckpt_path(ckpt_root: Path, mode: str, seed: int) -> Path | None:
    """Resolve final-checkpoint path for (mode, seed).

    The shipped layout for the OOD benchmark is:
      <ckpt_root>/gnn_hgt_ss/seed_<NNN>/gnn_hgt_final.pt   (5 seeds; HGT-SS)
      <ckpt_root>/gnn_mlp/gnn_mlp_final.zip                (single shared MLP)

    Note: ``gnn_hgt_ss/`` is the **single-seed-variant 5-seed ensemble**
    (trained on the original 14-scenario corpus) used for the OOD benchmark
    in paper Section 5.3 and Appendix D. It is distinct from ``gnn_hgt_ls/``
    (trained on the large-scale procedurally-generated corpus) which is used
    by E_hybrid_decomposition.py for Table 9.
    """
    if mode == "custom":
        cand = ckpt_root / "gnn_hgt_ss" / f"seed_{seed}" / "gnn_hgt_final.pt"
        return cand if cand.exists() else None
    # sb3 / MLP — one shared checkpoint, reused across all seeds
    cand = ckpt_root / "gnn_mlp" / "gnn_mlp_final.zip"
    return cand if cand.exists() else None


def _gnn_assign_fn(mode: str, ckpt: Path):
    if mode == "custom":
        from common import gnn_custom_assignments
        return lambda tg, agents: gnn_custom_assignments(tg, agents, model_path=str(ckpt))
    from common import gnn_mlp_assignments
    return lambda tg, agents: gnn_mlp_assignments(tg, agents, model_path=str(ckpt))


def _load_ood_scenarios(ood_dir: Path = OOD_DIR) -> list[dict]:
    scenarios = []
    for fp in sorted(ood_dir.glob("ood_*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        meta = d.get("metadata", {})
        scenarios.append({
            "name": fp.stem,
            "topology_class": meta.get("topology_class"),
            "n_agents_recommended": meta.get("n_agents_recommended", 12),
            "n_tasks": meta.get("structural_features", {}).get("n_tasks"),
            "taskgraph": TaskGraph(**d["taskgraph"]),
            "precomputed_greedy_cpm": d.get("baseline_sim_result", {}).get("greedy_cpm_makespan"),
        })
    return scenarios


# ────────────────────────────────────────────────────────────────────────────
# Subcommand: run
# ────────────────────────────────────────────────────────────────────────────
def cmd_run(args):
    ss_root = Path(args.checkpoint_root)
    if not ss_root.is_absolute():
        ss_root = PROJECT_ROOT / ss_root
    out_dir = Path(args.output)
    # Defensive: if user passed a path ending in .json (treating --output-dir
    # as a file), strip the .json extension to recover the intended dir name
    # and emit a warning. This avoids the silent-trap where a directory named
    # "X.json" gets created and downstream tools `open()` it as a file.
    if out_dir.suffix.lower() == ".json":
        import sys as _sys
        print(f"WARNING: --output-dir was given as '{out_dir}' (ends in .json). "
              f"Treating as directory name without the .json suffix: '{out_dir.with_suffix('')}'.",
              file=_sys.stderr)
        out_dir = out_dir.with_suffix("")
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    active_variants = {k: v for k, v in VARIANTS.items()
                       if args.variants is None or k in args.variants}

    scenarios = _load_ood_scenarios()
    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]

    print(f"OOD Benchmark: {len(active_variants)} variants × {len(scenarios)} OOD scenarios "
          f"× {len(SEEDS)} seeds × {args.rounds} rounds")
    print(f"  variants: {list(active_variants.keys())}")
    print(f"  agent fleet: per-scenario n_agents_recommended")

    # --- Greedy reference per scenario (deterministic, computed once) ---
    print(f"\nComputing greedy reference for {len(scenarios)} OOD scenarios...")
    greedy_ref = {}
    greedy_drift_check = []
    t0 = time.time()
    for scen in scenarios:
        n_agents = scen["n_agents_recommended"]
        agents = _build_ood_agents(n_agents)
        g_asgn, _ = greedy_assignments(scen["taskgraph"], agents)
        g_sched = assignments_to_schedule(scen["taskgraph"], g_asgn, agents,
                                          travel_model="eager")
        g_ms = float(simulate(scen["taskgraph"], g_sched, agents).makespan)
        greedy_ref[scen["name"]] = g_ms
        pre = scen["precomputed_greedy_cpm"]
        if pre is not None:
            diff_pct = 100.0 * abs(g_ms - pre) / pre if pre > 0 else 0.0
            greedy_drift_check.append({
                "scenario": scen["name"], "n_agents": n_agents,
                "precomputed_greedy_cpm": round(pre, 1),
                "this_run_greedy_cpm": round(g_ms, 1),
                "diff_pct": round(diff_pct, 3),
            })
    print(f"  done in {time.time()-t0:.1f}s")
    max_drift = max((r["diff_pct"] for r in greedy_drift_check), default=0)
    print(f"  greedy drift vs precomputed baseline: max={max_drift:.3f}% "
          f"(expect ≤ 0.01% if reproducible)")

    # --- GNN inference loop ---
    results = []
    missing = []
    inf_t0 = time.time()
    cell_count = 0
    total_cells = len(active_variants) * len(SEEDS) * len(scenarios) * args.rounds
    for variant, (mode, training) in active_variants.items():
        for seed in SEEDS:
            ckpt = _ckpt_path(ss_root, mode, seed)
            if ckpt is None:
                missing.append(f"{variant}_seed_{seed}")
                continue
            assign_fn = _gnn_assign_fn(mode, ckpt)
            seed_t0 = time.time()
            for scen in scenarios:
                tg = scen["taskgraph"]
                n_agents = scen["n_agents_recommended"]
                g_ms = greedy_ref[scen["name"]]
                for rnd in range(args.rounds):
                    agents = _build_ood_agents(n_agents)
                    try:
                        gnn_asgn, _ = assign_fn(tg, agents)
                        gnn_sched = assignments_to_schedule(tg, gnn_asgn, agents,
                                                            travel_model="eager")
                        gnn_ms = float(simulate(tg, gnn_sched, agents).makespan)
                        gap = 100.0 * (gnn_ms - g_ms) / g_ms if g_ms > 0 else None
                        results.append({
                            "variant": variant, "mode": mode, "training": training,
                            "seed": seed, "scenario": scen["name"],
                            "topology_class": scen["topology_class"],
                            "n_tasks": scen["n_tasks"], "n_agents": n_agents,
                            "round": rnd,
                            "gnn_makespan": round(gnn_ms, 1),
                            "greedy_makespan": round(g_ms, 1),
                            "gap_pct": round(gap, 2) if gap is not None else None,
                        })
                    except Exception as e:
                        results.append({
                            "variant": variant, "seed": seed,
                            "scenario": scen["name"],
                            "topology_class": scen["topology_class"],
                            "round": rnd, "error": str(e)[:200],
                        })
                    cell_count += 1
            elapsed = time.time() - seed_t0
            print(f"  done: {variant}_seed_{seed}  ({ckpt.name})  "
                  f"[{cell_count}/{total_cells} cells, +{elapsed:.0f}s, "
                  f"eta {((time.time()-inf_t0)/max(cell_count,1)*(total_cells-cell_count))/60:.0f}min]")

    # --- Aggregate ---
    by_vs = defaultdict(list)
    by_vt = defaultdict(list)
    by_v = defaultdict(list)
    for r in results:
        if r.get("gap_pct") is None:
            continue
        by_vs[(r["variant"], r["scenario"])].append(r["gap_pct"])
        by_vt[(r["variant"], r["topology_class"])].append(r["gap_pct"])
        by_v[r["variant"]].append(r["gap_pct"])

    summary = {
        "n_scenarios": len(scenarios),
        "seeds": SEEDS,
        "rounds": args.rounds,
        "agent_fleet_policy": "per-scenario n_agents_recommended",
        "missing_checkpoints": missing,
        "greedy_drift_check": {
            "max_diff_pct": round(max_drift, 4),
            "all_below_0_01_pct": max_drift <= 0.01,
            "per_scenario": greedy_drift_check,
        },
        "by_variant": {},
        "by_variant_topology": {},
    }
    for v, gaps in by_v.items():
        summary["by_variant"][v] = {
            "n": len(gaps),
            "mean_gap_pct": round(statistics.fmean(gaps), 2),
            "std_gap_pct": round(statistics.stdev(gaps), 2) if len(gaps) > 1 else 0.0,
            "min_gap_pct": round(min(gaps), 2),
            "max_gap_pct": round(max(gaps), 2),
        }
    for (v, t), gaps in by_vt.items():
        summary["by_variant_topology"][f"{v}|{t}"] = {
            "n": len(gaps),
            "mean_gap_pct": round(statistics.fmean(gaps), 2),
            "std_gap_pct": round(statistics.stdev(gaps), 2) if len(gaps) > 1 else 0.0,
        }

    # Cross-run determinism: per (variant, seed, scenario) std over rounds
    by_vss = defaultdict(list)
    for r in results:
        if r.get("gap_pct") is None:
            continue
        by_vss[(r["variant"], r["seed"], r["scenario"])].append(r["gap_pct"])
    non_zero_std = []
    for k, gaps in by_vss.items():
        if len(gaps) > 1 and statistics.stdev(gaps) > 0.001:
            non_zero_std.append({"key": "|".join(map(str, k)),
                                 "std": round(statistics.stdev(gaps), 3),
                                 "gaps": gaps})
    summary["cross_run_determinism"] = {
        "n_cells_checked": len(by_vss),
        "n_non_zero_std": len(non_zero_std),
        "examples": non_zero_std[:5],
    }

    out = {"summary": summary, "raw": results}
    out_path = out_dir / "ood_benchmark_metrics.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Total wall: {(time.time()-inf_t0)/60:.1f} min")
    print(f"Summary by variant:")
    for v, s in summary["by_variant"].items():
        print(f"  {v}: mean_gap={s['mean_gap_pct']:+.1f}%  std={s['std_gap_pct']:.1f}  "
              f"range=[{s['min_gap_pct']:+.1f}, {s['max_gap_pct']:+.1f}]  n={s['n']}")
    if missing:
        print(f"\n  WARNING: {len(missing)} missing checkpoints: {missing}")
    if non_zero_std:
        print(f"\n  WARNING: {len(non_zero_std)} cells have non-zero round-std "
              f"(GNN inference should be deterministic)")
    print(f"  greedy ref drift max: {max_drift:.4f}% vs precomputed baseline")
    print(f"\n  -> {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Subcommand: stats
# ────────────────────────────────────────────────────────────────────────────
def _ci_95_t(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, std, half-width of 95% CI using t-dist df=n-1)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.fmean(values)
    if n == 1:
        return mean, 0.0, float("nan")
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    try:
        from scipy import stats
        t_crit = float(stats.t.ppf(0.975, n - 1))
    except ImportError:
        t_crit = 1.96  # large-n approximation
    return mean, sd, t_crit * se


def _t_stat_paired(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    se = sd / math.sqrt(n)
    t = mean / se
    try:
        from scipy import stats
        p = float(stats.t.sf(abs(t), n - 1) * 2.0)
    except ImportError:
        p = float("nan")
    return t, p


def _verdict_for_variant(by_topo: dict, overall_mean: float, overall_ci: float):
    """Apply the Result A/B/C generalization framework.

    Returns (verdict_letter, justification_string).
    """
    worst_topo, worst_mean = None, -float("inf")
    for topo, st in by_topo.items():
        if st["mean_gap_pct"] > worst_mean:
            worst_mean = st["mean_gap_pct"]
            worst_topo = topo

    if overall_mean <= 5.0 and (overall_mean + overall_ci) <= 10.0:
        verdict = "A"
        why = (f"Overall mean {overall_mean:+.1f}% ≤ +5pp and upper CI95 bound "
               f"{overall_mean+overall_ci:+.1f}% ≤ +10pp; GNN within heuristic noise on OOD")
    elif overall_mean <= 20.0:
        verdict = "B"
        why = (f"Overall mean {overall_mean:+.1f}% in [+5, +20pp]; worst topology "
               f"{worst_topo} at {worst_mean:+.1f}%; topology-dependent generalization")
    else:
        verdict = "C"
        why = (f"Overall mean {overall_mean:+.1f}% > +20pp; worst topology {worst_topo} "
               f"at {worst_mean:+.1f}%; GNN limitation is architectural / training-bounded")
    return verdict, why


def cmd_stats(args):
    in_dir = Path(args.input)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept either a directory or a direct file path.
    src = in_dir / "ood_benchmark_metrics.json" if in_dir.is_dir() else in_dir
    if not src.exists():
        print(f"ERROR: {src} not found. Run `B_ood_benchmark.py run` first.")
        return 1

    data = json.load(open(src, encoding="utf-8"))
    raw = data["raw"]
    summary_in = data["summary"]

    # Per (variant, scenario, seed): one gap value (mean over rounds; std should be 0)
    by_vss = defaultdict(list)
    for r in raw:
        if r.get("gap_pct") is None:
            continue
        by_vss[(r["variant"], r["scenario"], r["seed"])].append(r["gap_pct"])
    by_vss_mean = {k: statistics.fmean(v) for k, v in by_vss.items()}

    scenario_topo = {r["scenario"]: r.get("topology_class") for r in raw}

    # === 1. Overall stats per variant ===
    overall = {}
    for variant in VARIANTS:
        gaps = [v for (var, _, _), v in by_vss_mean.items() if var == variant]
        mean, sd, ci = _ci_95_t(gaps)
        overall[variant] = {
            "n_cells": len(gaps),
            "mean_gap_pct": round(mean, 2),
            "std_gap_pct": round(sd, 2),
            "ci_95_half": round(ci, 2),
            "min_gap_pct": round(min(gaps), 2) if gaps else None,
            "max_gap_pct": round(max(gaps), 2) if gaps else None,
        }

    # Paired Δ (HGT - MLP) at same (seed, scenario)
    pair_keys = set()
    for (_, scen, seed) in by_vss_mean:
        pair_keys.add((scen, seed))
    diffs = []
    for (scen, seed) in pair_keys:
        h = by_vss_mean.get(("gnn_hgt_ss", scen, seed))
        m = by_vss_mean.get(("gnn_mlp_ss", scen, seed))
        if h is not None and m is not None:
            diffs.append(h - m)
    if diffs:
        d_mean, d_sd, d_ci = _ci_95_t(diffs)
        t, p = _t_stat_paired(diffs)
        paired_hgt_vs_mlp = {
            "n_pairs": len(diffs),
            "mean_diff_pp": round(d_mean, 2),
            "ci_95_half_pp": round(d_ci, 2),
            "t_stat": round(t, 2) if not math.isnan(t) else None,
            "p_two_sided": round(p, 5) if not math.isnan(p) else None,
            "interpretation": "positive = HGT worse than MLP on OOD",
        }
    else:
        paired_hgt_vs_mlp = {"n_pairs": 0}

    # === 2. Per-topology breakdown ===
    by_vt_gaps = defaultdict(list)
    for (variant, scen, seed), gap in by_vss_mean.items():
        topo = scenario_topo.get(scen)
        if topo:
            by_vt_gaps[(variant, topo)].append(gap)
    by_topology = {}
    for variant in VARIANTS:
        by_topology[variant] = {}
        for topo in TOPOLOGY_CLASSES:
            gaps = by_vt_gaps.get((variant, topo), [])
            if not gaps:
                continue
            mean, sd, ci = _ci_95_t(gaps)
            by_topology[variant][topo] = {
                "n_cells": len(gaps),
                "mean_gap_pct": round(mean, 2),
                "std_gap_pct": round(sd, 2),
                "ci_95_half": round(ci, 2),
                "min_gap_pct": round(min(gaps), 2),
                "max_gap_pct": round(max(gaps), 2),
            }

    # === 3. Per-scenario worst/best ===
    by_vsc = defaultdict(list)
    for (variant, scen, seed), gap in by_vss_mean.items():
        by_vsc[(variant, scen)].append(gap)
    per_scenario = {}
    for variant in VARIANTS:
        rows = []
        for scen in sorted(set(scenario_topo)):
            gaps = by_vsc.get((variant, scen), [])
            if not gaps:
                continue
            mean, sd, ci = _ci_95_t(gaps)
            rows.append({
                "scenario": scen, "topology": scenario_topo[scen],
                "n_seeds": len(gaps), "mean_gap_pct": round(mean, 2),
                "std_gap_pct": round(sd, 2),
            })
        rows.sort(key=lambda r: r["mean_gap_pct"])
        per_scenario[variant] = {
            "best_3": rows[:3],
            "worst_3": rows[-3:][::-1],
            "all": rows,
        }

    # === 4. Verdict ===
    verdicts = {}
    for variant in VARIANTS:
        v, why = _verdict_for_variant(by_topology[variant],
                                      overall[variant]["mean_gap_pct"],
                                      overall[variant]["ci_95_half"])
        verdicts[variant] = {"letter": v, "why": why}

    out = {
        "overall_by_variant": overall,
        "paired_hgt_vs_mlp": paired_hgt_vs_mlp,
        "by_topology": by_topology,
        "per_scenario": per_scenario,
        "generalization_verdict": verdicts,
        "source_metadata": {
            "n_scenarios": summary_in.get("n_scenarios"),
            "seeds": summary_in.get("seeds"),
            "rounds": summary_in.get("rounds"),
            "agent_fleet_policy": summary_in.get("agent_fleet_policy"),
            "missing_checkpoints": summary_in.get("missing_checkpoints"),
            "greedy_drift_max_pct": summary_in.get("greedy_drift_check", {}).get("max_diff_pct"),
            "determinism_n_non_zero": summary_in.get("cross_run_determinism", {}).get("n_non_zero_std"),
        },
    }
    out_path = out_dir / "ood_stats.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")

    # Optional CSV summary table
    if args.csv:
        import csv
        csv_path = out_dir / "ood_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant", "n_cells", "mean_gap_pct", "std_gap_pct",
                        "ci_95_half", "min_gap_pct", "max_gap_pct", "verdict"])
            for v in VARIANTS:
                s = overall[v]
                w.writerow([v, s["n_cells"], s["mean_gap_pct"], s["std_gap_pct"],
                            s["ci_95_half"], s["min_gap_pct"], s["max_gap_pct"],
                            verdicts[v]["letter"]])
        print(f"  -> {csv_path}")

    # === Markdown report ===
    lines = [
        "# OOD Benchmark Statistics",
        "",
        f"Source: `{src.name}` "
        f"({summary_in.get('n_scenarios')} OOD scenarios × {len(summary_in.get('seeds', []))} seeds × "
        f"{summary_in.get('rounds')} rounds × {len(VARIANTS)} SS variants)",
        f"Greedy reference drift vs precomputed baseline: max = "
        f"{summary_in.get('greedy_drift_check', {}).get('max_diff_pct', '?')}%",
        f"Cross-run determinism violations (gap std > 0.001 across rounds): "
        f"{summary_in.get('cross_run_determinism', {}).get('n_non_zero_std', '?')} / "
        f"{summary_in.get('cross_run_determinism', {}).get('n_cells_checked', '?')} cells",
        "",
        "## Table 1 — Overall by-variant summary (mean ± 95% CI)",
        "",
        "| Variant | n | mean gap_pct | std | 95% CI | range |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for v in VARIANTS:
        s = overall[v]
        lines.append(f"| {v} | {s['n_cells']} | {s['mean_gap_pct']:+.2f}% | "
                     f"{s['std_gap_pct']:.2f} | ±{s['ci_95_half']:.2f} | "
                     f"[{s['min_gap_pct']:+.1f}, {s['max_gap_pct']:+.1f}] |")
    lines += [
        "",
        f"**Paired Δ (HGT-SS − MLP-SS), same (seed, scenario)**: "
        f"n={paired_hgt_vs_mlp.get('n_pairs')}, "
        f"mean = {paired_hgt_vs_mlp.get('mean_diff_pp')}pp ± "
        f"{paired_hgt_vs_mlp.get('ci_95_half_pp')}pp (95% CI), "
        f"t = {paired_hgt_vs_mlp.get('t_stat')}, "
        f"two-sided p = {paired_hgt_vs_mlp.get('p_two_sided')}",
        "",
        "## Table 2 — Per-topology breakdown",
        "",
        "| Topology | n | HGT-SS mean ± CI95 | HGT-SS std | MLP-SS mean ± CI95 | MLP-SS std | HGT vs MLP Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for topo in TOPOLOGY_CLASSES:
        hgt = by_topology["gnn_hgt_ss"].get(topo, {})
        mlp = by_topology["gnn_mlp_ss"].get(topo, {})
        if not hgt or not mlp:
            continue
        delta = round(hgt["mean_gap_pct"] - mlp["mean_gap_pct"], 2)
        lines.append(f"| {topo} | {hgt['n_cells']} | {hgt['mean_gap_pct']:+.2f}% ±{hgt['ci_95_half']:.2f} | "
                     f"{hgt['std_gap_pct']:.2f} | {mlp['mean_gap_pct']:+.2f}% ±{mlp['ci_95_half']:.2f} | "
                     f"{mlp['std_gap_pct']:.2f} | {delta:+.2f}pp |")

    lines += [
        "",
        "## Generalization verdict (controlled OOD test)",
        "",
    ]
    for v in VARIANTS:
        ver = verdicts[v]
        lines.append(f"- **{v}**: **Result {ver['letter']}** — {ver['why']}")
    lines += [
        "",
        "**Result framework:**",
        "- A: variant mean ≤ +5pp AND upper CI95 ≤ +10pp → GNN generalizes within heuristic noise",
        "- B: mean in (+5, +20]pp → topology-dependent generalization",
        "- C: mean > +20pp → architectural / training-bounded limitation",
        "",
    ]

    (out_dir / "ood_stats_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {out_dir / 'ood_stats_table.md'}")

    print()
    print(f"=== Headline numbers ===")
    for v in VARIANTS:
        s = overall[v]
        ver = verdicts[v]
        print(f"  {v}: mean={s['mean_gap_pct']:+.1f}% ±{s['ci_95_half']:.1f} → Result {ver['letter']}")
    print(f"  HGT vs MLP paired Δ: {paired_hgt_vs_mlp.get('mean_diff_pp')}pp "
          f"(p={paired_hgt_vs_mlp.get('p_two_sided')})")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Subcommand: probe  (fast diagnostic on a small held-out probe set)
# ────────────────────────────────────────────────────────────────────────────
def _load_probe_scenarios(probe_set_path: Path):
    spec = json.load(open(probe_set_path, encoding="utf-8"))
    scenarios = []
    for name in spec["scenarios"]:
        d = json.load(open(OOD_DIR / name, encoding="utf-8"))
        scenarios.append({
            "name": name,
            "topology_class": d["metadata"]["topology_class"],
            "taskgraph": TaskGraph(**d["taskgraph"]),
        })
    return scenarios


def _find_probe_checkpoint(ckpt_root: Path, mode: str, seed: int) -> Path | None:
    """Locate the final checkpoint for (mode, seed) under the shipped layout."""
    return _ckpt_path(ckpt_root, mode, seed)


def cmd_probe(args):
    ckpt_root = Path(args.checkpoint_root)
    if not ckpt_root.is_absolute():
        ckpt_root = PROJECT_ROOT / ckpt_root
    scenarios = _load_probe_scenarios(Path(args.probe_set))
    print(f"OOD probe: {len(scenarios)} scenarios × {len(MODES)} modes × {len(SEEDS)} seeds")

    results = []
    missing_ckpts = []
    for mode in MODES:
        for seed in SEEDS:
            ckpt = _find_probe_checkpoint(ckpt_root, mode, seed)
            if ckpt is None:
                missing_ckpts.append(f"{mode}_seed_{seed}")
                print(f"  MISSING checkpoint: {mode}_seed_{seed}")
                continue
            gnn_asgn_fn = _gnn_assign_fn(mode, ckpt)
            for scen in scenarios:
                tg = scen["taskgraph"]
                # OOD scenarios have scale='ood'; use the large 8-agent fleet
                agents = make_agents({"scale": "large"})
                try:
                    g_asgn, _ = greedy_assignments(tg, agents)
                    g_sched = assignments_to_schedule(tg, g_asgn, agents, travel_model="eager")
                    g_ms = float(simulate(tg, g_sched, agents).makespan)

                    gnn_asgn, _ = gnn_asgn_fn(tg, agents)
                    gnn_sched = assignments_to_schedule(tg, gnn_asgn, agents, travel_model="eager")
                    gnn_ms = float(simulate(tg, gnn_sched, agents).makespan)

                    gap = 100.0 * (gnn_ms - g_ms) / g_ms if g_ms > 0 else None
                    results.append({
                        "mode": mode, "seed": seed,
                        "scenario": scen["name"],
                        "topology_class": scen["topology_class"],
                        "greedy_makespan": round(g_ms, 1),
                        "gnn_makespan": round(gnn_ms, 1),
                        "gap_pct": round(gap, 2) if gap is not None else None,
                    })
                except Exception as e:
                    results.append({
                        "mode": mode, "seed": seed,
                        "scenario": scen["name"],
                        "topology_class": scen["topology_class"],
                        "error": str(e)[:200],
                    })
            print(f"  done: {mode}_seed_{seed}  ({ckpt.name})")

    by_mode = defaultdict(list)
    by_mode_topo = defaultdict(list)
    for r in results:
        if r.get("gap_pct") is None:
            continue
        by_mode[r["mode"]].append(r["gap_pct"])
        by_mode_topo[(r["mode"], r["topology_class"])].append(r["gap_pct"])

    summary = {"probe_set": Path(args.probe_set).name,
               "seeds": SEEDS, "modes": MODES,
               "missing_checkpoints": missing_ckpts,
               "by_mode": {}, "by_mode_topology": {}}
    for mode, gaps in by_mode.items():
        if gaps:
            summary["by_mode"][mode] = {
                "n_cells": len(gaps),
                "mean_gap_pct": round(statistics.fmean(gaps), 2),
                "std_gap_pct": round(statistics.stdev(gaps), 2) if len(gaps) > 1 else 0.0,
                "min_gap_pct": round(min(gaps), 2),
                "max_gap_pct": round(max(gaps), 2),
            }
    for (mode, topo), gaps in by_mode_topo.items():
        if gaps:
            summary["by_mode_topology"][f"{mode}|{topo}"] = {
                "n_cells": len(gaps),
                "mean_gap_pct": round(statistics.fmean(gaps), 2),
                "std_gap_pct": round(statistics.stdev(gaps), 2) if len(gaps) > 1 else 0.0,
            }

    out = {"summary": summary, "raw": results}
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    for mode, s in summary["by_mode"].items():
        print(f"  {mode:<8} mean_gap={s['mean_gap_pct']:+.1f}%  std={s['std_gap_pct']:.1f}%  "
              f"range=[{s['min_gap_pct']:+.1f}, {s['max_gap_pct']:+.1f}]  n={s['n_cells']}")
    if missing_ckpts:
        print(f"\n  WARNING: {len(missing_ckpts)} missing checkpoints: {missing_ckpts}")
    print(f"\n  -> {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the full OOD benchmark sweep")
    p_run.add_argument("--checkpoint-root", default="models/checkpoints",
                       help="Root containing gnn_hgt_ls/seed_*/ and gnn_mlp/ "
                            "(default: models/checkpoints)")
    p_run.add_argument("--rounds", type=int, default=5)
    p_run.add_argument("--output-dir", "--output", required=True,
                       dest="output",
                       help="Output DIRECTORY (the script writes "
                            "ood_benchmark_metrics.json inside it; passing a "
                            "path that ends in .json by mistake will create "
                            "a directory with that name)")
    p_run.add_argument("--variants", nargs="+", default=None,
                       choices=list(VARIANTS.keys()),
                       help="Subset of variants to run (default: all)")
    p_run.add_argument("--max-scenarios", type=int, default=None,
                       help="Smoke-test cap: only first N scenarios (default: all)")
    p_run.set_defaults(func=cmd_run)

    p_stats = sub.add_parser("stats", help="Aggregate raw results into paper tables")
    p_stats.add_argument("--input", required=True,
                         help="Directory containing ood_benchmark_metrics.json "
                              "(or path to the file itself)")
    p_stats.add_argument("--output", required=True, help="Output directory")
    p_stats.add_argument("--csv", action="store_true",
                         help="Also emit ood_summary.csv")
    p_stats.set_defaults(func=cmd_stats)

    p_probe = sub.add_parser("probe", help="Quick diagnostic sweep on a probe set")
    p_probe.add_argument("--checkpoint-root", default="models/checkpoints",
                         help="Root containing gnn_hgt_ls/seed_*/ and gnn_mlp/ "
                              "(default: models/checkpoints)")
    p_probe.add_argument("--probe-set",
                         default=str(PROJECT_ROOT / "data" / "training_corpus_ood_probe_10.json"))
    p_probe.add_argument("--output", required=True, help="Output JSON path")
    p_probe.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
