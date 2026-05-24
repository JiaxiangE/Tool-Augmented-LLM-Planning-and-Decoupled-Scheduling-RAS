"""
Driver A — 9-Way Scheduler Benchmark Comparison.

Methods (all assignment-based methods use identical CPM timing):
  [1] Random+CPM         — type-compatible random assignment + CPM timing
  [2] Greedy (coupled)   — greedy_scheduler.schedule() with built-in timing
  [3] Greedy+CPM         — greedy assignments + CPM timing (isolates timing)
  [4] HEFT (native)      — HEFT with its own EFT-based timing
  [5] HEFT+CPM           — HEFT assignments + unified CPM timing
  [6] Auction+CPM        — auction assignments + CPM timing
  [7] CP-SAT             — exact solver, own timing (small/medium only)
  [8] GNN-MLP+CPM        — SB3 MaskablePPO assignments + CPM timing
  [9] GNN-HGT+CPM        — HGT policy assignments + CPM timing

The inclusion of both Greedy(coupled)/Greedy+CPM and HEFT(native)/HEFT+CPM
is intentional: the gaps quantify the CPM-decoupling contribution and the
assignment quality independently.

Output: results/A_benchmark_raw.csv, .json, summary stats, LaTeX tables.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    assignments_to_schedule, auction_assignments,
    build_paper_experiments_argparser, cpsat_schedule,
    ensure_results_dir, gnn_custom_assignments, gnn_custom_available,
    gnn_mlp_assignments, gnn_mlp_available, greedy_assignments,
    greedy_coupled_schedule, heft_assignments, heft_schedule,
    heft_insertion_assignments, heft_insertion_schedule,
    latex_booktabs, load_corpus, make_agents,
    random_assignments, save_csv, save_json, save_tex,
    seed_everything, simulate,
)


def run_all_benchmarks(travel_model: str = "eager", seed_offset: int = 0):
    corpus = load_corpus()
    mlp_ok = gnn_mlp_available()
    custom_ok = gnn_custom_available()
    print(f"Loaded {len(corpus)} scenarios | GNN-MLP: {'Y' if mlp_ok else 'N'} | "
          f"GNN-HGT: {'Y' if custom_ok else 'N'} | travel_model={travel_model} | seed_offset={seed_offset}")

    all_rows = []

    for entry in corpus:
        fname = entry["_filename"]
        from core.schema.taskgraph import TaskGraph
        tg = TaskGraph(**entry["taskgraph"])
        meta = entry["metadata"]
        agents = make_agents(meta)
        n_tasks = len(tg.nodes)
        scale = meta.get("scale", "?")

        # Re-seed per scenario so the order of scenarios doesn't affect Random+CPM.
        seed_everything(seed_offset * 1000 + abs(hash(fname)) % 1000)

        print(f"\n  {fname} ({n_tasks} tasks, {scale})")

        def _run_assignment(label, get_asgn_fn):
            """Run assignment method -> CPM timing -> simulate."""
            try:
                asgn, wall = get_asgn_fn()
                sched = assignments_to_schedule(tg, asgn, agents, travel_model=travel_model)
                sr = simulate(tg, sched, agents)
                energy_total = sum(sr.energy_consumption.values()) if sr.energy_consumption else 0.0
                print(f"    {label:<20} ms={sr.makespan:>7.0f}s  comp={sr.task_completion_rate:.2f}  "
                      f"viol={len(sr.constraint_violations)}  wall={wall:.3f}s")
                return {
                    "makespan": sr.makespan, "completion_rate": sr.task_completion_rate,
                    "energy_total_wh": round(energy_total, 1),
                    "violations": len(sr.constraint_violations), "wall_s": round(wall, 4),
                    "n_assigned": len(asgn),
                }
            except Exception as e:
                print(f"    {label:<20} FAILED: {e}")
                return None

        row = {"scenario": fname, "scale": scale, "n_tasks": n_tasks}

        # [1] Random+CPM — seed varies per round to give a true noise floor
        rand_seed = 1000 * seed_offset + 42
        r = _run_assignment("Random+CPM", lambda: random_assignments(tg, agents, seed=rand_seed))
        if r:
            for k, v in r.items():
                row[f"random_cpm_{k}"] = v

        # [2] Greedy (coupled)
        try:
            sched_coupled, wall_coupled = greedy_coupled_schedule(tg, agents)
            sr_coupled = simulate(tg, sched_coupled, agents)
            energy_c = sum(sr_coupled.energy_consumption.values()) if sr_coupled.energy_consumption else 0.0
            row["greedy_coupled_makespan"] = sr_coupled.makespan
            row["greedy_coupled_completion_rate"] = sr_coupled.task_completion_rate
            row["greedy_coupled_energy_total_wh"] = round(energy_c, 1)
            row["greedy_coupled_violations"] = len(sr_coupled.constraint_violations)
            row["greedy_coupled_wall_s"] = round(wall_coupled, 4)
            print(f"    {'Greedy (coupled)':<20} ms={sr_coupled.makespan:>7.0f}s  comp={sr_coupled.task_completion_rate:.2f}  "
                  f"viol={len(sr_coupled.constraint_violations)}")
        except Exception as e:
            print(f"    {'Greedy (coupled)':<20} FAILED: {e}")

        # [3] Greedy+CPM
        r = _run_assignment("Greedy+CPM", lambda: greedy_assignments(tg, agents))
        if r:
            for k, v in r.items():
                row[f"greedy_cpm_{k}"] = v

        # [4] HEFT (native) — own EFT-based timing
        try:
            sched_heft, wall_heft = heft_schedule(tg, agents)
            sr_heft = simulate(tg, sched_heft, agents)
            energy_h = sum(sr_heft.energy_consumption.values()) if sr_heft.energy_consumption else 0.0
            row["heft_native_makespan"] = sr_heft.makespan
            row["heft_native_completion_rate"] = sr_heft.task_completion_rate
            row["heft_native_energy_total_wh"] = round(energy_h, 1)
            row["heft_native_violations"] = len(sr_heft.constraint_violations)
            row["heft_native_wall_s"] = round(wall_heft, 4)
            print(f"    {'HEFT (native)':<20} ms={sr_heft.makespan:>7.0f}s  comp={sr_heft.task_completion_rate:.2f}  "
                  f"viol={len(sr_heft.constraint_violations)}")
        except Exception as e:
            print(f"    {'HEFT (native)':<20} FAILED: {e}")

        # [5] HEFT+CPM — HEFT assignments + unified CPM timing
        r = _run_assignment("HEFT+CPM", lambda: heft_assignments(tg, agents))
        if r:
            for k, v in r.items():
                row[f"heft_cpm_{k}"] = v

        # [5a] HEFT-Insertion (native) — strong baseline (own timing)
        try:
            sched_hi, wall_hi = heft_insertion_schedule(tg, agents)
            sr_hi = simulate(tg, sched_hi, agents)
            energy_hi = sum(sr_hi.energy_consumption.values()) if sr_hi.energy_consumption else 0.0
            row["heft_insertion_native_makespan"] = sr_hi.makespan
            row["heft_insertion_native_completion_rate"] = sr_hi.task_completion_rate
            row["heft_insertion_native_energy_total_wh"] = round(energy_hi, 1)
            row["heft_insertion_native_violations"] = len(sr_hi.constraint_violations)
            row["heft_insertion_native_wall_s"] = round(wall_hi, 4)
            print(f"    {'HEFT-Insertion(nat)':<20} ms={sr_hi.makespan:>7.0f}s  comp={sr_hi.task_completion_rate:.2f}  "
                  f"viol={len(sr_hi.constraint_violations)}")
        except Exception as e:
            print(f"    {'HEFT-Insertion(nat)':<20} FAILED: {e}")

        # [5b] HEFT-Insertion+CPM — strong baseline assignment + unified CPM
        r = _run_assignment("HEFT-Insertion+CPM", lambda: heft_insertion_assignments(tg, agents))
        if r:
            for k, v in r.items():
                row[f"heft_insertion_cpm_{k}"] = v

        # [6] Auction+CPM
        r = _run_assignment("Auction+CPM", lambda: auction_assignments(tg, agents))
        if r:
            for k, v in r.items():
                row[f"auction_cpm_{k}"] = v

        # [7] CP-SAT (own timing, small/medium only)
        if n_tasks <= 30 and len(agents) <= 8:
            try:
                cs, wall_cs = cpsat_schedule(tg, agents)
                if cs:
                    sr_cs = simulate(tg, cs, agents)
                    energy_cs = sum(sr_cs.energy_consumption.values()) if sr_cs.energy_consumption else 0.0
                    row["cpsat_makespan"] = sr_cs.makespan
                    row["cpsat_completion_rate"] = sr_cs.task_completion_rate
                    row["cpsat_energy_total_wh"] = round(energy_cs, 1)
                    row["cpsat_violations"] = len(sr_cs.constraint_violations)
                    row["cpsat_wall_s"] = round(wall_cs, 4)
                    print(f"    {'CP-SAT':<20} ms={sr_cs.makespan:>7.0f}s  comp={sr_cs.task_completion_rate:.2f}  "
                          f"viol={len(sr_cs.constraint_violations)}  [optimal]")
                else:
                    row["cpsat_makespan"] = None
            except Exception as e:
                print(f"    {'CP-SAT':<20} FAILED: {e}")
                row["cpsat_makespan"] = None
        else:
            row["cpsat_makespan"] = "N/A"

        # [8] GNN-MLP+CPM
        if mlp_ok:
            r = _run_assignment("GNN-MLP+CPM", lambda: gnn_mlp_assignments(tg, agents))
            if r:
                for k, v in r.items():
                    row[f"gnn_mlp_cpm_{k}"] = v
        else:
            row["gnn_mlp_cpm_makespan"] = "N/A"

        # [9] GNN-HGT+CPM — single canonical checkpoint (paper Table 4 protocol).
        # The shipped models/checkpoints/gnn_hgt/gnn_hgt_final.pt reproduces
        # every Table 4 GNN-HGT cell exactly (small_b 1774, medium_b 4458,
        # large_e 5085, etc.). Override the checkpoint via GNN_HGT_PATH.
        if custom_ok:
            r = _run_assignment("GNN-HGT+CPM", lambda: gnn_custom_assignments(tg, agents))
            if r:
                for k, v in r.items():
                    row[f"gnn_custom_cpm_{k}"] = v
        else:
            row["gnn_custom_cpm_makespan"] = "N/A"

        # gap calculations
        greedy_cpm_ms = row.get("greedy_cpm_makespan")
        cpsat_ms = row.get("cpsat_makespan")
        for prefix in ["random_cpm", "greedy_coupled", "heft_native", "heft_cpm", "auction_cpm", "gnn_mlp_cpm", "gnn_custom_cpm"]:
            ms = row.get(f"{prefix}_makespan")
            if not isinstance(ms, (int, float)):
                continue
            if isinstance(greedy_cpm_ms, (int, float)) and greedy_cpm_ms > 0:
                row[f"{prefix}_vs_greedy_cpm_pct"] = round((ms - greedy_cpm_ms) / greedy_cpm_ms * 100, 1)
            if isinstance(cpsat_ms, (int, float)) and cpsat_ms > 0:
                row[f"{prefix}_vs_cpsat_pct"] = round((ms - cpsat_ms) / cpsat_ms * 100, 1)

        all_rows.append(row)

    return all_rows


def compute_summary(all_rows):
    """Compute per-method summary statistics."""
    methods = {
        "Random+CPM": "random_cpm",
        "Greedy (coupled)": "greedy_coupled",
        "Greedy+CPM": "greedy_cpm",
        "HEFT (native)": "heft_native",
        "HEFT+CPM": "heft_cpm",
        "Auction+CPM": "auction_cpm",
        "CP-SAT": "cpsat",
        "GNN-MLP+CPM": "gnn_mlp_cpm",
        "GNN-HGT+CPM": "gnn_custom_cpm",
    }
    import numpy as np
    summary = {}
    for label, prefix in methods.items():
        ms_values = [r[f"{prefix}_makespan"] for r in all_rows
                     if isinstance(r.get(f"{prefix}_makespan"), (int, float))]
        gap_vs_greedy = [r[f"{prefix}_vs_greedy_cpm_pct"] for r in all_rows
                         if isinstance(r.get(f"{prefix}_vs_greedy_cpm_pct"), (int, float))]
        gap_vs_cpsat = [r[f"{prefix}_vs_cpsat_pct"] for r in all_rows
                        if isinstance(r.get(f"{prefix}_vs_cpsat_pct"), (int, float))]

        s = {"n_scenarios": len(ms_values)}
        if ms_values:
            s["mean_makespan"] = round(float(np.mean(ms_values)), 1)
            s["std_makespan"] = round(float(np.std(ms_values)), 1)
        if gap_vs_greedy:
            s["mean_gap_vs_greedy_cpm_pct"] = round(float(np.mean(gap_vs_greedy)), 1)
        if gap_vs_cpsat:
            s["mean_gap_vs_cpsat_pct"] = round(float(np.mean(gap_vs_cpsat)), 1)
        summary[label] = s

    return summary


def generate_comparison_table(all_rows):
    """LaTeX table: Scenario x Scheduler makespan."""
    headers = ["Scenario", "$|V|$", "Rand.", "Greedy", "G+CPM",
               "HEFT", "H+CPM", "Auct.", "CP-SAT", "MLP", "HGT"]

    def _ms(row, key):
        v = row.get(key)
        if isinstance(v, (int, float)):
            return f"{v:.0f}"
        return "N/A"

    rows = []
    for r in all_rows:
        short = r["scenario"].replace(".json", "").replace("_", r"\_")
        rows.append([
            short, str(r["n_tasks"]),
            _ms(r, "random_cpm_makespan"),
            _ms(r, "greedy_coupled_makespan"),
            _ms(r, "greedy_cpm_makespan"),
            _ms(r, "heft_native_makespan"),
            _ms(r, "heft_cpm_makespan"),
            _ms(r, "auction_cpm_makespan"),
            _ms(r, "cpsat_makespan"),
            _ms(r, "gnn_mlp_cpm_makespan"),
            _ms(r, "gnn_custom_cpm_makespan"),
        ])

    return latex_booktabs(
        headers, rows,
        caption="Makespan (seconds) across 9 scheduling methods and 14 scenarios. "
                "All +CPM methods use identical CPM timing. "
                "Greedy and HEFT are also shown with their native timing for attribution analysis.",
        label="tab:benchmark_comparison",
        col_fmt="lcrrrrrrrrrr",
    )


def generate_gap_table(all_rows):
    """LaTeX table showing gap% vs Greedy+CPM and vs CP-SAT."""
    headers = ["Scenario", "$|V|$",
               r"Rand/G\%", r"HEFT/G\%", r"Auct/G\%", r"MLP/G\%", r"HGT/G\%",
               r"HEFT/Opt\%", r"MLP/Opt\%", r"HGT/Opt\%"]

    def _gp(row, key):
        v = row.get(key)
        if isinstance(v, (int, float)):
            return f"{v:+.1f}"
        return "---"

    rows = []
    for r in all_rows:
        short = r["scenario"].replace(".json", "").replace("_", r"\_")
        rows.append([
            short, str(r["n_tasks"]),
            _gp(r, "random_cpm_vs_greedy_cpm_pct"),
            _gp(r, "heft_cpm_vs_greedy_cpm_pct"),
            _gp(r, "auction_cpm_vs_greedy_cpm_pct"),
            _gp(r, "gnn_mlp_cpm_vs_greedy_cpm_pct"),
            _gp(r, "gnn_custom_cpm_vs_greedy_cpm_pct"),
            _gp(r, "heft_cpm_vs_cpsat_pct"),
            _gp(r, "gnn_mlp_cpm_vs_cpsat_pct"),
            _gp(r, "gnn_custom_cpm_vs_cpsat_pct"),
        ])

    return latex_booktabs(
        headers, rows,
        caption="Percentage gap of each method relative to Greedy+CPM and CP-SAT optimal. "
                "HEFT/G = HEFT+CPM vs Greedy+CPM. Negative values indicate improvement.",
        label="tab:benchmark_gaps",
        col_fmt="lcrrrrrrrrr",
    )


def main():
    args = build_paper_experiments_argparser().parse_args()
    seed_everything(args.seed_offset)
    res_dir = ensure_results_dir(override=args.output_dir)
    all_rows = run_all_benchmarks(travel_model=args.travel_model, seed_offset=args.seed_offset)

    # Stamp meta into outputs for downstream aggregation
    for r in all_rows:
        r["_seed_offset"] = args.seed_offset
        r["_travel_model"] = args.travel_model

    # Save raw data
    save_csv(all_rows, res_dir / "A_benchmark_raw.csv")
    save_json(all_rows, res_dir / "A_benchmark_raw.json")

    # Summary stats
    summary = compute_summary(all_rows)
    summary["_meta"] = {"seed_offset": args.seed_offset, "travel_model": args.travel_model}
    save_json(summary, res_dir / "A_summary_stats.json")

    # LaTeX tables
    save_tex(generate_comparison_table(all_rows), res_dir / "A_table_comparison.tex")
    save_tex(generate_gap_table(all_rows), res_dir / "A_table_gaps.tex")

    # Print summary
    print("\n" + "=" * 80)
    print("  SUMMARY (gap vs Greedy+CPM)")
    print("=" * 80)
    for label, stats in summary.items():
        line = f"  {label:<20}"
        if "mean_gap_vs_greedy_cpm_pct" in stats:
            line += f"  vs G+CPM: {stats['mean_gap_vs_greedy_cpm_pct']:+.1f}%"
        if "mean_gap_vs_cpsat_pct" in stats:
            line += f"  vs CP-SAT: {stats['mean_gap_vs_cpsat_pct']:+.1f}%"
        print(line)
    print()


if __name__ == "__main__":
    main()
