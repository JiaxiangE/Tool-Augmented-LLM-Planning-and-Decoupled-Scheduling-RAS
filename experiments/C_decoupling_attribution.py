"""
Driver C - Eager-vs-Lazy CPM Decoupling (paper Tables 5 and 6).

For each scenario in the 14-scenario benchmark, compute six makespans:

  Greedy variants:
    [G1] greedy_coupled    : built-in coupled timing (HEFT-style insertion)
    [G2] greedy_cpm_eager  : same assignments, CPM re-timed under eager travel
    [G3] greedy_cpm_lazy   : same assignments, CPM re-timed under lazy travel

  HEFT variants:
    [H1] heft_native       : HEFT with its own EFT-based timing
    [H2] heft_cpm_eager    : HEFT assignments, CPM re-timed (eager)
    [H3] heft_cpm_lazy     : HEFT assignments, CPM re-timed (lazy)

For each scenario the per-method gains are:

    eager_gain = (coupled - cpm_eager) / coupled * 100
    lazy_gain  = (coupled - cpm_lazy ) / coupled * 100

The corpus-mean gains are the published Table 5 cells:
  Greedy  : eager = 32.2%,  lazy = 0.2%
  HEFT    : eager = 34.9%,  lazy = 0.6%

Interpretation. The near-zero lazy-gain isolates the *pure-algorithmic*
contribution of CPM re-timing (zero by construction when travel is laid out
identically to the coupled baseline). The much larger eager-gain captures the
*foresight advantage*: planner-side agent pre-positioning enabled by knowing
the full DAG before committing to a timeline. Together they decompose the
32-35% mean improvement reported in the paper abstract into a near-zero pure
component plus a topology-dependent foresight component (Table 6).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    assignments_to_schedule, build_paper_experiments_argparser,
    ensure_results_dir, greedy_assignments, greedy_coupled_schedule,
    heft_assignments, heft_schedule, latex_booktabs, load_corpus,
    make_agents, save_csv, save_json, save_tex, seed_everything, simulate,
)


def _ms(tg, asgn, agents, travel_model):
    """Run assignment dict -> CPM(travel_model) -> simulate -> makespan."""
    sched = assignments_to_schedule(tg, asgn, agents, travel_model=travel_model)
    return simulate(tg, sched, agents).makespan


def _coupled_ms(coupled_sched_fn, tg, agents):
    sched, _ = coupled_sched_fn(tg, agents)
    return simulate(tg, sched, agents).makespan


def run_decoupling(seed_offset: int = 0):
    corpus = load_corpus()
    print(f"C: eager-vs-lazy decoupling over {len(corpus)} scenarios "
          f"(seed_offset={seed_offset})")

    results = []
    for entry in corpus:
        fname = entry["_filename"]
        from core.schema.taskgraph import TaskGraph
        tg = TaskGraph(**entry["taskgraph"])
        meta = entry["metadata"]
        agents = make_agents(meta)
        n_tasks = len(tg.nodes)
        scale = meta.get("scale", "?")
        seed_everything(seed_offset * 1000 + abs(hash(fname)) % 1000)

        # Greedy family
        greedy_coupled = _coupled_ms(greedy_coupled_schedule, tg, agents)
        asgn_g, _ = greedy_assignments(tg, agents)
        greedy_eager = _ms(tg, asgn_g, agents, "eager")
        greedy_lazy  = _ms(tg, asgn_g, agents, "lazy")

        # HEFT family
        heft_native_sched, _ = heft_schedule(tg, agents)
        heft_native = simulate(tg, heft_native_sched, agents).makespan
        asgn_h, _ = heft_assignments(tg, agents)
        heft_eager = _ms(tg, asgn_h, agents, "eager")
        heft_lazy  = _ms(tg, asgn_h, agents, "lazy")

        def pct(coupled, cpm):
            return round((coupled - cpm) / coupled * 100, 2) if coupled > 0 else 0.0

        row = {
            "scenario": fname, "scale": scale, "n_tasks": n_tasks,
            "greedy_coupled_ms": greedy_coupled,
            "greedy_cpm_eager_ms": greedy_eager,
            "greedy_cpm_lazy_ms":  greedy_lazy,
            "greedy_eager_gain_pct": pct(greedy_coupled, greedy_eager),
            "greedy_lazy_gain_pct":  pct(greedy_coupled, greedy_lazy),
            "heft_native_ms": heft_native,
            "heft_cpm_eager_ms": heft_eager,
            "heft_cpm_lazy_ms":  heft_lazy,
            "heft_eager_gain_pct": pct(heft_native, heft_eager),
            "heft_lazy_gain_pct":  pct(heft_native, heft_lazy),
        }
        results.append(row)

        print(f"  {fname:<35}  G:eager={row['greedy_eager_gain_pct']:>5.1f}%  "
              f"lazy={row['greedy_lazy_gain_pct']:>5.1f}%   "
              f"H:eager={row['heft_eager_gain_pct']:>5.1f}%  "
              f"lazy={row['heft_lazy_gain_pct']:>5.1f}%")

    return results


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def generate_table5(results):
    """Per-scenario eager/lazy gains for Greedy and HEFT (paper Table 5 layout)."""
    headers = ["Scenario", r"$|V|$",
               "Coupled",  "G+CPM",   "G+CPM",   "Lazy\\%", "Eager\\%",
               "Native",   "H+CPM",   "H+CPM",   "Lazy\\%", "Eager\\%"]
    subhead = [""]*3 + ["lazy", "eager"] + [""]*3 + ["lazy", "eager"] + [""]*2
    rows = []
    for r in results:
        short = r["scenario"].replace(".json", "").replace("_", r"\_")
        rows.append([
            short, str(r["n_tasks"]),
            f"{r['greedy_coupled_ms']:.0f}",
            f"{r['greedy_cpm_lazy_ms']:.0f}",
            f"{r['greedy_cpm_eager_ms']:.0f}",
            f"{r['greedy_lazy_gain_pct']:.1f}",
            f"{r['greedy_eager_gain_pct']:.1f}",
            f"{r['heft_native_ms']:.0f}",
            f"{r['heft_cpm_lazy_ms']:.0f}",
            f"{r['heft_cpm_eager_ms']:.0f}",
            f"{r['heft_lazy_gain_pct']:.1f}",
            f"{r['heft_eager_gain_pct']:.1f}",
        ])

    # Mean row
    rows.append([
        r"\textbf{Mean}", "", "", "", "",
        f"\\textbf{{{_mean([r['greedy_lazy_gain_pct']  for r in results]):.1f}}}",
        f"\\textbf{{{_mean([r['greedy_eager_gain_pct'] for r in results]):.1f}}}",
        "", "", "",
        f"\\textbf{{{_mean([r['heft_lazy_gain_pct']  for r in results]):.1f}}}",
        f"\\textbf{{{_mean([r['heft_eager_gain_pct'] for r in results]):.1f}}}",
    ])

    return latex_booktabs(
        headers, rows,
        caption=("Per-scenario eager-vs-lazy decoupling gains. "
                 "\\emph{Lazy} CPM re-timing isolates the pure-algorithmic "
                 "contribution; \\emph{Eager} additionally captures the "
                 "foresight advantage from planner-side agent pre-positioning."),
        label="tab:decoupling",
        col_fmt="lc" + "r"*5 + "r"*5,
    )


def main():
    args = build_paper_experiments_argparser().parse_args()
    seed_everything(args.seed_offset)
    res_dir = ensure_results_dir(override=args.output_dir)
    print(f"Driver C: Eager-vs-Lazy CPM Decoupling (Table 5)\n")

    results = run_decoupling(seed_offset=args.seed_offset)
    for r in results:
        r["_seed_offset"] = args.seed_offset

    save_csv(results, res_dir / "C_decoupling.csv")
    save_json(results, res_dir / "C_decoupling.json")
    save_tex(generate_table5(results), res_dir / "C_table_decoupling.tex")

    print(f"\n  Mean Greedy:  eager = {_mean([r['greedy_eager_gain_pct'] for r in results]):5.2f}%  "
          f"lazy = {_mean([r['greedy_lazy_gain_pct'] for r in results]):5.2f}%")
    print(f"  Mean HEFT:    eager = {_mean([r['heft_eager_gain_pct']   for r in results]):5.2f}%  "
          f"lazy = {_mean([r['heft_lazy_gain_pct']   for r in results]):5.2f}%")
    print(f"\n  Paper Table 5 reference:  Greedy 32.2 / 0.2,  HEFT 34.9 / 0.6")


if __name__ == "__main__":
    main()
