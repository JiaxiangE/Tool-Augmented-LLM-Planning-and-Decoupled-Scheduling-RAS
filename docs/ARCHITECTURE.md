# Architecture overview

The implementation is organised as a three-layer pipeline matching the
main-paper figure.

```
                +------------------------------------------+
   natural      |  core/llm/                                |
   language --> |    PlanningController  (ReAct loop)       | --> task graph
   mission      |    tools/             (constraint /       |     (typed DAG)
   spec         |                        coverage / relay)  |
                +------------------------------------------+
                                       |
                                       v
                +------------------------------------------+
                |  core/scheduler/                          |
                |    greedy_scheduler.py                    |
                |    heft_scheduler.py                      |
                |    heft_insertion_scheduler.py            |
                |    auction_scheduler.py                   | --> schedule
                |    cpsat_scheduler.py     (optimal ref.)  |     (per-agent
                |    hybrid_scheduler.py    (GNN + greedy)  |      time line)
                |    gnn/                   (learned policy)|
                +------------------------------------------+
                                       |
                                       v
                +------------------------------------------+
                |  core/simulator/                          |
                |    lunar_sim.py     (discrete-event sim) |
                |    terrain.py       (LOLA DEM, slope)    | --> metrics
                |    energy_model.py  (battery + solar)    |     (makespan,
                |    comm_model.py    (PtP / relay LoS)    |      success,
                |                                           |      energy)
                +------------------------------------------+
```

## Layer 1 — Mission decomposition (`core/llm/`)

A `PlanningController` runs a tool-augmented ReAct loop over an LLM
backend. The default backend is Qwen3-Max via the DashScope API
(OpenAI-compatible interface); the `LLMBackend` abstraction in
`core/llm/llm_backend.py` accepts drop-in replacements with other
providers and ships an `OpenAIBackend` scaffolding stub alongside it.

The tool registry is in `core/llm/tools/`:

- `constraint_checker.py` — feasibility checks against the action library
- `coverage_planner.py`   — area-coverage decomposition into ExploreCell
- `relay_planner.py`      — communication-relay placement on the DEM
- `sampling_template.py`  — multi-step sample-collection sub-graphs
- `graph_assembler.py`    — combines per-task fragments into the final DAG

## Layer 2 — Scheduling (`core/scheduler/`)

All schedulers consume a `TaskGraph` (typed DAG with skill / location /
duration attributes per node and four edge types: Order / Data / Mutex /
Sync) and a list of `AgentInfo`, and produce a `Schedule` (per-agent
ordered list of `ScheduleItem`s with start / end times).

| File | Algorithm |
|---|---|
| `greedy_scheduler.py` | Forward-time-greedy, optional CPM-priority topological order |
| `heft_scheduler.py` | Heterogeneous Earliest Finish Time (Topcuoglu 2002) |
| `heft_insertion_scheduler.py` | HEFT with idle-slot insertion |
| `auction_scheduler.py` | Sequential Single-Item bidding |
| `cpsat_scheduler.py` | OR-Tools CP-SAT (optimal reference, ≤ 30 tasks) |
| `hybrid_scheduler.py` | Learned task ordering + greedy assignment + portfolio |
| `gnn/` | Heterogeneous Graph Transformer (HGT) policy trained with custom PPO |

The published main benchmark evaluates ten configurations:
Random+CPM, Greedy (coupled), Greedy+CPM, HEFT (native), HEFT+CPM,
HEFT-Insertion (native), HEFT-Insertion+CPM, Auction+CPM, CP-SAT,
GNN-MLP+CPM, and GNN-HGT+CPM. The (native) and (+CPM) variants of
the same assignment algorithm isolate the contribution of the
decoupling pass.

## Layer 3 — Execution (`core/simulator/`)

A discrete-event simulator advances each agent through its assigned
schedule, accounting for:

- **terrain**: 30° slope-traversability bound, PSR shading, base-to-task
  travel distance from the LOLA-derived elevation grid
- **energy**: linear-power motion model + solar-charging recovery
  windows (relevant on long 5 km missions with recharge logistics)
- **communication**: line-of-sight relay links, optional dedicated
  relay agents

The simulator reports a `SimulationResult` containing makespan,
per-agent energy / distance summaries, and any constraint violations.

## Data flow

A scenario file (`data/training_corpus/*.json`) contains both an
already-decomposed `taskgraph` and an agent-fleet specification. For
**ablation** experiments, this taskgraph is fed directly to the
scheduler — the LLM layer is bypassed. For **end-to-end** experiments
(`H_nl_robustness.py`), the LLM layer rebuilds the taskgraph from a
natural-language mission brief before scheduling.
