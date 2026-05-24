# Tool-Augmented LLM Planning and Decoupled Scheduling for Heterogeneous Multi-Agent Lunar Missions: A Three-Layer Architecture

Reference implementation accompanying the manuscript

> *Tool-Augmented LLM Planning and Decoupled Scheduling for
> Heterogeneous Multi-Agent Lunar Missions: A Three-Layer Architecture*
> (under review, Robotics and Autonomous Systems).

> **Abstract.** Heterogeneous multi-robot task scheduling requires both
> semantic understanding to decompose high-level mission goals into
> executable task plans and combinatorial optimization to assign and
> sequence tasks across agents under physical constraints. LLM-based
> planners interpret natural language directives but lack
> structural-validity guarantees, while optimization-based schedulers
> produce high-quality schedules but assume the task graph is given.
> We present a three-layer hierarchical architecture in which an LLM
> with tool augmentation generates validated task graphs via a ReAct
> framework, a decoupled scheduling layer separates task assignment
> from temporal optimization, and a simulation-feedback loop verifies
> physical feasibility and drives iterative refinement. Evaluated on
> 14 synthetic scenarios plus two real-terrain Shackleton benchmarks
> with ten scheduling methods, the tool-augmented planning layer
> eliminates structural violations from direct LLM generation,
> producing TaskGraphs with 100 % lenient and 87.1 % strict structural
> equivalence across 70 paraphrases over 5 mission families. The
> assignment-scheduling decoupling yields a 32–35 % mean makespan
> improvement, decomposing into a near-zero pure-algorithmic component
> and a topology-dependent foresight advantage (mean 35.8 %, 0–57 %
> range). On real NASA LOLA DEM terrain at the Shackleton crater
> (1 km and 5 km scales), the system schedules all tasks without
> modifying any layer.

---

## Directory layout

```
.
├── README.md                          this file
├── LICENSE                            MIT
├── requirements.txt                   pinned Python dependencies
│
├── core/
│   ├── schema/                        agent / environment / task-graph dataclasses
│   ├── simulator/                     discrete-event sim + terrain/energy/comm models
│   ├── scheduler/                     classical schedulers + CPM pass + learned policy
│   └── llm/                           ReAct loop + tool registry + LLM backend
│
├── data/
│   ├── training_corpus/               14 in-distribution scenarios
│   ├── training_corpus_ood/           40 procedurally generated OOD scenarios
│   ├── dem/                           LOLA-derived Shackleton terrain (1 km, 5 km)
│   ├── lr_sensitivity/                pre-computed LR-sweep summary for Table 8
│   └── README.md                      data dictionary
│
├── experiments/
│   ├── A_benchmark_comparison.py      Table 4 main benchmark (10 schedulers x 14 scenarios)
│   ├── B_ood_benchmark.py             OOD evaluation (Tables 14-16)
│   ├── C_decoupling_attribution.py    Tables 5/6 eager-vs-lazy attribution
│   ├── D_case_study_dem.py            Shackleton 1 km / 5 km case study (Table 10)
│   ├── E_hybrid_decomposition.py      Table 9 hybrid decomposition (Figure 3)
│   ├── F_structural_regression.py     Appendix D structural regression
│   ├── G_lr_sensitivity.py            Table 8 learning-rate sensitivity
│   ├── H_nl_robustness.py             Appendix B 70-paraphrase robustness
│   ├── common.py                      shared utilities
│   └── README.md                      reproduction recipes
│
├── scripts/
│   └── figures/                       one script per figure in the paper
│
├── models/
│   ├── README.md                      how to load checkpoints
│   └── checkpoints/
│       ├── gnn_hgt/                   canonical GNN-HGT checkpoint (Table 4 / 10)
│       ├── gnn_hgt_ss/                GNN-HGT 5-seed SS ensemble (Tables 14-17 OOD)
│       ├── gnn_hgt_ls/                GNN-HGT 5-seed LS ensemble (Table 9 / Figure 3)
│       └── gnn_mlp/                   GNN-MLP single checkpoint (MaskablePPO)
│
├── prompts/
│   └── llm_prompt_template.md         Appendix A full LLM system + tool prompts
│
└── docs/
    ├── REPRODUCIBILITY.md             data -> table/figure recipes
    ├── ARCHITECTURE.md                three-layer overview
    └── TROUBLESHOOTING.md             install + common issues
```

---

## Quickstart

```bash
git clone https://github.com/JiaxiangE/Tool-Augmented-LLM-Planning-and-Decoupled-Scheduling-RAS.git
cd Tool-Augmented-LLM-Planning-and-Decoupled-Scheduling-RAS

# Binary assets (checkpoints + DEM rasters) are stored via Git LFS.
git lfs install && git lfs pull

python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows (PowerShell or cmd)
# venv\Scripts\activate

pip install -r requirements.txt

# Smoke test: regenerate Table 4 main benchmark (10 schedulers x 14 scenarios,
# ~1 minute on a laptop CPU).
python experiments/A_benchmark_comparison.py

# Render Figure 2 (uses the manuscript-Table-4 values reproduced above).
python scripts/figures/make_fig2_benchmark_bar.py
```

The complete table-by-table / figure-by-figure recipe is in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## LLM API key (only needed for one experiment)

**Tables 4–5, 8–10, 14–16 and Figures 2–5 are offline** — they run
against the shipped data and checkpoints with **no API key**.

Only `experiments/H_nl_robustness.py` (Appendix B paraphrase robustness)
calls an LLM at runtime. The paper used Qwen3-Max via DashScope; the
backend abstraction in `core/llm/llm_backend.py` works with any
OpenAI-compatible endpoint, so a reviewer can supply their own key:

```bash
# Option 1 — Qwen via DashScope (the paper's setup)
export DASHSCOPE_API_KEY='your-key'
python experiments/H_nl_robustness.py \
    --dataset data/nl_robustness/paraphrase_dataset.json

# Option 2 — OpenAI proper (or any OpenAI-compatible provider)
python experiments/H_nl_robustness.py \
    --dataset data/nl_robustness/paraphrase_dataset.json \
    --api-key  "$OPENAI_API_KEY" \
    --base-url https://api.openai.com/v1 \
    --model    gpt-4o
```

A working template is in [`.env.example`](.env.example). Real `.env`
files are git-ignored. No API key is embedded anywhere in this
repository.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

## Contact

- **Corresponding author**: Nan Li ([ln@csu.ac.cn](mailto:ln@csu.ac.cn))
- **First author / code maintainer**: Jiaxiang E ([jiaxiang.e@mail.mcgill.ca](mailto:jiaxiang.e@mail.mcgill.ca))