# Trained Model Checkpoints

This directory holds the trained policies used by the experimental
drivers in `experiments/`.

## Layout

```
models/checkpoints/
├── gnn_hgt/                              <- canonical single checkpoint
│   ├── gnn_hgt_final.pt                  (400 KB, 500K steps, original
│   │                                      14-scenario corpus, mean reward +1.72)
│   └── training_summary.json
│
├── gnn_hgt_ss/                           <- 5-seed SS ensemble (Section 5.3 OOD)
│   ├── seed_42/    gnn_hgt_final.pt + training_summary.json
│   ├── seed_137/   ...
│   ├── seed_271/   ...
│   ├── seed_419/   ...
│   ├── seed_613/   ...
│   (1.2 MB per seed, 500K steps, original 14-scenario corpus,
│    mean reward ~+1.77; consumed by B_ood_benchmark.py + F)
│
├── gnn_hgt_ls/                           <- 5-seed LS ensemble (Section 5.4 hybrid)
│   ├── seed_42/    gnn_hgt_final.pt + training_summary.json
│   ├── seed_137/   ...
│   ├── seed_271/   ...
│   ├── seed_419/   ...
│   ├── seed_613/   ...
│   (1.2 MB per seed, 500K steps, large-scale procedurally-generated corpus
│    n_tasks in [100, 500], staged curriculum; consumed by
│    E_hybrid_decomposition.py for Table 9)
│   └── README.md
│
└── gnn_mlp/                              <- single MaskablePPO MLP policy
    └── gnn_mlp_final.zip                 (1.2 MB, stable-baselines3 format)
```

## Which checkpoint is used by which experiment

| Experiment script | Checkpoint(s) | Paper artifact |
|---|---|---|
| `A_benchmark_comparison.py` | `gnn_hgt/gnn_hgt_final.pt` + `gnn_mlp/` | Table 4 |
| `D_case_study_dem.py` | `gnn_hgt/gnn_hgt_final.pt` + `gnn_mlp/` | Table 10 (Shackleton) |
| `B_ood_benchmark.py` | `gnn_hgt_ss/seed_*/` + `gnn_mlp/` | Tables 14-16 (Section 5.3 OOD) |
| `F_structural_regression.py` | (consumes B output) | Tables 17-20 (Appendix D) |
| `E_hybrid_decomposition.py` | `gnn_hgt_ls/seed_*/` | Table 9 / Figure 3 (Section 5.4) |

Override the default HGT path for A and D via the `GNN_HGT_PATH`
environment variable.

## Variants

### `gnn_hgt/` (canonical Table-4 checkpoint)

Heterogeneous Graph Transformer encoder + custom PPO policy/value heads,
trained 500,224 steps (977 PPO iterations, 44,081 episodes) on the
14-scenario in-distribution corpus. Reproduces every GNN-HGT+CPM cell in
the paper's Table 4 deterministically (e.g. `large_e_mixed_dag` = 5085).

### `gnn_hgt_ls/` (5-seed ensemble)

Five independently-seeded HGT runs (seeds 42, 137, 271, 419, 613) trained
under a staged curriculum on a wider corpus mix. Used by the OOD
benchmark (`B_ood_benchmark.py`) and the hybrid decomposition study
(`E_hybrid_decomposition.py`) where multi-seed statistics are reported
in the paper.

### `gnn_mlp/` (MaskablePPO baseline)

MLP-encoder policy trained with stable-baselines3 MaskablePPO. The
encoder's frozen action space is 128 tasks x 10 agents (= 1280 logits);
`experiments/common.py` reshapes the current `StateEncoder` mask to fit
this shape, so scenarios with > 128 tasks or > 10 agents are not
supported by the MLP variant.

> **Note on `.zip`**: `gnn_mlp_final.zip` is stable-baselines3's native
> checkpoint container (a single zip wrapping `policy.pth`,
> `policy.optimizer.pth`, hyperparameter metadata, and a sb3 version
> marker). **Do not extract it manually** — `MaskablePPO.load()` reads
> the zip directly. The path passed to `.load()` omits the `.zip`
> suffix (sb3 appends it). See `gnn_mlp_assignments()` in
> `experiments/common.py` for the canonical call site.

## Loading

### GNN-HGT (custom PPO, `.pt`)

```python
import torch
from core.scheduler.gnn.policy_network import SchedulingPolicyNet
from core.scheduler.gnn.state_encoder import TASK_FEAT_DIM, AGENT_FEAT_DIM

policy = SchedulingPolicyNet(
    task_feat_dim=TASK_FEAT_DIM, agent_feat_dim=AGENT_FEAT_DIM,
    hidden_dim=64, encoder_type="hgt", num_heads=4, num_layers=2,
    metadata=(
        ["task", "agent"],
        [("task", "depends_on", "task"),
         ("agent", "comm_with", "agent"),
         ("agent", "can_exec",  "task")],
    ),
)
ckpt = torch.load("models/checkpoints/gnn_hgt/gnn_hgt_final.pt",
                  map_location="cpu", weights_only=True)
state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) \
        and "policy_state_dict" in ckpt else ckpt
policy.load_state_dict(state)
policy.eval()
```

### GNN-MLP (sb3 MaskablePPO, `.zip`)

```python
from sb3_contrib import MaskablePPO
policy = MaskablePPO.load("models/checkpoints/gnn_mlp/gnn_mlp_final")
# Note: sb3 appends .zip automatically.
```

## Git LFS

The `.pt` and `.zip` files are stored via Git Large File Storage. After
cloning, install LFS once per machine and pull the binary blobs:

```bash
git lfs install
git lfs pull
```

If `git lfs install` is missing, install Git LFS first
(https://git-lfs.com/).

## Re-training

The training driver is `core/scheduler/gnn/trainer.py`. Re-training is
not required to reproduce any paper artifact; the shipped checkpoints
are sufficient. A full HGT training pass takes ~35 hours per seed on a
single CPU.
