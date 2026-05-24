# GNN-HGT large-scale 5-seed ensemble

Five independently-seeded checkpoints of the GNN-HGT (Heterogeneous Graph
Transformer) policy trained on the large-scale procedurally-generated
corpus described in the paper (Section: *Scale-appropriate retrained
variant*; n_tasks ∈ [100, 500]) for 500K steps each. The 5-seed protocol
matches the paper's evaluation protocol: every reported HGT-LS number is
either a 5-seed mean with 95 % CI or a per-cell statistic over the 50
cells (5 seeds × 10 hold-out scenarios) produced by this ensemble.

## Files

```
seed_42/   gnn_hgt_final.pt   training_summary.json
seed_137/  gnn_hgt_final.pt   training_summary.json
seed_271/  gnn_hgt_final.pt   training_summary.json
seed_419/  gnn_hgt_final.pt   training_summary.json
seed_613/  gnn_hgt_final.pt   training_summary.json
```

Each `gnn_hgt_final.pt` is a **PyTorch state dict** (~1.2 MB) for a
`SchedulingPolicyNet` (policy + value heads on an HGT-encoded task
graph). The training driver is `core/scheduler/gnn/trainer.py`; the
policy class is `core.scheduler.gnn.policy_network.SchedulingPolicyNet`.

> **Note on the file format.** These are **not** stable-baselines3
> MaskablePPO checkpoints (which are `.zip` containers and live at
> `models/checkpoints/gnn_mlp/`). They are bare PyTorch state dicts
> produced by `torch.save` and must be loaded with `torch.load` +
> `state_dict.load_state_dict(...)`, **not** `MaskablePPO.load`.

## Training summary (per seed)

| Seed | Total steps | Episodes | Final mean reward | Wall time (h) |
|------|-------------|----------|-------------------|---------------|
|  42  | 500,736 | 3,785 | -152.62 | 35.3 |
| 137  | 500,736 | 3,927 | -133.99 | 35.8 |
| 271  | 500,736 | 3,866 | -138.51 | 35.8 |
| 419  | 500,736 | 3,829 | -118.42 | 35.8 |
| 613  | 500,736 | 3,813 | -130.36 | 35.1 |

Cross-seed mean final reward: **-134.78**, standard deviation: 12.4.

Shared hyperparameters: learning rate 3e-4 with cosine decay to 1e-5;
reward normalisation enabled; staged-curriculum training corpus.

## Loading

The canonical loader is in `experiments/E_hybrid_decomposition.py`
(`load_policy()`). The pattern:

```python
import torch
from core.scheduler.gnn.policy_network import SchedulingPolicyNet
from core.scheduler.gnn.state_encoder import TASK_FEAT_DIM, AGENT_FEAT_DIM

pyg_meta = (
    ["task", "agent"],
    [("task",  "depends_on", "task"),
     ("agent", "comm_with",  "agent"),
     ("agent", "can_exec",   "task")],
)

def load_seed(seed: int) -> SchedulingPolicyNet:
    policy = SchedulingPolicyNet(
        task_feat_dim=TASK_FEAT_DIM, agent_feat_dim=AGENT_FEAT_DIM,
        hidden_dim=64, encoder_type="hgt",
        num_heads=4, num_layers=2, metadata=pyg_meta,
    )
    ckpt = torch.load(
        f"models/checkpoints/gnn_hgt_ls/seed_{seed}/gnn_hgt_final.pt",
        map_location="cpu", weights_only=True,
    )
    state = ckpt["policy_state_dict"] \
            if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
    policy.load_state_dict(state)
    policy.eval()
    return policy

# Ensemble of all 5 seeds (paper's evaluation protocol)
ensemble = [load_seed(s) for s in [42, 137, 271, 419, 613]]
```

## How this ensemble is consumed by paper experiments

- `experiments/B_ood_benchmark.py` — Tables 14-16. Iterates over all 5
  seeds × 40 OOD scenarios; reports per-seed metrics and the 5-seed mean.
- `experiments/E_hybrid_decomposition.py` — Table 9 / Figure 3. Iterates
  over all 5 seeds × 10 hold-out scenarios (= 50 cells); reports the
  per-cell mean hybrid gap.

For Table 4 and Table 10 (paper main benchmark + Shackleton case study)
the paper uses the **single canonical checkpoint** at
`models/checkpoints/gnn_hgt/gnn_hgt_final.pt` — a separate, earlier
training run on the original 14-scenario corpus. See the top-level
`models/README.md` for full details.
