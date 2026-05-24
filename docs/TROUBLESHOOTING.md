# Troubleshooting

Common issues and their fixes.

---

## Install

### `torch==2.12.0+cpu` does not resolve

The pinned wheel is the PyTorch CPU build. On a system with CUDA, use
the matching CUDA wheel instead, or install torch first from the
PyTorch index:

```bash
pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If the second command later fails with `ModuleNotFoundError: No module
named 'pydantic'` (or any other package), it means the original
`pip install -r requirements.txt` aborted at torch and the remaining
packages were never installed. The two-step install above fixes it.

### Windows: `python` is hijacked by the Microsoft Store alias

If `python` opens the Microsoft Store instead of running Python, it's
the Windows app-execution-alias. Two fixes:

1. Use `py` instead of `python` to bootstrap the venv:
   ```cmd
   py -m venv venv
   venv\Scripts\activate
   ```
   Once the venv is activated, `python` correctly resolves to the venv's
   `python.exe`.

2. Or disable the alias permanently:
   `Settings → Apps → Advanced app settings → App execution aliases`,
   toggle off `python.exe` and `python3.exe`.

### `torch-geometric` install fails

`torch-geometric` requires `torch` to already be installed. Install
torch first, then re-run `pip install -r requirements.txt`.

### `ortools` fails on Apple Silicon

Use the universal2 wheel:

```bash
pip install ortools==9.15.6755 --force-reinstall --no-deps
```

### My local working copy is huge (1+ GB) — did I do something wrong?

No. `python -m venv venv` creates a `venv/` directory inside the repo
that holds ~1 GB of installed packages (torch alone is ~700 MB
unpacked). The repo's `.gitignore` lists `venv/` so `git status` /
`git add .` will correctly ignore it; only the ~24 MB of actual repo
content gets pushed. You can verify:

```bash
git check-ignore venv/  # should print: venv/
```

If you prefer to keep the venv outside the repo, put it in a sibling
directory (`python -m venv ../venv` then activate from there).

---

## Data

### Checkpoint files are zero bytes

This means `git lfs pull` did not run. Fix:

```bash
git lfs install
git lfs pull
ls -lh models/checkpoints/gnn_hgt_ls/seed_*/gnn_hgt_final.pt
# each file should be ~1.2 MB
```

### DEM files are missing

`data/dem/shackleton_*.npz` and `real_shackleton_*.json` are required
for `experiments/D_case_study_dem.py` and figures 4 / 5. If they are
missing the scripts will exit with `FileNotFoundError`. Re-pull with
`git lfs pull` (the `.npz` files are also tracked by LFS) or re-clone.

---

## Runtime

### `experiments/A_benchmark_comparison.py` is slow

The full benchmark loops over 10 schedulers × 14 scenarios. With the
shipped checkpoints it takes ~1 minute on a laptop CPU end-to-end. If
yours is taking notably longer, check that `torch` is the `+cpu` build
and not pulling in CUDA initialisation overhead.

### CP-SAT scheduler times out on large scenarios

CP-SAT is the optimal reference and is intended for small problem
instances (≤ 30 tasks). On larger task graphs (the `large_*` scenarios)
it will hit the per-run wall-clock limit and report `N/A` — this is
expected and is how the published Figure 2 reports it.

### `experiments/H_nl_robustness.py` errors with "missing API key"

This experiment calls a hosted LLM. Set the API key for the chosen
backend before running:

```bash
export DASHSCOPE_API_KEY=your-key      # Qwen (default)
# OR
export OPENAI_API_KEY=your-key         # OpenAI
```

Without an API key the natural-language layer cannot run; the rest of
the repository (schedulers, simulator, all other tables) does not
require any LLM call.

### `F_structural_regression.py` errors with `PermissionError: [Errno 13] Permission denied`

Caused by feeding the OOD benchmark's **output directory** (not the
JSON file inside it) to `F`'s `--ood-metrics` argument:

```bash
# Wrong — results/B_ood is a DIRECTORY, F open()s it and the OS refuses
python experiments/F_structural_regression.py --ood-metrics results/B_ood

# Right — point at the actual file inside the dir
python experiments/F_structural_regression.py \
    --ood-metrics results/B_ood/ood_benchmark_metrics.json
```

On Linux/macOS the same mistake surfaces as `IsADirectoryError`. The
file `ood_benchmark_metrics.json` is what `B_ood_benchmark.py run`
writes inside `--output-dir`.

### `B_ood_benchmark.py run` created a directory named `<something>.json`

The `--output-dir` flag is interpreted as a **directory**, not a file.
If you pass a path that ends in `.json`, the script will warn and strip
the suffix; if you ignore the warning the next driver in the pipeline
(`F_structural_regression.py`) will fail with `PermissionError` /
`IsADirectoryError` when it tries to `open()` a directory. Always pass
a directory-style path:

```bash
python experiments/B_ood_benchmark.py run --output-dir results/B_ood --rounds 5
#                                                       ^^^^^^^^^^^^
#                                                       directory, no .json suffix
```

---

## Figures

### Matplotlib raises `UserWarning: This figure includes Axes that are not compatible with tight_layout`

Harmless. Recent matplotlib versions emit this for figures combining
`GridSpec` with inline colorbars. The output PDF / PNG is unaffected.

### Chinese characters in plot labels

The plotting scripts use ASCII-only labels. If you see Chinese
characters anywhere in the rendered figures, please open an issue.

