"""
Driver F — Structural regression on OOD gap vs. task-graph features.

Uses the per-cell data emitted by ``B_ood_benchmark.py run`` (40 OOD scenarios
× 5 seeds × 5 rounds × 2 SS variants) together with the OOD scenarios'
structural features to fit:

    gap_pct ~ f(n_tasks, mutex_ratio, longest_chain_depth, parallelism_index,
                chain_depth_frac, parallelism_frac, [topology_class one-hot])

Two regression types:

  1. **Linear regression** (sklearn) — interpretable coefficients,
     R^2 + per-feature contribution
  2. **Random forest** (sklearn) — non-linear feature importance ranking

Output:
  results/structural_regression/
    regression_results.json   — coefficients, importances, R^2
    regression_report.md      — paper-ready table + interpretation

Usage:
    python experiments/F_structural_regression.py \
        --ood-metrics results/ood_benchmark_metrics.json \
        --output results/structural_regression

Pure CPU; runs in well under a minute.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OOD_METRICS = PROJECT_ROOT / "results" / "ood_benchmark_metrics.json"
DEFAULT_OOD_DIR = PROJECT_ROOT / "data" / "training_corpus_ood"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "structural_regression"

VARIANTS = ["gnn_hgt_ss", "gnn_mlp_ss"]
TOPOLOGY_CLASSES = ["deep_chain", "mixed", "mutex_dense", "parallel_pure", "sampling_burst"]
FEATURES_NUMERIC = ["n_tasks", "mutex_ratio", "longest_chain_depth",
                    "parallelism_index", "chain_depth_frac", "parallelism_frac"]


def load_data(ood_metrics: Path, ood_dir: Path):
    """Load per-cell OOD data + scenario structural features.

    Returns a list of dicts, one per (variant, scenario, seed) — typically
    200 × 2 variants = 400 rows. Each row aggregates 5 rounds (deterministic,
    so mean equals any single value).
    """
    data = json.load(open(ood_metrics, encoding="utf-8"))
    raw = data["raw"]

    # Group by (variant, scenario, seed); mean over rounds
    by_vss = defaultdict(list)
    for r in raw:
        if r.get("gap_pct") is None:
            continue
        by_vss[(r["variant"], r["scenario"], r["seed"])].append(r["gap_pct"])

    # Load scenario features
    feats_by_scen = {}
    for fp in sorted(ood_dir.glob("ood_*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        meta = d["metadata"]
        sf = meta["structural_features"]
        feats_by_scen[fp.stem] = {
            "topology_class": meta["topology_class"],
            "n_tasks": sf["n_tasks"],
            "n_edges": sf["n_edges"],
            "mutex_edges": sf["mutex_edges"],
            "mutex_ratio": sf["mutex_ratio"],
            "longest_chain_depth": sf["longest_chain_depth"],
            "parallelism_index": sf["parallelism_index"],
            "chain_depth_frac": round(sf["longest_chain_depth"] / max(sf["n_tasks"], 1), 3),
            "parallelism_frac": round(sf["parallelism_index"] / max(sf["n_tasks"], 1), 3),
        }

    rows = []
    for (variant, scen, seed), gaps in by_vss.items():
        feats = feats_by_scen[scen]
        row = {
            "variant": variant, "scenario": scen, "seed": seed,
            "gap_pct": statistics.fmean(gaps),
            **feats,
        }
        rows.append(row)
    return rows


def linear_regression(X, y, feature_names):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(Xs, y)
    r2 = model.score(Xs, y)
    return {
        "r2": round(r2, 4),
        "coefficients_standardized": {n: round(c, 3) for n, c in zip(feature_names, model.coef_)},
        "intercept_standardized": round(model.intercept_, 3),
        "scaler_mean": {n: round(m, 3) for n, m in zip(feature_names, scaler.mean_)},
        "scaler_scale": {n: round(s, 3) for n, s in zip(feature_names, scaler.scale_)},
    }


def random_forest(X, y, feature_names, n_estimators=300, max_depth=8, seed=42):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=seed, n_jobs=-1)
    model.fit(X, y)
    r2 = model.score(X, y)
    importances = sorted(zip(feature_names, model.feature_importances_),
                         key=lambda kv: -kv[1])
    return {
        "r2_train": round(r2, 4),
        "feature_importance": [{"name": n, "importance": round(float(imp), 4)}
                               for n, imp in importances],
    }


def main():
    ap = argparse.ArgumentParser(description="Structural regression on OOD gap")
    ap.add_argument("--ood-metrics", type=Path, default=DEFAULT_OOD_METRICS,
                    help="Path to ood_benchmark_metrics.json (from B_ood_benchmark.py run)")
    ap.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR,
                    help="Directory containing ood_*.json scenario files")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading per-cell OOD data from {args.ood_metrics.name}...")
    rows = load_data(args.ood_metrics, args.ood_dir)
    print(f"  {len(rows)} rows ((variant, scenario, seed) cells)")

    results = {"variants": {}}

    for variant in VARIANTS:
        var_rows = [r for r in rows if r["variant"] == variant]
        print(f"\n--- {variant} ({len(var_rows)} rows) ---")

        # Numeric-only features
        X = [[r[f] for f in FEATURES_NUMERIC] for r in var_rows]
        y = [r["gap_pct"] for r in var_rows]

        # Linear regression
        lin = linear_regression(X, y, FEATURES_NUMERIC)
        print(f"  Linear regression R^2 = {lin['r2']}")
        print(f"  Top 3 coefficients (standardized): ")
        coefs = sorted(lin['coefficients_standardized'].items(),
                       key=lambda kv: -abs(kv[1]))[:3]
        for n, c in coefs:
            print(f"    {n}: {c:+.3f}")

        # Random forest
        rf = random_forest(X, y, FEATURES_NUMERIC)
        print(f"  Random forest R^2(train) = {rf['r2_train']}")
        print(f"  Top 3 feature importances: ")
        for r in rf['feature_importance'][:3]:
            print(f"    {r['name']}: {r['importance']:.3f}")

        # With one-hot topology
        topo_cols = [f"topo_{t}" for t in TOPOLOGY_CLASSES]
        X_ext = []
        for r in var_rows:
            row_feat = [r[f] for f in FEATURES_NUMERIC]
            for t in TOPOLOGY_CLASSES:
                row_feat.append(1.0 if r["topology_class"] == t else 0.0)
            X_ext.append(row_feat)
        lin_ext = linear_regression(X_ext, y, FEATURES_NUMERIC + topo_cols)
        rf_ext = random_forest(X_ext, y, FEATURES_NUMERIC + topo_cols)
        print(f"  + one-hot topology: linear R^2 = {lin_ext['r2']}, RF R^2(train) = {rf_ext['r2_train']}")

        results["variants"][variant] = {
            "n_rows": len(var_rows),
            "linear_numeric_only": lin,
            "rf_numeric_only": rf,
            "linear_with_topology": lin_ext,
            "rf_with_topology": rf_ext,
        }

    json.dump(results, open(args.output / "regression_results.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n  -> {args.output / 'regression_results.json'}")

    # ── Markdown report ────────────────────────────────────────────────────
    lines = [
        "# OOD Gap Structural Regression",
        "",
        "## Method",
        "",
        f"- Data: {len(rows)} (variant, scenario, seed) cells from the OOD benchmark "
        "(40 OOD × 5 seeds × 2 variants = 400 rows total)",
        f"- Features (numeric): {', '.join(FEATURES_NUMERIC)}",
        f"- Features (extended): + one-hot for {', '.join(TOPOLOGY_CLASSES)}",
        "- Models: sklearn LinearRegression (standardized features) + RandomForestRegressor "
        "(n_est=300, max_depth=8, seed=42)",
        "- Metric: R^2 (in-sample for both; no train/test split given the small n=200 per "
        "variant and high seed-collinearity)",
        "",
        "## Results",
        "",
        "### Table A — R^2 across model × variant × feature set",
        "",
        "| Model | Feature set | HGT-SS R^2 | MLP-SS R^2 |",
        "|---|---|---:|---:|",
    ]
    for label, key_lin, key_rf in [
        ("Linear", "linear_numeric_only", None),
        ("Linear (+ topo one-hot)", "linear_with_topology", None),
        ("RF (numeric only)", None, "rf_numeric_only"),
        ("RF (+ topo one-hot)", None, "rf_with_topology"),
    ]:
        h = results["variants"]["gnn_hgt_ss"]
        m = results["variants"]["gnn_mlp_ss"]
        if key_lin:
            h_r2 = h[key_lin]["r2"]
            m_r2 = m[key_lin]["r2"]
        else:
            h_r2 = h[key_rf]["r2_train"]
            m_r2 = m[key_rf]["r2_train"]
        lines.append(f"| {label} | (varies) | {h_r2} | {m_r2} |")

    lines += ["", "### Table B — Top features by Random Forest importance (numeric+topology)",
              "",
              "| Rank | Feature | HGT-SS importance | Feature | MLP-SS importance |",
              "|---:|---|---:|---|---:|"]
    h_rf = results["variants"]["gnn_hgt_ss"]["rf_with_topology"]["feature_importance"]
    m_rf = results["variants"]["gnn_mlp_ss"]["rf_with_topology"]["feature_importance"]
    for i, (h, m) in enumerate(zip(h_rf[:8], m_rf[:8]), 1):
        lines.append(f"| {i} | {h['name']} | {h['importance']:.3f} | "
                     f"{m['name']} | {m['importance']:.3f} |")

    lines += ["",
              "### Table C — Linear coefficients (standardized) — numeric features only",
              "",
              "Sign + magnitude shows which features push gap higher (positive coef) "
              "or lower (negative coef) per standardized unit.",
              "",
              "| Feature | HGT-SS coef | MLP-SS coef |",
              "|---|---:|---:|"]
    h_lin = results["variants"]["gnn_hgt_ss"]["linear_numeric_only"]["coefficients_standardized"]
    m_lin = results["variants"]["gnn_mlp_ss"]["linear_numeric_only"]["coefficients_standardized"]
    for f in FEATURES_NUMERIC:
        lines.append(f"| {f} | {h_lin[f]:+.3f} | {m_lin[f]:+.3f} |")

    # Interpretation
    h_lin_r2 = results["variants"]["gnn_hgt_ss"]["linear_numeric_only"]["r2"]
    h_lin_ext_r2 = results["variants"]["gnn_hgt_ss"]["linear_with_topology"]["r2"]
    m_lin_r2 = results["variants"]["gnn_mlp_ss"]["linear_numeric_only"]["r2"]
    m_lin_ext_r2 = results["variants"]["gnn_mlp_ss"]["linear_with_topology"]["r2"]
    h_rf_r2 = results["variants"]["gnn_hgt_ss"]["rf_numeric_only"]["r2_train"]
    m_rf_r2 = results["variants"]["gnn_mlp_ss"]["rf_numeric_only"]["r2_train"]

    lines += [
        "",
        "## Interpretation",
        "",
        f"### HGT-SS",
        f"- Linear model R^2 (numeric only): **{h_lin_r2}** — structural features explain "
        f"{int(h_lin_r2*100)}% of HGT-SS gap variance",
        f"- Adding one-hot topology raises R^2 to **{h_lin_ext_r2}** "
        f"(+{int((h_lin_ext_r2-h_lin_r2)*100)}pp): the topology class carries information beyond raw features",
        f"- Random Forest R^2(train) = **{h_rf_r2}** (non-linear interactions matter)",
        f"- Top predictive feature: **{h_rf[0]['name']}** "
        f"(importance {h_rf[0]['importance']:.3f})",
        "",
        f"### MLP-SS",
        f"- Linear model R^2 (numeric only): **{m_lin_r2}**",
        f"- + topology one-hot: **{m_lin_ext_r2}** "
        f"(+{int((m_lin_ext_r2-m_lin_r2)*100)}pp)",
        f"- Random Forest R^2(train) = **{m_rf_r2}**",
        f"- Top predictive feature: **{m_rf[0]['name']}** "
        f"(importance {m_rf[0]['importance']:.3f})",
        "",
    ]

    (args.output / "regression_report.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
    print(f"  -> {args.output / 'regression_report.md'}")


if __name__ == "__main__":
    main()
