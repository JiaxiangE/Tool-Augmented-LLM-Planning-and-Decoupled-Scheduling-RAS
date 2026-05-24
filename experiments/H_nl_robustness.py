"""
Driver H — Natural-language robustness experiment.

For each instruction family in the paraphrase dataset:

  1. Form a stable *reference* TaskGraph by majority-vote of K runs on the
     family's canonical NL prompt (handles LLM run-to-run stochasticity).
  2. For each paraphrase, plan a TaskGraph from the paraphrased prompt.
  3. Compute dual semantic-equivalence (SE) scores between paraphrase and
     reference: a strict structural-deviation test (±10% on |V|, |E|) and a
     lenient test (tool-set + agent-type-reqs match + op-count cosine ≥ 0.85).

Failures are classified by ``ErrorType`` so true LLM regressions can be
separated from transient API issues (with exponential backoff retry).

The LLM API key must be set via the ``DASHSCOPE_API_KEY`` (or equivalent
provider-specific) environment variable; it is never hardcoded and never
written to disk.

Usage:
    python experiments/H_nl_robustness.py \
        --dataset data/nl_robustness/paraphrase_dataset.json \
        --n-paraphrases 5 \
        --output-dir results/nl_robustness
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.schema.environment import EnvironmentState
from core.schema.taskgraph import TaskGraph
from core.llm.controller import ControllerConfig, PlanningController
from core.llm.llm_backend import QwenBackend


# ─────────────────────────────────────────────────────────────────────────────
# Error classification
# ─────────────────────────────────────────────────────────────────────────────
class ErrorType(str, Enum):
    NONE = "none"
    API_QUOTA = "api_quota_exhausted"           # HTTP 403 free-tier / rate
    API_TRANSIENT = "api_transient_error"        # 429, 5xx
    LLM_MAX_ITER = "llm_hit_max_iter"            # info.iterations == max_iters
    LLM_SILENT_EXIT = "llm_silent_exit"          # tg=None, iters < max_iters
    LLM_INVALID_TG = "llm_produced_invalid_tg"   # tg returned but failed validation
    SYSTEM_ERROR = "system_error"                # any other exception


def _classify_api_error(exc: Exception) -> ErrorType:
    """Map an OpenAI/httpx exception to ErrorType."""
    msg = str(exc).lower()
    if ("quota" in msg or "freetier" in msg or "free tier" in msg
            or " 403 " in msg or "code: 403" in msg
            or "overdue" in msg or "access denied" in msg
            or "good standing" in msg or "arrearage" in msg):
        return ErrorType.API_QUOTA
    for c in ("429", "500", "502", "503", "504"):
        if c in msg:
            return ErrorType.API_TRANSIENT
    if "rate" in msg or "ratelimiterror" in msg.replace(" ", ""):
        return ErrorType.API_TRANSIENT
    return ErrorType.SYSTEM_ERROR


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def _features_from_tg(tg: TaskGraph) -> Dict[str, Any]:
    op_counts = Counter(n.op_type for n in tg.nodes)
    return {
        "n_nodes": len(tg.nodes),
        "n_edges": len(tg.edges),
        "op_type_counts": dict(op_counts),
        "agent_type_reqs_set": sorted({tuple(sorted(n.agent_type_reqs or []))
                                       for n in tg.nodes}),
    }


def _cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _tg_signature(features: Dict[str, Any]) -> int:
    """Hash used for majority vote on the stable reference."""
    return hash((features["n_nodes"], features["n_edges"],
                  tuple(sorted(features["op_type_counts"].items()))))


# ─────────────────────────────────────────────────────────────────────────────
# Dual SE metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(tg: Optional[TaskGraph], ref_features: Dict[str, Any]) -> Dict[str, Any]:
    if tg is None or ref_features is None:
        return {
            "tg_produced": False,
            "strict_se": False,
            "lenient_se": False,
            "op_type_distribution_match_score": 0.0,
            "tool_call_sequence_match": False,
            "agent_type_reqs_consistency": False,
        }

    f = _features_from_tg(tg)
    ref_n = ref_features["n_nodes"]; ref_e = ref_features["n_edges"]
    node_dev = abs(f["n_nodes"] - ref_n) / max(ref_n, 1)
    edge_dev = abs(f["n_edges"] - ref_e) / max(ref_e, 1)
    strict_se = node_dev <= 0.10 and edge_dev <= 0.10

    op_sim = _cosine(Counter(f["op_type_counts"]),
                      Counter(ref_features["op_type_counts"]))
    tool_match = set(f["op_type_counts"]) == set(ref_features["op_type_counts"])
    agent_match = (set(map(tuple, f["agent_type_reqs_set"])) ==
                   set(map(tuple, ref_features["agent_type_reqs_set"])))
    # Lenient SE = same tool set + same agent-type-reqs + op-counts cosine >= 0.85
    lenient_se = tool_match and agent_match and op_sim >= 0.85

    return {
        "tg_produced": True,
        "n_nodes": f["n_nodes"],
        "n_edges": f["n_edges"],
        "op_type_counts": f["op_type_counts"],
        "strict_se": bool(strict_se),
        "lenient_se": bool(lenient_se),
        "node_dev_pct": round(100 * node_dev, 1),
        "edge_dev_pct": round(100 * edge_dev, 1),
        "op_type_distribution_match_score": round(op_sim, 3),
        "tool_call_sequence_match": bool(tool_match),
        "agent_type_reqs_consistency": bool(agent_match),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single trial with retry + error classification
# ─────────────────────────────────────────────────────────────────────────────
def run_trial(nl_text: str, max_iters: int = 8,
              max_api_retries: int = 5,
              model: str = None,
              api_key: str = None,
              base_url: str = None) -> Tuple[Optional[TaskGraph], Dict[str, Any]]:
    """Run one planning trial with retry/backoff on transient API errors.

    Returns:
        (tg or None, info_dict_with_error_type)
    """
    backend_kwargs = {}
    if model:
        backend_kwargs["model"] = model
    if api_key:
        backend_kwargs["api_key"] = api_key
    if base_url:
        backend_kwargs["base_url"] = base_url

    last_err: Optional[Exception] = None
    for attempt in range(max_api_retries):
        backend = QwenBackend(**backend_kwargs)
        config = ControllerConfig(max_iterations=max_iters)
        controller = PlanningController(llm_backend=backend,
                                         env_state=EnvironmentState(),
                                         config=config)
        t0 = time.time()
        try:
            tg, info = controller.plan(nl_text)
            wall = time.time() - t0
            iters = info.get("iterations", 0) if info else 0
            tool_calls = info.get("tool_calls", []) if info else []
            tool_call_names = [tc.get("tool_name") if isinstance(tc, dict) else str(tc)
                                for tc in tool_calls]
            if tg is None:
                err_type = (ErrorType.LLM_MAX_ITER if iters >= max_iters
                            else ErrorType.LLM_SILENT_EXIT)
                return None, {
                    "wall_s": round(wall, 2),
                    "iterations": iters,
                    "error_type": err_type.value,
                    "error_msg": None,
                    "tool_calls_made": tool_call_names,
                    "attempt": attempt + 1,
                    "model_used": backend.model_name,
                }
            return tg, {
                "wall_s": round(wall, 2),
                "iterations": iters,
                "error_type": ErrorType.NONE.value,
                "error_msg": None,
                "tool_calls_made": tool_call_names,
                "attempt": attempt + 1,
                "model_used": backend.model_name,
            }
        except Exception as e:
            wall = time.time() - t0
            err_type = _classify_api_error(e)
            last_err = e
            if err_type == ErrorType.API_TRANSIENT and attempt < max_api_retries - 1:
                sleep_s = (2 ** attempt) + random.random()
                print(f"    [retry {attempt+1}/{max_api_retries}] {err_type.value}; "
                      f"sleeping {sleep_s:.1f}s — {str(e)[:80]}")
                time.sleep(sleep_s)
                continue
            return None, {
                "wall_s": round(wall, 2),
                "iterations": 0,
                "error_type": err_type.value,
                "error_msg": str(e)[:300],
                "tool_calls_made": [],
                "attempt": attempt + 1,
                "model_used": backend.model_name,
            }
    return None, {
        "wall_s": 0.0,
        "iterations": 0,
        "error_type": _classify_api_error(last_err).value if last_err else ErrorType.SYSTEM_ERROR.value,
        "error_msg": str(last_err)[:300] if last_err else "unknown",
        "tool_calls_made": [],
        "attempt": max_api_retries,
        "model_used": model or "unknown",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reference: K-run majority vote on the canonical NL prompt
# ─────────────────────────────────────────────────────────────────────────────
def get_stable_reference(nl_text: str, n_runs: int = 3, max_iters: int = 8,
                          model: str = None, api_key: str = None,
                          base_url: str = None) -> Tuple[Optional[TaskGraph], Dict[str, Any]]:
    """Run reference up to n_runs times; majority-vote on structure signature."""
    candidates = []
    attempts_log = []
    for i in range(n_runs):
        tg, info = run_trial(nl_text, max_iters=max_iters, model=model,
                              api_key=api_key, base_url=base_url)
        attempts_log.append({"attempt_idx": i, **info,
                             "tg_n_nodes": (len(tg.nodes) if tg else None),
                             "tg_n_edges": (len(tg.edges) if tg else None)})
        if tg is not None:
            candidates.append((tg, _features_from_tg(tg)))

    if not candidates:
        return None, {"reference_status": "all_failed",
                       "n_attempts": n_runs,
                       "attempts_log": attempts_log}
    if len(candidates) == 1:
        return candidates[0][0], {"reference_status": "single_success",
                                    "n_attempts": n_runs,
                                    "attempts_log": attempts_log}

    sigs = [_tg_signature(f) for _, f in candidates]
    most_common = Counter(sigs).most_common(1)[0]
    if most_common[1] >= 2:
        idx = sigs.index(most_common[0])
        return candidates[idx][0], {"reference_status": "majority_stable",
                                      "n_agree": most_common[1],
                                      "n_attempts": n_runs,
                                      "attempts_log": attempts_log}
    return candidates[0][0], {"reference_status": "unstable",
                                "warning": f"all {len(candidates)} successful runs differ; results suspect",
                                "n_attempts": n_runs,
                                "attempts_log": attempts_log}


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset",
                    default=str(PROJECT_ROOT / "data" / "nl_robustness" / "paraphrase_dataset.json"))
    ap.add_argument("--n-paraphrases", type=int, default=5)
    ap.add_argument("--ref-runs", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--output-dir",
                    default=str(PROJECT_ROOT / "results" / "nl_robustness"))
    ap.add_argument("--families", nargs="*", default=None,
                    help="Subset of families to run")
    ap.add_argument("--model", default=None,
                    help="Override LLM model (default: QwenBackend.DEFAULT_MODEL = 'qwen3-max')")
    ap.add_argument("--api-key", default=None,
                    help="API key. Defaults to $DASHSCOPE_API_KEY (or $OPENAI_API_KEY if "
                         "--base-url is set to an OpenAI-compatible endpoint).")
    ap.add_argument("--base-url", default=None,
                    help="Override API base URL (e.g. for OpenAI-compatible providers).")
    args = ap.parse_args()

    # API key resolution: CLI > $DASHSCOPE_API_KEY > $OPENAI_API_KEY
    resolved_key = (args.api_key
                    or os.environ.get("DASHSCOPE_API_KEY")
                    or os.environ.get("OPENAI_API_KEY"))
    if not resolved_key:
        print(
            "ERROR: no LLM API key found.\n"
            "  Set one of:\n"
            "    export DASHSCOPE_API_KEY='your-key'   (Qwen via DashScope, default)\n"
            "    export OPENAI_API_KEY='your-key'      (any OpenAI-compatible provider)\n"
            "  Or pass via --api-key on the command line.\n"
            "  For non-DashScope providers, also pass --base-url\n"
            "  (e.g. https://api.openai.com/v1 for OpenAI proper).",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = json.load(open(args.dataset, encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = args.families or list(dataset.keys())
    print(f"Families: {families}")
    print(f"Paraphrases per family: {args.n_paraphrases}")
    print(f"Reference runs per family (majority vote): {args.ref_runs}")
    print(f"Max iters/trial: {args.max_iters}")
    print(f"Model: {args.model or '(QwenBackend default)'}\n")

    all_results = []
    for fam_id in families:
        fam = dataset[fam_id]
        print(f"=== {fam_id} ===")
        print(f"  Reference NL: {fam['reference'][:80]}")

        # --- Stable reference via K-run majority ---
        ref_tg, ref_info = get_stable_reference(fam["reference"],
                                                  n_runs=args.ref_runs,
                                                  max_iters=args.max_iters,
                                                  model=args.model,
                                                  api_key=args.api_key,
                                                  base_url=args.base_url)
        print(f"  Reference status: {ref_info['reference_status']}")
        for log in ref_info.get("attempts_log", []):
            print(f"    [ref attempt {log['attempt_idx']}] "
                  f"nodes={log['tg_n_nodes']} edges={log['tg_n_edges']} "
                  f"err={log['error_type']} iters={log['iterations']} wall={log['wall_s']}s")
        if ref_tg is None:
            all_results.append({"family": fam_id, "is_reference": True,
                                "nl": fam["reference"], "metrics": None,
                                "info": ref_info, "ref_features": None})
            continue

        ref_features = _features_from_tg(ref_tg)
        all_results.append({"family": fam_id, "is_reference": True,
                            "nl": fam["reference"],
                            "metrics": {"tg_produced": True, **ref_features},
                            "info": ref_info, "ref_features": ref_features})

        # --- Paraphrase trials ---
        paras = fam["paraphrases"][:args.n_paraphrases]
        for i, nl in enumerate(paras):
            tg, info = run_trial(nl, max_iters=args.max_iters, model=args.model,
                                  api_key=args.api_key, base_url=args.base_url)
            metrics = compute_metrics(tg, ref_features)
            all_results.append({"family": fam_id, "is_reference": False,
                                "paraphrase_idx": i, "nl": nl,
                                "metrics": metrics, "info": info,
                                "ref_features": ref_features})
            print(f"  P{i:>2}  nodes={metrics.get('n_nodes', 'F'):<4} "
                  f"strict={'Y' if metrics['strict_se'] else 'N'} "
                  f"lenient={'Y' if metrics['lenient_se'] else 'N'} "
                  f"op_sim={metrics.get('op_type_distribution_match_score', 0):.2f} "
                  f"err={info.get('error_type')} wall={info.get('wall_s')}s")

    # --- Save ---
    raw_path = out_dir / "nl_robustness_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # --- Summary with error-type breakdown ---
    by_fam: Dict[str, Dict[str, Any]] = {}
    err_counter: Counter = Counter()
    for r in all_results:
        if r["is_reference"]:
            continue
        fam = r["family"]
        if fam not in by_fam:
            by_fam[fam] = {"trials": 0, "strict_se": 0, "lenient_se": 0,
                            "op_sim_sum": 0.0, "tool_match": 0, "agent_match": 0,
                            "tg_produced": 0}
        s = by_fam[fam]
        m = r["metrics"]; info = r["info"]
        s["trials"] += 1
        if m["tg_produced"]:
            s["tg_produced"] += 1
            s["op_sim_sum"] += m.get("op_type_distribution_match_score", 0.0)
        if m["strict_se"]: s["strict_se"] += 1
        if m["lenient_se"]: s["lenient_se"] += 1
        if m.get("tool_call_sequence_match"): s["tool_match"] += 1
        if m.get("agent_type_reqs_consistency"): s["agent_match"] += 1
        err_counter[info.get("error_type", "?")] += 1

    total = sum(s["trials"] for s in by_fam.values())
    summary = {
        "model": args.model or QwenBackend.DEFAULT_MODEL,
        "n_paraphrases_per_family": args.n_paraphrases,
        "ref_runs": args.ref_runs,
        "overall": {
            "total_paraphrase_trials": total,
            "strict_se_rate": round(
                sum(s["strict_se"] for s in by_fam.values()) / max(total, 1), 3),
            "lenient_se_rate": round(
                sum(s["lenient_se"] for s in by_fam.values()) / max(total, 1), 3),
            "tg_produced_rate": round(
                sum(s["tg_produced"] for s in by_fam.values()) / max(total, 1), 3),
        },
        "error_type_distribution": dict(err_counter),
        "by_family": {
            fam: {
                "trials": s["trials"],
                "strict_se_rate": round(s["strict_se"] / max(s["trials"], 1), 3),
                "lenient_se_rate": round(s["lenient_se"] / max(s["trials"], 1), 3),
                "tg_produced_rate": round(s["tg_produced"] / max(s["trials"], 1), 3),
                "mean_op_sim": round(s["op_sim_sum"] / max(s["tg_produced"], 1), 3),
                "tool_match_rate": round(s["tool_match"] / max(s["trials"], 1), 3),
                "agent_match_rate": round(s["agent_match"] / max(s["trials"], 1), 3),
            } for fam, s in by_fam.items()
        },
    }
    sum_path = out_dir / "nl_robustness_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"OVERALL  total={total}  strict_se={summary['overall']['strict_se_rate']:.1%}  "
          f"lenient_se={summary['overall']['lenient_se_rate']:.1%}  "
          f"tg_produced={summary['overall']['tg_produced_rate']:.1%}")
    print(f"Error-type distribution: {dict(err_counter)}")
    print()
    for fam, s in summary["by_family"].items():
        print(f"  {fam:<30} strict={s['strict_se_rate']:.1%} "
              f"lenient={s['lenient_se_rate']:.1%} produced={s['tg_produced_rate']:.1%}")
    print(f"\n  -> {raw_path}")
    print(f"  -> {sum_path}")


if __name__ == "__main__":
    main()
