"""
Few-Shot Examples for the LLM Task Planner.

Each example is a complete ReAct trace showing the thought chain
and tool call sequence. These are embedded in Section 5 of the
System Prompt.

IMPORTANT: When tool parameters_schema changes, the examples here
must be updated too. The schema-drift test in the test suite validates
that all example tool calls still pass validate_params().
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Example format: list of (thought, tool_call, result_summary) tuples
# ---------------------------------------------------------------------------

EXAMPLE_1_LINEAR = {
    "title": "Basic PSR Relay + Scan + Sample",
    "user_directive": "Deploy a relay to cover sector A at (400,400), scan the sector, and sample the anomaly at (420,420).",
    "trace": [
        {
            "thought": (
                "The target sector is at (400,400), which is about 566m from the base at (0,0,0). "
                "This exceeds the 500m comm range, so I need to deploy a relay first. "
                "Let me call relay_planner to establish communication coverage."
            ),
            "tool_call": {
                "name": "relay_planner",
                "arguments": {
                    "target_pos": [400.0, 400.0, 0.0],
                    "base_pos": [0.0, 0.0, 0.0],
                    "comm_range": 500.0,
                    "prefix": "sector_a_relay",
                },
            },
            "result_summary": "Generated 1 DeployRelay node at (200, 200, 0).",
        },
        {
            "thought": (
                "Relay planned. Now I need to scan sector A to produce map data. "
                "I'll use coverage_planner with a bounding box around the target area."
            ),
            "tool_call": {
                "name": "coverage_planner",
                "arguments": {
                    "region_bounds": [[380.0, 380.0], [420.0, 420.0]],
                    "cell_size": 10.0,
                    "prefix": "scan_sector_a",
                },
            },
            "result_summary": "Generated 16 ExploreCell nodes covering a 4x4 grid, each producing map_chunk_scan_sector_a_N.",
        },
        {
            "thought": (
                "Scan planned. The anomaly at (420,420) needs sampling. "
                "The sampling step requires map data from the scan, so I specify the "
                "relevant map chunks in requires_data."
            ),
            "tool_call": {
                "name": "sampling_template",
                "arguments": {
                    "site_id": "anomaly_1",
                    "location": [420.0, 420.0, 0.0],
                    "requires_data": ["map_chunk_scan_sector_a_15"],
                },
            },
            "result_summary": "Generated 4-step pipeline: Approach -> DeployArm -> Extract -> Store for site anomaly_1.",
        },
        {
            "thought": (
                "All SubGraphs are ready. I need to assemble them with cross-edges: "
                "relay deployment must complete before any scanning begins."
            ),
            "tool_call": {
                "name": "graph_assembler",
                "arguments": {
                    "mission_id": "psr_sector_a_survey",
                    "cross_edges": [
                        {"src": "sector_a_relay_deploy_0", "dst": "scan_sector_a_cell_0"},
                    ],
                },
            },
            "result_summary": "Assembled TaskGraph with 21 nodes, 23 edges (including 1 auto-injected DATA edge). Validation passed.",
        },
    ],
}

EXAMPLE_2_PARALLEL = {
    "title": "Two-Sector Parallel Survey with Sampling",
    "user_directive": "Survey sector A (300-350, 300-350) and sector B (300-350, 400-450) in parallel, then collect samples from both.",
    "trace": [
        {
            "thought": (
                "Both sectors are within ~495m of the base, just barely within comm range. "
                "I'll plan coverage for both sectors. They can be explored in parallel."
            ),
            "tool_call": {
                "name": "coverage_planner",
                "arguments": {
                    "region_bounds": [[300.0, 300.0], [350.0, 350.0]],
                    "cell_size": 25.0,
                    "prefix": "scan_A",
                },
            },
            "result_summary": "Generated 4 ExploreCell nodes for sector A.",
        },
        {
            "thought": "Now plan coverage for sector B.",
            "tool_call": {
                "name": "coverage_planner",
                "arguments": {
                    "region_bounds": [[300.0, 400.0], [350.0, 450.0]],
                    "cell_size": 25.0,
                    "prefix": "scan_B",
                },
            },
            "result_summary": "Generated 4 ExploreCell nodes for sector B.",
        },
        {
            "thought": (
                "Now add sampling for both sectors. Sector A sample depends on A's map data, "
                "sector B sample depends on B's map data."
            ),
            "tool_call": {
                "name": "sampling_template",
                "arguments": {
                    "site_id": "site_A",
                    "location": [325.0, 325.0, 0.0],
                    "requires_data": ["map_chunk_scan_A_3"],
                },
            },
            "result_summary": "Generated 4-step sampling pipeline for site_A.",
        },
        {
            "thought": "Sampling for sector B.",
            "tool_call": {
                "name": "sampling_template",
                "arguments": {
                    "site_id": "site_B",
                    "location": [325.0, 425.0, 0.0],
                    "requires_data": ["map_chunk_scan_B_3"],
                },
            },
            "result_summary": "Generated 4-step sampling pipeline for site_B.",
        },
        {
            "thought": (
                "All SubGraphs ready. No cross-edges needed between the two sectors -- "
                "they can be explored in parallel. The DATA edges between scan and sampling "
                "will be auto-injected by the assembler."
            ),
            "tool_call": {
                "name": "graph_assembler",
                "arguments": {
                    "mission_id": "dual_sector_survey",
                    "cross_edges": [],
                },
            },
            "result_summary": "Assembled TaskGraph with 16 nodes. 2 DATA edges auto-injected. Validation passed.",
        },
    ],
}

EXAMPLE_3_REMOTE_PSR = {
    "title": "Remote PSR Crater Investigation (with Constraint Feedback)",
    "user_directive": "Investigate the deep PSR crater at (800, 800). Scan it and collect ice samples.",
    "trace": [
        {
            "thought": (
                "The crater at (800,800) is ~1131m from the base. This is well beyond the 500m comm range. "
                "I need a relay chain. Let me call relay_planner."
            ),
            "tool_call": {
                "name": "relay_planner",
                "arguments": {
                    "target_pos": [800.0, 800.0, 0.0],
                    "base_pos": [0.0, 0.0, 0.0],
                    "comm_range": 500.0,
                    "prefix": "psr_relay",
                },
            },
            "result_summary": "Generated 2 DeployRelay nodes at (267, 267, 0) and (533, 533, 0).",
        },
        {
            "thought": "Relays planned. Now scan the crater area.",
            "tool_call": {
                "name": "coverage_planner",
                "arguments": {
                    "region_bounds": [[780.0, 780.0], [820.0, 820.0]],
                    "cell_size": 10.0,
                    "prefix": "crater_scan",
                },
            },
            "result_summary": "Generated 16 ExploreCell nodes for the crater grid.",
        },
        {
            "thought": "Scan planned. Now add ice sampling at the crater center.",
            "tool_call": {
                "name": "sampling_template",
                "arguments": {
                    "site_id": "ice_core_1",
                    "location": [800.0, 800.0, -2.0],
                    "requires_data": ["map_chunk_crater_scan_7"],
                },
            },
            "result_summary": "Generated 4-step sampling pipeline for ice_core_1.",
        },
        {
            "thought": (
                "All SubGraphs ready. I must ensure relays are deployed before scanning starts. "
                "The last relay must be done before the first scan cell."
            ),
            "tool_call": {
                "name": "graph_assembler",
                "arguments": {
                    "mission_id": "psr_crater_investigation",
                    "cross_edges": [
                        {"src": "psr_relay_deploy_1", "dst": "crater_scan_cell_0"},
                    ],
                },
            },
            "result_summary": "Assembled TaskGraph with 22 nodes. 1 DATA edge auto-injected. Validation passed.",
        },
    ],
}


EXAMPLE_4_FEEDBACK_FIX = {
    "title": "Fixing a Constraint Violation from Feedback",
    "user_directive": "Reassess your previous graph in light of simulation feedback.",
    "trace": [
        {
            "thought": (
                "The feedback indicates a constraint violation: 'Comm Violation at Node scan_target_cell_0: distance to base 2121.0m exceeds range 500.0m'. "
                "I must deploy a relay chain to cover the distance to the target at (1500, 1500). "
                "I will call the relay_planner to fix this connectivity gap."
            ),
            "tool_call": {
                "name": "relay_planner",
                "arguments": {
                    "target_pos": [1500.0, 1500.0, 0.0],
                    "base_pos": [0.0, 0.0, 0.0],
                    "comm_range": 500.0,
                    "prefix": "patch_relay_chain",
                },
            },
            "result_summary": "Generated 4 DeployRelay nodes.",
        },
        {
            "thought": (
                "The relays are now generated. I must re-assemble the entire TaskGraph, injecting the new relay subgraph "
                "along with the previous scan/sample subgraphs that the system kept in cache. I will add a cross-edge "
                "so the last relay deploys before the scanning begins."
            ),
            "tool_call": {
                "name": "graph_assembler",
                "arguments": {
                    "mission_id": "patched_scan_mission",
                    "cross_edges": [
                        {"src": "patch_relay_chain_deploy_3", "dst": "scan_target_cell_0"},
                    ],
                },
            },
            "result_summary": "Assembled TaskGraph with 24 nodes. Validation passed.",
        },
    ],
}


def format_examples_for_prompt() -> str:
    """Format all examples into a prompt-ready string."""
    examples = [EXAMPLE_1_LINEAR, EXAMPLE_2_PARALLEL, EXAMPLE_3_REMOTE_PSR, EXAMPLE_4_FEEDBACK_FIX]
    parts = []

    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}: {ex['title']}")
        parts.append(f"**User**: \"{ex['user_directive']}\"")
        parts.append("")

        for j, step in enumerate(ex["trace"], 1):
            parts.append(f"**Step {j}**")
            parts.append(f"*Thought*: {step['thought']}")
            parts.append(f"*Tool Call*: `{step['tool_call']['name']}`")
            parts.append(f"```json\n{_safe_json(step['tool_call']['arguments'])}\n```")
            parts.append(f"*Result*: {step['result_summary']}")
            parts.append("")

        parts.append("---\n")

    return "\n".join(parts)


def get_all_example_tool_calls() -> List[Dict[str, Any]]:
    """
    Return all tool calls from all examples for schema-drift validation.

    Used by the test suite to verify examples match current tool schemas.
    """
    examples = [EXAMPLE_1_LINEAR, EXAMPLE_2_PARALLEL, EXAMPLE_3_REMOTE_PSR, EXAMPLE_4_FEEDBACK_FIX]
    calls = []
    for ex in examples:
        for step in ex["trace"]:
            calls.append(step["tool_call"])
    return calls


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
