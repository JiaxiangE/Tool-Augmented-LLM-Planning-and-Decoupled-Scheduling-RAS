"""
Planning Controller — Dual-layer orchestrator.

Architecture:
  Pre-Rules -> ReAct Loop (LLM <-> Tools) -> Post-Rules

Features:
  - Hard deterministic rules (pre/post/format)
  - ReAct loop with tool call dedup cache
  - Error recovery with retry (max 3 per failure)
  - Token management interface (counting + history trimming)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.schema.environment import EnvironmentState
from core.schema.taskgraph import TaskGraph

from .llm_backend import LLMBackend
from .output_parser import coerce_params, parse_tool_call, repair_json, validate_tool_call
from .prompts.few_shot_examples import format_examples_for_prompt
from .prompts.system_prompt import build_system_prompt
from .tools.base import BaseTool, SubGraph
from .tools.constraint_checker import ConstraintCheckerTool, ConstraintReport
from .tools.coverage_planner import CoveragePlannerTool
from .tools.graph_assembler import GraphAssemblerTool
from .tools.relay_planner import RelayPlannerTool
from .tools.sampling_template import SamplingTemplateTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    """Configuration for the PlanningController."""
    max_iterations: int = 15
    max_retries_per_failure: int = 3
    token_budget: int = 100_000  # Rough token limit before trimming
    auto_relay_distance_threshold: float = 500.0  # Auto-insert relay if target > this
    enable_hard_rules: bool = True


# ---------------------------------------------------------------------------
# Dedup Cache
# ---------------------------------------------------------------------------

def _cache_key(tool_name: str, params: Dict[str, Any]) -> str:
    """Generate a deterministic cache key from tool name + params."""
    raw = json.dumps({"name": tool_name, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PlanningController:
    """
    Dual-layer orchestrator for LLM-driven task planning.

    Layer 1 (Hard Rules): Deterministic checks and auto-interventions
    Layer 2 (ReAct Loop): LLM flexible reasoning with tool calls
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        env_state: Optional[EnvironmentState] = None,
        config: Optional[ControllerConfig] = None,
    ):
        self._llm = llm_backend
        self._env = env_state or EnvironmentState()
        self._config = config or ControllerConfig()

        # Register tools
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

        # State accumulators
        self._subgraphs: List[SubGraph] = []
        self._messages: List[Dict[str, Any]] = []
        self._dedup_cache: Dict[str, SubGraph] = {}
        self._total_tokens: int = 0
        self._event_log: List[Dict[str, Any]] = []

    def _log_event(self, event_type: str, **kwargs) -> None:
        """Record a diagnostic event for debugging."""
        entry = {"time": time.time(), "type": event_type, **kwargs}
        self._event_log.append(entry)
        logger.info("[EVENT] %s: %s", event_type, json.dumps({k: str(v)[:200] for k, v in kwargs.items()}, ensure_ascii=False))

    def _register_default_tools(self) -> None:
        """Register the 5 core domain tools."""
        tools = [
            CoveragePlannerTool(),
            RelayPlannerTool(),
            SamplingTemplateTool(),
            ConstraintCheckerTool(),
            GraphAssemblerTool(),
        ]
        for t in tools:
            self._tools[t.name] = t

    def register_tool(self, tool: BaseTool) -> None:
        """Register an additional tool."""
        self._tools[tool.name] = tool

    @property
    def tool_specs(self) -> List[Dict[str, Any]]:
        """Generate OpenAI-format tool specs for all registered tools."""
        return [t.to_function_spec() for t in self._tools.values()]

    # -------------------------------------------------------------------
    # Hard Rules
    # -------------------------------------------------------------------

    def _apply_pre_rules(self, user_directive: str) -> List[Dict[str, Any]]:
        """
        Pre-rules: deterministic checks before the ReAct loop starts.

        Returns additional tool calls to auto-execute.
        """
        if not self._config.enable_hard_rules:
            return []

        auto_calls: List[Dict[str, Any]] = []

        # Rule: If user mentions coordinates beyond comm range, auto-inject relay
        # We parse simple coordinate patterns from the directive
        import re
        coord_pattern = re.compile(r"\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)")
        matches = coord_pattern.findall(user_directive)

        base_pos = self._env.config.base_station_pos
        threshold = self._config.auto_relay_distance_threshold

        for match in matches:
            try:
                x, y = float(match[0]), float(match[1])
                import math
                dist = math.sqrt((x - base_pos[0]) ** 2 + (y - base_pos[1]) ** 2)
                if dist > threshold:
                    logger.info(
                        "Pre-rule: Target (%.0f, %.0f) is %.0fm from base (> %.0fm). "
                        "Auto-injecting relay_planner call.",
                        x, y, dist, threshold,
                    )
                    auto_calls.append({
                        "name": "relay_planner",
                        "arguments": {
                            "target_pos": [x, y, 0.0],
                            "base_pos": list(base_pos),
                            "comm_range": self._env.config.comm_range_m,
                            "prefix": f"auto_relay_{int(x)}_{int(y)}",
                        },
                    })
                    break  # One relay chain is usually sufficient
            except (ValueError, IndexError):
                continue

        return auto_calls

    def _apply_post_rules(self, task_graph: TaskGraph) -> Tuple[TaskGraph, ConstraintReport]:
        """
        Post-rules: auto-validate and optionally patch the TaskGraph.

        Always runs constraint_checker.
        If comm gaps found, auto-patches with relay_planner.
        """
        checker = self._tools.get("constraint_checker")
        if not isinstance(checker, ConstraintCheckerTool):
            # No checker available, return as-is
            return task_graph, ConstraintReport(is_valid=True)

        report = checker.check(task_graph, self._env)

        if report.is_valid:
            return task_graph, report

        # Auto-patch: comm violations -> inject relays
        comm_violations = [v for v in report.violations if v.type == "comm"]
        if comm_violations and self._config.enable_hard_rules:
            logger.info("Post-rule: %d comm violations found. Attempting auto-patch.", len(comm_violations))
            # For each comm violation, get the node location and add a relay
            assembler = self._tools.get("graph_assembler")
            if isinstance(assembler, GraphAssemblerTool):
                relay_tool = self._tools.get("relay_planner")
                if isinstance(relay_tool, RelayPlannerTool):
                    node_map = {n.id: n for n in task_graph.nodes}
                    patch_subgraphs = list(self._subgraphs)  # Copy current subgraphs

                    for v in comm_violations:
                        if v.node_id and v.node_id in node_map:
                            node = node_map[v.node_id]
                            if node.location:
                                relay_sg = relay_tool.execute({
                                    "target_pos": list(node.location),
                                    "base_pos": list(self._env.config.base_station_pos),
                                    "comm_range": self._env.config.comm_range_m,
                                    "prefix": f"patch_relay_{v.node_id}",
                                })
                                if relay_sg.nodes:
                                    patch_subgraphs.append(relay_sg)

                    # Re-assemble with patches
                    patched_tg, meta = assembler.assemble(
                        patch_subgraphs,
                        task_graph.mission_id,
                    )
                    if patched_tg:
                        # Re-check
                        new_report = checker.check(patched_tg, self._env)
                        return patched_tg, new_report

        return task_graph, report

    def _apply_format_rules(
        self,
        tool_call: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Format rules: fix malformed tool call parameters.

        Returns (fixed_call, was_modified).
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        tool = self._tools.get(name)
        if not tool:
            return tool_call, False

        # Coerce parameters using the tool's schema
        schema = tool.parameters_schema
        fixed_args = coerce_params(args, schema)

        was_modified = fixed_args != args
        if was_modified:
            logger.debug("Format rule: coerced params for %s", name)

        return {"name": name, "arguments": fixed_args, **{k: v for k, v in tool_call.items() if k not in ("name", "arguments")}}, was_modified

    # -------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Tuple[SubGraph, str]:
        """
        Execute a tool call with dedup cache and error handling.

        Returns (result_subgraph, result_summary_string).
        """
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        # Dedup check
        cache_k = _cache_key(name, args)
        if cache_k in self._dedup_cache:
            cached = self._dedup_cache[cache_k]
            summary = (
                f"[System] This tool '{name}' was already called with identical parameters. "
                f"Cached result returned ({len(cached.nodes)} nodes, {len(cached.edges)} edges). "
                f"Please proceed to the next step."
            )
            logger.info("Dedup cache hit for %s", name)
            return cached, summary

        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        # Special handling for graph_assembler -- inject accumulated subgraphs
        if name == "graph_assembler":
            args["_subgraphs"] = self._subgraphs

        result = tool.execute(args)

        # Cache the result
        self._dedup_cache[cache_k] = result

        # Determine if this goes into subgraphs accumulator
        # constraint_checker results do NOT go into subgraphs
        if name != "constraint_checker" and name != "graph_assembler":
            self._subgraphs.append(result)

        # Build summary
        summary = (
            f"Tool '{name}' executed successfully. "
            f"Generated {len(result.nodes)} nodes, {len(result.edges)} edges."
        )
        if result.metadata:
            # Add key metadata points
            for k in ("total_cells", "num_relays", "data_edges_injected", "validation", "report"):
                if k in result.metadata:
                    summary += f" {k}={result.metadata[k]}"

        return result, summary

    # -------------------------------------------------------------------
    # Token Management
    # -------------------------------------------------------------------

    def _trim_history_if_needed(self) -> None:
        """Trim conversation history if token count exceeds budget."""
        self._total_tokens = self._llm.count_tokens(self._messages)

        if self._total_tokens <= self._config.token_budget:
            return

        logger.warning(
            "Token count %d exceeds budget %d. Trimming history.",
            self._total_tokens, self._config.token_budget,
        )

        # Keep system prompt (first message) + last 3 turns (6 messages)
        if len(self._messages) > 7:
            system_msg = self._messages[0]
            recent = self._messages[-6:]
            self._messages = [system_msg] + recent
            self._total_tokens = self._llm.count_tokens(self._messages)
            logger.info("Trimmed to %d tokens", self._total_tokens)

    # -------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------

    def plan(self, user_directive: str) -> Tuple[Optional[TaskGraph], Dict[str, Any]]:
        """
        Execute the full planning pipeline.

        Args:
            user_directive: Natural language mission description.

        Returns:
            (task_graph, run_metadata) where task_graph may be None if planning failed.
        """
        # Reset state
        self._subgraphs = []
        self._messages = []
        self._dedup_cache = {}
        self._total_tokens = 0
        self._event_log = []

        run_meta: Dict[str, Any] = {
            "model": self._llm.model_name,
            "iterations": 0,
            "tool_calls": [],
            "errors": [],
            "pre_rules_applied": [],
            "post_rules_report": None,
        }

        self._log_event("pipeline_start", directive=user_directive, model=self._llm.model_name)

        # --- Phase 1: Pre-Rules ---
        auto_calls = self._apply_pre_rules(user_directive)
        self._log_event("pre_rules", count=len(auto_calls), tools=[ac["name"] for ac in auto_calls])
        for ac in auto_calls:
            self._log_event("pre_rule_exec", tool=ac["name"], args=ac["arguments"])
            try:
                result, summary = self._execute_tool(ac)
                run_meta["pre_rules_applied"].append({
                    "tool": ac["name"],
                    "result_nodes": len(result.nodes),
                })
                self._log_event("pre_rule_result", tool=ac["name"], nodes=len(result.nodes), edges=len(result.edges))
            except Exception as e:
                logger.error("Pre-rule tool execution failed: %s", e)
                run_meta["errors"].append(f"Pre-rule {ac['name']} failed: {e}")
                self._log_event("pre_rule_error", tool=ac["name"], error=str(e))

        # --- Phase 2: Build System Prompt ---
        few_shot = format_examples_for_prompt()
        system_prompt = build_system_prompt(
            tools=list(self._tools.values()),
            env_state=self._env,
            few_shot_examples=few_shot,
        )

        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_directive},
        ]

        # If pre-rules auto-called tools, inform the LLM via a system-level note
        if auto_calls:
            pre_info = (
                "Pre-planning rules have already executed the following tools:\n"
            )
            for ac in auto_calls:
                pre_info += f"  - {ac['name']}({json.dumps(ac['arguments'], ensure_ascii=False)})\n"
            pre_info += (
                "These SubGraphs are already in the accumulator. "
                "You should now continue planning: call coverage_planner to scan the area, "
                "sampling_template for any sampling sites, and finally graph_assembler to assemble the TaskGraph."
            )
            # Use role=system to avoid the LLM thinking the assistant already responded
            self._messages.append({"role": "system", "content": pre_info})
            self._log_event("pre_rule_context_injected", message_role="system", content_preview=pre_info[:200])

        # --- Phase 3: ReAct Loop ---
        final_task_graph: Optional[TaskGraph] = None

        for iteration in range(self._config.max_iterations):
            run_meta["iterations"] = iteration + 1
            self._log_event("react_iteration_start", iteration=iteration + 1, max=self._config.max_iterations,
                            message_count=len(self._messages), subgraphs=len(self._subgraphs))

            # Token management
            self._trim_history_if_needed()

            # Call LLM
            try:
                thought, tool_call = self._llm.call_with_tools(
                    self._messages, self.tool_specs,
                )
            except Exception as e:
                self._log_event("llm_call_error", iteration=iteration + 1, error=str(e))
                run_meta["errors"].append(f"LLM call failed at iteration {iteration + 1}: {e}")
                break

            self._log_event("llm_response", iteration=iteration + 1,
                            thought_len=len(thought) if thought else 0,
                            thought_preview=(thought or "")[:300],
                            has_tool_call=tool_call is not None,
                            tool_name=tool_call["name"] if tool_call else None)

            # Append thought to messages
            msg: Dict[str, Any] = {"role": "assistant", "content": thought or ""}
            if tool_call:
                msg["tool_calls"] = [{
                    "id": tool_call.get("id", "call_unknown"),
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call.get("arguments", {}), ensure_ascii=False)
                    }
                }]
            self._messages.append(msg)

            # No tool call -> LLM says "done"
            if not tool_call:
                self._log_event("react_loop_exit", reason="no_tool_call", iteration=iteration + 1,
                                thought_preview=(thought or "")[:500])
                break

            # Validate tool call
            is_valid, err = validate_tool_call(tool_call, self._tools)
            if not is_valid:
                # Error recovery: feed error back to LLM
                self._messages.append({
                    "role": "tool",
                    "content": f"[Error] {err}. Please try again with a valid tool name.",
                    "tool_call_id": tool_call.get("id", "unknown"),
                })
                run_meta["errors"].append(err)
                continue

            # Apply format rules
            tool_call, was_fixed = self._apply_format_rules(tool_call)
            if was_fixed:
                logger.info("Format rule applied to %s", tool_call["name"])

            # Execute tool with retry
            retries = 0
            result: Optional[SubGraph] = None
            summary = ""

            while retries <= self._config.max_retries_per_failure:
                try:
                    result, summary = self._execute_tool(tool_call)
                    break
                except Exception as e:
                    retries += 1
                    error_msg = f"Tool '{tool_call['name']}' failed: {e}"
                    logger.warning("%s (retry %d/%d)", error_msg, retries, self._config.max_retries_per_failure)

                    if retries > self._config.max_retries_per_failure:
                        # Feed error to LLM for manual correction
                        self._messages.append({
                            "role": "tool",
                            "content": f"[Error] {error_msg}. Max retries exceeded. Please adjust parameters.",
                            "tool_call_id": tool_call.get("id", "unknown"),
                        })
                        run_meta["errors"].append(error_msg)
                        break

                    # Feed error to LLM
                    self._messages.append({
                        "role": "tool",
                        "content": f"[Error] {error_msg}. Please fix the parameters and try again.",
                        "tool_call_id": tool_call.get("id", "unknown"),
                    })

            if result is None:
                continue

            # Log tool call
            run_meta["tool_calls"].append({
                "iteration": iteration + 1,
                "tool": tool_call["name"],
                "nodes_generated": len(result.nodes),
            })

            # If graph_assembler was called, extract TaskGraph
            if tool_call["name"] == "graph_assembler":
                tg_json = result.metadata.get("task_graph_json")
                if tg_json:
                    try:
                        final_task_graph = TaskGraph.model_validate_json(tg_json)
                        summary += " TaskGraph assembled successfully."
                    except Exception as e:
                        summary += f" TaskGraph parsing failed: {e}"
                        run_meta["errors"].append(f"TaskGraph parse error: {e}")
                elif result.metadata.get("validation") == "failed":
                    summary += f" Validation failed: {result.metadata.get('validation_error', 'unknown')}"

            # Feed result to LLM
            self._messages.append({
                "role": "tool",
                "content": summary,
                "tool_call_id": tool_call.get("id", "unknown"),
            })

            # If we got a TaskGraph from assembler, break out
            if final_task_graph is not None:
                break

        # --- Phase 4: Post-Rules ---
        if final_task_graph is not None:
            logger.info("Applying post-rules to assembled TaskGraph...")
            final_task_graph, report = self._apply_post_rules(final_task_graph)
            run_meta["post_rules_report"] = report.model_dump()

            if not report.is_valid:
                logger.warning("Post-rule constraint check found violations: %s", report.violations)
        else:
            logger.warning("Planning completed without producing a TaskGraph.")
            self._log_event("pipeline_end_no_taskgraph")

        run_meta["total_tokens"] = self._total_tokens
        run_meta["event_log"] = self._event_log
        return final_task_graph, run_meta
