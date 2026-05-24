# Appendix A: LLM Prompt Architecture

This appendix specifies the structured prompt used by the LLM planner, the
controller that wraps it, and the few-shot ReAct traces it learns from.

## A.1 Overview

The Planning-layer LLM is driven by a **five-section structured system
prompt**, assembled at run time by the `build_system_prompt()` function. Each
section targets a specific failure mode observed during early development.
Table A.1 summarises the role of each section.

**Table A.1: Five-Section Prompt Architecture**

| Section | Name | Length (approx.) | Failure mode addressed | Design principle |
|---------|------|-------------------|------------------------|------------------|
| Sec. 1 | Role Definition | ~200 tokens | LLM fabricates task nodes (hallucination) | Identity anchoring + hard constraint declaration |
| Sec. 2 | Available Tools | ~400 tokens (dynamic) | Wrong argument format, call to non-existent tools | Auto-generated schema from code |
| Sec. 3 | Output Format | ~200 tokens | Tool call without reasoning (skipped chain-of-thought) | Mandatory `[THOUGHT] -> [ACTION]` paradigm |
| Sec. 4 | Domain Constraints | ~300 tokens (dynamic) | Physical constraint violations (comm range, slope) | Injected from `EnvironmentState` at run time |
| Sec. 5 | Few-Shot Examples | ~800 tokens | Wrong tool order, missing cross-edges | Four complete ReAct trace demonstrations |

Total prompt length is approximately 1,900 tokens (varies with tool count and
agent-list size).

---

## A.2 Section 1: Role Definition

**Failure mode addressed**: Without an explicit role anchor, the LLM tends to
emit raw JSON task nodes directly ("hallucinated nodes"), bypassing the tool
chain. The resulting nodes lack required metadata (`agent_type_reqs`,
`produces`/`consumes`).

**Full text**:

```
You are a Lunar Surface Multi-Agent Task Planner.

Your job is to decompose high-level natural language mission directives into
a structured TaskGraph -- a directed acyclic graph of concrete, executable
tasks for a team of heterogeneous lunar rovers, relays, and samplers.

You work by calling specialized planning tools one at a time. Each tool
returns a SubGraph (a fragment of the final TaskGraph). After generating all
needed SubGraphs, you call the graph_assembler to merge them into a
complete, validated TaskGraph.

You MUST NOT fabricate task nodes or edges directly -- all nodes must come
from tool outputs.

CRITICAL -- MANDATORY REASONING: Before EVERY tool call, you MUST output
your step-by-step reasoning as text in the `content` field of your response.
Structure your thinking as:

[THOUGHT] I need to... because... The current state is...
          The target is at distance... which means...
[ACTION]  I will now call <tool_name> to...

You are FORBIDDEN from making a tool call with an empty content field.
Always explain your reasoning first, then call the tool. This chain-of-
thought is essential for debugging mission plans.
```

**Key design points**:

- **"MUST NOT fabricate"**: this hard constraint restricts the LLM's role
  from *generator* to *orchestrator*; every node must be produced by a tool.
- **Mandatory reasoning**: the forced chain-of-thought makes the planning
  process auditable and also lowers parameter-error rates (the distance
  arithmetic the model writes in its thought is typically consistent with
  the arguments it then passes to the tool).
- **Model-agnostic wording**: the prompt does not name a specific LLM. The
  reference implementation uses a Qwen-family model, but the prompt itself
  is not coupled to one provider.

---

## A.3 Section 2: Available Tools (Auto-Generated)

**Failure mode addressed**: Hard-coded tool descriptions drift out of sync
with the actual code (parameter names and types diverge), producing tool
calls the runtime cannot parse.

**Generation mechanism**: Every tool subclasses `BaseTool` and implements
`to_function_spec()`, which returns the OpenAI function-calling JSON Schema.
`build_system_prompt()` walks the registered tools and renders this section
at run time.

**Example fragment** (rendered for `relay_planner`):

```
### `relay_planner`
Deploy a chain of communication relay nodes between the base station and a
remote target location. Automatically calculates the number of relays
needed based on comm_range.

**Parameters:**
{
  "type": "object",
  "properties": {
    "target_pos":  {"type": "array", "items": {"type": "number"},
                    "description": "Target [x, y, z] in meters"},
    "base_pos":    {"type": "array", "items": {"type": "number"},
                    "description": "Base station [x, y, z]"},
    "comm_range":  {"type": "number",
                    "description": "Max comm range in meters (default 500)"},
    "prefix":      {"type": "string",
                    "description": "Node ID prefix for the relay chain"}
  },
  "required": ["target_pos"]
}
```

**Schema-drift protection**: the test suite includes a dedicated test that
extracts every tool call embedded in the few-shot examples and runs
`validate_params()` against the current schema. Any schema change that is
not reflected in the few-shot examples breaks CI.

**Tool registry overview**:

| Tool | Inputs | Output (SubGraph) | Purpose |
|------|--------|-------------------|---------|
| `relay_planner` | target_pos, base_pos, comm_range | DeployRelay node chain | Establish comm relay chain |
| `coverage_planner` | region_bounds, cell_size | ExploreCell grid nodes | Area coverage scanning |
| `sampling_template` | site_id, location, requires_data | 4-step sampling pipeline | Approach -> DeployArm -> Extract -> Store |
| `graph_assembler` | mission_id, cross_edges | Merged TaskGraph | Combine all SubGraphs into a validated DAG |
| `constraint_checker` | (auto-invoked) | ConstraintReport | Validate comm / terrain / energy constraints |

---

## A.4 Section 3: Output Format

**Failure mode addressed**: The LLM sometimes emits a tool call with no
accompanying reasoning text (especially in late dialogue turns). The
consequences are: (1) debugging is hard because no rationale is recorded,
and (2) parameter quality drops -- without an explicit distance calculation,
the model is more likely to misjudge whether a relay is needed.

**Full text**:

```
## Output Format

For EVERY turn, you MUST output BOTH of these in order:
1. **Thought** (in the text/content field): Your reasoning about what needs
   to happen next. Include:
   - What you observe about the current state
   - Distance/energy/comm calculations if relevant
   - Why you chose this specific tool and these parameters
2. **Tool Call**: Specify exactly one tool to invoke with its parameters.

EXAMPLE of correct output:
  Content: "[THOUGHT] The target sector is at (600,600), which is
  sqrt(600^2+600^2) ~= 849m from base. This exceeds the 500m comm range,
  so I need a relay. [ACTION] Calling coverage_planner to scan the
  40x40m area around the target."
  Tool Call: coverage_planner({...})

Your final step MUST be calling `graph_assembler` to merge all SubGraphs
into the final TaskGraph.

After the graph is assembled, the system will automatically run
`constraint_checker`. If violations are found, you will be asked to fix
them.

When you believe the plan is complete and validated, respond with NO tool
call and a summary of the plan.
```

**Supplementary mechanism**: in addition to the system prompt, the
controller appends a short reminder to each user turn:

```
Remember: you MUST include your reasoning in the content/text field
before making any tool call. Explain your [THOUGHT] then describe
your [ACTION].
```

This dual reminder (system prompt + per-turn injection) cut the empty-thought
rate from roughly 35% to under 5% in our experiments.

---

## A.5 Section 4: Domain Constraints (Dynamic)

**Failure mode addressed**: Without explicit physical rules, the LLM
frequently produces infeasible plans. The three most common failures are:

- **Comm link broken**: tasks scheduled beyond 500m from the base with no
  relay deployed.
- **PSR energy violation**: long tasks scheduled in permanently shadowed
  regions without accounting for the lack of solar recharge.
- **Missing sampling prerequisite**: sampling scheduled without a prior
  exploration scan to produce the required map data.

**Generation mechanism**: this section is built by
`_build_domain_constraints(env_state)`, which reads the current
`EnvironmentState` so that constraint parameters (comm range, agent list,
base station position) always match the active scenario.

**Full template** (dynamic fields shown as `{...}`):

```
## Domain Constraints (Lunar Surface Physics)

These are hard physical constraints. Violating them will cause mission
failure.

### Communication
- Base station comm range: {comm_range_m}m
- Any task with `requires_connected=True` MUST be within {comm_range_m}m
  of either the base station or an active relay
- If your target area is beyond {comm_range_m}m from the base, you MUST
  deploy relays using `relay_planner` BEFORE scheduling exploration or
  sampling tasks in that area

### Terrain
- Slopes > 30 degrees are IMPASSABLE -- do not plan tasks at locations
  with extreme elevation changes
- PSR (Permanently Shadowed Region) areas have NO solar power -- agents
  operating there consume battery only, no recharge
- Obstacle cells are completely blocked -- route around them

### Sampling Protocol
- Sampling REQUIRES prior exploration data -- you must run
  `coverage_planner` on the area BEFORE running `sampling_template`
- The `sampling_template` generates a 4-step pipeline:
  Approach -> DeployArm -> Extract -> Store
- Analysis (if needed) is a SEPARATE task, not part of sampling

### Agent Capabilities
- ROVER: mobility + exploration (ExploreCell)
- RELAY: communication relay deployment (DeployRelay)
- SAMPLER: sample collection (Approach, DeployArm, Extract, Store)
- ANALYZER: data analysis (Analyze)
- Tools set `agent_type_reqs` but NEVER assign specific agents --
  that is the scheduler's job

### Available Agents
{dynamically generated agent list with type, energy, position}

### Base Station
- Position: {base_station_pos}
```

**Key design decision**: tools set `agent_type_reqs` but never bind a
specific agent ID. This is the central planning/scheduling decoupling
contract; the LLM decides only what *type* of agent is needed, and the
downstream scheduler optimises the actual assignment.

---

## A.6 Section 5: Few-Shot Examples

**Failure mode addressed**: Even with the previous four sections in place,
the LLM still makes mistakes in:

- tool ordering (e.g., sampling before scanning),
- omitting critical cross-edges (e.g., no dependency between relay
  deployment and scanning), and
- responding ineffectively to constraint-violation feedback.

Few-shot examples address these by providing complete ReAct traces (sequence
of thought -> tool_call -> result triples) that demonstrate the correct
pattern. The four traces are reproduced in full in Appendix B.

---

## A.7 Controller Integration: Pre-Rules -> ReAct -> Post-Rules

The prompt is embedded in a larger **dual-layer controller** architecture:

```
+-------------------------------------------------------------+
|                    PlanningController.run()                  |
|                                                              |
|  Phase 1: Pre-Rules (deterministic)                          |
|    - Distance check: target > 500m? -> auto-call             |
|      relay_planner                                           |
|    - Result is injected into the system message               |
|                                                              |
|  Phase 2: Build System Prompt                                |
|    - build_system_prompt(tools, env_state, few_shot)         |
|                                                              |
|  Phase 3: ReAct Loop (max 15 iterations)                     |
|    - Token management (trim history if > budget)             |
|    - LLM call -> parse thought + tool_call                   |
|    - Tool execution -> SubGraph accumulation                 |
|    - Dedup cache (prevent repeated identical calls)          |
|    - graph_assembler -> constraint_checker -> done or retry  |
|                                                              |
|  Phase 4: Post-Rules (deterministic)                         |
|    - Verify all connected nodes have comm coverage           |
|    - Patch missing relay chains                              |
|    - Repair missing DATA edges                               |
|                                                              |
|  Output: Validated TaskGraph                                  |
+-------------------------------------------------------------+
```

Pre-Rules and Post-Rules act as a deterministic safety net that catches the
LLM's omissions. The ablation study in the main paper confirms that the
Post-Rules layer cannot be removed without a significant rise in
constraint-violation rate.

---

# Appendix B: Few-Shot ReAct Traces

The four traces below are reproduced verbatim from Section 5 of the system
prompt. Each trace covers a distinct planning pattern and walks through the
full thought -> tool_call -> result loop.

---

## B.1 Example 1: Linear Relay + Scan + Sample

**Pattern**: basic end-to-end planning -- relay -> area scan -> point
sampling -> DAG assembly.

**User Directive**: *"Deploy a relay to cover sector A at (400,400), scan
the sector, and sample the anomaly at (420,420)."*

**Step 1 -- Relay Planning**

*Thought*:
> The target sector is at (400,400), which is about 566m from the base at
> (0,0,0). This exceeds the 500m comm range, so I need to deploy a relay
> first. Let me call relay_planner to establish communication coverage.

*Tool Call*: `relay_planner`
```json
{
  "target_pos": [400.0, 400.0, 0.0],
  "base_pos": [0.0, 0.0, 0.0],
  "comm_range": 500.0,
  "prefix": "sector_a_relay"
}
```

*Result*: Generated 1 DeployRelay node at (200, 200, 0).

**Step 2 -- Coverage Scanning**

*Thought*:
> Relay planned. Now I need to scan sector A to produce map data. I'll use
> coverage_planner with a bounding box around the target area.

*Tool Call*: `coverage_planner`
```json
{
  "region_bounds": [[380.0, 380.0], [420.0, 420.0]],
  "cell_size": 10.0,
  "prefix": "scan_sector_a"
}
```

*Result*: Generated 16 ExploreCell nodes covering a 4x4 grid, each
producing `map_chunk_scan_sector_a_N`.

**Step 3 -- Sampling Pipeline**

*Thought*:
> Scan planned. The anomaly at (420,420) needs sampling. The sampling step
> requires map data from the scan, so I specify the relevant map chunks in
> `requires_data`.

*Tool Call*: `sampling_template`
```json
{
  "site_id": "anomaly_1",
  "location": [420.0, 420.0, 0.0],
  "requires_data": ["map_chunk_scan_sector_a_15"]
}
```

*Result*: Generated 4-step pipeline: Approach -> DeployArm -> Extract ->
Store for site anomaly_1.

**Step 4 -- Graph Assembly**

*Thought*:
> All SubGraphs are ready. I need to assemble them with cross-edges: relay
> deployment must complete before any scanning begins.

*Tool Call*: `graph_assembler`
```json
{
  "mission_id": "psr_sector_a_survey",
  "cross_edges": [
    {"src": "sector_a_relay_deploy_0", "dst": "scan_sector_a_cell_0"}
  ]
}
```

*Result*: Assembled TaskGraph with 21 nodes, 23 edges (including 1
auto-injected DATA edge). Validation passed.

**Key teaching points**:

- The explicit distance computation `sqrt(400^2+400^2) ~= 566m > 500m`
  triggers the relay decision.
- The `requires_data` field encodes the scan -> sample data dependency.
- The `cross_edges` argument encodes the relay -> scan temporal dependency.
- `graph_assembler` auto-injects a DATA edge between scan `produces` and
  sampling `consumes`.

---

## B.2 Example 2: Two-Sector Parallel Survey

**Pattern**: two independent sectors that can be explored in parallel,
demonstrating a DAG topology with **no cross-edges**.

**User Directive**: *"Survey sector A (300-350, 300-350) and sector B
(300-350, 400-450) in parallel, then collect samples from both."*

**Step 1 -- Coverage for Sector A**

*Thought*:
> Both sectors are within ~495m of the base, just barely within comm range.
> I'll plan coverage for both sectors. They can be explored in parallel.

*Tool Call*: `coverage_planner`
```json
{
  "region_bounds": [[300.0, 300.0], [350.0, 350.0]],
  "cell_size": 25.0,
  "prefix": "scan_A"
}
```

*Result*: Generated 4 ExploreCell nodes for sector A.

**Step 2 -- Coverage for Sector B**

*Thought*:
> Now plan coverage for sector B.

*Tool Call*: `coverage_planner`
```json
{
  "region_bounds": [[300.0, 400.0], [350.0, 450.0]],
  "cell_size": 25.0,
  "prefix": "scan_B"
}
```

*Result*: Generated 4 ExploreCell nodes for sector B.

**Step 3 -- Sampling for Sector A**

*Thought*:
> Now add sampling for both sectors. Sector A sample depends on A's map
> data, sector B sample depends on B's map data.

*Tool Call*: `sampling_template`
```json
{
  "site_id": "site_A",
  "location": [325.0, 325.0, 0.0],
  "requires_data": ["map_chunk_scan_A_3"]
}
```

*Result*: Generated 4-step sampling pipeline for site_A.

**Step 4 -- Sampling for Sector B**

*Thought*:
> Sampling for sector B.

*Tool Call*: `sampling_template`
```json
{
  "site_id": "site_B",
  "location": [325.0, 425.0, 0.0],
  "requires_data": ["map_chunk_scan_B_3"]
}
```

*Result*: Generated 4-step sampling pipeline for site_B.

**Step 5 -- Graph Assembly (No Cross-Edges)**

*Thought*:
> All SubGraphs ready. No cross-edges needed between the two sectors --
> they can be explored in parallel. The DATA edges between scan and
> sampling will be auto-injected by the assembler.

*Tool Call*: `graph_assembler`
```json
{
  "mission_id": "dual_sector_survey",
  "cross_edges": []
}
```

*Result*: Assembled TaskGraph with 16 nodes. 2 DATA edges auto-injected.
Validation passed.

**Key teaching points**:

- `cross_edges: []` indicates that the two parallel branches have no
  temporal coupling, so the scheduler is free to dispatch them concurrently.
- Each branch has its own `requires_data` keys, so data dependencies do not
  bleed across sectors.
- Together with Example 1 this shows the decision boundary between "needs
  cross-edge" and "doesn't need cross-edge".

---

## B.3 Example 3: Remote PSR Crater Investigation

**Pattern**: very long range (>1000m) requires a **multi-hop relay chain**,
demonstrating `relay_planner`'s automatic chain construction.

**User Directive**: *"Investigate the deep PSR crater at (800, 800). Scan
it and collect ice samples."*

**Step 1 -- Multi-Hop Relay Chain**

*Thought*:
> The crater at (800,800) is ~1131m from the base. This is well beyond the
> 500m comm range. I need a relay chain. Let me call relay_planner.

*Tool Call*: `relay_planner`
```json
{
  "target_pos": [800.0, 800.0, 0.0],
  "base_pos": [0.0, 0.0, 0.0],
  "comm_range": 500.0,
  "prefix": "psr_relay"
}
```

*Result*: Generated 2 DeployRelay nodes at (267, 267, 0) and (533, 533, 0).

**Step 2 -- Crater Scanning**

*Thought*:
> Relays planned. Now scan the crater area.

*Tool Call*: `coverage_planner`
```json
{
  "region_bounds": [[780.0, 780.0], [820.0, 820.0]],
  "cell_size": 10.0,
  "prefix": "crater_scan"
}
```

*Result*: Generated 16 ExploreCell nodes for the crater grid.

**Step 3 -- Ice Sampling**

*Thought*:
> Scan planned. Now add ice sampling at the crater center.

*Tool Call*: `sampling_template`
```json
{
  "site_id": "ice_core_1",
  "location": [800.0, 800.0, -2.0],
  "requires_data": ["map_chunk_crater_scan_7"]
}
```

*Result*: Generated 4-step sampling pipeline for ice_core_1.

**Step 4 -- Graph Assembly with Relay Dependency**

*Thought*:
> All SubGraphs ready. I must ensure relays are deployed before scanning
> starts. The last relay must be done before the first scan cell.

*Tool Call*: `graph_assembler`
```json
{
  "mission_id": "psr_crater_investigation",
  "cross_edges": [
    {"src": "psr_relay_deploy_1", "dst": "crater_scan_cell_0"}
  ]
}
```

*Result*: Assembled TaskGraph with 22 nodes. 1 DATA edge auto-injected.
Validation passed.

**Key teaching points**:

- Distance 1131m -> `ceil(1131/500) - 1 = 2` relays are automatically
  generated.
- The cross-edge targets the **last** relay (`deploy_1`), ensuring the
  whole chain is up before scanning begins.
- The negative z (-2.0) for the ice-core site demonstrates use of 3D
  coordinates for sub-surface targets such as crater floors.

---

## B.4 Example 4: Feedback-Driven Constraint Repair

**Pattern**: after the simulator reports a constraint violation, the LLM
performs **responsive repair** of the existing TaskGraph rather than
rebuilding it from scratch.

**User Directive**: *"Reassess your previous graph in light of simulation
feedback."*

**Simulation Feedback** (produced by `generate_feedback_prompt()`):
```
### Simulation Feedback Report
- Makespan: 2450.0 seconds
- Task Completion Rate: 75.0%

#### Constraint Violations:
- Comm Violation at Node scan_target_cell_0:
  distance to base 2121.0m exceeds range 500.0m

#### Action Required:
Your previous graph failed to complete all tasks.
Fix the dependency or connectivity issues and regenerate
the TaskGraph JSON by calling the appropriate tools.
```

**Step 1 -- Diagnose and Deploy Patch Relay Chain**

*Thought*:
> The feedback indicates a constraint violation: 'Comm Violation at Node
> scan_target_cell_0: distance to base 2121.0m exceeds range 500.0m'. I
> must deploy a relay chain to cover the distance to the target at (1500,
> 1500). I will call the relay_planner to fix this connectivity gap.

*Tool Call*: `relay_planner`
```json
{
  "target_pos": [1500.0, 1500.0, 0.0],
  "base_pos": [0.0, 0.0, 0.0],
  "comm_range": 500.0,
  "prefix": "patch_relay_chain"
}
```

*Result*: Generated 4 DeployRelay nodes.

**Step 2 -- Reassemble with Patched Subgraph**

*Thought*:
> The relays are now generated. I must re-assemble the entire TaskGraph,
> injecting the new relay subgraph along with the previous scan/sample
> subgraphs that the system kept in cache. I will add a cross-edge so the
> last relay deploys before the scanning begins.

*Tool Call*: `graph_assembler`
```json
{
  "mission_id": "patched_scan_mission",
  "cross_edges": [
    {"src": "patch_relay_chain_deploy_3", "dst": "scan_target_cell_0"}
  ]
}
```

*Result*: Assembled TaskGraph with 24 nodes. Validation passed.

**Key teaching points**:

- The LLM extracts both the violation type (Comm Violation) and the exact
  distance (2121m) from the feedback message.
- The SubGraph accumulator's cache is reused: no need to regenerate the
  scan or sample SubGraphs, only the new relay chain is added.
- `patch_relay_chain_deploy_3` (the 4th relay, index 3) is wired to
  `scan_target_cell_0`, ensuring the cross-edge targets the end of the
  chain.
- Distance 2121m -> `ceil(2121/500) - 1 = 4` relays, matching the
  generated result.

---

## B.5 Feedback Prompt Template

When the simulation layer returns a result, `generate_feedback_prompt()`
converts the `FeedbackReport` into a structured LLM prompt that triggers the
repair behaviour demonstrated in Example 4:

```
### Simulation Feedback Report
- **Makespan (Total Time):** {makespan_s} seconds
- **Task Completion Rate:** {task_completion_rate}%

#### Constraint Violations:
- {violation_1}
- {violation_2}
- ...

#### Critical Path Bottlenecks:
The following tasks are on the critical path and consuming significant time:
> {bottleneck_task_1}, {bottleneck_task_2}, ...

#### Structural Optimization Suggestions:
Based on the simulation, consider these STRUCTURAL adjustments to the
TaskGraph:
*(CRITICAL: You are only responsible for the graph structure. DO NOT
assign agents to tasks. Output the optimized JSON.)*
- {suggestion_1}
- {suggestion_2}
- ...

**Action Required:** [complete/incomplete conditional message]

Based on the above analysis, you MUST now fix the TaskGraph by calling
the appropriate tools.
Do NOT explain the problem -- the analysis above already explains it.
Your next response MUST be a tool call. Start with the most critical fix.
```

**Key design points**:

- **"DO NOT assign agents to tasks"** reinforces the planning/scheduling
  separation contract.
- **"Do NOT explain the problem"** prevents the conversational-trap mode
  where the model spends turns re-explaining the diagnosis instead of
  emitting a fix.
- **"Your next response MUST be a tool call"** enforces an action-oriented
  loop and prevents multi-turn discussion that produces no plan change.

---

## B.6 Schema-Drift Validation

To keep the few-shot examples synchronised with the live tool schema, the
test suite includes the following extraction helper:

```python
# In few_shot_examples.py
def get_all_example_tool_calls() -> List[Dict[str, Any]]:
    """Return all tool calls from all examples for schema-drift
    validation. Used by the test suite to verify examples match
    current tool schemas."""
    examples = [EXAMPLE_1_LINEAR, EXAMPLE_2_PARALLEL,
                EXAMPLE_3_REMOTE_PSR, EXAMPLE_4_FEEDBACK_FIX]
    calls = []
    for ex in examples:
        for step in ex["trace"]:
            calls.append(step["tool_call"])
    return calls  # 15 tool calls total
```

CI runs the following procedure:

1. Extract all 15 example tool calls.
2. For each call, run `tool.validate_params(call["arguments"])`.
3. Any schema mismatch fails the test.

This closes the loop **code change -> schema change -> few-shot examples
become invalid -> CI fails -> developer updates the example**.
