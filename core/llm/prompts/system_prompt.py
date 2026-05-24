"""
System Prompt Builder.

Auto-generates the 5-section system prompt from code:
  1. Role Definition
  2. Available Tools (from BaseTool.to_function_spec())
  3. Output Format
  4. Domain Constraints (from DomainConfig + hardcoded rules)
  5. Few-Shot Examples
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm.tools.base import BaseTool

from core.schema.environment import EnvironmentState


# ---------------------------------------------------------------------------
# Section 1: Role Definition
# ---------------------------------------------------------------------------

ROLE_DEFINITION = """You are a Lunar Surface Multi-Agent Task Planner.

Your job is to decompose high-level natural language mission directives into a structured TaskGraph -- a directed acyclic graph of concrete, executable tasks for a team of heterogeneous lunar rovers, relays, and samplers.

You work by calling specialized planning tools one at a time. Each tool returns a SubGraph (a fragment of the final TaskGraph). After generating all needed SubGraphs, you call the graph_assembler to merge them into a complete, validated TaskGraph.

You MUST NOT fabricate task nodes or edges directly -- all nodes must come from tool outputs.

**CRITICAL -- MANDATORY REASONING**: Before EVERY tool call, you MUST output your step-by-step reasoning as text in the `content` field of your response. Structure your thinking as:

[THOUGHT] I need to... because... The current state is... The target is at distance... which means...
[ACTION] I will now call <tool_name> to...

You are FORBIDDEN from making a tool call with an empty content field. Always explain your reasoning first, then call the tool. This chain-of-thought is essential for debugging mission plans."""


# ---------------------------------------------------------------------------
# Section 3: Output Format
# ---------------------------------------------------------------------------

OUTPUT_FORMAT = """## Output Format

For EVERY turn, you MUST output BOTH of these in order:
1. **Thought** (in the text/content field): Your reasoning about what needs to happen next. Include:
   - What you observe about the current state
   - Distance/energy/comm calculations if relevant
   - Why you chose this specific tool and these parameters
2. **Tool Call**: Specify exactly one tool to invoke with its parameters.

EXAMPLE of correct output:
```
Content: "[THOUGHT] The target sector is at (600,600), which is sqrt(600^2+600^2) ~= 849m from base. This exceeds the 500m comm range, so I need a relay. [ACTION] Calling coverage_planner to scan the 40x40m area around the target."
Tool Call: coverage_planner({...})
```

Your final step MUST be calling `graph_assembler` to merge all SubGraphs into the final TaskGraph.

After the graph is assembled, the system will automatically run `constraint_checker`. If violations are found, you will be asked to fix them.

When you believe the plan is complete and validated, respond with NO tool call and a summary of the plan."""


# Lightweight reminder injected into user messages to nudge thought output
THOUGHT_CHAIN_REMINDER = (
    "Remember: you MUST include your reasoning in the content/text field "
    "before making any tool call. Explain your [THOUGHT] then describe your [ACTION]."
)


# ---------------------------------------------------------------------------
# Section 4: Domain Constraints
# ---------------------------------------------------------------------------

def _build_domain_constraints(env_state: EnvironmentState) -> str:
    """Build domain constraint rules from config + hardcoded physics."""
    config = env_state.config
    comm_range = config.comm_range_m

    agents_desc = ""
    if env_state.agents:
        lines = []
        for a in env_state.agents.values():
            lines.append(f"  - {a.id}: type={a.agent_type.value}, energy={a.energy_wh}Wh, pos={a.position}")
        agents_desc = "\n".join(lines)
    else:
        agents_desc = "  (No agents specified -- planner should generate tasks without assignment)"

    return f"""## Domain Constraints (Lunar Surface Physics)

These are hard physical constraints. Violating them will cause mission failure.

### Communication
- Base station comm range: {comm_range}m
- Any task with `requires_connected=True` MUST be within {comm_range}m of either the base station or an active relay
- If your target area is beyond {comm_range}m from the base, you MUST deploy relays using `relay_planner` BEFORE scheduling exploration or sampling tasks in that area

### Terrain
- Slopes > 30 degrees are IMPASSABLE -- do not plan tasks at locations with extreme elevation changes
- PSR (Permanently Shadowed Region) areas have NO solar power -- agents operating there consume battery only, no recharge
- Obstacle cells are completely blocked -- route around them

### Sampling Protocol
- Sampling REQUIRES prior exploration data -- you must run `coverage_planner` on the area BEFORE running `sampling_template`
- The `sampling_template` generates a 4-step pipeline: Approach -> DeployArm -> Extract -> Store
- Analysis (if needed) is a SEPARATE task, not part of sampling

### Agent Capabilities
- ROVER: mobility + exploration (ExploreCell)
- RELAY: communication relay deployment (DeployRelay)
- SAMPLER: sample collection (Approach, DeployArm, Extract, Store)
- ANALYZER: data analysis (Analyze)
- Tools set `agent_type_reqs` but NEVER assign specific agents -- that is the scheduler's job

### Available Agents
{agents_desc}

### Base Station
- Position: {config.base_station_pos}
"""


# ---------------------------------------------------------------------------
# Build the complete prompt
# ---------------------------------------------------------------------------

def build_system_prompt(
    tools: List["BaseTool"],
    env_state: EnvironmentState,
    few_shot_examples: str = "",
) -> str:
    """
    Build the complete 5-section system prompt.

    Args:
        tools: List of registered BaseTool instances
        env_state: Current environment state
        few_shot_examples: Pre-formatted few-shot example string

    Returns:
        Complete system prompt string
    """
    # Section 2: Available Tools (auto-generated)
    tool_section = "## Available Tools\n\n"
    tool_section += "You may call the following tools. Each tool accepts specific parameters.\n\n"
    for tool in tools:
        spec = tool.to_function_spec()
        fn = spec["function"]
        tool_section += f"### `{fn['name']}`\n"
        tool_section += f"{fn['description']}\n\n"
        tool_section += f"**Parameters:**\n```json\n{json.dumps(fn['parameters'], indent=2)}\n```\n\n"

    # Section 4: Domain constraints
    domain_section = _build_domain_constraints(env_state)

    # Section 5: Few-shot
    few_shot_section = ""
    if few_shot_examples:
        few_shot_section = f"## Examples\n\n{few_shot_examples}\n"

    # Assemble
    sections = [
        "# System Prompt: Lunar Multi-Agent Task Planner\n",
        f"## Role\n\n{ROLE_DEFINITION}\n",
        tool_section,
        OUTPUT_FORMAT + "\n",
        domain_section,
        few_shot_section,
    ]

    return "\n---\n\n".join(s for s in sections if s)
