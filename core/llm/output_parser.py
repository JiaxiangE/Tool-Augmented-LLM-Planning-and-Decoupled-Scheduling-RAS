"""
Output Parser.

Handles LLM output format issues:
  - JSON repair (trailing commas, unquoted keys, markdown fences)
  - Schema coercion (fill defaults, cast types)
  - Multi-format tool call extraction
  - Tool name validation against registry
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def repair_json(raw: str) -> str:
    """
    Attempt to fix common LLM JSON errors.

    Fixes:
      - Strip markdown code fences (```json ... ```)
      - Remove trailing commas before } or ]
      - Remove single-line comments (// ...)
    """
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?\s*\n?", "", raw)
    raw = raw.strip("`").strip()

    # Remove single-line comments
    raw = re.sub(r"//.*$", "", raw, flags=re.MULTILINE)

    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    return raw.strip()


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract the first JSON object or array from text.

    Looks for {...} or [...] patterns with balanced braces.
    """
    # Try to find fenced JSON first
    fenced = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if fenced:
        return fenced.group(1).strip()

    # Find first { or [
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx = text.find(start_char)
        if idx == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(idx, len(text)):
            c = text[i]

            if escape_next:
                escape_next = False
                continue

            if c == "\\":
                escape_next = True
                continue

            if c == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return text[idx : i + 1]

    return None


def parse_tool_call(
    response_content: str,
    tool_calls_native: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract a tool call from LLM output.

    Supports multiple formats:
      1. Native OpenAI tool_calls (from API response)
      2. JSON block in text with {"name": ..., "arguments": ...}
      3. Inline function call pattern: tool_name(arg1=val1, ...)

    Returns:
      {"name": "tool_name", "arguments": {...}} or None
    """
    # 1. Native tool calls (already parsed by API)
    if tool_calls_native and len(tool_calls_native) > 0:
        tc = tool_calls_native[0]
        if isinstance(tc, dict):
            return tc
        # Handle OpenAI-style objects
        if hasattr(tc, "function"):
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            return {"name": tc.function.name, "arguments": args}

    # 2. JSON block in text
    json_str = extract_json_block(response_content)
    if json_str:
        repaired = repair_json(json_str)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict) and "name" in parsed:
                return {
                    "name": parsed["name"],
                    "arguments": parsed.get("arguments", parsed.get("parameters", {})),
                }
        except json.JSONDecodeError:
            pass

    return None


def coerce_params(
    params: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Coerce parameter values to match expected types and fill defaults.

    - Converts string numbers to float/int
    - Fills missing optional fields with defaults from schema
    """
    properties = schema.get("properties", {})
    result = dict(params)

    for key, prop in properties.items():
        if key not in result:
            # Fill default if available
            if "default" in prop:
                result[key] = prop["default"]
            continue

        value = result[key]
        expected_type = prop.get("type")

        # String -> number coercion
        if expected_type == "number" and isinstance(value, str):
            try:
                result[key] = float(value)
            except ValueError:
                pass
        elif expected_type == "integer" and isinstance(value, str):
            try:
                result[key] = int(value)
            except ValueError:
                pass
        # String -> array coercion (from comma-separated)
        elif expected_type == "array" and isinstance(value, str):
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = [v.strip() for v in value.split(",")]

    return result


def validate_tool_call(
    tool_call: Dict[str, Any],
    tool_registry: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Validate a tool call against the tool registry.

    Returns:
      (is_valid, error_message_or_None)
    """
    name = tool_call.get("name")
    if not name:
        return False, "Tool call missing 'name' field"

    if name not in tool_registry:
        available = list(tool_registry.keys())
        return False, f"Unknown tool '{name}'. Available: {available}"

    return True, None
