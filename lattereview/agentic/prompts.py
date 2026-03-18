"""System prompt builders for AgenticReviewer."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from pydantic import BaseModel


def build_system_prompt(
    *,
    name: str,
    backstory: str,
    system_prompt: str,
    output_type: Type[BaseModel],
    max_iterations: int,
    agentic_effort: str = "medium",
    enabled_skill_descriptions: Optional[List[Dict[str, str]]] = None,
    memory_summaries: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the full system prompt for an AgenticReviewer.

    In non-agentic mode (max_iterations=1), produces a minimal prompt focused
    on structured output. In agentic mode, adds tool usage guidance, memory
    context, and skill descriptions.

    Args:
        name: Agent name/identity.
        backstory: Agent backstory/persona description.
        system_prompt: User-provided system instructions.
        output_type: The Pydantic model defining expected output schema.
        max_iterations: Iteration budget (1 = non-agentic).
        agentic_effort: Tool usage guidance level (low/medium/high).
        enabled_skill_descriptions: List of {"name": ..., "description": ...} for enabled skills.
        memory_summaries: List of {"id": ..., "brief": ...} for loaded memories.

    Returns:
        Complete system prompt string.
    """
    sections = []

    # Identity
    sections.append(f"# Identity\nYou are {name}.")
    if backstory:
        sections.append(f"{backstory}")

    # User instructions
    if system_prompt:
        sections.append(f"# Instructions\n{system_prompt}")

    # Output schema guidance
    schema_desc = _describe_output_schema(output_type)
    sections.append(f"# Output Format\n{schema_desc}")

    # Agentic sections (only when max_iterations > 1)
    if max_iterations > 1:
        sections.append(_build_effort_guidance(agentic_effort, max_iterations))

        if enabled_skill_descriptions:
            sections.append(_build_skill_section(enabled_skill_descriptions))

        if memory_summaries:
            sections.append(_build_memory_section(memory_summaries))

    return "\n\n".join(sections)


def build_task_prompt(
    task_prompt_template: str,
    item_text: str,
) -> str:
    """Substitute the item text into the task prompt template.

    Args:
        task_prompt_template: Template string containing ${item}$ placeholder.
        item_text: The actual item text to substitute.

    Returns:
        The rendered task prompt.
    """
    return task_prompt_template.replace("${item}$", item_text)


def _describe_output_schema(output_type: Type[BaseModel]) -> str:
    """Generate a human-readable description of the output schema."""
    lines = ["Your response must be a structured JSON object with the following fields:"]
    schema = output_type.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for field_name, field_info in properties.items():
        field_type = field_info.get("type", "any")
        description = field_info.get("description", "")
        req = " (required)" if field_name in required else " (optional)"
        line = f"- **{field_name}** ({field_type}{req}): {description}"
        lines.append(line)

    return "\n".join(lines)


def _build_effort_guidance(effort: str, max_iterations: int) -> str:
    """Build tool usage guidance based on agentic effort level."""
    header = "# Tool Usage Guidance"

    if effort == "low":
        body = (
            "Use tools sparingly. Only call a tool when the information is clearly "
            "necessary to produce an accurate response. Prefer to answer directly "
            "when you have sufficient context from the item text alone."
        )
    elif effort == "high":
        body = (
            "Use tools proactively and thoroughly. Search for additional context, "
            "verify claims, consult your memories, and use all available skills "
            "to produce the most well-informed and accurate response possible."
        )
    else:  # medium (default)
        body = (
            "Use tools when they would meaningfully improve your response quality. "
            "Search for context when the item is ambiguous or when verification "
            "would strengthen your confidence. Don't over-research straightforward items."
        )

    budget = f"You have a budget of {max_iterations} iterations for this review."
    if max_iterations <= 4:
        budget += " This is a limited budget — be efficient with tool calls."

    return f"{header}\n{body}\n\n{budget}"


def _build_skill_section(skill_descriptions: List[Dict[str, str]]) -> str:
    """Build the enabled skills section for the system prompt."""
    lines = ["# Available Skills"]
    for skill in skill_descriptions:
        lines.append(f"- **{skill['name']}**: {skill['description']}")
    return "\n".join(lines)


def _build_memory_section(memory_summaries: List[Dict[str, str]]) -> str:
    """Build the memory context section for the system prompt."""
    lines = [f"# Your Memories ({len(memory_summaries)} total)"]
    for mem in memory_summaries:
        lines.append(f"- {mem['id']}: {mem['brief']}")
    lines.append("\nUse the load_memory tool to read full details when relevant.")
    return "\n".join(lines)
