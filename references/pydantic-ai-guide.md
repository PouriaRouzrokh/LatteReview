# Pydantic AI Production Guide

A comprehensive guide to building production-grade AI agent systems with [Pydantic AI](https://ai.pydantic.dev/), based on real-world patterns from a multi-agent chatbot serving a Yale University department. This guide covers agent creation, tool/skill architecture, multi-model support, conversation history management, streaming, and best practices.

> **Pydantic AI version**: `>=1.50.0` (V1 stable API, committed until V2 — April 2026 at earliest)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Agent Creation & Configuration](#2-agent-creation--configuration)
3. [Dependency Injection with RunContext](#3-dependency-injection-with-runcontext)
4. [Tool Definitions](#4-tool-definitions)
5. [Skill System: Folder-Based Tool Instructions](#5-skill-system-folder-based-tool-instructions)
6. [Multi-Model & Multi-Provider Support](#6-multi-model--multi-provider-support)
7. [System Prompts (Static + Dynamic)](#7-system-prompts-static--dynamic)
8. [Agent Registry & Self-Registration](#8-agent-registry--self-registration)
9. [Multi-Agent Dispatch (Router → Specialist)](#9-multi-agent-dispatch-router--specialist)
10. [Message History & Serialization](#10-message-history--serialization)
11. [Conversation History Compaction](#11-conversation-history-compaction)
12. [Streaming & Real-Time Status Updates](#12-streaming--real-time-status-updates)
13. [Multi-Iteration Agent Loops](#13-multi-iteration-agent-loops)
14. [Multimodal Tool Returns (Images, Binary)](#14-multimodal-tool-returns-images-binary)
15. [Configuration Patterns](#15-configuration-patterns)
16. [Testing & Development](#16-testing--development)
17. [Best Practices Summary](#17-best-practices-summary)
18. [Pydantic AI API Reference Cheat Sheet](#18-pydantic-ai-api-reference-cheat-sheet)

---

## 1. Architecture Overview

The system uses a **hierarchical multi-agent pattern** where a fast Router Agent handles simple queries directly and dispatches complex ones to specialist agents:

```
User Message
    ↓
Router Agent (gpt-4o-mini, fast, cheap)
    ├── Handles directly: greetings, chitchat, phone lookups
    └── Dispatches to specialists:
        ├── Policy Agent (gpt-5-mini, reasoning) → RAG search, regex search
        └── Schedule Agent (gpt-5-mini, reasoning) → external API queries
```

### Key Design Principles

- **Router is fast and cheap**: Use a small model (gpt-4o-mini) for routing decisions
- **Specialists are powerful**: Use reasoning models (gpt-5-mini with `reasoning_effort`) for domain tasks
- **Self-registration**: Agents register themselves with a central registry on module import
- **Independent histories**: Each agent maintains its own conversation history, compacted independently
- **Status streaming**: Tools emit real-time status updates to the frontend via `asyncio.Queue`

### File Structure

```
agent/
├── router_agent.py          # Router agent (dispatch + simple tools)
├── policy_agent.py          # Policy specialist (RAG, regex, images)
├── schedule_agent.py        # Schedule specialist (API queries)
├── registry.py              # Agent registry (self-registration)
├── skill_loader.py          # Skill folder loader (YAML + markdown)
├── prompts/
│   ├── router_prompt.py     # Router system prompt template
│   ├── policy_prompt.py     # Policy agent system prompt
│   └── schedule_prompt.py   # Schedule agent system prompt
├── tools/
│   └── decorators.py        # @tool_display() decorator
└── skills/                  # Skill instruction folders
    ├── agent_dispatch/
    │   └── SKILL.md         # When to dispatch vs handle directly
    ├── phone_lookup/
    │   └── SKILL.md         # Phone directory search strategy
    ├── policy_search/
    │   └── SKILL.md         # RAG search strategy
    └── schedule_lookup/
        └── SKILL.md         # Schedule API query strategy
```

---

## 2. Agent Creation & Configuration

### Basic Agent Creation

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModelSettings

# 1. Define model settings
settings = OpenAIChatModelSettings(
    max_tokens=2000,
)
# Conditionally add optional settings
settings["temperature"] = 0.3
# For reasoning models (o1, gpt-5-mini, etc.):
settings["openai_reasoning_effort"] = "low"  # "low", "medium", "high"

# 2. Create the agent
agent = Agent(
    "openai:gpt-4o-mini",          # Model identifier (provider:model)
    deps_type=MyDeps,               # Type for dependency injection
    system_prompt="You are a helpful assistant.",
    model_settings=settings,
)
```

### Agent Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str \| Model \| None` | LLM identifier, e.g. `"openai:gpt-5-mini"` |
| `deps_type` | `type[T]` | Dependency injection type (pass the **type**, not instance) |
| `system_prompt` | `str \| Sequence[str]` | Static system prompts (preserved in history) |
| `instructions` | `str \| callable` | Task directives (excluded from history when `message_history` is passed) |
| `model_settings` | `ModelSettings` | Default request settings |
| `retries` | `int` | Default retry count (default: 1) |
| `end_strategy` | `str` | `'early'` (default) or `'exhaustive'` |
| `tools` | `list` | Explicit tool list |
| `toolsets` | `list` | Reusable tool collections |

### Production Example: Router Agent

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModelSettings

from myapp.config import config

# Build settings from environment-driven config
_router_settings = OpenAIChatModelSettings(
    max_tokens=config.ROUTER.MAX_TOKENS,       # 2000
)
if config.ROUTER.TEMPERATURE:
    _router_settings["temperature"] = config.ROUTER.TEMPERATURE  # 0.3
if config.ROUTER.REASONING_EFFORT:
    _router_settings["openai_reasoning_effort"] = config.ROUTER.REASONING_EFFORT

# Create the agent
router_agent = Agent(
    config.ROUTER.MODEL,              # "openai:gpt-4o-mini"
    deps_type=RouterDeps,
    system_prompt=get_router_prompt(agent_registry.get_descriptions_for_prompt()),
    model_settings=_router_settings,
)
```

### Production Example: Specialist Agent

```python
_policy_settings = OpenAIChatModelSettings(
    max_tokens=config.POLICY_AGENT.MAX_TOKENS,   # 4000
)
if config.POLICY_AGENT.REASONING_EFFORT:
    _policy_settings["openai_reasoning_effort"] = config.POLICY_AGENT.REASONING_EFFORT  # "low"

policy_agent = Agent(
    config.POLICY_AGENT.MODEL,              # "openai:gpt-5-mini"
    deps_type=PolicyAgentDeps,
    system_prompt=POLICY_AGENT_PROMPT,
    model_settings=_policy_settings,
)
```

---

## 3. Dependency Injection with RunContext

Dependencies are how you pass runtime context (database sessions, API clients, callbacks) to tools and system prompts.

### Define Dependencies as Dataclasses

```python
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class RouterDeps:
    """Dependencies for the router agent."""
    session: AsyncSession                               # DB access
    phone_entries: List[dict]                          # Cached data
    status_callback: Callable[[str], Awaitable[None]]  # UI status updates
    agent_histories: Dict[str, List[ModelMessage]] = field(default_factory=dict)
    chat_id: Optional[int] = None
    restricted_document_type_ids: Optional[List[int]] = None
    user_full_name: Optional[str] = None


@dataclass
class PolicyAgentDeps:
    """Dependencies for the policy agent (minimal)."""
    session: AsyncSession
    status_callback: Callable[[str], Awaitable[None]]
    restricted_document_type_ids: Optional[List[int]] = None


@dataclass
class ScheduleAgentDeps:
    """Dependencies for the schedule agent (no DB needed)."""
    status_callback: Callable[[str], Awaitable[None]]
    user_full_name: Optional[str] = None
```

### Key Principles

1. **Pass the type to `deps_type`** (not an instance): `Agent(..., deps_type=RouterDeps)`
2. **Pass the instance to `deps=`** at runtime: `agent.run("query", deps=my_deps)`
3. **Keep deps minimal per agent** — specialists only get what they need
4. **Use `Callable` for callbacks** — this enables decoupled status streaming

### Access Dependencies in Tools

```python
from pydantic_ai import RunContext

@agent.tool
async def my_tool(ctx: RunContext[MyDeps], query: str) -> str:
    """Tool has type-safe access to dependencies."""
    await ctx.deps.status_callback("Working on it...")
    result = await ctx.deps.session.execute(...)
    return str(result)
```

### RunContext Properties

| Property | Description |
|----------|-------------|
| `ctx.deps` | The dependency instance |
| `ctx.retry` | Current retry attempt number |
| `ctx.usage` | Token usage statistics so far |
| `ctx.model` | Current model info |

---

## 4. Tool Definitions

### Registration Methods

**Method 1: `@agent.tool` — Context-aware (receives `RunContext`):**

```python
@agent.tool
async def search_db(ctx: RunContext[MyDeps], query: str, limit: int = 10) -> str:
    """Search the database for relevant results.

    Args:
        ctx: Run context with dependencies.
        query: Search query text.
        limit: Maximum results to return (default 10).

    Returns:
        Formatted string of results.
    """
    results = await ctx.deps.db.search(query, limit=limit)
    return format_results(results)
```

**Method 2: `@agent.tool_plain` — Stateless (no context):**

```python
@agent.tool_plain
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
```

**Method 3: `Tool()` class — Explicit construction:**

```python
from pydantic_ai import Tool

agent = Agent('openai:gpt-5-mini', tools=[
    Tool(my_function, takes_ctx=True, name='custom_name', description='...')
])
```

### Tool Docstrings → LLM Instructions

Pydantic AI automatically extracts tool name, description, and parameter descriptions from docstrings. **Google-style docstrings are recommended:**

```python
@agent.tool
async def rag_search(
    ctx: RunContext[PolicyAgentDeps],
    query: str,
    k: int = 10,
    threshold: Optional[float] = None,
) -> str:
    """Search for policy chunks semantically similar to the query.

    Args:
        ctx: Run context with dependencies.
        query: Search query text.
        k: Maximum number of results to return (default 10).
        threshold: Minimum similarity score 0-1 (default uses system config).

    Returns:
        Formatted string of matching policy chunks with metadata.
    """
```

The LLM sees: tool name `rag_search`, the full description, and each parameter with its type, default, and description.

### Custom Tool Display Decorator

Attach user-facing status text to tools without a centralized mapping:

```python
# tools/decorators.py
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

def tool_display(text: str) -> Callable[[F], F]:
    """Attach a display_text attribute to a tool function.

    Usage:
        @tool_display("Searching policies...")
        @agent.tool
        async def my_tool(ctx, query): ...

        # Access: my_tool._display_text -> "Searching policies..."
    """
    def decorator(func: F) -> F:
        func._display_text = text
        return func
    return decorator
```

**Usage pattern** (note: `@tool_display` goes ABOVE `@agent.tool`):

```python
@tool_display("Searching through policies...")
@agent.tool
async def rag_search(ctx: RunContext[PolicyAgentDeps], query: str) -> str:
    """Search for policy chunks semantically similar to the query."""
    # Emit status to UI
    await ctx.deps.status_callback(rag_search._display_text)
    # ... tool implementation
```

### Tool Retry with ModelRetry

When a tool encounters a recoverable error, use `ModelRetry` to ask the LLM to adjust its approach:

```python
from pydantic_ai import ModelRetry

@agent.tool
async def search(ctx: RunContext[MyDeps], query: str) -> str:
    """Search the knowledge base."""
    results = await ctx.deps.db.search(query)
    if not results:
        raise ModelRetry("No results found. Try rephrasing the query or using different keywords.")
    return format_results(results)
```

### Tool Return Types

Tools can return different types depending on complexity:

```python
# Simple: return a string
return "Found 5 results: ..."

# Multimodal: return ToolReturn with binary content (images, etc.)
from pydantic_ai import ToolReturn, BinaryContent

return ToolReturn(
    return_value=text_result,        # What the LLM sees as text
    content=[                         # Rich content for the LLM
        text_result,
        BinaryContent(data=image_bytes, media_type="image/png"),
    ],
)

# Metadata-only: return ToolReturn with app-only metadata
return ToolReturn(
    return_value="Image ready.",
    content=["Include this HTML: <img src='...'>"],
)
```

---

## 5. Skill System: Folder-Based Tool Instructions

Skills are **markdown files with YAML frontmatter** that provide workflow instructions for HOW and WHEN to use tools. They are organized as folders, enabling each skill to include supporting files (scripts, reference docs, examples).

### Folder Structure

```
agent/skills/
├── policy_search/
│   ├── SKILL.md                 # Main skill definition (YAML + markdown)
│   ├── examples.md              # Optional: example queries and expected behavior
│   └── search_strategies.py     # Optional: helper scripts referenced by skill
├── phone_lookup/
│   └── SKILL.md
├── schedule_lookup/
│   ├── SKILL.md
│   └── api_reference.md         # Optional: API endpoint documentation
└── agent_dispatch/
    └── SKILL.md
```

### Skill File Format (instructions.md)

Each skill folder MUST contain a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: policy_search
description: Search and retrieve department policy documents
triggers:
  - policy
  - procedure
  - guideline
  - protocol
  - handbook
---

# Policy Search Strategy

When a user asks about policies, procedures, or department information:

1. Start with `rag_search` using a clear semantic query derived from the user's question
2. Review results — if good matches (similarity > 0.5), use `retrieve_lines` to get more context
3. If no good semantic matches, try `regex_search` for exact terms, acronyms, or names
4. Combine information from multiple sources when needed
5. Always cite the policy title and source URL
6. When contradicting policies are found, note which is more recent
7. Never hallucinate — only report what the tools return
```

### YAML Frontmatter Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique skill identifier (defaults to folder name if omitted) |
| `description` | `str` | Yes | One-line description for system prompt injection |
| `triggers` | `list[str]` | No | Keywords that indicate this skill is relevant |

### Skill Loader Implementation

```python
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


@dataclass
class Skill:
    """A loaded skill with metadata and instructions."""
    name: str
    description: str
    triggers: List[str]
    instructions: str


class SkillLoader:
    """Loads and caches skill files from the skills directory."""

    def __init__(self) -> None:
        self._cache: Dict[str, Skill] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazily discover and index all skills on first access."""
        if self._loaded:
            return

        if not os.path.isdir(SKILLS_DIR):
            logger.warning(f"Skills directory not found: {SKILLS_DIR}")
            self._loaded = True
            return

        for entry in os.listdir(SKILLS_DIR):
            skill_dir = os.path.join(SKILLS_DIR, entry)
            skill_path = os.path.join(skill_dir, "SKILL.md")
            if os.path.isfile(skill_path):
                try:
                    skill = self._parse_skill(skill_path)
                    self._cache[skill.name] = skill
                except Exception as e:
                    logger.warning(f"Failed to load skill from {skill_path}: {e}")

        logger.info(f"Loaded {len(self._cache)} skills: {list(self._cache.keys())}")
        self._loaded = True

    def _parse_skill(self, path: str) -> Skill:
        """Parse a SKILL.md file with YAML frontmatter and markdown body."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split frontmatter from body
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                body = parts[2].strip()
            else:
                frontmatter = {}
                body = content
        else:
            frontmatter = {}
            body = content

        return Skill(
            name=frontmatter.get("name", os.path.basename(os.path.dirname(path))),
            description=frontmatter.get("description", ""),
            triggers=frontmatter.get("triggers", []),
            instructions=body,
        )

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        self._ensure_loaded()
        return self._cache.get(name)

    def get_descriptions_for_prompt(self) -> str:
        """Get formatted skill list for inclusion in system prompts."""
        self._ensure_loaded()
        if not self._cache:
            return ""
        lines = ["Available skills:"]
        for skill in self._cache.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def list_skills(self) -> List[str]:
        """List all available skill names."""
        self._ensure_loaded()
        return list(self._cache.keys())


# Module-level singleton
skill_loader = SkillLoader()
```

### Loading Supporting Files from Skill Folders

A skill folder can contain additional files that the agent loads on demand:

```python
import os

def load_skill_reference(skill_name: str, filename: str) -> Optional[str]:
    """Load a supporting file from a skill folder.

    Example:
        load_skill_reference("schedule_lookup", "api_reference.md")
    """
    path = os.path.join(SKILLS_DIR, skill_name, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None
```

### Injecting Skills into System Prompts

Skills can be injected into agent system prompts so the LLM knows how to use its tools:

```python
from myapp.agent.skill_loader import skill_loader

def get_router_prompt(agent_descriptions: str) -> str:
    skill_descriptions = skill_loader.get_descriptions_for_prompt()
    return f"""You are a routing agent.

{agent_descriptions}

{skill_descriptions}

Route queries to the appropriate specialist agent based on the skill triggers."""
```

Or inject specific skill instructions for a specialist agent:

```python
skill = skill_loader.get("policy_search")
if skill:
    system_prompt += f"\n\n## Tool Usage Instructions\n{skill.instructions}"
```

### Example: Complex Skill with Supporting Files

```
skills/
└── data_analysis/
    ├── SKILL.md                  # Main skill definition
    ├── examples.md               # Example queries → expected tool usage
    ├── schema_reference.md       # Database schema the agent can reference
    └── validation_rules.py       # Script with validation functions
```

**SKILL.md:**
```markdown
---
name: data_analysis
description: Analyze structured data from the database
triggers:
  - analyze
  - report
  - statistics
  - trend
  - compare
---

# Data Analysis Strategy

When analyzing data:

1. Use `query_database` to fetch raw data
2. If the schema is unclear, load `schema_reference.md` for table definitions
3. For complex aggregations, use `run_analysis_script` with appropriate parameters
4. See `examples.md` for expected output formats

## Reference Files
- `schema_reference.md`: Database table and column definitions
- `examples.md`: Sample queries and expected behavior
```

---

## 6. Multi-Model & Multi-Provider Support

### Provider Prefixes

Pydantic AI supports 11+ built-in providers. Model identifiers use a `provider:model` format:

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai:` | `openai:gpt-5-mini`, `openai:gpt-4o-mini` |
| Anthropic | `anthropic:` | `anthropic:claude-sonnet-4-6` |
| Google Gemini | `google-gla:` | `google-gla:gemini-3-flash-preview` |
| xAI (Grok) | `xai:` | `xai:grok-3` |
| Groq | `groq:` | `groq:llama-4-scout-17b` |
| Mistral | `mistral:` | `mistral:mistral-large-latest` |
| AWS Bedrock | `bedrock:` | `bedrock:anthropic.claude-sonnet-4-6` |
| OpenRouter | `openrouter:` | `openrouter:meta-llama/llama-4` |

### Switching Providers with Zero Code Changes

The key pattern: **store model identifiers in environment variables**, not in code.

```python
# config.py
import os

_config = {
    "ROUTER": {
        "MODEL": os.environ.get("ROUTER_MODEL", "openai:gpt-4o-mini"),
    },
    "POLICY_AGENT": {
        "MODEL": os.environ.get("POLICY_AGENT_MODEL", "openai:gpt-5-mini"),
    },
    "SCHEDULE_AGENT": {
        "MODEL": os.environ.get("SCHEDULE_AGENT_MODEL", "openai:gpt-5-mini"),
    },
}
```

```bash
# .env — switch to Anthropic with one line change
ROUTER_MODEL=anthropic:claude-haiku-4-5
POLICY_AGENT_MODEL=anthropic:claude-sonnet-4-6
SCHEDULE_AGENT_MODEL=anthropic:claude-sonnet-4-6
```

```python
# Agent creation — model comes from config, not hardcoded
router_agent = Agent(
    config.ROUTER.MODEL,    # Reads from env var at startup
    deps_type=RouterDeps,
    system_prompt=system_prompt,
    model_settings=settings,
)
```

### Runtime Model Override

You can override the model at runtime without changing the agent definition:

```python
# Use a different model for a specific run
result = await agent.run(
    "complex question",
    deps=deps,
    model="anthropic:claude-sonnet-4-6",   # Override just for this run
)
```

### Custom Provider Configuration

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

# Custom base URL (Azure, local LLM, etc.)
client = AsyncOpenAI(
    api_key="...",
    base_url="https://my-azure-endpoint.openai.azure.com/v1",
    max_retries=3,
)
model = OpenAIChatModel("gpt-5-mini", provider=OpenAIProvider(openai_client=client))

agent = Agent(model, deps_type=MyDeps)
```

### FallbackModel (Resilience)

```python
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel(
    "openai:gpt-5-mini",
    "anthropic:claude-sonnet-4-6",
    fallback_on=lambda resp: resp.finish_reason == "length",
)

agent = Agent(model, deps_type=MyDeps)
```

### Provider-Specific Model Settings

```python
from pydantic_ai.models.openai import OpenAIChatModelSettings

# OpenAI-specific settings
settings = OpenAIChatModelSettings(
    max_tokens=4000,
    temperature=0.3,
)
# Reasoning effort (OpenAI o1, gpt-5-mini, etc.)
settings["openai_reasoning_effort"] = "low"
```

```python
# For Anthropic, use prompt caching
settings = {
    "anthropic_cache_instructions": "5m",
    "anthropic_cache_tool_definitions": "1h",
    "anthropic_cache_messages": True,
}
```

---

## 7. System Prompts (Static + Dynamic)

### Static System Prompt

Passed at agent creation. Good for instructions that don't change between runs:

```python
POLICY_AGENT_PROMPT = """You are a policy search specialist. You have access to
a document database containing department policies and procedures.

Always:
- Cite policy titles and IDs in your responses
- Use HTML format for all responses
- Never hallucinate — only report what tools return
"""

policy_agent = Agent(
    "openai:gpt-5-mini",
    deps_type=PolicyAgentDeps,
    system_prompt=POLICY_AGENT_PROMPT,
)
```

### Dynamic System Prompt

Decorated functions that inject runtime context (current date, user identity, etc.):

```python
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

@router_agent.system_prompt
async def _inject_router_context(ctx: RunContext[RouterDeps]) -> str:
    """Inject current date/time and user identity into the router prompt."""
    now = datetime.now(EASTERN)
    parts = [
        f"Current date: {now.strftime('%A, %B %d, %Y')} (Eastern Time).",
    ]
    if ctx.deps.user_full_name:
        parts.append(f"The current user's name is: {ctx.deps.user_full_name}")
    return "\n".join(parts)
```

**Important**: You can have **both** a static `system_prompt` and `@agent.system_prompt` decorators. They are concatenated.

### Instructions vs System Prompts

| Feature | `system_prompt` | `instructions` |
|---------|-----------------|----------------|
| In message history | Yes | No (when `message_history` is set) |
| Use case | Multi-turn context | Single-turn directives |
| Recommendation | Use when chaining agents need to see it | Default choice for most cases |

**In production**: We use `system_prompt` because our agents maintain conversation history across multiple turns, and the prompt needs to be part of that history.

---

## 8. Agent Registry & Self-Registration

The registry pattern decouples the router from specialist agents. Specialists register themselves on module import.

### Registry Implementation

```python
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from pydantic_ai import Agent

logger = logging.getLogger(__name__)


@dataclass
class RegisteredAgent:
    """A specialist agent registered with the registry."""
    name: str                                          # "policy", "schedule"
    agent: Agent                                       # The Agent instance
    description: str                                   # For router's system prompt
    dispatch_display_text: str                         # Status text on dispatch
    deps_factory: Callable                             # Creates deps for this agent
    keywords: List[str] = field(default_factory=list)  # Trigger keywords
    max_history_messages: int = 16                     # Compaction threshold
    request_limit: int = 15                            # Max tool calls per run
    # Config info for admin dashboard
    model_id: str = ""
    max_tokens: int = 0
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None


class AgentRegistry:
    """Registry of specialist agents. Agents self-register on import."""

    def __init__(self) -> None:
        self._agents: Dict[str, RegisteredAgent] = {}

    def register(self, agent_info: RegisteredAgent) -> None:
        """Register a specialist agent."""
        if agent_info.name in self._agents:
            logger.warning(f"Overwriting existing agent: {agent_info.name}")
        self._agents[agent_info.name] = agent_info
        logger.info(f"Registered agent: {agent_info.name}")

    def get(self, name: str) -> Optional[RegisteredAgent]:
        """Get a registered agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def get_descriptions_for_prompt(self) -> str:
        """Get formatted descriptions for the router's system prompt."""
        if not self._agents:
            return "No specialist agents available."
        lines = ["Available specialist agents:"]
        for info in self._agents.values():
            keywords_str = ", ".join(info.keywords) if info.keywords else "general"
            lines.append(
                f'- "{info.name}": {info.description} (keywords: {keywords_str})'
            )
        return "\n".join(lines)


# Module-level singleton
agent_registry = AgentRegistry()
```

### Self-Registration Pattern

Each specialist agent registers itself at the module level:

```python
# policy_agent.py

from myapp.agent.registry import RegisteredAgent, agent_registry

# ... define policy_agent, tools, etc. ...

# Self-register with the agent registry
def _create_policy_deps(session, status_callback, restricted_ids=None, **kwargs):
    """Factory for creating PolicyAgentDeps."""
    return PolicyAgentDeps(
        session=session,
        status_callback=status_callback,
        restricted_document_type_ids=restricted_ids,
    )

agent_registry.register(
    RegisteredAgent(
        name="policy",
        agent=policy_agent,
        description="Searches and analyzes department policy documents.",
        dispatch_display_text="Researching your question...",
        deps_factory=_create_policy_deps,
        keywords=["policy", "procedure", "guideline", "protocol"],
        max_history_messages=config.POLICY_AGENT.MAX_HISTORY_MESSAGES,
        request_limit=config.POLICY_AGENT.REQUEST_LIMIT,
        model_id=config.POLICY_AGENT.MODEL,
        max_tokens=config.POLICY_AGENT.MAX_TOKENS,
        reasoning_effort=config.POLICY_AGENT.REASONING_EFFORT or None,
    )
)
```

### Import Order Matters

The router must import specialist agent modules **before** it reads the registry, so they have time to register:

```python
# chat_service.py (orchestration layer)

# Import specialist agents to ensure registry is populated BEFORE router
import myapp.agent.policy_agent   # noqa: F401 (side-effect: registers)
import myapp.agent.schedule_agent # noqa: F401 (side-effect: registers)

# Now import router (which reads the registry for its system prompt)
from myapp.agent.router_agent import router_agent, RouterDeps
```

---

## 9. Multi-Agent Dispatch (Router → Specialist)

The dispatch tool lets the router delegate complex queries to specialist agents.

### Dispatch Tool Implementation

```python
import asyncio

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.messages import ModelMessage

from myapp.agent.registry import agent_registry
from myapp.services.context_compaction import compact_history, needs_compaction

DISPATCH_TIMEOUT = 60  # seconds


@tool_display("Researching your question...")
@router_agent.tool
async def dispatch_agent(
    ctx: RunContext[RouterDeps],
    agent_name: str,
    query: str,
) -> str:
    """Dispatch a query to a specialist agent for detailed research.

    Args:
        ctx: Run context with dependencies.
        agent_name: Name of the specialist agent (e.g., "policy").
        query: The user's question to research.

    Returns:
        The specialist agent's response.
    """
    registered = agent_registry.get(agent_name)
    if not registered:
        available = ", ".join(agent_registry.list_agents())
        return f"Unknown agent '{agent_name}'. Available agents: {available}"

    await ctx.deps.status_callback(registered.dispatch_display_text)

    try:
        # 1. Load agent's conversation history from shared mutable dict
        agent_history = ctx.deps.agent_histories.get(agent_name, [])

        # 2. Compact history if it exceeds the agent's threshold
        if needs_compaction(agent_history, registered.max_history_messages):
            agent_history = await compact_history(
                agent_history,
                registered.max_history_messages,
                config.CONTEXT.MIN_RECENT_MESSAGES,
                ctx.deps.status_callback,
            )

        # 3. Create deps for the specialist agent via its factory
        specialist_deps = registered.deps_factory(
            ctx.deps.session,
            ctx.deps.status_callback,
            restricted_document_type_ids=ctx.deps.restricted_document_type_ids,
            user_full_name=ctx.deps.user_full_name,
        )

        # 4. Run the specialist with history and usage limits
        result = await asyncio.wait_for(
            registered.agent.run(
                query,
                deps=specialist_deps,
                message_history=agent_history,
                usage_limits=UsageLimits(request_limit=registered.request_limit),
            ),
            timeout=DISPATCH_TIMEOUT,
        )

        # 5. Save updated history back (includes this turn's messages)
        ctx.deps.agent_histories[agent_name] = result.all_messages()

        return str(result.output) if result.output else ""

    except asyncio.TimeoutError:
        return "<p>The search took too long. Please try a more specific question.</p>"
    except Exception as e:
        logger.error(f"dispatch_agent: '{agent_name}' failed: {e}", exc_info=True)
        return "<p>An error occurred. Please try again.</p>"
```

### Key Design Decisions

1. **Shared mutable dict**: `ctx.deps.agent_histories` is a dict passed by reference. Each dispatch call updates it, and it's persisted to DB after the chat completes.

2. **Per-agent history compaction**: Each specialist compacts its own history independently, with its own threshold.

3. **Deps factory pattern**: Each registered agent provides a factory function that creates its specific deps from the router's shared context.

4. **Timeout protection**: `asyncio.wait_for()` prevents specialist agents from hanging indefinitely.

5. **UsageLimits**: Prevents runaway tool calling by capping the number of LLM requests per run.

---

## 10. Message History & Serialization

### Message Types

Pydantic AI uses structured message types:

```python
from pydantic_ai.messages import (
    ModelMessage,            # Base type (union of Request + Response)
    ModelMessagesTypeAdapter, # For serialization/deserialization
    ModelRequest,            # User messages + tool returns
    ModelResponse,           # Assistant messages + tool calls
    UserPromptPart,          # User text content
    TextPart,                # Assistant text content
    ToolCallPart,            # Tool invocation by the model
    ToolReturnPart,          # Tool result sent back to model
)
```

### Serialization (Save to DB)

```python
from pydantic_core import to_jsonable_python

def serialize_messages(messages: List[ModelMessage]) -> List[dict]:
    """Serialize PydanticAI messages to JSON-serializable dicts."""
    return to_jsonable_python(messages)
```

### Deserialization (Load from DB)

```python
from pydantic_ai.messages import ModelMessagesTypeAdapter

def deserialize_messages(messages_json: List[dict]) -> List[ModelMessage]:
    """Deserialize stored JSON back to PydanticAI message objects."""
    return ModelMessagesTypeAdapter.validate_python(messages_json)
```

### Passing History to Agent Runs

```python
# Continue a conversation with full history
result = await agent.run(
    user_prompt,
    deps=deps,
    message_history=previous_messages,    # List[ModelMessage]
    usage_limits=UsageLimits(request_limit=10),
)

# Get updated history (includes this turn)
all_messages = result.all_messages()      # Full history
new_messages = result.new_messages()      # Only this turn's messages
```

### Backward Compatibility: Text-Only Fallback

For chats that predate PydanticAI (or don't have stored JSON history), reconstruct basic history from text messages:

```python
def format_history_fallback(text_messages: List[DBMessage]) -> List[ModelMessage]:
    """Reconstruct message history from text-only DB records."""
    messages: List[ModelMessage] = []
    for msg in text_messages:
        if msg.role == "user":
            messages.append(
                ModelRequest(parts=[UserPromptPart(content=msg.content)])
            )
        elif msg.role == "assistant":
            messages.append(
                ModelResponse(parts=[TextPart(content=msg.content)])
            )
    return messages
```

### Per-Agent History Storage

Each specialist agent maintains its own history, stored separately:

```python
# Load per-agent histories from DB
agent_histories_raw = await agent_history_repo.get_histories_for_chat(chat_id)
# Returns: {"policy": [dict, dict, ...], "schedule": [dict, dict, ...]}

# Deserialize each agent's history
agent_histories: Dict[str, List[ModelMessage]] = {}
for name, messages_json in agent_histories_raw.items():
    msgs = deserialize_messages(messages_json)
    if msgs:
        agent_histories[name] = msgs

# Pass to router deps (shared mutable dict)
deps = RouterDeps(
    ...,
    agent_histories=agent_histories,
)

# After agent run, save updated histories
for agent_name, agent_msgs in deps.agent_histories.items():
    serialized = serialize_messages(agent_msgs)
    await agent_history_repo.upsert_history(
        chat_id=chat_id,
        agent_name=agent_name,
        messages_json=serialized,
        message_count=len(serialized),
    )
```

---

## 11. Conversation History Compaction

When conversation history grows too long, older messages are summarized by an LLM into a concise recap.

### When to Compact

```python
def needs_compaction(messages: List[ModelMessage], max_history_messages: int) -> bool:
    """Check if history exceeds the compaction threshold."""
    return len(messages) > max_history_messages
```

### Compaction Algorithm

```python
async def compact_history(
    messages: List[ModelMessage],
    max_history_messages: int,
    min_recent: int,          # Keep this many recent messages verbatim
    status_callback=None,
) -> List[ModelMessage]:
    """Compact history by summarizing older messages."""
    old_messages, recent_messages = split_history(messages, min_recent)

    if not old_messages:
        return messages

    try:
        summary = await summarize_messages(old_messages)  # LLM call
    except Exception:
        summary = fallback_summary(old_messages)  # Extract user questions

    # Build summary as request/response pair (fits naturally in history)
    summary_pair = [
        ModelRequest(parts=[UserPromptPart(content="[Summary of our earlier conversation]")]),
        ModelResponse(parts=[TextPart(content=summary)]),
    ]

    return summary_pair + recent_messages
```

### Safe Split at Request Boundaries

Never split in the middle of a tool call/return pair:

```python
def split_history(
    messages: List[ModelMessage], min_recent: int
) -> Tuple[List[ModelMessage], List[ModelMessage]]:
    """Split history, aligning to ModelRequest boundaries."""
    if len(messages) <= min_recent:
        return [], messages

    split_idx = len(messages) - min_recent

    # Walk backward to find a ModelRequest boundary
    while split_idx > 0 and not isinstance(messages[split_idx], ModelRequest):
        split_idx -= 1

    if split_idx <= 0:
        return [], messages  # Can't safely split

    return messages[:split_idx], messages[split_idx:]
```

### Summarizer LLM

Use a cheap, fast model for summarization:

```python
COMPACTION_PROMPT = """Summarize the earlier portion of a conversation.
Preserve: policy names/IDs, key facts, user preferences, unresolved questions.
Omit: tool call mechanics, greetings, redundant info.
Start with "Here is what was discussed earlier:"

CONVERSATION:
{conversation_text}"""

async def summarize_messages(messages: List[ModelMessage]) -> str:
    """Call the summarizer LLM."""
    conversation_text = messages_to_text(messages)
    client = get_openai_client()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",           # Fast and cheap
        messages=[{"role": "user", "content": COMPACTION_PROMPT.format(
            conversation_text=conversation_text
        )}],
        max_tokens=1000,
        temperature=0.3,
    )
    return response.choices[0].message.content
```

### Converting Messages to Text

```python
def messages_to_text(messages: List[ModelMessage]) -> str:
    """Convert message list to readable text for the summarizer."""
    lines = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f"User: {part.content}")
                elif isinstance(part, ToolReturnPart):
                    content = str(part.content)[:300] + "..."
                    lines.append(f"[Tool result for {part.tool_name}: {content}]")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f"Agent: {part.content}")
                elif isinstance(part, ToolCallPart):
                    args = str(part.args)[:100] + "..."
                    lines.append(f"[Agent called: {part.tool_name}({args})]")
    return "\n".join(lines)
```

### Two-Level Compaction

Compaction happens at two levels:
1. **Router history**: Compacted in the chat service before running the router
2. **Agent histories**: Compacted in the dispatch tool before running each specialist

```
Router history (20 msgs max) → Compact if needed → Run router
    └── dispatch_agent →
        Agent history (16 msgs max) → Compact if needed → Run specialist
```

---

## 12. Streaming & Real-Time Status Updates

### The asyncio.Queue Pattern

Instead of using Pydantic AI's built-in `run_stream()`, this architecture uses `agent.run()` in a background task with an `asyncio.Queue` for status updates. This gives fine-grained control over what streams to the frontend.

```python
import asyncio

async def process_message_stream(user_id, message, chat_id):
    """Process a message with real-time status streaming."""

    # 1. Create status queue
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def emit_status(text: str) -> None:
        await status_queue.put(text)

    # 2. Create deps with the callback
    deps = RouterDeps(
        session=session,
        status_callback=emit_status,
        # ...
    )

    # 3. Run agent in background task
    agent_task = asyncio.create_task(
        router_agent.run(
            message,
            deps=deps,
            message_history=history,
            usage_limits=UsageLimits(request_limit=10),
        )
    )

    # 4. Stream status updates while agent works
    while not agent_task.done():
        try:
            status_text = await asyncio.wait_for(
                status_queue.get(), timeout=0.3
            )
            yield StreamChunk(type="tool_status", data={"text": status_text})
        except asyncio.TimeoutError:
            continue

    # 5. Drain remaining status messages
    while not status_queue.empty():
        try:
            status_text = status_queue.get_nowait()
            yield StreamChunk(type="tool_status", data={"text": status_text})
        except asyncio.QueueEmpty:
            break

    # 6. Check for errors
    if agent_task.exception():
        raise agent_task.exception()

    result = agent_task.result()

    # 7. Extract and stream tool calls/outputs from message history
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    yield StreamChunk(type="tool_call", data={
                        "id": part.tool_call_id,
                        "name": part.tool_name,
                        "input": part.args,
                    })
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    yield StreamChunk(type="tool_output", data={
                        "tool_call_id": part.tool_call_id,
                        "output": str(part.content)[:500],
                    })

    # 8. Stream the final HTML response
    html = extract_html(str(result.output))
    for i in range(0, len(html), 500):
        yield StreamChunk(type="html_chunk", data={"html_chunk": html[i:i+500]})
        await asyncio.sleep(0.01)  # Small delay for streaming effect

    yield StreamChunk(type="html_message", data={"html": html})
    yield StreamChunk(type="status", data={"status": "complete"})
```

### SSE Event Types

| Type | Payload | When |
|------|---------|------|
| `chat_info` | `{chat_id, title}` | Chat created/title updated |
| `tool_status` | `{text}` | Tool starts executing |
| `tool_call` | `{id, name, input}` | LLM invoked a tool |
| `tool_output` | `{tool_call_id, output}` | Tool returned a result |
| `html_chunk` | `{html_chunk}` | Incremental response HTML |
| `html_message` | `{html}` | Complete response HTML |
| `status` | `{status, chat_id}` | Final status ("complete" or "error") |
| `error` | `{message}` | Error occurred |

### Alternative: Pydantic AI Native Streaming

For simpler use cases, Pydantic AI has built-in streaming:

```python
# Text streaming
async with agent.run_stream("prompt", deps=deps) as stream:
    async for chunk in stream.stream_text(delta=True):
        yield chunk  # Incremental text deltas

# Event streaming
async for event in agent.run_stream_events("prompt", deps=deps):
    if isinstance(event, PartStartEvent):
        ...
    elif isinstance(event, PartDeltaEvent):
        ...
    elif isinstance(event, FunctionToolCallEvent):
        ...
    elif isinstance(event, FinalResultEvent):
        ...
```

---

## 13. Multi-Iteration Agent Loops

Pydantic AI agents automatically iterate: they call tools, receive results, and decide whether to call more tools or produce a final response. The LLM controls the loop.

### Controlling Iterations with UsageLimits

```python
from pydantic_ai import UsageLimits

result = await agent.run(
    query,
    deps=deps,
    usage_limits=UsageLimits(
        request_limit=15,           # Max LLM requests (tool iterations)
        total_tokens_limit=100000,  # Max total tokens
        tool_calls_limit=20,        # Max total tool invocations
    ),
)
```

### How the Iteration Loop Works

```
1. Agent receives user message
2. LLM generates response
   ├── If response contains tool calls:
   │   a. Execute all tool calls (potentially in parallel)
   │   b. Send tool results back to LLM
   │   c. Go to step 2
   └── If response is text only:
       └── Return final result
```

Each iteration = one LLM request. The `request_limit` caps the total number of iterations.

### Example: Agent Uses Multiple Tools Iteratively

```python
@agent.tool
async def rag_search(ctx: RunContext[Deps], query: str) -> str:
    """Semantic search for relevant documents."""
    ...

@agent.tool
async def retrieve_lines(ctx: RunContext[Deps], doc_id: int, start: int, end: int) -> str:
    """Read specific lines from a document for more context."""
    ...

@agent.tool
async def regex_search(ctx: RunContext[Deps], pattern: str) -> str:
    """Search for exact terms when semantic search misses."""
    ...
```

The LLM might:
1. Call `rag_search("vacation policy")` → gets chunk results
2. Call `retrieve_lines(42, 10, 30)` → reads more context around a match
3. Call `regex_search("PTO|paid time off")` → searches for exact terms
4. Finally produce a synthesized HTML response

All of this happens automatically — the LLM decides when to use tools and when to stop.

### Manual Iteration with `agent.iter()`

For fine-grained control over each step:

```python
async with agent.iter("prompt", deps=deps) as run:
    node = run.next_node
    while not isinstance(node, End):
        # Inspect or modify behavior between steps
        if isinstance(node, ToolCallNode):
            print(f"About to call: {node.tool_name}")
        node = await run.next(node)
    result = run.result
```

### Nested Multi-Iteration (Dispatch Pattern)

The dispatch pattern enables nested iteration: the router iterates (calling dispatch), and within each dispatch, the specialist iterates (calling its own tools).

```
Router iteration 1:
  └── dispatch_agent("policy", "What is the call schedule?")
      Policy Agent iteration 1: rag_search("call schedule")
      Policy Agent iteration 2: retrieve_lines(15, 1, 50)
      Policy Agent iteration 3: → produces final answer
  Router receives specialist response
Router iteration 2: → produces final HTML
```

---

## 14. Multimodal Tool Returns (Images, Binary)

### ToolReturn with Binary Content

When tools need to return images alongside text:

```python
from pydantic_ai import ToolReturn, BinaryContent

@tool_display("Searching through policies...")
@agent.tool
async def rag_search(ctx: RunContext[Deps], query: str) -> Union[str, ToolReturn]:
    """Search for policy chunks, including any embedded images."""
    results = await search_chunks(query)
    result_text = format_results(results)

    # Check for image placeholders in results
    image_ids = extract_image_ids(results)

    if image_ids:
        images = await fetch_images(ctx.deps.session, image_ids)
        content_parts: list[Union[str, BinaryContent]] = [result_text]
        for img in images:
            resized = resize_for_llm(img.data, img.mime_type)
            content_parts.append(f"\nImage {img.id} — {img.alt_text}:")
            content_parts.append(
                BinaryContent(data=resized, media_type=img.mime_type)
            )
        return ToolReturn(
            return_value=result_text,   # Text version for history
            content=content_parts,       # Rich content for LLM
        )

    return result_text  # Simple string when no images
```

### Image Presentation Tool

```python
@tool_display("Loading image...")
@agent.tool
async def present_image(
    ctx: RunContext[Deps],
    image_id: str,
    caption: str,
) -> ToolReturn:
    """Show a policy image to the user.

    Args:
        image_id: The UUID from the [IMAGE: id=...] placeholder.
        caption: A brief caption describing what the image shows.
    """
    await ctx.deps.status_callback(present_image._display_text)

    img = await fetch_image(ctx.deps.session, image_id)
    if not img:
        return ToolReturn(
            return_value="Image not found.",
            content=["Image not found. Do not include in response."],
        )

    html_snippet = (
        f'<figure class="policy-image">'
        f'<img src="/images/{image_id}" alt="{escape(caption)}">'
        f'<figcaption>{escape(caption)}</figcaption>'
        f'</figure>'
    )

    return ToolReturn(
        return_value=f"Image {image_id} ready for display.",
        content=[f"Include this EXACT HTML:\n{html_snippet}"],
    )
```

---

## 15. Configuration Patterns

### Environment-Driven Config

Store all agent configuration in environment variables with sensible defaults:

```python
import os
from types import SimpleNamespace

_config_dict = {
    "ROUTER": {
        "MODEL": os.environ.get("ROUTER_MODEL", "openai:gpt-4o-mini"),
        "TEMPERATURE": float(os.environ.get("ROUTER_TEMPERATURE", "0.3")),
        "MAX_TOKENS": int(os.environ.get("ROUTER_MAX_TOKENS", "2000")),
        "REASONING_EFFORT": os.environ.get("ROUTER_REASONING_EFFORT", ""),
        "MAX_HISTORY_MESSAGES": int(os.environ.get("ROUTER_MAX_HISTORY_MESSAGES", "20")),
        "REQUEST_LIMIT": int(os.environ.get("ROUTER_REQUEST_LIMIT", "10")),
    },
    "POLICY_AGENT": {
        "MODEL": os.environ.get("POLICY_AGENT_MODEL", "openai:gpt-5-mini"),
        "MAX_TOKENS": int(os.environ.get("POLICY_AGENT_MAX_TOKENS", "4000")),
        "REASONING_EFFORT": os.environ.get("POLICY_AGENT_REASONING_EFFORT", "low"),
        "MAX_HISTORY_MESSAGES": int(os.environ.get("POLICY_AGENT_MAX_HISTORY_MESSAGES", "16")),
        "REQUEST_LIMIT": int(os.environ.get("POLICY_AGENT_REQUEST_LIMIT", "15")),
    },
    "SUMMARIZER": {
        "MODEL": os.environ.get("SUMMARIZER_MODEL", "gpt-4o-mini"),
        "MAX_TOKENS": int(os.environ.get("SUMMARIZER_MAX_TOKENS", "1000")),
        "TEMPERATURE": float(os.environ.get("SUMMARIZER_TEMPERATURE", "0.3")),
    },
    "CONTEXT": {
        "MIN_RECENT_MESSAGES": int(os.environ.get("MIN_RECENT_MESSAGES", "6")),
    },
}

def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d

config = _dict_to_namespace(_config_dict)
# Access: config.ROUTER.MODEL, config.POLICY_AGENT.REASONING_EFFORT, etc.
```

### Configuration Best Practices

| Setting | Router | Specialist | Summarizer |
|---------|--------|------------|------------|
| Model | Small, fast (gpt-4o-mini) | Powerful, reasoning (gpt-5-mini) | Small, cheap (gpt-4o-mini) |
| Temperature | 0.3 (consistent routing) | 0.0 or reasoning_effort | 0.3 |
| Max tokens | 2000 (short responses) | 4000 (detailed responses) | 1000 (summaries) |
| History limit | 20 messages | 16 messages | N/A |
| Request limit | 10 (few tool calls) | 15 (many tool iterations) | N/A |

---

## 16. Testing & Development

### Test Models (No API Calls)

```python
from pydantic_ai.models.test import TestModel

# Fixed response
agent = Agent(TestModel(custom_result_text="test response"))
result = agent.run_sync("test")
assert result.output == "test response"
```

### Override for Testing

```python
with agent.override(deps=mock_deps):
    result = await agent.run("test prompt")
```

### Function Model (Custom Logic)

```python
from pydantic_ai.models.function import FunctionModel

def my_model(messages, info):
    # Custom response logic for testing
    return ModelResponse(parts=[TextPart(content="mocked")])

agent = Agent(FunctionModel(my_model))
```

### Debugging with capture_run_messages

```python
from pydantic_ai import capture_run_messages

with capture_run_messages() as messages:
    try:
        result = agent.run_sync(prompt)
    except Exception:
        # Inspect the full message exchange
        for msg in messages:
            print(msg)
```

---

## 17. Best Practices Summary

### Agent Architecture

1. **Use a router + specialist pattern** — fast routing model for dispatch, powerful models for domain work
2. **Self-register agents** — decouples the router from specialist implementations
3. **Independent histories per agent** — each specialist maintains its own context
4. **Environment-driven model selection** — switch providers without code changes

### Tool Design

5. **Use `@agent.tool` with `RunContext`** for tools needing dependencies; `@agent.tool_plain` for pure functions
6. **Write Google-style docstrings** — they become the LLM's tool documentation
7. **Use `@tool_display()` decorator** — self-documenting status text per tool
8. **Use `ModelRetry`** for recoverable errors — the LLM adjusts its approach
9. **Return `ToolReturn`** for multimodal content — separate text value from rich content

### History & Compaction

10. **Serialize with `to_jsonable_python()`**, deserialize with `ModelMessagesTypeAdapter.validate_python()`
11. **Split history at `ModelRequest` boundaries** — never break tool call/return pairs
12. **Use a cheap model for summarization** — gpt-4o-mini is fast and sufficient
13. **Compact at two levels** — router history and per-agent histories independently

### Streaming

14. **Use `asyncio.Queue` for status updates** — fine-grained control over what streams to frontend
15. **Run agents as background tasks** — `asyncio.create_task()` enables concurrent status streaming
16. **Emit status from tools** — `await ctx.deps.status_callback(tool._display_text)`

### Safety

17. **Set `UsageLimits`** on every agent run — prevent runaway tool calling and token usage
18. **Use `asyncio.wait_for(timeout=60)`** on specialist dispatch — prevent hanging
19. **Store full message history** — tool calls/returns are critical context for multi-turn conversations
20. **Provide text-only fallback** — backward compatibility for chats without stored JSON history

### Skills

21. **Organize skills as folders** — each skill gets its own directory with `instructions.md`
22. **Use YAML frontmatter** — structured metadata (name, description, triggers)
23. **Include supporting files** — reference docs, examples, helper scripts alongside `instructions.md`
24. **Inject skills into system prompts** — the LLM needs to know which skills are available

---

## 18. Pydantic AI API Reference Cheat Sheet

### Core Classes

```python
from pydantic_ai import Agent, RunContext, UsageLimits, Tool, ModelRetry, ToolReturn, BinaryContent
```

### Message Types

```python
from pydantic_ai.messages import (
    ModelMessage, ModelMessagesTypeAdapter,
    ModelRequest, ModelResponse,
    UserPromptPart, TextPart, ToolCallPart, ToolReturnPart,
)
```

### Serialization

```python
from pydantic_core import to_jsonable_python

# Serialize
json_data = to_jsonable_python(messages)

# Deserialize
messages = ModelMessagesTypeAdapter.validate_python(json_data)
```

### Model Settings

```python
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.settings import ModelSettings
```

### Agent Run Methods

```python
# Async run (most common)
result = await agent.run(prompt, deps=deps, message_history=history)

# Sync run (convenience)
result = agent.run_sync(prompt, deps=deps)

# Streaming
async with agent.run_stream(prompt, deps=deps) as stream:
    async for chunk in stream.stream_text(delta=True): ...

# Event streaming
async for event in agent.run_stream_events(prompt, deps=deps): ...

# Manual iteration
async with agent.iter(prompt, deps=deps) as run:
    node = run.next_node
    while not isinstance(node, End):
        node = await run.next(node)
```

### Result Properties

```python
result.output          # Final typed response
result.all_messages()  # Full history including this turn
result.new_messages()  # Only this turn's messages
result.usage()         # Token/request metrics
```

### Toolsets (Reusable Collections)

```python
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset()

@toolset.tool
async def search(ctx: RunContext[Deps], query: str) -> str: ...

agent = Agent("openai:gpt-5-mini", toolsets=[toolset])
```

### FallbackModel

```python
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel("openai:gpt-5-mini", "anthropic:claude-sonnet-4-6")
```

---

## License

This guide is based on Pydantic AI (MIT License) and production patterns from the YDRP.chat project.
