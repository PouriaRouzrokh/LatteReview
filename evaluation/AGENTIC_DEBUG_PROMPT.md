# Agentic Reviewer Debug & Optimization Task

## Context

We just ran a full evaluation comparing v1 (old models), v2 non-agentic (latest models, `max_iterations=1`), and v2 agentic (latest models, `max_iterations=15`, skills enabled) on 978 articles across 3 search criteria of increasing difficulty.

**Expected**: Agentic reviewers should outperform non-agentic because they can search the web, build memory across items, flag uncertain cases, and search within text.

**Actual**: Agentic reviewers performed **worse** than non-agentic on all 3 tasks. Non-agentic with updated models beat v1, so the models themselves are fine — it's the agentic machinery that's hurting performance.

## Observed Problems

### Problem 1: GPT-5.4-mini ignores tools entirely

Agent2 (`openai:gpt-5.4-mini`) does **exactly 3 iterations with 2 tool calls on 96% of items** (938/978). It appears to be calling `get_skill_details` once and then immediately producing output. It never calls `duckduckgo_search`, never saves memory, never flags items. The entire agentic capability is wasted.

Meanwhile, Agent1 (`google-gla:gemini-3.1-flash-lite-preview`) does use tools — some items get 5-16 iterations with actual search calls. But even Gemini's tool usage is inconsistent.

**Questions to investigate**:
- Is the system prompt too long/complex, causing GPT to skip to output?
- Is `get_skill_details` the meta-tool? If so, why is the model calling it instead of the actual tools?
- Look at `lattereview/agentic/skills/_meta_tool.py` — is this L3 progressive disclosure meta-tool being presented as a required first step? Should it be optional?
- Are the actual tool functions (from `searching-duckduckgo/tools.py`, `managing-memory/tools.py`, etc.) being registered on the agent, or only the meta-tool?

### Problem 2: System prompt overload

The system prompt includes:
1. Agent identity (name, backstory)
2. System instructions
3. Iteration/effort guidance
4. Enabled skill descriptions (L1 metadata for all 4 skills)
5. Memory summaries (grows as memories accumulate)
6. Flag summaries
7. Output schema instructions

**Questions to investigate**:
- Read `lattereview/agentic/prompts.py` — how big does the system prompt get? Print an actual system prompt for an agentic reviewer with 4 skills, 10 memories, and 2 flags.
- Is the system prompt growing too large, especially after 50 memories accumulate?
- Are we injecting both skill descriptions AND tool schemas? Pydantic AI already injects tool schemas into the LLM call — so if we ALSO put skill descriptions in the system prompt, the model sees tool info twice (once in our prompt, once from Pydantic AI). This could be confusing.
- Should we remove skill descriptions from the system prompt entirely and rely on Pydantic AI's native tool description injection?

### Problem 3: Memory content quality varies

- For search1 (easy), Gemini saved **item-specific memories** like "Study A-293 uses CT for GGN analysis and doesn't mention PET." These are useless for future items — they're just restating what's in the abstract.
- For search2/search3 (harder), Gemini saved **useful patterns** like "Radiomics is not deep learning" and "Prognosis is not diagnosis."
- GPT-5.4-mini **never saves memories at all** across any task.

**Questions to investigate**:
- Read `lattereview/agentic/skills/builtin/managing-memory/SKILL.md` — does it clearly instruct the agent to save generalizable patterns, not item-specific facts?
- Is the memory tool description clear enough about what TO save vs NOT save?
- Is the max_memories=50 cap causing issues? After hitting 50, does the system prompt with 50 memory summaries bloat the context?

### Problem 4: Flagging barely fires

Only 1 item flagged per agent per task. For search3, where "external validation" info is genuinely absent from most abstracts, we expected many more flags. The flags that were raised are correct and well-reasoned — there just aren't enough of them.

**Questions to investigate**:
- Read `lattereview/agentic/skills/builtin/flagging-items/SKILL.md` — is the guidance too conservative about when to flag?
- Is `agentic_effort="high"` actually reflected in the prompts? Read the prompt builder to see what "high" does.

### Problem 5: The `get_skill_details` meta-tool may be a bottleneck

The meta-tool exists for L3 progressive disclosure — agents call it to get detailed SKILL.md content when they need to understand how to use a skill. But if the model calls it reflexively on every item, it wastes an iteration and potentially confuses the model with a wall of text.

**Questions to investigate**:
- Read `lattereview/agentic/skills/_meta_tool.py` — what does this tool return?
- Is it returning the full SKILL.md body every time?
- Should we preload tool instructions into the system prompt instead of requiring a meta-tool call?
- Or should we make the meta-tool description clearer: "Only call this if you need detailed instructions for a skill you haven't used before"?

## Key Files to Read

1. `lattereview/agentic/prompts.py` — system prompt construction (this is the most important file)
2. `lattereview/agentic/reviewer.py` — how skills/toolsets are registered on the Pydantic AI Agent
3. `lattereview/agentic/skills/_meta_tool.py` — the L3 progressive disclosure meta-tool
4. `lattereview/agentic/skills/registry.py` — how skills are discovered, enabled, and their toolsets returned
5. `lattereview/agentic/skills/loader.py` — how SKILL.md is parsed and toolsets loaded
6. `lattereview/agentic/skills/builtin/managing-memory/SKILL.md` and `tools.py` — memory skill
7. `lattereview/agentic/skills/builtin/flagging-items/SKILL.md` and `tools.py` — flagging skill
8. `lattereview/agentic/skills/builtin/searching-duckduckgo/SKILL.md` and `tools.py` — search skill
9. `lattereview/agentic/skills/builtin/searching-content/SKILL.md` and `tools.py` — content search skill

## What I Want

1. **Root cause analysis**: Why does GPT-5.4-mini ignore tools? Is it the prompt, the tool registration, or the meta-tool?
2. **Prompt audit**: Print the full system prompt for an agentic reviewer and assess whether it's too long, redundant, or poorly structured.
3. **Concrete fixes**: Propose specific code changes to:
   - Make models actually use tools (especially search and memory)
   - Reduce system prompt bloat
   - Fix the meta-tool to not waste iterations
   - Improve memory quality guidance
   - Make flagging more aggressive on genuinely uncertain items
4. **Implement the fixes** in the code (edit `prompts.py`, `_meta_tool.py`, SKILL.md files, etc.)
5. **Test**: After fixes, run a quick test with 10 items to verify the model actually calls tools.

## Important Notes

- The v2 agentic code is at `lattereview/agentic/`
- Use `uv run pytest tests/agentic/ -m "not live" -v` to verify nothing breaks
- The project uses Pydantic AI (`pydantic-ai>=1.69.0`). Check `references/pydantic-ai-guide.md` for patterns.
- Do NOT use `from __future__ import annotations` in tool modules — it breaks Pydantic AI's type resolution.
- Read `.claude/checkpoints/checkpoint-1/snapshot.md` for the full technical state.
