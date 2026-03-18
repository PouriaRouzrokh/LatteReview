# LatteReview Technical Snapshot

**Date:** 2026-03-18 (updated after RFD-7)
**Version:** 1.1.1 (v2 in development on `dev/v2` branch)
**Python:** >=3.12
**License:** CC BY-NC 4.0

## Overview

LatteReview is a literature review framework for LLM-powered document screening, scoring, and structured data abstraction. It runs multi-agent, multi-round review pipelines over DataFrames, with support for images, RAG via async callables, RIS file ingestion, and cost tracking. Published as arXiv:2501.05468.

## v2 Development Status

**Branch:** `dev/v2`
**Plan:** `/home/pouria/.claude/plans/binary-noodling-owl.md`

| RFD | Status | Tests |
|-----|--------|-------|
| RFD-1: Core AgenticReviewer | COMPLETE | 46 unit + 6 live |
| RFD-2: AgenticWorkflow | COMPLETE | 29 unit + 2 live |
| RFD-3: Skills System + Registry | COMPLETE | 43 unit + 1 live |
| RFD-4: Memory System | COMPLETE | 44 unit + 2 live |
| RFD-5: Flagging + Revisit | COMPLETE | 39 unit + 2 live |
| RFD-6: Helper Agents | COMPLETE | 30 unit + 2 live |
| RFD-7: Checkpoint/Resume | COMPLETE | 53 unit + 2 live |
| RFD-8: Search Skills Bundle | **NEXT** | — |
| RFD-9: Preset Reviewer Types | Planned | — |
| RFD-10: Backward Compatibility | Planned | — |
| RFD-11: Docs + Tutorials | Planned | — |

**Total tests:** 284 unit + 17 live = 301 (all passing)

### v2 Code Location
All v2 code lives under `lattereview/agentic/`. v1 code is untouched.

### What's Built (RFD-1 through RFD-6)

#### RFD-1: Core AgenticReviewer
- `lattereview/agentic/reviewer.py` — `AgenticReviewer` class wrapping Pydantic AI `Agent`
- `lattereview/agentic/deps.py` — `ReviewDeps` dataclass (dependency injection)
- `lattereview/agentic/prompts.py` — System prompt builder (iteration/effort/memory/skill aware)
- `lattereview/agentic/output_models.py` — `ScoringOutput`, `EvaluationOutput`, `AbstractionOutput`, `build_dynamic_output_model()`
- `lattereview/agentic/__init__.py` — Public exports

#### RFD-2: AgenticWorkflow
- `lattereview/agentic/workflow.py` — `AgenticWorkflow` + `AgenticWorkflowError`
- Same schema format as v1: `round`, `reviewers`, `text_inputs`, `filter`
- Same column naming: `round-{ROUND_ID}_{REVIEWER_NAME}_{KEYWORD}`
- Output column stores dict (not string) for v1 filter lambda compat
- `reviewer_costs` uses tuple keys `(round_id, reviewer_name)` matching v1
- Uses `print()` for verbose output matching v1

#### RFD-3: Skills System + Registry
- `lattereview/agentic/skills/base.py` — `SkillManifest`, `validate_skill_name()`
- `lattereview/agentic/skills/loader.py` — `parse_skill_md()`, `discover_skills()`, `load_toolset()`
- `lattereview/agentic/skills/registry.py` — `SkillRegistry` (discover, enable, disable, get toolsets/descriptions)
- `lattereview/agentic/skills/_meta_tool.py` — `get_skill_details` meta-tool for L3 progressive disclosure
- `lattereview/agentic/skills/builtin/searching-content/` — First builtin skill (regex_search, keyword_search)
- **Integration**: `AgenticReviewer._setup_skills()` auto-wires skills into `review_item()`; `review_items()` caches toolsets and passes them down to avoid per-item filesystem scanning
- **Key gotcha**: Do NOT use `from __future__ import annotations` in modules that define Pydantic AI tool functions — it breaks runtime type resolution for `RunContext[ReviewDeps]`

#### RFD-4: Memory System
- `lattereview/agentic/memory/index.py` — `MemoryIndex` (file-backed `_index.json` with asyncio.Lock)
- `lattereview/agentic/memory/store.py` — `MemoryStore` (wraps MemoryIndex + `mem_XXX.md` files, max_memories=50, asyncio.Lock on all read/write operations)
- `lattereview/agentic/memory/__init__.py` — Exports MemoryStore, MemoryIndex
- `lattereview/agentic/skills/builtin/managing-memory/` — Skill with 5 tools (save, load, list, load_multiple, delete)
- **Integration**: `review_item()` auto-creates MemoryStore from working_dir; `review_items()` shares store across items; `AgenticWorkflow` creates per-agent/round stores
- **Storage**: `working_dir/round_{ID}/agent_{NAME}/memory/{_index.json, mem_001.md, ...}`
- **System prompt**: Memory summaries (id + brief) injected into "# Your Memories" section

#### RFD-5: Flagging + Revisit System
- `lattereview/agentic/flags/store.py` — `FlagStore` (file-backed `flags.json` with asyncio.Lock on all read/write operations)
- `lattereview/agentic/flags/__init__.py` — Exports FlagStore
- `lattereview/agentic/skills/builtin/flagging-items/` — Skill with 3 tools (flag_for_revisit, list_flagged_items, resolve_current_flag)
- **Integration**: `review_item()` auto-creates FlagStore from working_dir; `review_items()` shares store across items; `AgenticWorkflow` creates per-agent/round stores
- **Revisit**: `AgenticWorkflow._revisit_flagged_items()` re-reviews flagged items sequentially after initial pass; auto-resolves flags on successful revisit output
- **DataFrame columns**: `round-{ID}_{NAME}_flagged` (bool) and `round-{ID}_{NAME}_flag_reason` (str) for unresolved flags
- **Storage**: `working_dir/round_{ID}/agent_{NAME}/flags/flags.json`
- **System prompt**: Flag summaries (item_id + reason) injected into "# Flagged Items" section

#### RFD-6: Helper Agent System
- `lattereview/agentic/helpers/manager.py` — `HelperAgentManager` (ordered helper delegation, priority-based selection, recursion safety)
- `lattereview/agentic/helpers/__init__.py` — Exports HelperAgentManager
- `lattereview/agentic/skills/builtin/discussing-with-helpers/` — Skill with 1 tool (discuss_with_helper)
- **Integration**: `review_item()` creates HelperAgentManager when helpers configured in agentic mode; passed via `ReviewDeps.helper_manager`
- **Safety**: Helpers never get discussing-with-helpers skill (filtered out in `_run_helper()`), max depth=1
- **Helper output**: Plain text (`output_type=str`), not structured — advisory responses only
- **Helper memory**: `working_dir/round_{ID}/helper_{NAME}/memory/` — separate from primary reviewer
- **Storage**: Helper memories persist across consultations within a round

#### RFD-7: Checkpoint/Resume System
- `lattereview/agentic/checkpoint/state.py` — `RunState` (serializable run progress) + `compute_schema_hash()`
- `lattereview/agentic/checkpoint/manager.py` — `CheckpointManager` (atomic per-item saves via write-to-temp-then-rename, DataFrame snapshots as parquet, resume detection with schema hash comparison)
- `lattereview/agentic/logging/action_log.py` — `ActionLogger` (per-item JSONL logging with timestamp, action_type, details)
- `lattereview/agentic/checkpoint/__init__.py` — Exports RunState, CheckpointManager, compute_schema_hash
- `lattereview/agentic/logging/__init__.py` — Exports ActionLogger
- **Integration**: `AgenticWorkflow` gains `resume: bool = False` param; when working_dir is set, creates CheckpointManager, saves per-item results atomically, saves DataFrame snapshots after each round, marks state completed/failed; on resume, loads prior state, warns if schema changed, skips completed items
- **Reviewer integration**: `review_item()` accepts `action_logger` and `checkpoint_manager` params; logs review_start/review_complete; saves per-item checkpoint after each review
- **Storage**: `run_metadata.json` at working_dir root, `round_{ID}/agent_{NAME}/results/item_{ID}.json` per item, `round_{ID}/agent_{NAME}/logs/item_{ID}.jsonl` per item, `output/after_round_{ID}.parquet` snapshots, `output/final.parquet`

### Empty Directory Structure (ready for future RFDs)
```
lattereview/agentic/
└── skills/builtin/{searching-google,searching-duckduckgo,searching-pubmed,
                    searching-semantic-scholar,searching-arxiv}/  (empty, awaiting RFD-8)
```

---

## v2 Critical Implementation Notes

### Provider Model Strings — STRING-ONLY SWITCHING
Users switch providers by changing **one string**. Pydantic AI natively handles all major providers:

| Provider | String format | Env var |
|---|---|---|
| OpenAI | `openai:gpt-5.4-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google-gla:gemini-3.1-flash-lite-preview` | `GEMINI_API_KEY` |
| OpenRouter | `openrouter:google/gemini-2.5-flash` | `OPENROUTER_API_KEY` |
| Groq | `groq:llama-3.3-70b` | `GROQ_API_KEY` |
| Together | `together:meta-llama/...` | `TOGETHER_API_KEY` |
| Fireworks | `fireworks:accounts/fireworks/...` | `FIREWORKS_API_KEY` |
| Ollama | `ollama:llama3.2` | (none) |

**DO NOT** build custom provider wrappers. Check Pydantic AI docs first: https://ai.pydantic.dev/models/

### Pydantic AI Reference Docs
Always check these before implementing features:
- **Providers:** https://ai.pydantic.dev/api/providers/ (incl. `GatewayProvider` for proxies)
- **Models:** https://ai.pydantic.dev/models/ (OpenRouter, Groq, Together, Fireworks, etc.)
- **OpenRouter specifics:** https://ai.pydantic.dev/models/openrouter/
- **Testing:** Use `TestModel` / `FunctionModel` from `pydantic_ai.models.test`
- **Tools/Toolsets:** `FunctionToolset` from `pydantic_ai.toolsets`
- **Usage limits:** `UsageLimits` from `pydantic_ai.usage`
- **Message serialization:** `to_jsonable_python()` / `ModelMessagesTypeAdapter.validate_python()`
- **Local guide:** `references/pydantic-ai-guide.md` (Yale chatbot patterns)

### Model Choices for Tests
- **OpenAI:** `openai:gpt-5.4-mini` (latest non-pro, fast, cheap)
- **Anthropic:** `anthropic:claude-haiku-4-5-20251001` (may get transient 529 overloaded)
- **Gemini:** `google-gla:gemini-3.1-flash-lite-preview` (verify with `client.models.list()`)
- **OpenRouter:** `openrouter:google/gemini-2.5-flash`
- **Unit tests:** Always use `TestModel()` from `pydantic_ai.models.test`

### Testing Pattern
- All tests in `tests/agentic/`
- `uv run pytest tests/agentic/ -m "not live" -v` — unit tests (no API keys)
- `uv run pytest tests/agentic/ -m "live" -v` — live tests (API keys from `.env`)
- Live tests: `@pytest.mark.live` + `@pytest.mark.asyncio`, skip if key missing
- `asyncio_mode = "auto"` configured in `pyproject.toml`
- Conftest at `tests/conftest.py` with `env_keys` fixture

### Dependencies
- `pydantic-ai>=1.69.0` (core v2 dependency)
- `pyyaml>=6.0` (skill YAML frontmatter parsing)
- Dev: `pytest>=8.0.0`, `pytest-asyncio>=0.24.0` (install with `uv sync --extra dev`)

### Known Gotchas
- **`from __future__ import annotations`**: Do NOT use in modules that define Pydantic AI tool functions. It turns type hints into strings, breaking `get_type_hints()` which Pydantic AI uses to resolve `RunContext[ReviewDeps]`. The `_meta_tool.py` was created specifically to avoid this issue.
- **Transient API errors**: OpenAI may return connection errors. Live tests may need a retry — this is not a code bug.

---

## v1 Architecture (unchanged)

### Module Structure
```
lattereview/
├── agents/          # v1 reviewer classes (BasicReviewer, ScoringReviewer, etc.)
├── providers/       # v1 LLM provider abstractions (OpenAI, Google, LiteLLM, Ollama)
├── workflows/       # v1 multi-round pipeline orchestration (ReviewWorkflow)
└── utils/           # Data handlers (RIS parser)
```

### Agent Hierarchy
```
BasicReviewer (base, pydantic.BaseModel)
├── ScoringReviewer    — response_format: {reasoning, score, certainty}
├── TitleAbstractReviewer — response_format: {reasoning, evaluation}
└── AbstractionReviewer   — response_format: dynamic from abstraction_keys
```

### Workflow Column Naming
`round-{ROUND_ID}_{REVIEWER_NAME}_{KEYWORD}` (e.g., `round-A_ScoringReviewer_score`)

### Data Flow
```
Input (DataFrame / CSV / XLSX / RIS)
    → ReviewWorkflow.run() → iterates rounds
    → filter(row) → eligible rows
    → _format_text_input() → "=== col ===\nvalue" blocks
    → BasicReviewer.review_items() → async batch with semaphore
    → Provider.get_json_response() → LLM API
    → _safe_parse_output() → DataFrame columns
```

### v1 Dependencies
| Category | Packages |
|---|---|
| LLM routing | `litellm>=1.55.2`, `openai>=1.57.4`, `google-genai>=1.15.0`, `ollama>=0.4.4` |
| Data | `pandas>=2.2.2`, `openpyxl>=3.1.5` |
| Image | `pillow>=11.0.0`, `opencv-python>=4.10.0.84` |
| Validation | `pydantic>=2.10.3` |
| Cost tracking | `tokencost>=0.1.17` |
