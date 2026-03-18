# LatteReview

[![PyPI version](https://badge.fury.io/py/lattereview.svg)](https://badge.fury.io/py/lattereview)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![View on arXiv](https://img.shields.io/badge/arXiv-View%20Paper-orange)](https://arxiv.org/abs/2501.05468)
[![Sponsor me on GitHub](https://img.shields.io/badge/Sponsor%20me-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/PouriaRouzrokh)
[![Support me on Ko-fi](https://img.shields.io/badge/Support%20me-Ko--fi-orange.svg?logo=ko-fi&logoColor=white)](http://ko-fi.com/pouriarouzrokh)
[![Website](https://img.shields.io/badge/Website-pouriarouzrokh.com-blue.svg)](https://pouriarouzrokh.com)

<p><img src="docs/images/robot.png" width="400"></p>

**LatteReview** is an agentic literature review framework for LLM-powered document screening, scoring, and abstraction. Built on [Pydantic AI](https://ai.pydantic.dev/), it gives your AI reviewers agentic reasoning loops, built-in skills, persistent memory, helper agents, and checkpoint/resume -- so reviewing hundreds of papers is as smooth as enjoying a cup of latte.

---

## What's New in v2

v2 transforms reviewers from one-shot LLM calls into **agentic entities** that think, search, remember, consult experts, and self-correct:

- **Agentic reasoning loops** -- reviewers iterate up to `max_iterations` times, using tools and refining their reasoning before committing a final answer. Control effort with `agentic_effort` (low/medium/high).
- **9 built-in skills** -- memory management, web search (DuckDuckGo, Google, PubMed, Semantic Scholar, arXiv), content search, item flagging, and helper-agent discussion. Enable any combination with `skills=[...]`.
- **Persistent memory** -- reviewers accumulate insights across items in a batch, learning patterns and building expertise as they review.
- **Helper agents** -- attach specialist sub-agents that the primary reviewer can consult mid-review for second opinions or domain expertise.
- **Checkpoint/resume** -- atomic per-item saves let you stop and restart long workflows without losing progress. Set `working_dir` and `resume=True`.
- **Preset reviewer types** -- `ScoringReviewer`, `TitleAbstractReviewer`, and `AbstractionReviewer` work out of the box with sensible defaults.
- **Multi-provider via model strings** -- switch between any LLM with a single string: `"openai:gpt-5.4-mini"`, `"anthropic:claude-sonnet-4-6"`, `"google-gla:gemini-3-flash-preview"`, and more.
- **Custom skills** -- create your own skill folders with a `SKILL.md` manifest and `tools.py`, and pass them via `custom_skill_paths=[...]`.

---

## Installation

```bash
pip install lattereview
```

**Extras:**

```bash
# Search skills (DuckDuckGo, Google, PubMed, Semantic Scholar, arXiv)
pip install lattereview[search]

# Everything (all agentic extras)
pip install lattereview[agentic-all]
```

Requires **Python >= 3.12**.

---

## Quick Start (v2 Agentic)

### 1. Score a single item

The simplest use case: a `ScoringReviewer` evaluates one item. With `max_iterations=1`, this behaves like v1 (a single LLM call). Increase `max_iterations` to enable the agentic loop.

```python
from lattereview.agentic import ScoringReviewer
import asyncio

reviewer = ScoringReviewer(
    model="openai:gpt-5.4-mini",
    name="Scorer",
    scoring_task="Rate the relevance of this article to AI in healthcare",
    scoring_set=[1, 2, 3, 4, 5],
    max_iterations=1,  # Single LLM call (non-agentic, like v1)
)
result, cost = asyncio.run(reviewer.review_item("A study on deep learning for chest X-ray diagnosis..."))
print(result)  # {"reasoning": "...", "score": 4, "certainty": 85}
```

### 2. Agentic review with skills and memory

This is where v2 shines. Enable skills so the reviewer can search the literature, build memory across items, and flag uncertain cases for human review:

```python
from lattereview.agentic import ScoringReviewer
import asyncio

reviewer = ScoringReviewer(
    model="anthropic:claude-sonnet-4-6",
    name="Analyst",
    scoring_task="Rate the methodological rigor of this study",
    scoring_set=[1, 2, 3, 4, 5],
    scoring_rules="1=anecdotal, 2=weak methodology, 3=adequate, 4=strong, 5=gold standard",
    # -- v2 agentic capabilities --
    max_iterations=20,         # Allow up to 20 reasoning steps
    agentic_effort="high",     # Encourage thorough tool use
    skills=[
        "searching-pubmed",        # Search PubMed to verify claims
        "searching-duckduckgo",    # General web search for context
        "managing-memory",         # Remember patterns across items
        "flagging-items",          # Flag uncertain items for human review
    ],
)

# The reviewer will search, reason, build memory, and may flag uncertain items
result, cost = asyncio.run(reviewer.review_item(
    "A novel GNN architecture predicts drug interactions with F1=0.94 on DrugBank, "
    "outperforming all existing methods by 8%."
))
print(f"Score: {result['score']}, Cost: ${cost:.4f}")
```

### 3. Multi-agent workflow with helper agents

Attach expert helper agents that the primary reviewer can consult mid-review. Run everything through `AgenticWorkflow` with checkpoint/resume:

```python
from lattereview.agentic import TitleAbstractReviewer, ScoringReviewer, AgenticWorkflow
from pathlib import Path
import pandas as pd
import asyncio

# Expert helper -- a specialist the main reviewer can consult
expert = ScoringReviewer(
    model="anthropic:claude-sonnet-4-6",
    name="MethodsExpert",
    backstory="You are a biostatistician who evaluates study methodology.",
    scoring_task="Assess the statistical methodology",
    scoring_set=[1, 2, 3, 4, 5],
    max_iterations=5,
)

# Primary reviewer with helper access, memory, and search
screener = TitleAbstractReviewer(
    model="openai:gpt-5.4-mini",
    name="Screener",
    inclusion_criteria="Studies applying AI/ML to diagnostic medical imaging",
    exclusion_criteria="Non-English, conference abstracts only, non-clinical",
    max_iterations=15,
    agentic_effort="medium",
    skills=["searching-pubmed", "managing-memory", "discussing-with-helpers"],
    helpers=[expert],
    helper_max_iterations=5,
)

# Multi-round workflow with checkpoint/resume
workflow = AgenticWorkflow(
    workflow_schema=[{
        "round": "A",
        "reviewers": [screener],
        "text_inputs": ["title", "abstract"],
    }],
    working_dir=Path("./review_output"),  # Saves progress here
    verbose=True,
)

data = pd.read_csv("articles.csv")
results = asyncio.run(workflow(data))

# If interrupted, just re-run with resume=True to continue where you left off:
# workflow = AgenticWorkflow(..., working_dir=Path("./review_output"), resume=True)
```

### 4. Custom skills

Create your own skills by adding a folder with `SKILL.md` (manifest) and `tools.py` (tool functions):

```python
from lattereview.agentic import ScoringReviewer
from pathlib import Path

reviewer = ScoringReviewer(
    model="google-gla:gemini-3-flash-preview",
    name="Reviewer",
    scoring_task="Rate this article",
    custom_skill_paths=[Path("./my_skills/clinical_lookup")],
    skills=["clinical-lookup"],  # Name from your SKILL.md
)
```

---

## Quick Start (v1 Classic)

The original v1 API is still available. v1 modules emit deprecation warnings; see the [migration guide](docs/migration.md) for upgrade instructions.

```python
from lattereview.providers import LiteLLMProvider
from lattereview.agents import TitleAbstractReviewer
from lattereview.workflows import ReviewWorkflow
import asyncio

reviewer = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-5.4-mini"),
    name="Alice",
    inclusion_criteria="AI in radiology",
)

workflow = ReviewWorkflow(
    workflow_schema=[{
        "round": "A",
        "reviewers": [reviewer],
        "text_inputs": ["title", "abstract"],
    }]
)

results = asyncio.run(workflow("articles.xlsx"))
```

---

## Key Features

| Feature | v1 | v2 |
|---------|----|----|
| Structured LLM output | Single call | Agentic loop (up to N iterations) |
| Tool use (search, memory) | -- | 9 built-in skills + custom skills |
| Cross-item learning | -- | Persistent memory system |
| Expert consultation | -- | Helper agents with priority delegation |
| Crash recovery | -- | Checkpoint/resume with atomic saves |
| Model configuration | Provider wrapper classes | One-line model strings |
| Action logging | -- | Per-item JSONL action logs |
| Item flagging | -- | Flag uncertain items for human review |

**Built-in skills:** `managing-memory`, `searching-duckduckgo`, `searching-google`, `searching-pubmed`, `searching-semantic-scholar`, `searching-arxiv`, `searching-content`, `flagging-items`, `discussing-with-helpers`.

---

## Supported Providers

| Provider | Model String | Example |
|----------|-------------|---------|
| OpenAI | `openai:model-name` | `openai:gpt-5.4-mini` |
| Anthropic | `anthropic:model-name` | `anthropic:claude-sonnet-4-6` |
| Google Gemini | `google-gla:model-name` | `google-gla:gemini-3-flash-preview` |
| OpenRouter | `openrouter:provider/model` | `openrouter:google/gemini-3-flash-preview` |
| Groq | `groq:model-name` | `groq:llama-4-scout-17b-16e-instruct` |
| Ollama | `ollama:model-name` | `ollama:llama3.2` |

Any provider supported by [Pydantic AI](https://ai.pydantic.dev/) works with LatteReview.

---

## Documentation and Tutorials

- **Documentation site**: [https://pouriarouzrokh.github.io/LatteReview](https://pouriarouzrokh.github.io/LatteReview)
- **v2 Agentic tutorials**: [`tutorials_agentic/`](tutorials_agentic/) -- scoring, screening, abstraction, skills, checkpoint/resume
- **v1 Classic tutorials**: [`tutorials/`](tutorials/)
- **Migration guide**: [`docs/migration.md`](docs/migration.md)

---

## Citation

If you use LatteReview in your research, please cite our paper:

```bibtex
@misc{rouzrokh2025lattereview,
    title={LatteReview: A Multi-Agent Framework for Systematic Review Automation Using Large Language Models},
    author={Pouria Rouzrokh and Moein Shariatnia},
    year={2025},
    eprint={2501.05468},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

---

## Author

<table border="0">
<tr>
<td style="width: 80px;">
<img src="https://github.com/PouriaRouzrokh.png?size=80" alt="Pouria Rouzrokh" style="border-radius: 50%;" />
</td>
<td>
<strong>Pouria Rouzrokh, MD, MPH, MHPE</strong><br>
Medical Practitioner and Machine Learning Engineer<br>
Incoming Radiology Resident @Yale University<br>
Former Data Scientist @Mayo Clinic AI Lab<br>
<a href="https://x.com/prouzrokh">
  <img src="https://img.shields.io/twitter/follow/prouzrokh?style=social" alt="Twitter Follow" />
</a>
<a href="https://linkedin.com/in/pouria-rouzrokh">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-blue" alt="LinkedIn" />
</a>
<a href="https://scholar.google.com/citations?user=Ksv9I0sAAAAJ&hl=en">
  <img src="https://img.shields.io/badge/Google%20Scholar-Profile-green" alt="Google Scholar" />
</a>
<a href="https://github.com/PouriaRouzrokh">
  <img src="https://img.shields.io/badge/GitHub-Profile-black" alt="GitHub" />
</a>
<a href="mailto:po.rouzrokh@gmail.com">
  <img src="https://img.shields.io/badge/Email-Contact-red" alt="Email" />
</a>
</td>
</tr>
</table>

---

## Support

If you find LatteReview helpful in your research or work, consider supporting its continued development:

- [Become a sponsor](https://github.com/sponsors/PouriaRouzrokh) on GitHub
- [Support me on Ko-fi](http://ko-fi.com/pouriarouzrokh)
- [Star the repository](https://github.com/PouriaRouzrokh/LatteReview) to help others discover the project

---

## Acknowledgement

Heartfelt gratitude to [Moein Shariatnia](https://github.com/moein-shariatnia) for his invaluable support and contributions to this project.

---

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/).
