# LatteReview 🤖☕

[![PyPI version](https://badge.fury.io/py/lattereview.svg)](https://badge.fury.io/py/lattereview)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintained: yes](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/prouzrokh/lattereview)
[![View on arXiv](https://img.shields.io/badge/arXiv-View%20Paper-orange)](https://arxiv.org/abs/2501.05468)
[![Sponsor me on GitHub](https://img.shields.io/badge/Sponsor%20me-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/PouriaRouzrokh)
[![Support me on Ko-fi](https://img.shields.io/badge/Support%20me-Ko--fi-orange.svg?logo=ko-fi&logoColor=white)](http://ko-fi.com/pouriarouzrokh)

<p><img src="docs/images/robot.png" width="400"></p>

---

🚨 **NEW in v1.4.0**: Decision reviewers now also run on **Perplexity's pplx-decider** and **OpenAI's gpt-6-luna** through their new Decisions APIs, next to TypeSafe's **Jev**. In our evaluation on 11,793 articles, pplx-decider ranked articles best, for about $0.05 per 1,000 abstracts. Perplexity's Sonar works as an LLM reviewer too. See [What's New](#-whats-new-in-v140) and [Decision models vs LLM reviewers](#-decision-models).

---

LatteReview is a powerful Python package designed to automate academic literature review processes through AI-powered agents. Just like enjoying a cup of latte ☕, reviewing numerous research articles should be a pleasant, efficient experience that doesn't consume your entire day!

## 🆕 What's New in v1.4.0

- **Perplexity's Decisions API**: decision reviewers run on Perplexity's **pplx-decider** with `SystemOneProvider(backend="perplexity")` (`PERPLEXITY_API_KEY`), or through OpenRouter with `SystemOneProvider(backend="openrouter", model="perplexity/pplx-decider-v1.1-27b")`.
- **OpenAI's Decisions API**: `SystemOneProvider(backend="openai")` runs **gpt-6-luna** through OpenAI's new Decisions API. The provider translates OpenAI's request format, so every decision reviewer works unchanged. Questions that OpenAI declines to answer come back as None instead of stopping the run.
- **Three decision models compared on 11,793 articles**: pplx-decider ranked best (mean AUC 0.895, best on 7 of 9 datasets, about $0.05 per 1,000 abstracts), then Jev (0.878, $0.06) and the v1 LLM reviewers (0.828). gpt-6-luna's Decisions API scored 0.779 ($0.16): on long, multi-part criteria it returned 0.00 for almost every article. See the [comparison](https://pouriarouzrokh.github.io/LatteReview/decision_models/#comparing-jev-pplx-decider-and-gpt-6-luna).
- **Perplexity's models as LLM reviewers**: `LiteLLMProvider(model="perplexity/sonar")` reaches Sonar, and other vendors' models such as `perplexity/openai/gpt-6-luna`, through Perplexity's new Agent API (Perplexity no longer serves Sonar as chat completions). About $0.20 per 1,000 abstracts with Sonar, without web search.
- **Correct costs for per-request fees**: `LiteLLMProvider` now uses the cost the API reports (OpenRouter and Perplexity), so fees such as Sonar's search fee are counted. Before, a Sonar request through OpenRouter was recorded at about 1/18 of its price.
- **Same input, same answer**: decision reviewers leave the workflow's `Review Task ID` line out of the model's input. With it, a borderline article's answer could depend on its row number.

A decision-model + LLM pipeline in a few lines: Perplexity's pplx-decider screens every article, and only the articles it is unsure about go to an LLM (set `PERPLEXITY_API_KEY` and `OPENAI_API_KEY`):

```python
import asyncio
import pandas as pd
from lattereview.providers import SystemOneProvider, OpenAIProvider
from lattereview.agents import DecisionTitleAbstractReviewer, TitleAbstractReviewer
from lattereview.workflows import ReviewWorkflow

inclusion = {1: "The study must involve CT scans.", 2: "The study must use deep learning."}
exclusion = {1: "The study must not include PET scans."}

decider = DecisionTitleAbstractReviewer(
    provider=SystemOneProvider(backend="perplexity"),  # or SystemOneProvider() for TypeSafe's Jev, backend="openai"
    name="Decider", inclusion_criteria=inclusion, exclusion_criteria=exclusion,
)
llm = TitleAbstractReviewer(
    provider=OpenAIProvider(model="gpt-6-luna"),
    name="LLM", inclusion_criteria=str(inclusion), exclusion_criteria=str(exclusion),
)

workflow = ReviewWorkflow(workflow_schema=[
    {"round": "A", "reviewers": [decider], "text_inputs": ["title", "abstract"]},
    {"round": "B", "reviewers": [llm], "text_inputs": ["title", "abstract"],
     "filter": lambda row: pd.isna(p := row["round-A_Decider_include_probability"]) or 0.1 <= p < 0.9},  # uncertain or unanswered
])
results = asyncio.run(workflow(pd.read_csv("articles.csv")))  # columns: title, abstract
```

Try it in the notebooks:
[Screening with Jev, pplx-decider and gpt-6-luna](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/decision_review_jev/decision_review_jev.ipynb) ·
[Hybrid Jev + LLM review with measurements](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/hybrid_review_jev_llm/hybrid_review_jev_llm.ipynb) ·
[Decision Models docs](https://pouriarouzrokh.github.io/LatteReview/decision_models/).

Existing reviewers and defaults work as before. See the [CHANGELOG](./CHANGELOG.md) for the full list.

## What Was New in v1.3.0

- **Decision reviewers with Jev**: LatteReview can now review with System One decision models such as TypeSafe's [Jev](https://pouriarouzrokh.github.io/LatteReview/decision_models/), which answer typed questions with probabilities instead of generating text. `DecisionTitleAbstractReviewer`, `DecisionScoringReviewer` and the generic `DecisionReviewer` work in any `ReviewWorkflow`, next to LLM reviewers.
- **One provider, any backend**: `SystemOneProvider` works with TypeSafe and OpenRouter, and with any other `/v1/systemone` server via `base_url`, including a self-hosted OpenJev model on your own machine.
- **Evaluated at full scale**: on all 11,793 articles of LatteReview's evaluation datasets, Jev ranked articles better than the v1 LLM reviewers on every dataset (mean AUC 0.88 vs 0.83), for about $0.06 per 1,000 articles. See the [evaluation](https://pouriarouzrokh.github.io/LatteReview/decision_models/#evaluation), including where Jev falls short.
- **Thresholds for a target recall**: `suggest_threshold` fits a probability cutoff on labeled data.
- **Hybrid workflows**: let Jev screen everything and send only uncertain articles to an LLM ([tutorial](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/hybrid_review_jev_llm/hybrid_review_jev_llm.ipynb)).

## What Was New in v1.2.0

- **Current models**: tested with OpenAI GPT-6 (`gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`) and GPT-5.x, Anthropic Claude Opus 5.5, Sonnet 5, Haiku 4.5 and Fable 5.1, and Google Gemini 3.x (`gemini-3.8-flash`, `gemini-3.5-flash-lite`). Older models such as `gpt-4o-mini` and `gemini-2.5-flash` keep working.
- **No more rejected-parameter errors**: if a model rejects a setting in `model_args` (e.g., `temperature` on GPT-6 or Claude 5, or `max_tokens` on OpenAI reasoning models), LatteReview drops or renames it with a one-time warning and retries. If a reasoning model runs out of tokens before finishing its answer, the call is retried without the limit.
- **New default models**: `OpenAIProvider` and `LiteLLMProvider` default to `gpt-6-luna`, `GoogleProvider` to `gemini-3.8-flash`, and `OllamaProvider` to `qwen3.8:27b`. Pass `model=` to choose another.
- **Better local models**: `OllamaProvider` constrains answers to the reviewer's JSON schema, maps `reasoning_effort` to Ollama's `think` setting, passes other `model_args` (e.g., `top_p`) as model options instead of failing, and `close()` works again. Tested with `qwen3.8:27b` on a 32 GB Apple Silicon Mac.
- **More accurate costs**: computed from the token usage each API reports, including hidden reasoning tokens.
- **Python 3.10 or later** is now required. On Python 3.9, `pip` installs 1.1.1.

## 🎯 Key Features

- Multi-agent review system with customizable roles and expertise levels for each reviewer
- Support for multiple review rounds with hierarchical decision-making workflows
- Review diverse content types including article titles, abstracts, custom texts, and even **images** using LLM-powered reviewer agents
- Define reviewer agents with specialized backgrounds and distinct evaluation capabilities (e.g., scoring or concept abstraction or custom reviewers of your own preferance)
- Create flexible review workflows where multiple agents operate in parallel or sequential arrangements
- Enable reviewer agents to analyze peer feedback, cast votes, and propose corrections to other reviewers' assessments
- Enhance reviews with item-specific context integration, supporting use cases like **Retrieval Augmented Generation (RAG)**
- Broad compatibility with LLM providers through LiteLLM, including OpenAI and Ollama
- Model-agnostic integration supporting OpenAI, Gemini, Claude, Groq, DeepSeek, Perplexity, OpenRouter, and local models via Ollama
- **NEW**: Decision-model reviewers (Perplexity's pplx-decider, TypeSafe's Jev, OpenAI's gpt-6-luna, or a self-hosted OpenJev) that return probabilities for fast, cheap screening
- High-performance asynchronous processing for efficient batch reviews
- Standardized output format featuring detailed scoring metrics and reasoning transparency
- Robust cost tracking and memory management systems
- Extensible architecture supporting custom review workflow implementation
- **NEW**: Support for RIS (Research Information Systems) file format for academic literature review

## 💾Installation

```bash
pip install lattereview
```

LatteReview requires Python 3.10 or later. Please refer to the [installation guide](./docs/installation.md) for detailed instructions.

## 🚀 Quick Start and Documentation

LatteReview enables you to create custom literature review workflows with multiple AI reviewers. Each reviewer can use different models and providers based on your needs. Below is a working example of how you can use LatteReview for doing a quick title/abstract review with two junior and one senior reviewers (all AI agents)! And this is just the beginning! Beyond study screening, LatteReview can handle data abstraction, customized pipelines, image analysis, and much more. Explore the [Tutorials](#-tutorials) for more examples!

Please refer to the [Quick Start](./docs/quickstart.md) page and [Documentation](https://pouriarouzrokh.github.io/LatteReview/) page for detailed instructions.

The example below is fully self-contained: set your API keys, install the package, and run it as-is. It uses one OpenAI and one Gemini model, so it needs `OPENAI_API_KEY` and `GEMINI_API_KEY` (in a `.env` file or exported in your shell). You can swap in any LiteLLM-supported model — see [Model Support](#-model-support).

```python
from lattereview.providers import LiteLLMProvider
from lattereview.agents import TitleAbstractReviewer
from lattereview.workflows import ReviewWorkflow
import pandas as pd
import asyncio
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY, GEMINI_API_KEY) from a .env file
load_dotenv()

# First Reviewer: Conservative approach
reviewer1 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-6-luna"),
    name="Alice",
    backstory="a radiologist with expertise in systematic reviews",
    inclusion_criteria="The study must focus on applications of artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies that are not peer-reviewed or not written in English.",
    model_args={"reasoning_effort": "low"},
)

# Second Reviewer: More exploratory approach
reviewer2 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gemini/gemini-3.8-flash"),
    name="Bob",
    backstory="a computer scientist specializing in medical AI",
    inclusion_criteria="The study must focus on applications of artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies that are not peer-reviewed or not written in English.",
    model_args={"reasoning_effort": "low"},
)

# Expert Reviewer: Resolves disagreements
expert = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-6-sol"),
    name="Carol",
    backstory="a professor of AI in medical imaging",
    inclusion_criteria="The study must focus on applications of artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies that are not peer-reviewed or not written in English.",
    model_args={"reasoning_effort": "high"},
    additional_context="Alice and Bob disagree with each other on whether or not to include this article. You can find their reasonings above.",
)

# Define workflow
workflow = ReviewWorkflow(
    workflow_schema=[
        {
            "round": 'A',  # First round: Initial review by both reviewers
            "reviewers": [reviewer1, reviewer2],
            "text_inputs": ["title", "abstract"]
        },
        {
            "round": 'B',  # Second round: Expert reviews only disagreements
            "reviewers": [expert],
            "text_inputs": ["title", "abstract", "round-A_Alice_output", "round-A_Bob_output"],
            "filter": lambda row: row["round-A_Alice_evaluation"] != row["round-A_Bob_evaluation"]
        }
    ]
)

# Prepare your data: a DataFrame (or .csv/.xlsx/.ris file path) with 'title' and 'abstract' columns
data = pd.DataFrame(
    {
        "title": [
            "Deep learning for automated detection of pneumonia on chest radiographs",
            "Effects of mindfulness meditation on stress levels in college students",
        ],
        "abstract": [
            "We developed a convolutional neural network to detect pneumonia on chest X-rays.",
            "A randomized trial of mindfulness training in 200 undergraduates reduced stress.",
        ],
    }
)
# Or load from a file: data = pd.read_excel("articles.xlsx")

results = asyncio.run(workflow(data))  # Returns a pandas DataFrame with all original and output columns

# Save results
results.to_csv("review_results.csv", index=False)
```

### 🎯 Decision Models

Decision reviewers use a decision model instead of an LLM. Three models are supported, all through `SystemOneProvider`:

| Model | Provider | API key | Mean AUC* | Cost per 1,000 abstracts* | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Perplexity's **pplx-decider** | `SystemOneProvider(backend="perplexity")` | `PERPLEXITY_API_KEY` | **0.895** | **$0.05** | Best in our evaluation; open weights (Apache-2.0); bills the abstract once per question |
| TypeSafe's **Jev** | `SystemOneProvider()` (default) | `TYPESAFE_API_KEY` | 0.878 | $0.06 | Nearly free to add questions; uncertain answers vary slightly between runs |
| OpenAI's **gpt-6-luna** | `SystemOneProvider(backend="openai")` | `OPENAI_API_KEY` | 0.779 | $0.16 | Weak on long, multi-part criteria; may decline single questions (answer = None) |

\* `DecisionTitleAbstractReviewer` on all 11,793 articles of LatteReview's evaluation datasets; the v1 LLM reviewers scored 0.828. Jev and pplx-decider are also available through OpenRouter (`backend="openrouter"`, `OPENROUTER_API_KEY`).

```python
from lattereview.providers import SystemOneProvider
from lattereview.agents import DecisionTitleAbstractReviewer

decider = DecisionTitleAbstractReviewer(
    provider=SystemOneProvider(backend="perplexity"),  # pplx-decider-v1.1-27b
    name="Decider",
    inclusion_criteria={1: "The study must involve CT scans.", 2: "The study must use deep learning."},
    exclusion_criteria={1: "The study must not include PET scans."},
)
# Use it in a ReviewWorkflow like any reviewer. Columns: evaluation (1-5), include_probability,
# confidence, criteria (per-criterion probabilities) and reasoning (generated from the probabilities).
```

**Decision models vs LLM reviewers.** An LLM reviewer writes its answer and a reasoning; a decision model writes nothing and returns a probability for every allowed answer, so its answers are always on-schema and can be thresholded for a target recall. Decision models are fast (under a second per article) and cheap (about $0.05-0.16 per 1,000 abstracts), and in our evaluation pplx-decider and Jev ranked articles better than the v1 LLM reviewers. But they read text only, write no reasoning, do not do multi-step reasoning, and a 0.5 cutoff is often too strict for long, multi-part criteria: rank by `include_probability` or fit a cutoff per model with `suggest_threshold`. A good pattern is to let a decision model screen everything and send only uncertain articles to an LLM. Read [Decision Models](https://pouriarouzrokh.github.io/LatteReview/decision_models/) for how they work, backends, costs, self-hosting and the full evaluation.

## 🔌 Model Support

LatteReview offers flexible model integration through multiple providers:

- **LiteLLMProvider** (Recommended): Supports OpenAI, Anthropic (Claude), Gemini, Groq, DeepSeek, Perplexity (Sonar), OpenRouter, and more
- **OpenAIProvider**: Direct integration with OpenAI and Gemini APIs
- **GoogleProvider**: Direct integration with Gemini through Google's `google-genai` SDK
- **OllamaProvider**: Optimized for local models via Ollama
- **SystemOneProvider**: decision models: TypeSafe's Jev, Perplexity's pplx-decider and OpenAI's gpt-6-luna (Decisions APIs), via TypeSafe, Perplexity, OpenAI, OpenRouter, or any `/v1/systemone` server (e.g., a self-hosted OpenJev)

If you don't pass a `model`, `OpenAIProvider` and `LiteLLMProvider` use `gpt-6-luna`, `GoogleProvider` uses `gemini-3.8-flash`, and `OllamaProvider` uses `qwen3.8:27b` (run `ollama pull qwen3.8:27b` first). For Claude, pass e.g. `LiteLLMProvider(model="anthropic/claude-sonnet-5")` or `"anthropic/claude-haiku-4-5"` for a cheaper option. For Perplexity, pass `LiteLLMProvider(model="perplexity/sonar")` (`PERPLEXITY_API_KEY`); LatteReview sends it through Perplexity's Agent API.

Note: Models should support async operations and structured JSON outputs for optimal performance.

### Model compatibility

Newer reasoning models reject some request parameters that older models accept. OpenAI's GPT-5/GPT-6 families and o-series models reject `max_tokens` (they use `max_completion_tokens`) and non-default `temperature`/`top_p`. Anthropic's Claude Opus 4.7+, Claude 5 and Fable models reject `temperature`, `top_p` and `top_k`. LatteReview handles this for you. If a model rejects a parameter in `model_args`, it is dropped (or `max_tokens` is sent as `max_completion_tokens`) with a one-time warning. If a reasoning model runs out of tokens before finishing its answer, the call is retried without the token limit. Models that accept these parameters are called exactly as before, so existing code keeps working.

For reasoning models, we recommend leaving out `max_tokens` and `temperature` and using `reasoning_effort` (e.g., `"low"` for screening, `"high"` for an expert reviewer) instead. See [Model compatibility](https://pouriarouzrokh.github.io/LatteReview/api/providers/#model-compatibility) in the docs for details.

## 📖 Documentation

Full documentation and API reference are available at: [https://pouriarouzrokh.github.io/LatteReview](https://pouriarouzrokh.github.io/LatteReview)

## 🎓 Tutorials

✅ TitleAbstractReviewer: 
    🔸[1.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/title_abstract_review/title_abstract_review.ipynb) A simple task of abstract screening based on 1-5 scoring + inclusion and exclusion criteria
✅ AbstractionReviewer:
    🔸[1.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/abstraction_review_simple/abstraction_review_sample.ipynb) Data abstraction from abstracts/manuscripts
✅ ScoringReviewer: 
    🔸[1.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/scoring_review_simple/scoring_review_simple.ipynb) A simple task of abstract screening based on custom scoring by multiple agents
    🔸[2.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/scoring_review_rag/scoring_review_rag.ipynb) Question answering with RAG (Retrieval Augmented Generation)
    🔸[3.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/scoring_review_image/scoring_review_image.ipynb) Image analysis by LatteReview  
✅ Custom Reviewer:
    🔸[1.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/custom_reviewer/abstraction_review_literature_analysis.ipynb) How to Customize the AbstractReviewer Agent for Your Needs
    🔸[2.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/base_functionalities/base_functionalities.ipynb) Chat with the agents and other base functionalities
    🔸[3.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/abstraction_review_literature_analysis/abstraction_review_literature_analysis.ipynb): Combination of differnet agents for a comprehensive literature review
✅ Decision Reviewers (Jev, pplx-decider, gpt-6-luna):
    🔸[1.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/decision_review_jev/decision_review_jev.ipynb) Screening, scoring and categorical extraction with Jev, and the same screening with pplx-decider and gpt-6-luna
    🔸[2.](https://github.com/PouriaRouzrokh/LatteReview/blob/main/tutorials/hybrid_review_jev_llm/hybrid_review_jev_llm.ipynb) Hybrid review: Jev screens everything, an LLM handles the uncertain articles

## 🛣️ Roadmap for Future Features

- [x] Implementing LiteLLM to add support for additional model providers
- [x] Draft the package full documentation
- [x] Enable agents to return a percentage of certainty
- [x] Enable agents to be grounded in static references (text provided by the user)
- [x] Enable agents to be grounded in dynamic references (i.e., recieve a function that outputs a text based on the input text. This function could, e.g., be a RAG function.)
- [x] Support for image-based inputs and multimodal analysis
- [x] Development of `AbstractionReviewer` class for automated paper summarization
- [x] Showcase how `AbstractionReviewer` class could be used to analyse the literature around a certain topic.
- [x] Adding a tutorial example and also a section to the docs on how to create custom reviewer agents.
- [x] Adding a `TitleAbstractReviewer` agent and adding a tutorial for it.
- [x] Evaluating LatteReview.
- [x] Writing the white paper for the package and public launch
- [x] Addign support for `RIS` files.
- [x] Adding support for models without structured-output (json_schema) capability via an automatic JSON-mode fallback (e.g., DeepSeek).
- [x] Supporting the newest reasoning models (GPT-6, Claude 5, Gemini 3.x) with automatic handling of parameters they reject.
- [x] Supporting System One decision models (Jev) for fast, probability-based screening, including hybrid Jev + LLM workflows.
- [x] Supporting Perplexity's and OpenAI's Decisions APIs, and Perplexity's Sonar LLMs.
- [ ] Development of a no-code web application
- [ ] (for v>) Adding conformal prediction tool for calibrating agents on their certainty scores
- [ ] (for v>2.0.0) Adding a dialogue tool for enabling agents to seek external help (from helper agents or parallel reviewer agents) during review.
- [ ] (for v>2.0.0) Adding a memory component to the agents for saving their own insights or insightful feedback they receive from the helper agents.

## 👨‍💻 Author

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

## ❤️ Support LatteReview

If you find LatteReview helpful in your research or work, consider supporting its continued development. Since we're already sharing a virtual coffee break while reviewing papers, maybe you'd like to treat me to a real one? ☕ 😊

### Ways to Support:

- [Become my sponsor](https://github.com/sponsors/PouriaRouzrokh) on GitHub
- [Treat me to a cup of coffee](http://ko-fi.com/pouriarouzrokh) on Ko-fi ☕
- [Star the repository](https://github.com/PouriaRouzrokh/LatteReview) to help others discover the project
- Submit bug reports, feature requests, or contribute code
- Share your experience using LatteReview in your research

## 📜 License

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (see the [LICENSE](./LICENSE) file).
To view a copy of this license, visit [creativecommons.org/licenses/by-nc-nd/4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## 🤝 Contributing

I welcome contributions! Please feel free to submit a Pull Request.

## Acknowledgement

I would like to express my heartfelt gratitude to [Moein Shariatnia](https://github.com/moein-shariatnia) for his invaluable support and contributions to this project.

## 📚 Citation

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
