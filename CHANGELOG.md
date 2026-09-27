# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Adding MCP support.

## [1.2.0] - 2026-9-27

This release makes LatteReview work with the newest OpenAI, Anthropic and Google models while keeping older models working unchanged. It also includes all changes from 1.1.2, which was never published to PyPI; they are listed below under 1.1.2.

### Added

- Support for current models, tested with real API calls: OpenAI GPT-6 (`gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`) and GPT-5.x, Anthropic Claude Opus 5.5, Opus 5, Sonnet 5, Haiku 4.5 and Fable 5.1, and Google Gemini 3.x (`gemini-3.8-flash`, `gemini-3.5-flash-lite`). See "Model compatibility" in the providers documentation.
- Automatic handling of request parameters that newer models reject. When the API rejects a parameter in `model_args` (e.g., `temperature` on GPT-5/GPT-6 and Claude Opus 4.7+/Claude 5, or `max_tokens` on OpenAI reasoning models), `OpenAIProvider`, `LiteLLMProvider` and `GoogleProvider` drop it, or send `max_completion_tokens` instead of `max_tokens`, print a one-time warning, and retry. Models that accept these parameters are called exactly as before.
- When a reasoning model runs out of tokens before finishing a structured answer (hidden reasoning counts against `max_tokens`), the call is retried once without the token limit, with a warning. Previously such items failed after all retries.
- `GoogleProvider` passes Gemini's `thinking_config` from `model_args` (e.g., `{"thinking_level": "low"}` on Gemini 3.x).
- `LiteLLMProvider` falls back to JSON mode when a model rejects the forced tool call that older LiteLLM releases use for structured output (Claude Opus 5.5 and Fable 5.1), and parses JSON wrapped in a markdown code fence.

### Fixed

- `OpenAIProvider` no longer fails reviews on models that `tokencost` does not know (e.g., all GPT-5.5+ and GPT-6 models). Costs are now computed from the token usage the API reports, including reasoning tokens, and priced with LiteLLM's model map. A model missing from every price map costs 0 with a warning instead of failing.
- `GoogleProvider` costs now come from Gemini's reported usage (including thinking tokens) and current prices. The previous hard-coded rates were several times too high for current models, and each review made two extra `count_tokens` API calls.
- `LiteLLMProvider` now reports costs for Groq models, whose responses name a model that is not in LiteLLM's price map.
- Image inputs are sent with standard MIME types (`image/jpeg` for `.jpg` files). Claude rejected the previous `image/jpg`.
- `OpenAIProvider` uses `client.chat.completions.parse` when available and falls back to `client.beta.chat.completions.parse` on older `openai` releases (the `beta` path no longer exists in `openai` 3.x).
- Cost warnings are printed once per model instead of once per reviewed item.
- `OllamaProvider` no longer fails on `model_args` other than `temperature`/`max_tokens`: `reasoning_effort` maps to Ollama's `think` setting, and other keys (e.g., `top_p`, `seed`) are passed as model `options` instead of crashing `AsyncClient.chat`.
- `OllamaProvider` requests output that follows the reviewer's JSON schema instead of generic JSON mode.
- `OllamaProvider.close()` works again (it called a method the `ollama` client does not have).
- `OllamaProvider` streaming (`get_response(..., stream=True)`) works; it iterated over the unawaited `chat()` coroutine. The docs' streaming example now shows the required `await`.

### Changed

- Python 3.10 or later is now required (`requires-python>=3.10`). Current releases of LiteLLM, `openai` and `google-genai` no longer support Python 3.9; Python 3.9 users keep getting LatteReview 1.1.1 from `pip`.
- Dependency floors raised: `litellm>=1.94.0` (native Claude structured outputs), `google-genai>=1.51.0` (Gemini 3 `thinking_level`) and `ollama>=0.5.3` (`think` levels).
- README, documentation and all tutorial notebooks now use current models and were re-executed with saved outputs. Examples use `reasoning_effort` instead of `max_tokens`/`temperature` for reasoning models. The `evaluation/` notebooks are unchanged records of the original evaluation runs.
- Fixed tutorial bugs found while re-running them: a misspelled CSV path in the literature-analysis tutorial, a wrong column name in the scoring tutorial, and image generation that could give two target colors the same digit.
- **Default models updated** to current cost-efficient workhorse models: `OpenAIProvider` and `LiteLLMProvider` now default to `gpt-6-luna` (was `gpt-4o-mini`), `GoogleProvider` defaults to `gemini-3.8-flash` (was `gemini-2.5-pro`, which Google now limits to accounts that have used it before), and `OllamaProvider` defaults to `qwen3.8:27b` (was `llama3.2-vision:latest`; run `ollama pull qwen3.8:27b` first). This changes the model, and so the cost and results, for code that relies on the default; pass `model=` explicitly to keep the old one.

## [1.1.2] - 2026-6-10 (never published to PyPI; included in 1.2.0)

### Fixed

- Surfaced the real underlying error when an item review fails after all retries. Previously every failure (retired model, rejected response format, etc.) collapsed into the unhelpful `Error running workflow: Error running workflow: Error reviewing items: Error reviewing item!` message.
- Cost calculation failures in `LiteLLMProvider` no longer crash reviews. Models missing from LiteLLM's pricing map (e.g., Groq and OpenRouter models) now report a cost of 0 with a warning instead of discarding the successful review.
- `LiteLLMProvider.get_json_response` now falls back to basic JSON mode (`json_object`) when a provider rejects `json_schema` response formats (e.g., DeepSeek).
- `ReviewWorkflow` schema validation now actually runs (the old `__post_init__` hook was never invoked under Pydantic v2) and produces clear messages when a round is missing required keys or has an invalid reviewer.
- `BasicReviewer.review_items` now returns the total cost of all items in the batch instead of only the last item's cost, so per-reviewer costs reported by `ReviewWorkflow` are correct.
- `ReviewWorkflowError` messages are no longer double-wrapped.

### Changed

- Replaced retired Gemini models in defaults, README, docs, and tutorials: `gemini-1.5-flash` → `gemini-2.5-flash`, `gemini-2.5-pro-preview-05-06` → `gemini-2.5-pro`, `gemini-2.5-flash-preview-04-17` → `gemini-2.5-flash`. Google has retired the Gemini 1.x/2.0 models and the 2.5 preview aliases, which caused workflows following the previous documentation to fail with 404 errors.
- Cleaned up packaging: `pyproject.toml` is now the single source of metadata (removed `setup.py`). Runtime dependencies no longer include development tools (black, flake8, mkdocs, twine, opencv, etc.); they moved to the `[dev]`, `[docs]`, and `[all]` extras that the installation guide already documented. `requires-python` is back to `>=3.9` as documented.
- Documentation accuracy pass: fixed the quickstart round-B example to use the reviewers' names in column references (`round-A_Alice_output`, not `round-A_reviewer1_output`), corrected the `OpenAIProvider` Gemini example (no `gemini/` prefix outside LiteLLM), refreshed OpenRouter model examples, aligned all license references with the CC BY-NC-ND 4.0 LICENSE file, documented required API-key environment variables, and made the README quick-start example self-contained and runnable as-is.

## [1.1.1] - 2026-1-7

### Fixed

- Fixed infinite retry loop in `review_item` function in `BasicReviewer` that occurred when JSON parsing errors happened with LiteLLM provider. The `num_tried` counter is now properly incremented, preventing infinite retries that consumed API credits.

## [1.1.0] - 2025-5-14

### Fixed

- Added google-genai to requirements.txt.

## [1.0.9] - 2025-5-14

### Added

- Added the `GoogleProvider` class to providers.
- Added support for Gemini 2.5 family of models.

### Changed

- Updated the docs to reflect the above changes.

## [1.0.8] - 2025-4-29

### Added

- Swithced to UV instead of Pypi.

## [1.0.7] - 2025-4-29

### Added

- Swithced to UV instead of Pypi.

## [1.0.6] - 2025-4-29

### Fixed

- Attempted fixing the bug in handling structured outputs by the Olama provider.

## [1.0.5] - 2025-3-16

### Fixed

- Fixed initalization of the `lattereview.utils.data_handler` module.

## [1.0.4] - 2025-3-16

### Added

- Added support for handling `RIS` input files.

### Changed

- The `review_workflow.py` will now accept string inputs with `.ris`, `.csv`, `.xls`, `.xlsx` formats.

## [1.0.3] - 2025-2-1

### Changed

- The `OpenAIProvider` now accepts a base_url, enabling it to be used with providers like OpenRouter.
- Providers now accept a calculate_cost argument to control whether or not calculate the chats. Mostly useful for models that are not available in tokencost package.
- Updated the docs to reflect the above features.

## [1.0.2] - 2025-2-1

### Fixed

- Fixed a bug in base_reviewer that caused the model_args to be printed in every call.

### Changed

- The model_args in base_agent is now an empty dictionary by default.

## [1.0.1] - 2025-1-30

### Fixed

- Fixed a bug in base_reviewer that caused the reviewers to fail when model_args were not provided.

## [1.0.0] - 2025-1-14

### Fixed

- Fixed some typos.

## [0.8.0] - 2025-1-14

### Added

- Added the arXiv citation.

### Fixed

- Fixed some typos.

## [0.7.0] - 2025-1-4

### Added

- Added a `_vesrion.py` file.

### Changed

- Updated the project license to `CC-BY-NC-ND-4.0`.
- Updated the readme file.

### Removed

- Removed the `Field` class in pydantic validations.

### Fixed

- Fixed an issue in the versioning of Pandas package that resulted in a warning when installing on colab.

## [0.6.0] - 2025-1-4

### Added

- Added `TitleAbstractReviewer` agent.
- Added a tutorial for the `TitleAbstractReviewer` agent.
- Evaluated lattereeview using the `TitleAbstractReviewer` agent.

### Changed

- Addressed a bug in prompts and `BaseAgent` which prevented the correct removal of additional_context and examples where they were not provided to the agents.
- Updated all the docs to reflect all the above changes.
- Updated the `README.md` file to reflect all the above changes.

### Removed

- The `ReasoningType` is now removed and `reasoning` in agents receives simple string variables.

### Fixed

- The `BasicReviewer` is now directly importable from the agents module.

## [0.5.1] - 2025-1-1

### Fixed

- The `BasicReviewer` is now directly importable from the agents module.

## [0.5.0] - 2024-12-31

### Added

- Added a section to the docs on how to create custom reviewer agents.

### Changed

- Renamed the `examples` folder to `tutorials`.
- Updated the `README.md` file to reflect all the above changes.

## [0.4.0] - 2024-12-27

### Added

- Added support for `AbstractionReviewer` agents.

### Changed

- Renamed `BaseAgent` to `BasicReviewer`.
- Moved many joint functionalities betwee `ScoringReviewer` and `AbstractionReviewer` to the `BasicReviewer`.
- The reviewer agents will not load promps from their own scripts.
- Addressed a bug in `base_prompt.py` that prevented the placeholders in generic propmt to be appropriately removed if their value is empty.
- Moved the `generic_prompt` attribute to `basic_reviewer.py`.
- Updated all the docs to reflect all the above changes.
- Updated the `README.md` file to reflect all the above changes.

### Removed

- The `generic_prompts` folder is removed. Generic prompts are now defined in the body of the script for each custom reviewer class.

## [0.3.0] - 2024-12-26

### Added

- Workflows and agents can now accept a list of images to process both textual and image input data (if supported by the chosen model)
- Added the `scoring_review_image` use case to the example folders.
- Added the `base_functionalites` use case to the example folders and removed it from the `scoring_review_simple` example.

### Changed

- Updated the variable and method names in all scripts to clearly signal if they are dealing with text or image data.
- Updated all the docs to reflect all the above changes.
- Updated the `README.md` file to reflect all the above changes.

### Removed

- Removed the hashing validation in the `review_workflow.py`.
- Removed the output validation in the `OllamaProvider.py`

### Fixed

- Addressed a bug in the `scoring_review_prompt.txt` that caused the reasoning and examples not to be read by the agents.

## [0.2.1] - 2024-12-23

### Added

- The agents can now accept an `additional_context` argument of type string or Callable. If callable, expects an async function that accepts a single input review item (e.g., to retrieve the relevant context for that unique item in RAG use cases)
- Added the `scoring_review_rag` use case to the example folders.

### Changed

- Moved `examples/scoring_review_test.ipynb` to `examples/scoring_review_simple/scoring_review_simple.ipynb`.
- Updated all provider classes so that they can now directly accept classes inheriting from pydantic.BaseModel as `response_format_class`.
- Updated the naming convention of prompts in the agent methods for further clarity.
- Added `_` to all internal methods of `BasicReviewer` class.
- All example data spreadsheets are now named as `data.csv`.
- Updated all the docs to reflect all the above changes.
- Updated the `README.md` file to reflect all the above changes.

### Fixed

- Addressed a bug in the `scoring_review_prompt.txt` that caused the reasoning and examples not to be read by the agents.

## [0.2.0] - 2024-12-21

### Added

- All agents now return a `certainty` score which is an integer between 0 to 100.
- It is now possible to pass `0` to `scoring_set` of the ScoringReviewer agents as 0 is not used for denoting uncertainty anymore.

### Changed

- Updated the `review_workflow` to dynamically add any output keys from reviewers to the workflow dataframe.

- Updated the `scoring_review_prompt` for clarity and to reflect the above changes.
- Renamed the `score_set` parameter of the `scooring_reviewer `to `scoring_set`.
- Renamed `score_review_test.md` to `scoring_review_test.md`.
- Updated the `scoring_review_test.ipynb` to reflect all the above changes.
- Updated the `README.md` file to reflect all the above changes.

### Deprecated

- `ScoringReviewer` agent now only accepts `brief` and `cot` for reasoning. the `long` reasoning is now deprecated.

## [0.1.1] - 2024-12-16

### Added

- Added the package documentation to the `docs` folder.
- Added a `data` folder within the `examples` folder.

### Changed

- Updated the `README.md` file.
- Moved `README.md` to the `docs` folder.
- Changed `notebooks` folder to `examples`.
- Changed `data.xlsx` to `test_article_data.csv` which now has cleaner Column names and only contains 20 rows.
- Moved the `review_workflow.py` to the `workflows` folder.
- Passing the `inputs_description` to agents are now optional. The default value is "article title/abstract."

### Fixed

- Bug in `OpenAIProvider.py` making it unable to read the environmental variable for OPENAI_API_KEY.
