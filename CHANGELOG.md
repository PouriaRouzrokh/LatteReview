# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Adding MCP support.

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
