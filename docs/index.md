# LatteReview

[![PyPI version](https://badge.fury.io/py/lattereview.svg)](https://badge.fury.io/py/lattereview)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintained: yes](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/prouzrokh/lattereview)
[![View on arXiv](https://img.shields.io/badge/arXiv-View%20Paper-orange)](https://arxiv.org/abs/2501.05468)
[![Sponsor me on GitHub](https://img.shields.io/badge/Sponsor%20me-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/PouriaRouzrokh)
[![Support me on Ko-fi](https://img.shields.io/badge/Support%20me-Ko--fi-orange.svg?logo=ko-fi&logoColor=white)](http://ko-fi.com/pouriarouzrokh)

<p><img src="images/robot.png" width="400"></p>

A framework for multi-agent review workflows using large language models.

## Overview

LatteReview is a powerful Python package designed to automate academic literature review processes through AI-powered agents. Just like enjoying a cup of latte, reviewing numerous research articles should be a pleasant, efficient experience that doesn't consume your entire day!

### What's New in v2

LatteReview v2 introduces a fully redesigned **agentic framework** built on [Pydantic AI](https://ai.pydantic.dev/). The new architecture provides:

- **Simplified model configuration** -- use model strings like `"openai:gpt-5.4-mini"`, `"anthropic:claude-sonnet-4-6"`, or `"google-gla:gemini-3-flash-preview"` instead of provider wrapper classes.
- **Skill system** -- extend reviewers with configurable and tool-based skills (web search, PubMed, arXiv, Semantic Scholar, and more).
- **Preset reviewer types** -- `ScoringReviewer`, `TitleAbstractReviewer`, and `AbstractionReviewer` are ready to use out of the box.
- **AgenticWorkflow** -- a new workflow engine with checkpoint/resume, per-item cost tracking, and action logging.
- **Memory and helpers** -- reviewers can carry context across items and use helper functions during review.

The v1 API remains fully functional but emits deprecation warnings. See the [Migration Guide](migration.md) for details on upgrading.

!!! tip "Not every task needs agentic mode"
    Agentic capabilities shine when the task requires information beyond the input text (e.g., verifying claims via search, building cross-item expertise). For simple extraction tasks where all the information is in the abstract, non-agentic mode (`max_iterations=1`) is faster, cheaper, and often equally accurate. See the [Quick Start guide](quickstart.md#when-to-use-agentic-mode) for detailed guidance.

## Features

- Multi-agent review system with customizable roles and expertise levels for each reviewer
- Support for multiple review rounds with hierarchical decision-making workflows
- Review diverse content types including article titles, abstracts, custom texts, and even **images** using LLM-powered reviewer agents
- Define reviewer agents with specialized backgrounds and distinct evaluation capabilities (e.g., scoring or concept abstraction or custom reviewers of your own preference)
- Create flexible review workflows where multiple agents operate in parallel or sequential arrangements
- Enable reviewer agents to analyze peer feedback, cast votes, and propose corrections to other reviewers' assessments
- Enhance reviews with item-specific context integration, supporting use cases like **Retrieval Augmented Generation (RAG)**
- Model-agnostic integration supporting OpenAI, Gemini, Claude, Groq, and local models via Ollama
- High-performance asynchronous processing for efficient batch reviews
- Standardized output format featuring detailed scoring metrics and reasoning transparency
- Robust cost tracking and memory management systems
- Extensible architecture supporting custom review workflow implementation
- Support for RIS (Research Information Systems) file format for academic literature review
- **v2**: Skill system with built-in search skills (Google, DuckDuckGo, PubMed, Semantic Scholar, arXiv)
- **v2**: Checkpoint/resume for long-running workflows with atomic per-item saves
- **v2**: Simplified model configuration via Pydantic AI model strings

## Quick Links

- [Installation Guide](installation.md)
- [Quick Start Guide](quickstart.md)
- [Migration Guide (v1 to v2)](migration.md)
- [Agentic API Reference (v2)](api/agentic.md)
- [Tutorial notebooks (v1)](https://github.com/PouriaRouzrokh/LatteReview/tree/main/tutorials)
- [Tutorial notebooks (v2)](https://github.com/PouriaRouzrokh/LatteReview/tree/main/tutorials_agentic)
- [API Reference (v1)](api/workflows.md)
- [GitHub Repository](https://github.com/PouriaRouzrokh/LatteReview)

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. See [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) for details.

## Authors

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

## Support LatteReview

If you find LatteReview helpful in your research or work, consider supporting its continued development.

### Ways to Support:

- [Become my sponsor](https://github.com/sponsors/PouriaRouzrokh) on GitHub
- [Treat me to a cup of coffee](http://ko-fi.com/pouriarouzrokh) on Ko-fi
- [Star the repository](https://github.com/PouriaRouzrokh/LatteReview) to help others discover the project
- Submit bug reports, feature requests, or contribute code
- Share your experience using LatteReview in your research

## Acknowledgement

I would like to express my heartfelt gratitude to [Moein Shariatnia](https://github.com/moein-shariatnia) for his invaluable support and contributions to this project.

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
