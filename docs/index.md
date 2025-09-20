# LatteReview 🤖☕

[![PyPI version](https://badge.fury.io/py/lattereview.svg)](https://badge.fury.io/py/lattereview)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintained: yes](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/prouzrokh/lattereview)
[![View on arXiv](https://img.shields.io/badge/arXiv-View%20Paper-orange)](https://arxiv.org/abs/2501.05468)
[![Sponsor me on GitHub](https://img.shields.io/badge/Sponsor%20me-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/PouriaRouzrokh)
[![Support me on Ko-fi](https://img.shields.io/badge/Support%20me-Ko--fi-orange.svg?logo=ko-fi&logoColor=white)](http://ko-fi.com/pouriarouzrokh)

<p><img src="images/robot.png" width="400"></p>

A framework for multi-agent review workflows using large language models.

🚨 **NEW**: Now supports the Gemini 2.5 family of models using a new GoogleProvider class.

## Overview

LatteReview is a powerful Python package designed to automate academic literature review processes through AI-powered agents. Just like enjoying a cup of latte ☕, reviewing numerous research articles should be a pleasant, efficient experience that doesn't consume your entire day!

## Features

- Multi-agent review system with customizable roles and expertise levels for each reviewer
- Support for multiple review rounds with hierarchical decision-making workflows
- Review diverse content types including article titles, abstracts, custom texts, and even **images** using LLM-powered reviewer agents
- Define reviewer agents with specialized backgrounds and distinct evaluation capabilities (e.g., scoring or concept abstraction or custom reviewers of your own preferance)
- Create flexible review workflows where multiple agents operate in parallel or sequential arrangements
- Enable reviewer agents to analyze peer feedback, cast votes, and propose corrections to other reviewers' assessments
- Enhance reviews with item-specific context integration, supporting use cases like **Retrieval Augmented Generation (RAG)**
- Broad compatibility with LLM providers through LiteLLM, including OpenAI and Ollama
- Model-agnostic integration supporting OpenAI, Gemini, Claude, Groq, and local models via Ollama
- High-performance asynchronous processing for efficient batch reviews
- Standardized output format featuring detailed scoring metrics and reasoning transparency
- Robust cost tracking and memory management systems
- Extensible architecture supporting custom review workflow implementation
- **NEW**: Support for RIS (Research Information Systems) file format for academic literature review

## Quick Links

- [Installation Guide](installation.md)
- [Quick Start Guide](quickstart.md)
- [Tutorial notebooks](https://github.com/PouriaRouzrokh/LatteReview/tree/main/tutorials)
- [API Reference](api/workflows.md)
- [GitHub Repository](https://github.com/PouriaRouzrokh/LatteReview)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PouriaRouzrokh/LatteReview/blob/main/LICENSE) file for details.

## 👨‍💻 Authors

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

If you find LatteReview helpful in your research or work, consider supporting its continued development. Since we're already sharing a virtual coffee break while reviewing papers, maybe you'd like to treat me to a real one? ☕ 😊

### Ways to Support:

- [Become my sponsor](https://github.com/sponsors/PouriaRouzrokh) on GitHub
- [Treat me to a cup of coffee](http://ko-fi.com/pouriarouzrokh) on Ko-fi ☕
- [Star the repository](https://github.com/PouriaRouzrokh/LatteReview) to help others discover the project
- Submit bug reports, feature requests, or contribute code
- Share your experience using LatteReview in your research

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
