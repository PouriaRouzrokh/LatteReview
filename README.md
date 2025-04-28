# LatteReview 🤖☕

[![PyPI version](https://badge.fury.io/py/lattereview.svg)](https://badge.fury.io/py/lattereview)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintained: yes](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/prouzrokh/lattereview)
[![View on arXiv](https://img.shields.io/badge/arXiv-View%20Paper-orange)](https://arxiv.org/abs/2501.05468)
[![Sponsor me on GitHub](https://img.shields.io/badge/Sponsor%20me-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/PouriaRouzrokh)
[![Support me on Ko-fi](https://img.shields.io/badge/Support%20me-Ko--fi-orange.svg?logo=ko-fi&logoColor=white)](http://ko-fi.com/pouriarouzrokh)

<p><img src="docs/images/robot.png" width="400"></p>

LatteReview is a powerful Python package designed to automate academic literature review processes through AI-powered agents. Just like enjoying a cup of latte ☕, reviewing numerous research articles should be a pleasant, efficient experience that doesn't consume your entire day!

## 🎯 Key Features

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

## ��️ Installation

```bash
pip install lattereview
```

Please refer to the [installation guide](./docs/installation.md) for detailed instructions.

## 🚀 Quick Start and Documentation

LatteReview enables you to create custom literature review workflows with multiple AI reviewers. Each reviewer can use different models and providers based on your needs. Below is a working example of how you can use LatteReview for doing a quick title/abstract review with two junior and one senior reviewers (all AI agents)!

Please refer to the [Quick Start](./docs/quickstart.md) page and [Documentation](https://pouriarouzrokh.github.io/LatteReview/) page for detailed instructions.

```python
from lattereview.providers import LiteLLMProvider
from lattereview.agents import TitleAbstractReviewer
from lattereview.workflows import ReviewWorkflow
import pandas as pd
import asyncio
from dotenv import load_dotenv

# Load environment variables from the .env file in the root directory of your project
load_dotenv()

# First Reviewer: Conservative approach
reviewer1 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-4o-mini"),
    name="Alice",
    backstory="a radiologist with expertise in systematic reviews",
    inclusion_criteria="The study must focus on applications of artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies that are not peer-reviewed or not written in English.",
    model_args={"temperature": 0.2},
)

# Second Reviewer: More exploratory approach
reviewer2 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gemini/gemini-1.5-flash"),
    name="Bob",
    backstory="a computer scientist specializing in medical AI",
    inclusion_criteria="The study must focus on applications of artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies that are not peer-reviewed or not written in English.",
    model_args={"temperature": 0.2},
)

# Expert Reviewer: Resolves disagreements
expert = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="o3-mini"),
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

# Load and process your data
data = pd.read_excel("articles.xlsx")  # Must have 'title' and 'abstract' columns
results = asyncio.run(workflow(data))  # Returns a pandas DataFrame with all original and output columns

# Save results
results.to_csv("review_results.csv", index=False)
```

## 🔌 Model Support

LatteReview offers flexible model integration through multiple providers:

- **LiteLLMProvider** (Recommended): Supports OpenAI, Anthropic (Claude), Gemini, Groq, and more
- **OpenAIProvider**: Direct integration with OpenAI and Gemini APIs
- **OllamaProvider**: Optimized for local models via Ollama

Note: Models should support async operations and structured JSON outputs for optimal performance.

## 📖 Documentation

Full documentation and API reference are available at: [https://pouriarouzrokh.github.io/LatteReview](https://pouriarouzrokh.github.io/LatteReview)

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
- [ ] Adding support for Deepseek R1 models (and models w/o structured output capablity in general).
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

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.  
To view a copy of this license, visit [LICENSE](http://creativecommons.org/licenses/by-nc/4.0/).

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
