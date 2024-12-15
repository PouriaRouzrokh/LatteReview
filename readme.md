# LatteReview 🤖☕

[![PyPI version](https://badge.fury.io/py/lattereview.svg)](https://badge.fury.io/py/lattereview)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintained: yes](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/prouzrokh/lattereview)

LatteReview is a powerful Python package designed to automate academic literature review processes through AI-powered agents. Just like enjoying a cup of latte ☕, reviewing numerous research articles should be a pleasant, efficient experience that doesn't consume your entire day!

## 🎯 Key Features

- Multi-agent review system with customizable roles and expertise
- Support for multiple review rounds with hierarchical decision-making
- Flexible model integration (OpenAI, Gemini, Claude, Groq, local models via Ollama)
- Asynchronous processing for high-performance batch reviews
- Structured output format with detailed scoring and reasoning
- Comprehensive cost tracking and memory management
- Extensible architecture for custom review workflows

## 🛠️ Installation

```bash
pip install lattereview
```

## 🚀 Quick Start

Here's a simple example of how to set up a review workflow with two primary reviewers and an expert reviewer for conflict resolution:

```python
from lattereview.providers import LiteLLMProvider
from lattereview.agents import ScoringReviewer
from lattereview.review_workflow import ReviewWorkflow
import pandas as pd
import asyncio

# First Reviewer: Conservative approach
reviewer1 = ScoringReviewer(
    provider=LiteLLMProvider(model="gpt-4o-mini"),
    name="Alice",
    backstory="a radiologist with expertise in systematic reviews",
    input_description="article title and abstract",
    scoring_task="Evaluate how relevant the article is to artificial intelligence applications in radiology",
    score_set=[1, 2, 3, 4, 5],
    scoring_rules="Rate the relevance on a scale of 1 to 5, where 1 means the article is not at all relevant to AI in radiology, and 5 means it directly focuses on AI applications in radiology.",
    model_args={"temperature": 0.1}  # Low temperature for consistent scoring
)

# Second Reviewer: More exploratory approach
reviewer2 = ScoringReviewer(
    provider=LiteLLMProvider(model="gemini/gemini-1.5-flash"),
    name="Bob",
    backstory="a computer scientist specializing in medical AI",
    input_description="article title and abstract",
    scoring_task="Evaluate how relevant the article is to artificial intelligence applications in radiology",
    score_set=[1, 2, 3, 4, 5],
    scoring_rules="Rate the relevance on a scale of 1 to 5, where 1 means the article is not at all relevant to AI in radiology, and 5 means it directly focuses on AI applications in radiology.",
    model_args={"temperature": 0.8}  # Higher temperature for creative interpretation
)

# Expert Reviewer: Resolves disagreements
expert = ScoringReviewer(
    provider=LiteLLMProvider(model="gpt-4o"),
    name="Carol",
    backstory="a professor of AI in medical imaging",
    input_description="article title, abstract, and previous reviews",
    scoring_task="Review Alice and Bob's relevance assessments of this article to AI in radiology",
    score_set=[1, 2],
    scoring_rules='Score 1 if you agree with Alice\'s assessment, 2 if you agree with Bob\'s assessment',
    model_args={"temperature": 0.1}  # Low temperature for careful judgment
)

# Define the multi-round review workflow
workflow = ReviewWorkflow([
    {
        "round": 'A',  # First round: Initial review by both reviewers
        "reviewers": [reviewer1, reviewer2],
        "inputs": ["title", "abstract"]
    },
    {
        "round": 'B',  # Second round: Expert reviews only disagreements
        "reviewers": [expert],
        "inputs": ["title", "abstract", "round-A_Alice_output", "round-A_Bob_output"],
        "filter": lambda row: row["round-A_Alice_score"] != row["round-A_Bob_score"]
    }
])

# Run the workflow on your data
data = pd.read_excel("articles.xlsx")
results = asyncio.run(workflow(data))
```

## 🔌 Model Support

LatteReview offers flexible model integration through multiple providers:

- **LiteLLMProvider** (Recommended): Supports OpenAI, Anthropic (Claude), Gemini, Groq, and more
- **OpenAIProvider**: Direct integration with OpenAI and Gemini APIs
- **OllamaProvider**: Optimized for local models via Ollama

Note: Models should support async operations and structured JSON outputs for optimal performance.

## 📖 Documentation

Full documentation and API reference will be available in our upcoming preprint paper. [Link to be added]

## 🛣️ Roadmap

- [ ] Development of `AbstractionReviewer` class for automated paper summarization
- [ ] Support for image-based inputs and multimodal analysis
- [ ] Development of a no-code web application
- [ ] Integration of RAG (Retrieval-Augmented Generation) tools
- [ ] Addition of graph-based analysis tools
- [ ] Enhanced visualization capabilities
- [ ] Support for additional model providers

## 👨‍💻 Author

**Pouria Rouzrokh, MD, MPH, MHPE**  
Medical Practitioner and Machine Learning Engineer  
Incoming Radiology Resident @Yale University  
Former Data Scientist @Mayo Clinic AI Lab

Find my work:
[![Twitter Follow](https://img.shields.io/twitter/follow/prouzrokh?style=social)](https://twitter.com/prouzrokh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/pouria-rouzrokh)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Profile-green)](https://scholar.google.com/citations?user=Ksv9I0sAAAAJ&hl=en)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:po.rouzrokh@gmail.com)

## ❤️ Support LatteReview

If you find LatteReview helpful in your research or work, consider supporting its continued development. Since we're already sharing a virtual coffee break while reviewing papers, maybe you'd like to treat me to a real one? ☕ 😊

### Ways to Support:

- [Treat me to a coffee](http://ko-fi.com/pouriarouzrokh) on Ko-fi ☕
- [Star the repository](https://github.com/PouriaRouzrokh/LatteReview) to help others discover the project
- Submit bug reports, feature requests, or contribute code
- Share your experience using LatteReview in your research

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📚 Citation

If you use LatteReview in your research, please cite our paper:

```bibtex
# Preprint citation to be added
```
