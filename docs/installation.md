# Installation

## Requirements

- **Python 3.12 or later**
- Core dependencies are installed automatically (pydantic-ai, pandas, pydantic, pyyaml, and others)

## 1. Install from PyPI (Recommended)

```bash
pip install lattereview
```

### Optional Extras

LatteReview provides optional dependency groups for extended functionality:

```bash
# v2 search skills (DuckDuckGo, Google, PubMed, arXiv)
pip install "lattereview[search]"

# v2 Semantic Scholar search skill
pip install "lattereview[scholar]"

# All v2 agentic extras (search + scholar)
pip install "lattereview[agentic-all]"

# Development tools
pip install "lattereview[dev]"

# Documentation tools
pip install "lattereview[docs]"

# All extras
pip install "lattereview[all]"
```

## 2. Install from Source Code

#### Option A: Using Git

```bash
# Clone the repository
git clone https://github.com/PouriaRouzrokh/LatteReview.git
cd LatteReview
```

#### Option B: Using ZIP Download

1. Go to https://github.com/PouriaRouzrokh/LatteReview
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file and navigate to the directory:

```bash
cd path/to/LatteReview-main
```

After obtaining the source code through either option, you can install it using one of these methods:

```bash
# Basic installation
pip install .

# Install from specific versions of dependencies mentioned in requirements.txt
pip install -r requirements.txt

# Development installation (all optional dependencies)
pip install -e ".[all]"
```

## Verify Installation

```python
import lattereview
print(lattereview.__version__)
```

## Core Dependencies

The following are installed automatically with the base package:

- **pydantic-ai** (>=1.69.0) -- v2 agentic framework backbone
- **pyyaml** (>=6.0) -- skill configuration loading
- **pandas** (>=2.2.3) -- data handling
- **pydantic** (>=2.10.3) -- data validation and output models
- **litellm** (>=1.55.2) -- v1 provider support
- **openai** (>=1.57.4) -- v1 OpenAI provider

## Troubleshooting

If you encounter installation issues:

```bash
# Check Python version
python --version  # Should be 3.12 or later

# Update pip
pip install --upgrade pip

# Install build dependencies
pip install build wheel setuptools
```
