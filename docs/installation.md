There are several ways to install LatteReview:

## 1. Install from PyPI (Recommended)

```bash
pip install lattereview
```

You can also install additional features using these extras:

```bash
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

# Development installation (editable, with all optional dependencies)
pip install -e ".[all]"
```

You can also install directly from GitHub without cloning:

```bash
pip install "git+https://github.com/PouriaRouzrokh/LatteReview.git"
```

## Verify Installation

```python
import lattereview
print(lattereview.__version__)
```

## Requirements

- Python 3.9 or later
- Core dependencies (automatically installed):
  - litellm (>=1.55.2)
  - openai (>=1.57.4)
  - google-genai (>=1.15.0)
  - pandas (>=2.2.2)
  - pydantic (>=2.10.3)
  - And others as specified in `pyproject.toml`

## Set Up API Keys

To call hosted LLMs you need API keys for the providers you plan to use (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`). See [Step 1 of the Quick Start guide](quickstart.md#step-1-set-up-api-keys) for the details. No keys are needed for local models served through Ollama.

## Troubleshooting

If you encounter installation issues:

```bash
# Check Python version
python --version  # Should be 3.9 or later

# Update pip
pip install --upgrade pip

# Install build dependencies
pip install build wheel setuptools
```
