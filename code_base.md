## requirements.txt

```txt
litellm>=1.55.2
nest-asyncio>=1.6.0
ollama>=0.4.4
openai>=1.57.4
pandas>=2.2.3
pydantic>=2.10.3
python-dotenv>=1.0.1
tokencost>=0.1.17
tqdm>=4.67.1
openpyxl>=3.1.5
ipykernel>=6.29.5
black>=24.10.0
flake8>=7.1.1
```

## collect_scripts.py

```py
"""Script to collect and organize code files into a markdown document."""

from pathlib import Path
from typing import List, Tuple, Set


def gather_code_files(
    root_dir: Path, extensions: Set[str], exclude_files: Set[str], exclude_folders: Set[str]
) -> Tuple[List[Path], List[Path]]:
    """Gather code files while respecting exclusion rules."""
    try:
        code_files: List[Path] = []
        excluded_files_found: List[Path] = []

        for file_path in root_dir.rglob("*"):
            if any(excluded in file_path.parts for excluded in exclude_folders):
                if file_path.is_file():
                    excluded_files_found.append(file_path)
                continue

            if file_path.is_file():
                if file_path.name in exclude_files:
                    excluded_files_found.append(file_path)
                elif file_path.suffix in extensions:
                    code_files.append(file_path)

        return code_files, excluded_files_found
    except Exception as e:
        raise RuntimeError(f"Error gathering code files: {str(e)}")


def write_to_markdown(code_files: List[Path], excluded_files: List[Path], output_file: Path) -> None:
    """Write collected files to a markdown document."""
    try:
        with output_file.open("w", encoding="utf-8") as md_file:
            for file_path in code_files:
                relative_path = file_path.relative_to(file_path.cwd())
                md_file.write(f"## {relative_path}\n\n")
                md_file.write("```" + file_path.suffix.lstrip(".") + "\n")
                md_file.write(file_path.read_text(encoding="utf-8"))
                md_file.write("\n```\n\n")
    except Exception as e:
        raise RuntimeError(f"Error writing markdown file: {str(e)}")


def create_markdown(
    root_dir: Path,
    extensions: Set[str],
    exclude_files: Set[str],
    exclude_folders: Set[str],
    output_file: Path = Path("code_base.md"),
) -> None:
    """Create a markdown file containing all code files."""
    try:
        code_files, excluded_files = gather_code_files(root_dir, extensions, exclude_files, exclude_folders)
        write_to_markdown(code_files, excluded_files, output_file)
        print(
            f"Markdown file '{output_file}' created with {len(code_files)} code files \
                and {len(excluded_files)} excluded files."
        )
    except Exception as e:
        raise RuntimeError(f"Error creating markdown: {str(e)}")


if __name__ == "__main__":
    root_directory = Path(__file__).parent
    extensions_to_look_for = {".py", ".ipynb", ".txt"}
    exclude_files_list = {".env", "__init__.py", "init.py"}
    exclude_folders_list = {"venv"}

    create_markdown(root_directory, extensions_to_look_for, exclude_files_list, exclude_folders_list)

```

## notebooks/score_review_test.ipynb

```ipynb
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "%reload_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import asyncio\n",
    "import json\n",
    "import nest_asyncio\n",
    "import os\n",
    "import sys\n",
    "from dotenv import load_dotenv\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "sys.path.append('../')\n",
    "from lattereview.providers.openai_provider import OpenAIProvider\n",
    "from lattereview.providers.ollama_provider import OllamaProvider\n",
    "from lattereview.providers.litellm_provider import LiteLLMProvider\n",
    "from lattereview.agents.scoring_reviewer import ScoringReviewer\n",
    "from lattereview.review_workflow import ReviewWorkflow"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setting up the notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Loading environment variables:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sk-cq_M0pNgHhCFnlDOCMnagYA1l2X7Yea5CL0ci5pZMNT3BlbkFJ0m0x9wm5M_EstX5SjLu_kdwGMDYjkUdviNPs4pe9cA\n"
     ]
    }
   ],
   "source": [
    "# Load environment variables from .env file\n",
    "load_dotenv('../.env')\n",
    "print(os.getenv('OPENAI_API_KEY'))\n",
    "\n",
    "# Enable asyncio in Jupyter\n",
    "nest_asyncio.apply()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Loading a dummy dataset:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Title</th>\n",
       "      <th>1st author</th>\n",
       "      <th>repo</th>\n",
       "      <th>year</th>\n",
       "      <th>abstract</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Segmentized quarantine policy for managing a t...</td>\n",
       "      <td>Kim, J.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>By the end of 2021, COVID-19 had spread to ove...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>AutoProteinEngine: A Large Language Model Driv...</td>\n",
       "      <td>Liu, Y.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Protein engineering is important for biomedica...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Integration of Large Vision Language Models fo...</td>\n",
       "      <td>Chen, Z.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Traditional natural disaster response involves...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Choice between Partial Trajectories</td>\n",
       "      <td>Marklund, H.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>As AI agents generate increasingly sophisticat...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Building Altruistic and Moral AI Agent with Br...</td>\n",
       "      <td>Zhao, F.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>As AI closely interacts with human society, it...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID                                              Title    1st author   repo  \\\n",
       "0   1  Segmentized quarantine policy for managing a t...       Kim, J.  arXiv   \n",
       "1   2  AutoProteinEngine: A Large Language Model Driv...       Liu, Y.  arXiv   \n",
       "2   3  Integration of Large Vision Language Models fo...      Chen, Z.  arXiv   \n",
       "3   4                Choice between Partial Trajectories  Marklund, H.  arXiv   \n",
       "4   5  Building Altruistic and Moral AI Agent with Br...      Zhao, F.  arXiv   \n",
       "\n",
       "   year                                           abstract  \n",
       "0  2024  By the end of 2021, COVID-19 had spread to ove...  \n",
       "1  2024  Protein engineering is important for biomedica...  \n",
       "2  2024  Traditional natural disaster response involves...  \n",
       "3  2024  As AI agents generate increasingly sophisticat...  \n",
       "4  2024  As AI closely interacts with human society, it...  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_excel('data.xlsx')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing the base functionalities"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing the OpenAI provider (with OpenAI and Gemini models):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning: model not found. Using cl100k_base encoding.\n",
      "Warning: model not found. Using cl100k_base encoding.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "('The capital of France is Paris.\\n',\n",
       " {'input_cost': 5.25e-07, 'output_cost': 2.1e-06, 'total_cost': 2.625e-06})"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# openanai_provider = OpenAIProvider(model=\"gpt-4o-mini\")\n",
    "openanai_provider = OpenAIProvider(model=\"gemini-1.5-flash\")\n",
    "question = \"What is the capital of France?\"\n",
    "asyncio.run(openanai_provider.get_response(question))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing the Ollama provider:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('The capital of France is Paris!',\n",
       " {'input_cost': 0, 'output_cost': 0, 'total_cost': 0})"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ollama_provider = OllamaProvider(model=\"llama3.2-vision:latest\", host=\"http://localhost:11434\")\n",
    "question = \"What is the capital of France?\"\n",
    "asyncio.run(ollama_provider.get_response(question))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing the LiteLLM provider:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('The capital of France is Paris.\\n', 3.45e-06)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# litellm_provider = LiteLLMProvider(model=\"gpt-4o-mini\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"claude-3-5-sonnet-20240620\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"ollama/llama3.2-vision:latest\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\")\n",
    "litellm_provider = LiteLLMProvider(model=\"gemini/gemini-1.5-flash\")\n",
    "\n",
    "question = \"What is the capital of France?\"\n",
    "asyncio.run(litellm_provider.get_response(question))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing the ScoringReviewer agent:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Inputs:\n",
      "\n",
      " segmentized quarantine policy for managing a tradeoff between containment of infectious disease and social cost of quarantine\n",
      "autoproteinengine: a large language model driven agent framework for multimodal automl in protein engineering\n",
      "integration of large vision language models for efficient post-disaster damage assessment and reporting \n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Reviewing 3 items - 2024-12-14 22:24:13: 100%|██████████| 3/3 [00:07<00:00,  2.43s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Outputs:\n",
      "{'reasoning': 'The title clearly indicates a policy-focused study addressing the balance between disease containment and social impacts through quarantine segmentation, which is a relevant and well-defined research topic.', 'score': 2}\n",
      "{'reasoning': 'The title suggests a novel framework combining LLMs with AutoML for protein engineering, indicating a clear and relevant contribution to automated machine learning applications.', 'score': 2}\n",
      "{'reasoning': 'The title effectively indicates the use of vision-language models for post-disaster assessment, which is a clear and relevant application of AI for disaster management.', 'score': 2}\n",
      "\n",
      "Costs:\n",
      "\n",
      "0.003042\n",
      "0.002931\n",
      "0.002925\n",
      "\n",
      "Total cost:\n",
      "\n",
      "0.002925\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "agent = ScoringReviewer(\n",
    "    # provider=OpenAIProvider(model=\"gpt-4o-mini\"),\n",
    "    # provider=OpenAIProvider(model=\"gemini-1.5-flash\"),\n",
    "    # provider=OllamaProvider(model=\"llama3.2-vision:latest\", host=\"http://localhost:11434\"),\n",
    "    # provider=LiteLLMProvider(model=\"gpt-4o-mini\"),\n",
    "    provider=LiteLLMProvider(model=\"claude-3-5-sonnet-20241022\"),\n",
    "    # provider=LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\"),\n",
    "    # provider=LiteLLMProvider(model=\"gemini/gemini-1.5-flash\"),\n",
    "    name=\"Pouria\",\n",
    "    max_concurrent_requests=1, \n",
    "    backstory=\"an expert reviewer and researcher!\",\n",
    "    input_description = \"article title\",\n",
    "    model_args={\"max_tokens\": 100, \"temperature\": 0.1},\n",
    "    reasoning = \"brief\",\n",
    "    review_criteria=\"Look for articles that certainly do not employ any AI or machine learning agents\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules='Score 1 if the paper does not meet the criteria, and 2 if the paper meets the criteria.',\n",
    ")\n",
    "\n",
    "\n",
    "# Dummy input\n",
    "text_list = data.Title.str.lower().tolist()\n",
    "print(\"Inputs:\\n\\n\", '\\n'.join(text_list[:3]), \"\\n\\n\")\n",
    "\n",
    "# Dummy review\n",
    "results, total_cost = asyncio.run(agent.review_items(text_list[:3]))\n",
    "print(\"Outputs:\")\n",
    "for result in results:\n",
    "    print(result)\n",
    "\n",
    "# Dummy costs\n",
    "print(\"\\nCosts:\\n\")\n",
    "for item in agent.memory:\n",
    "    print(item['cost'])\n",
    "\n",
    "print(\"\\nTotal cost:\\n\")\n",
    "print(total_cost)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing the main Functionalities"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### A multiagent review workflow for doing title/abstract analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Setting up the agents:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "pouria = ScoringReviewer(\n",
    "    # provider=OpenAIProvider(model=\"gemini-1.5-flash\"),\n",
    "    # provider=OllamaProvider(model=\"llama3.2-vision:latest\", host=\"http://localhost:11434\"),\n",
    "    # provider=LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\"),\n",
    "    # provider=LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\"),\n",
    "    provider=LiteLLMProvider(model=\"gemini/gemini-1.5-flash\"),\n",
    "    name=\"Pouria\",\n",
    "    max_concurrent_requests=20, \n",
    "    backstory=\"a radiologist with many years of background in statistcis and data science, who are famous among your colleagues for your systematic thinking, organizaton of thoughts, and being conservative\",\n",
    "    model_args={\"max_tokens\": 100, \"temperature\": 0.1},\n",
    "    input_description = \"tilte and abstract of scientific articles\",\n",
    "    reasoning = \"cot\",\n",
    "    scoring_task=\"Look for articles that disucss large languange models-based AI agents applied to medical imaging data\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules='Score 1 if the paper meets the criteria, and 2 if the paper does not meet the criteria.',\n",
    ")\n",
    "\n",
    "bardia = ScoringReviewer(\n",
    "    provider=OpenAIProvider(model=\"gpt-4o-mini\"),\n",
    "    name=\"Bardia\",\n",
    "    max_concurrent_requests=20, \n",
    "    backstory=\"an expert in data science with a background in developing ML models for healthcare, who are famous among your colleagues for your creativity and out of the box thinking\",\n",
    "    model_args={\"max_tokens\": 100, \"temperature\": 0.8},\n",
    "    input_description = \"tilte and abstract of scientific articles\",\n",
    "    reasoning = \"brief\",\n",
    "    scoring_task=\"Look for articles that disucss large languange models-based AI agents applied to medical imaging data\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules='Score 1 if the paper meets the criteria, and 2 if the paper does not meet the criteria.',\n",
    ")\n",
    "\n",
    "brad = ScoringReviewer(\n",
    "    provider=OpenAIProvider(model=\"gpt-4o\"),\n",
    "    name=\"Brad\",\n",
    "    max_concurrent_requests=20, \n",
    "    backstory=\"a senior radiologist with a PhD in computer science and years of experience as the director of a DL lab focused on developing ML models for radiology and healthcare\",\n",
    "    input_description = \"tilte and abstract of scientific articles\",\n",
    "    temperature=0.4,\n",
    "    reasoning = \"cot\",\n",
    "    max_tokens=100,\n",
    "    scoring_task=\"\"\"Pouria and Bardia have Looked for articles that disucss large languange models-based AI agents applied to medical imaging data. \n",
    "                       They scored an article 1 if they thought it does not meet this criteria, 2 if they thought it meets the criteria, 0 if they were uncertain of scoring.\n",
    "                       You will receive an article they have had different opinions about, as well as each of their scores and their reasoning for that score. Read their reviews and determine who you agree with. \n",
    "                    \"\"\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules=\"\"\"Score 1 if you agree with Pouria, and score 2 if you agree with Bardia.\"\"\",\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Setting up the review workflow:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "title_abs_review = ReviewWorkflow(\n",
    "    workflow_schema=[\n",
    "        {\n",
    "            \"round\": 'A',\n",
    "            \"reviewers\": [pouria, bardia],\n",
    "            \"inputs\": [\"Title\", \"abstract\"]\n",
    "        },\n",
    "        {\n",
    "            \"round\": 'B',\n",
    "            \"reviewers\": [brad],\n",
    "            \"inputs\": [\"Title\", \"abstract\", \"round-A_Pouria_output\", \"round-A_Bardia_output\"],\n",
    "            \"filter\": lambda row: row[\"round-A_Pouria_output\"][\"score\"] != row[\"round-A_Bardia_output\"][\"score\"]\n",
    "        }\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Applying the review workflow to a number of sample articles:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Starting review round A (1/2)...\n",
      "Processing 10 eligible rows\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "['round: A', 'reviewer_name: Pouria'] -                     2024-12-14 22:24:20: 100%|██████████| 10/10 [00:01<00:00,  8.03it/s]\n",
      "['round: A', 'reviewer_name: Bardia'] -                     2024-12-14 22:24:22: 100%|██████████| 10/10 [00:01<00:00,  6.55it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Starting review round B (2/2)...\n",
      "Skipping review round B - no eligible rows\n",
      "Total cost: \n",
      "0.000132\n",
      "\n",
      "Detailed cost:\n",
      "{('A', 'Pouria'): 5.01e-05, ('A', 'Bardia'): 8.19e-05}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Title</th>\n",
       "      <th>1st author</th>\n",
       "      <th>repo</th>\n",
       "      <th>year</th>\n",
       "      <th>abstract</th>\n",
       "      <th>round-A_Pouria_output</th>\n",
       "      <th>round-A_Pouria_score</th>\n",
       "      <th>round-A_Pouria_reasoning</th>\n",
       "      <th>round-A_Bardia_output</th>\n",
       "      <th>round-A_Bardia_score</th>\n",
       "      <th>round-A_Bardia_reasoning</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>19</td>\n",
       "      <td>JAILJUDGE: A COMPREHENSIVE JAILBREAK JUDGE BEN...</td>\n",
       "      <td>Liu, F.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Although significant research efforts have bee...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention m...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention medical imaging ...</td>\n",
       "      <td>{'reasoning': 'The article focuses on evaluati...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on evaluating the safety a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>85</td>\n",
       "      <td>An Autonomous GIS Agent Framework for Geospati...</td>\n",
       "      <td>Ning, H.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Powered by the emerging large language models ...</td>\n",
       "      <td>{'reasoning': 'The article discusses an autono...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses an autonomous GIS agent ...</td>\n",
       "      <td>{'reasoning': 'The paper discusses an autonomo...</td>\n",
       "      <td>2</td>\n",
       "      <td>The paper discusses an autonomous GIS agent fr...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>434</td>\n",
       "      <td>Bifurcation theory captures band formation in ...</td>\n",
       "      <td>Trenado, C.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>Collective behavior occurs ubiquitously in nat...</td>\n",
       "      <td>{'reasoning': 'The provided abstract does not ...</td>\n",
       "      <td>2</td>\n",
       "      <td>The provided abstract does not discuss large l...</td>\n",
       "      <td>{'reasoning': 'This article does not discuss l...</td>\n",
       "      <td>2</td>\n",
       "      <td>This article does not discuss large language m...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>191</td>\n",
       "      <td>A Call for Embodied AI</td>\n",
       "      <td>Paolo, G.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>We propose Embodied AI (E-AI) as the next fund...</td>\n",
       "      <td>{'reasoning': 'The abstract mentions Large Lan...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract mentions Large Language Models (L...</td>\n",
       "      <td>{'reasoning': 'The article focuses on Embodied...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on Embodied AI and its the...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>524</td>\n",
       "      <td>Learning from zero: how to make consumption-sa...</td>\n",
       "      <td>Shi, R.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2021</td>\n",
       "      <td>This exercise proposes a learning mechanism to...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article focuses on economic...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on economic decision-makin...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>251</td>\n",
       "      <td>Individual Variation Affects Outbreak Magnitud...</td>\n",
       "      <td>Lazebnik, T.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>Zoonotic disease transmission between animals ...</td>\n",
       "      <td>{'reasoning': 'The article focuses on a multi-...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on a multi-pathogen model ...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>707</td>\n",
       "      <td>Agent-based modeling predicts HDL-independent ...</td>\n",
       "      <td>Paalvast, Y.</td>\n",
       "      <td>bioRxiv</td>\n",
       "      <td>2018</td>\n",
       "      <td>A hallmark of the metabolic syndrome is low HD...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article discusses an agent-...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses an agent-based modeling ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>231</td>\n",
       "      <td>Moderate confirmation bias enhances collective...</td>\n",
       "      <td>Bergerot, C.</td>\n",
       "      <td>bioRxiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>Humans tend to give more weight to information...</td>\n",
       "      <td>{'reasoning': 'The provided abstract does not ...</td>\n",
       "      <td>2</td>\n",
       "      <td>The provided abstract does not discuss large l...</td>\n",
       "      <td>{'reasoning': 'The article discusses reinforce...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses reinforcement-learning a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>755</td>\n",
       "      <td>SPEW: Synthetic populations and ecosystems of ...</td>\n",
       "      <td>Gallagher, S.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2017</td>\n",
       "      <td>Agent-based models (ABMs) simulate interaction...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The article discusses agent-bas...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses agent-based models for s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>362</td>\n",
       "      <td>Neural Network Augmented Compartmental Pandemi...</td>\n",
       "      <td>Kummer, L.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>Compartmental models are a tool commonly used ...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article focuses on neural n...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on neural network augmente...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID                                              Title     1st author  \\\n",
       "0   19  JAILJUDGE: A COMPREHENSIVE JAILBREAK JUDGE BEN...        Liu, F.   \n",
       "1   85  An Autonomous GIS Agent Framework for Geospati...       Ning, H.   \n",
       "2  434  Bifurcation theory captures band formation in ...    Trenado, C.   \n",
       "3  191                             A Call for Embodied AI      Paolo, G.   \n",
       "4  524  Learning from zero: how to make consumption-sa...        Shi, R.   \n",
       "5  251  Individual Variation Affects Outbreak Magnitud...   Lazebnik, T.   \n",
       "6  707  Agent-based modeling predicts HDL-independent ...   Paalvast, Y.   \n",
       "7  231  Moderate confirmation bias enhances collective...   Bergerot, C.   \n",
       "8  755  SPEW: Synthetic populations and ecosystems of ...  Gallagher, S.   \n",
       "9  362  Neural Network Augmented Compartmental Pandemi...     Kummer, L.   \n",
       "\n",
       "      repo  year                                           abstract  \\\n",
       "0    arXiv  2024  Although significant research efforts have bee...   \n",
       "1    arXiv  2024  Powered by the emerging large language models ...   \n",
       "2    arXiv  2022  Collective behavior occurs ubiquitously in nat...   \n",
       "3    arXiv  2024  We propose Embodied AI (E-AI) as the next fund...   \n",
       "4    arXiv  2021  This exercise proposes a learning mechanism to...   \n",
       "5    arXiv  2023  Zoonotic disease transmission between animals ...   \n",
       "6  bioRxiv  2018  A hallmark of the metabolic syndrome is low HD...   \n",
       "7  bioRxiv  2023  Humans tend to give more weight to information...   \n",
       "8    arXiv  2017  Agent-based models (ABMs) simulate interaction...   \n",
       "9    arXiv  2022  Compartmental models are a tool commonly used ...   \n",
       "\n",
       "                               round-A_Pouria_output round-A_Pouria_score  \\\n",
       "0  {'reasoning': 'The abstract does not mention m...                    2   \n",
       "1  {'reasoning': 'The article discusses an autono...                    2   \n",
       "2  {'reasoning': 'The provided abstract does not ...                    2   \n",
       "3  {'reasoning': 'The abstract mentions Large Lan...                    2   \n",
       "4  {'reasoning': 'The article does not discuss la...                    2   \n",
       "5  {'reasoning': 'The article focuses on a multi-...                    2   \n",
       "6  {'reasoning': 'The article does not discuss la...                    2   \n",
       "7  {'reasoning': 'The provided abstract does not ...                    2   \n",
       "8  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "9  {'reasoning': 'The article does not discuss la...                    2   \n",
       "\n",
       "                            round-A_Pouria_reasoning  \\\n",
       "0  The abstract does not mention medical imaging ...   \n",
       "1  The article discusses an autonomous GIS agent ...   \n",
       "2  The provided abstract does not discuss large l...   \n",
       "3  The abstract mentions Large Language Models (L...   \n",
       "4  The article does not discuss large language mo...   \n",
       "5  The article focuses on a multi-pathogen model ...   \n",
       "6  The article does not discuss large language mo...   \n",
       "7  The provided abstract does not discuss large l...   \n",
       "8  The abstract does not mention large language m...   \n",
       "9  The article does not discuss large language mo...   \n",
       "\n",
       "                               round-A_Bardia_output round-A_Bardia_score  \\\n",
       "0  {'reasoning': 'The article focuses on evaluati...                    2   \n",
       "1  {'reasoning': 'The paper discusses an autonomo...                    2   \n",
       "2  {'reasoning': 'This article does not discuss l...                    2   \n",
       "3  {'reasoning': 'The article focuses on Embodied...                    2   \n",
       "4  {'reasoning': 'The article focuses on economic...                    2   \n",
       "5  {'reasoning': 'The article does not discuss la...                    2   \n",
       "6  {'reasoning': 'The article discusses an agent-...                    2   \n",
       "7  {'reasoning': 'The article discusses reinforce...                    2   \n",
       "8  {'reasoning': 'The article discusses agent-bas...                    2   \n",
       "9  {'reasoning': 'The article focuses on neural n...                    2   \n",
       "\n",
       "                            round-A_Bardia_reasoning  \n",
       "0  The article focuses on evaluating the safety a...  \n",
       "1  The paper discusses an autonomous GIS agent fr...  \n",
       "2  This article does not discuss large language m...  \n",
       "3  The article focuses on Embodied AI and its the...  \n",
       "4  The article focuses on economic decision-makin...  \n",
       "5  The article does not discuss large language mo...  \n",
       "6  The article discusses an agent-based modeling ...  \n",
       "7  The article discusses reinforcement-learning a...  \n",
       "8  The article discusses agent-based models for s...  \n",
       "9  The article focuses on neural network augmente...  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Reload the data if needed.\n",
    "sample_data = pd.read_excel('data.xlsx').sample(10).reset_index(drop=True)\n",
    "updated_data = asyncio.run(title_abs_review(sample_data))\n",
    "\n",
    "print(\"Total cost: \")\n",
    "print(title_abs_review.get_total_cost())\n",
    "\n",
    "print(\"\\nDetailed cost:\")\n",
    "print(title_abs_review.reviewer_costs)\n",
    "\n",
    "updated_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "        Title: JAILJUDGE: A COMPREHENSIVE JAILBREAK JUDGE BENCHMARK WITH MULTI-AGENT ENHANCED EXPLANATION EVALUATION FRAMEWORK\n",
      "        Abstract: Although significant research efforts have been dedicated to enhancing the safety of large language models (LLMs) by understanding and defending against jailbreak attacks, evaluating the defense capabilities of LLMs against jailbreak attacks also attracts lots of attention. Current evaluation methods lack explainability and do not generalize well to complex scenarios, resulting in incomplete and inaccurate assessments (e.g., direct judgment without reasoning explainability, the F1 score of the GPT-4 judge is only 55% in complex scenarios and bias evaluation on multilingual scenarios, etc.). To address these challenges, we have developed a comprehensive evaluation benchmark, JAILJUDGE, which includes a wide range of risk scenarios with complex malicious prompts (e.g., synthetic, adversarial, in-the-wild, and multi-language scenarios, etc.) along with high-quality humanannotated test datasets. Specifically, the JAILJUDGE dataset comprises training data of JAILJUDGE, with over 35k+ instruction-tune training data with reasoning explainability, and JAILJUDGETEST, a 4.5k+ labeled set of broad risk scenarios and a 6k+ labeled set of multilingual scenarios in ten languages. To provide reasoning explanations (e.g., explaining why an LLM is jailbroken or not) and fine-grained evaluations (jailbroken score from 1 to 10), we propose a multi-agent jailbreak judge framework, JailJudge MultiAgent, making the decision inference process explicit and interpretable to enhance evaluation quality. Using this framework, we construct the instruction-tuning ground truth and then instruction-tune an end-to-end jailbreak judge model, JAILJUDGE Guard, which can also provide reasoning explainability with fine-grained evaluations without API costs. Additionally, we introduce JailBoost, an attacker-agnostic attack enhancer, and GuardShield, a safety moderation defense method, both based on JAILJUDGE Guard. Comprehensive experiments demonstrate the superiority of our JAILJUDGE benchmark and jailbreak judge methods. Our jailbreak judge methods (JailJudge MultiAgent and JAILJUDGE Guard) achieve SOTA performance in closed-source models (e.g., GPT-4) and safety moderation models (e.g., Llama-Guard and ShieldGemma, etc.), across a broad range of complex behaviors (e.g., JAILJUDGE benchmark, etc.) to zero-shot scenarios (e.g., other open data, etc.). Importantly, JailBoost and Guard- Shield, based on JAILJUDGE Guard, can enhance downstream tasks in jailbreak attacks and defenses under zero-shot settings with significant improvement (e.g., JailBoost can increase the average performance by approximately 29.24%, while GuardShield can reduce the average defense ASR from 40.46% to 0.15%).\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention medical imaging or its applications.  The focus is on large language models and their safety, specifically concerning jailbreak attacks. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on evaluating the safety and performance of large language models in the context of jailbreak attacks, without any reference to medical imaging data or applications in that field, thus it does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: An Autonomous GIS Agent Framework for Geospatial Data Retrieval\n",
      "        Abstract: Powered by the emerging large language models (LLMs), autonomous geographic information systems (GIS) agents have the potential to accomplish spatial analyses and cartographic tasks. However, a research gap exists to support fully autonomous GIS agents: how to enable agents to discover and download the necessary data for geospatial analyses. This study proposes an autonomous GIS agent framework capable of retrieving required geospatial data by generating, executing, and debugging programs. The framework utilizes the LLM as the decision-maker, selects the appropriate data source (s) from a pre-defined source list, and fetches the data from the chosen source. Each data source has a handbook that records the metadata and technical details for data retrieval. The proposed framework is designed in a plug-and-play style to ensure flexibility and extensibility. Human users or autonomous data scrawlers can add new data sources by adding new handbooks. We developed a prototype agent based on the framework, released as a QGIS plugin (GeoData Retrieve Agent) and a Python program. Experiment results demonstrate its capability of retrieving data from various sources including OpenStreetMap, administrative boundaries and demographic data from the US Census Bureau, satellite basemaps from ESRI World Imagery, global digital elevation model (DEM) from OpenTopography.org, weather data from a commercial provider, the COVID-19 cases from the NYTimes GitHub. Our study is among the first attempts to develop an autonomous geospatial data retrieval agent.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article discusses an autonomous GIS agent framework powered by large language models (LLMs).  However, this framework is applied to geospatial data, not medical imaging data. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The paper discusses an autonomous GIS agent framework that utilizes large language models, but it focuses on geospatial data retrieval rather than medical imaging data, which does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Bifurcation theory captures band formation in the Vicsek model of flock formation\n",
      "        Abstract: Collective behavior occurs ubiquitously in nature and it plays a key role in bacterial colonies, mammalian cells or flocks of birds. Here, we examine the average density and velocity of self-propelled particles, which are described by a system of partial differential equations near the flocking transition of the Vicsek model. This agent-based model illustrates the trend towards flock formation of animals that align their velocities to an average of those of their neighbors. Near the flocking transition, particle density and velocity obey partial differential equations that include a parameter ε measuring the distance to the bifurcation point. We have obtained analytically the Riemann invariants in one and two spatial dimensions for the hyperbolic (ε = 0) and parabolic (ε 6= 0) system and, under periodic initial-boundary value conditions, we show that the solutions include wave trains. Additionally, we have found wave trains having oscillation frequencies that agree with those predicted by a linearization approximation and that may propagate at angles depending on the initial condition. The wave amplitudes increase with time for the hyperbolic system but are stabilized to finite values for the parabolic system. To integrate the partial differential equations, we design a basic numerical scheme which is first order in time and space. To mitigate numerical dissipation and ensure good resolution of the wave features, we also use a high order accurate WENO5 reconstruction procedure in space and a third order accurate Runge-Kutta scheme in time. Comparisons with direct simulations of the Vicsek model confirm these predictions.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The provided abstract does not discuss large language models, AI agents, or medical imaging.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: This article does not discuss large language models or their application to medical imaging data, but rather focuses on collective behavior in self-propelled particle systems.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: A Call for Embodied AI\n",
      "        Abstract: We propose Embodied AI (E-AI) as the next fundamental step in the pursuit of Artificial General Intelligence (AGI), juxtaposing it against current AI advancements, particularly Large Language Models (LLMs). We traverse the evolution of the embodiment concept across diverse fields (philosophy, psychology, neuroscience, and robotics) to highlight how E-AI distinguishes itself from the classical paradigm of static learning. By broadening the scope of E-AI, we introduce a theoretical framework based on cognitive architectures, emphasizing perception, action, memory, and learning as essential components of an embodied agent. This framework is aligned with Friston’s active inference principle, offering a comprehensive approach to E-AI development. Despite the progress made in the field of AI, substantial challenges, such as the formulation of a novel AI learning theory and the innovation of advanced hardware, persist. Our discussion lays down a foundational guideline for future E-AI research. Highlighting the importance of creating E-AI agents capable of seamless communication, collaboration, and coexistence with humans and other intelligent entities within real-world environments, we aim to steer the AI community towards addressing the multifaceted challenges and seizing the opportunities that lie ahead in the quest for AGI.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract mentions Large Language Models (LLMs), but the focus of the paper is on Embodied AI, a broader concept that does not directly address the application of LLMs to medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on Embodied AI and its theoretical framework rather than specifically addressing large language models applied to medical imaging data, making it not relevant to the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Learning from zero: how to make consumption-saving decisions in a stochastic environment with an AI algorithm\n",
      "        Abstract: This exercise proposes a learning mechanism to model economic agent’s decision-making process using an actor-critic structure in the literature of artificial intelligence It is motivated by the psychology literature of learning through reinforcing good or bad decisions In a model of an environment, to learn to make decisions, this AI agent needs to interact with its environment and make explorative actions Each action in a given state brings a reward signal to the agent These interactive experience is saved in the agent’s memory, which is then used to update its subjective belief of the world The agent’s decision-making strategy is formed and adjusted based on this evolving subjective belief This agent does not only take an action that it knows would bring a high reward, it also explores other possibilities This is the process of taking explorative actions, and it ensures that the agent notices changes in its environment and adapt its subjective belief and decisions accordingly Through a model of stochastic optimal growth, I illustrate that the economic agent under this proposed learning structure is adaptive to changes in an underlying stochastic process of the economy AI agents can differ in their levels of exploration, which leads to different experience in the same environment This reflects on to their different learning behaviours and welfare obtained The chosen economic structure possesses the fundamental decision making problems of macroeconomic models, i.e., how to make consumption-saving decisions in a lifetime, and it can be generalised to other decision-making processes and economic models.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  The abstract describes an AI agent used for economic modeling, not medical imaging. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on economic decision-making and does not discuss large language models or their application to medical imaging data, which is the specified criterion.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Individual Variation Affects Outbreak Magnitude and Predictability in an Extended Multi-Pathogen SIR Model of Pigeons Vising Dairy Farms\n",
      "        Abstract: Zoonotic disease transmission between animals and humans is a growing risk and the agricultural context acts as a likely point of transition, with individual heterogeneity acting as an important contributor. Livestock often occurs at high local densities, facilitating spread within sites (e.g. among cows in a dairy farm), while wildlife is often more mobile, potentially connecting spatially isolated sites. Thus, understanding the dynamics of disease spread in the wildlife-livestock interface is crucial for mitigating these risks of transmission. Specifically, the interactions between pigeons and in-door cows at dairy farms can lead to significant disease transmission and economic losses for farmers; putting livestock, adjacent human populations, and other wildlife species at risk. In this paper, we propose a novel spatio-temporal multi-pathogen model with continuous spatial movement. The model expands on the Susceptible-Exposed-Infected-Recovered-Dead (SEIRD) framework and accounts for both within-species and cross-species transmission of pathogens, as well as the exploration-exploitation movement dynamics of pigeons, which play a critical role in the spread of infection agents. In addition to model formulation, we also implement it as an agent-based simulation approach and use empirical field data to investigate different biologically realistic scenarios, evaluating the effect of various parameters on the epidemic spread. Namely, in agreement with theoretical expectations, the model predicts that the heterogeneity of the pigeons’ movement dynamics can drastically affect both the magnitude and stability of outbreaks. In addition, joint infection by multiple pathogens can have an interactive effect unobservable in single-pathogen SIR models, reflecting a non-intuitive inhibition of the outbreak. Our findings highlight the impact of heterogeneity in host behavior on their pathogens and allow realistic predictions of outbreak dynamics in the multi-pathogen wildlife-livestock interface with consequences to zoonotic diseases in various systems.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article focuses on a multi-pathogen model for disease transmission in pigeons and cows, using an agent-based simulation.  It does not involve large language models or AI agents applied to medical imaging data. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data, instead focusing on disease transmission modeling in livestock and wildlife interactions.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Agent-based modeling predicts HDL-independent pathway of removal of excess surface lipids from very low density lipoprotein\n",
      "        Abstract: A hallmark of the metabolic syndrome is low HDL-cholesterol coupled with high plasma triglycerides (TG), but it is unclear what drives this close association. Plasma triglycerides and HDL cholesterol are thought to communicate through two distinct mechanisms. Firstly, excess surface lipids from VLDL released during lipolysis are transferred to HDL, thereby contributing to HDL directly but also indirectly through providing substrate for LCAT. Secondly, high plasma TG increases clearance of HDL through core-lipid exchange between VLDL and HDL via CETP and subsequent hydrolysis of the TG in HDL, resulting in smaller HDL and thus increased clearance rates. To test our understanding of how high plasma TG induces low HDL-cholesterol, making use of established knowledge, we developed a comprehensive agent-based model of lipoprotein metabolism which was validated using monogenic disorders of lipoprotein metabolism. By perturbing plasma TG in the model, we tested whether the current theoretical framework reproduces experimental findings. Interestingly, while increasing plasma TG through simulating decreased lipolysis of VLDL resulted in the expected decrease in HDL cholesterol, perturbing plasma TG through simulating increased VLDL production rates did not result in the expected HDL-TG relation at physiological lipid fluxes. However, model perturbations and experimental findings can be reconciled if we assume a pathway removing excess surface-lipid from VLDL that does not contribute to HDL cholesterol ester production through LCAT. In conclusion, our model simulations suggest that excess surface lipid from VLDL is cleared in part independently from HDL.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  The abstract describes an agent-based model, but this is a different type of agent model used in systems biology and not related to AI agents. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses an agent-based modeling approach for lipoprotein metabolism but does not mention large language models or their application to medical imaging data, hence it does not meet the criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Moderate confirmation bias enhances collective decision-making in reinforcement-learning agents\n",
      "        Abstract: Humans tend to give more weight to information confirming their beliefs than to information that disconfirms them. Nevertheless, this apparent irrationality has been shown to improve individual decision-making under uncertainty. However, little is known about this bias’ impact on collective decision-making. Here, we investigate the conditions under which confirmation bias is beneficial or detrimental to collective decision-making. To do so, we develop a Collective Asymmetric Reinforcement Learning (CARL) model in which artificial agents observe others’ actions and rewards, and update this information asymmetrically. We use agent-based simulations to study how confirmation bias affects collective performance on a two-armed bandit task, and how resource scarcity, group size and bias strength modulate this effect. We find that a confirmation bias benefits group learning across a wide range of resource-scarcity conditions. Moreover, we discover that, past a critical bias strength, resource abundance favors the emergence of two different performance regimes, one of which is suboptimal. In addition, we find that this regime bifurcation comes with polarization in small groups of agents. Overall, our results suggest the existence of an optimal, moderate level of confirmation bias for collective decision-making.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The provided abstract does not discuss large language models or AI agents applied to medical imaging data.  The research focuses on reinforcement learning agents and collective decision-making, a different area of AI.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses reinforcement-learning agents and confirmation bias, but it does not focus on large language models or their application in medical imaging data, therefore it does not meet the criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: SPEW: Synthetic populations and ecosystems of the world\n",
      "        Abstract: Agent-based models (ABMs) simulate interactions between autonomous agents in constrained environments over time. ABMs are often used for modeling the spread of infectious diseases. In order to simulate disease outbreaks or other phenomena, ABMs rely on “synthetic ecosystems,” or information about agents and their environments that is representative of the real world. Previous approaches for generating synthetic ecosystems have some limitations: they are not open-source, cannot be adapted to new or updated input data sources, and do not allow for alternative methods for sampling agent characteristics and locations. We introduce a general framework for generating Synthetic Populations and Ecosystems of the World (SPEW), implemented as an open-source R package. SPEW allows researchers to choose from a variety of sampling methods for agent characteristics and locations when generating synthetic ecosystems for any geographic region. SPEW can produce synthetic ecosystems for any agent (e.g. humans, mosquitoes, etc), provided that appropriate data is available. We analyze the accuracy and computational efficiency of SPEW given different sampling methods for agent characteristics and locations and provide a suite of diagnostics to screen our synthetic ecosystems. SPEW has generated over five billion human agents across approximately 100,000 geographic regions in about 70 countries, available online.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models, AI agents, or medical imaging.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses agent-based models for simulating interactions in ecosystems, but it does not address large language models or their application to medical imaging data, making it unrelated to the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Neural Network Augmented Compartmental Pandemic Models\n",
      "        Abstract: Compartmental models are a tool commonly used in epidemiology for the mathematical modelling of the spread of infectious diseases, with their most popular representative being the Susceptible-Infected-Removed (SIR) model and its derivatives. However, current SIR models are bounded in their capabilities to model government policies in the form of non-pharmaceutical interventions (NPIs) and weather effects and offer limited predictive power. More capable alternatives such as agent based models (ABMs) are computationally expensive and require specialized hardware. We introduce a neural network augmented SIR model that can be run on commodity hardware, takes NPIs and weather effects into account and offers improved predictive power as well as counterfactual analysis capabilities. We demonstrate our models improvement of the state-of-the-art modeling COVID-19 in Austria during the 03.2020 to 03.2021 period and provide an outlook for the future up to 01.2024.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on neural network augmented compartmental models for epidemiology rather than large language models applied to medical imaging data, making it irrelevant to the criteria specified.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n"
     ]
    }
   ],
   "source": [
    "for i, row in updated_data.iterrows():\n",
    "    print(\n",
    "        f\"\"\"\n",
    "        Title: {row.Title}\n",
    "        Abstract: {row.abstract}\n",
    "        Pouria's score: {row[\"round-A_Pouria_score\"]}\n",
    "        Pouria's reasoning: {row[\"round-A_Pouria_reasoning\"]}\n",
    "        Bardia's score: {row[\"round-A_Bardia_score\"]}\n",
    "        Bardia's reasoning: {row[\"round-A_Bardia_reasoning\"]}\n",
    "        Brad's score: {None if \"round-B_Brad_score\" not in row else row[\"round-B_Brad_score\"]}\n",
    "        Brad's reasoning: {None if \"round-B_Brad_reasoning\" not in row else row[\"round-B_Brad_reasoning\"]}\n",
    "        \"\"\"\n",
    "    )"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

```

## lattereview/review_workflow.py

```py
import pydantic
from typing import List, Dict, Any, Union
import pandas as pd
import json
import hashlib

from .agents.scoring_reviewer import ScoringReviewer


class ReviewWorkflowError(Exception):
    """Base exception for workflow-related errors."""

    pass


class ReviewWorkflow(pydantic.BaseModel):
    workflow_schema: List[Dict[str, Any]]
    memory: List[Dict] = list()
    reviewer_costs: Dict = dict()
    total_cost: float = 0.0
    verbose: bool = True

    def __post_init__(self, __context):
        """Initialize after Pydantic model initialization."""
        try:
            for review_task in self.workflow_schema:
                round_id = review_task["round"]
                reviewers = (
                    review_task["reviewers"]
                    if isinstance(review_task["reviewers"], list)
                    else [review_task["reviewers"]]
                )
                reviewer_names = [f"round-{round_id}_{reviewer.name}" for reviewer in reviewers]
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                initial_inputs = [col for col in inputs if "_output_" not in col]

                # Validate reviewers
                for reviewer in reviewers:
                    if not isinstance(reviewer, ScoringReviewer):
                        raise ReviewWorkflowError(f"Invalid reviewer: {reviewer}")

                # Validate input columns
                for input_col in initial_inputs:
                    if input_col not in __context["data"].columns:
                        if input_col.split("_output")[0] not in reviewer_names:
                            raise ReviewWorkflowError(f"Invalid input column: {input_col}")
        except Exception as e:
            raise ReviewWorkflowError(f"Error initializing Review Workflow: {e}")

    async def __call__(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """Run the workflow."""
        try:
            if isinstance(data, pd.DataFrame):
                return await self.run(data)
            elif isinstance(data, dict):
                return await self.run(pd.DataFrame(data))
            else:
                raise ReviewWorkflowError(f"Invalid data type: {type(data)}")
        except Exception as e:
            raise ReviewWorkflowError(f"Error running workflow: {e}")

    def _create_content_hash(self, content: str) -> str:
        """Create a hash of the content for tracking."""
        return hashlib.md5(content.encode()).hexdigest()

    def _format_input_text(self, row: pd.Series, inputs: List[str]) -> tuple:
        """Format input text with content tracking."""
        parts = []
        content_keys = []

        for input_col in inputs:
            if "_output_" not in input_col:
                value = str(row[input_col]).strip()
                parts.append(f"=== {input_col} ===\n{value}")
                content_keys.append(self._create_content_hash(value))

        return "\n\n".join(parts), "-".join(content_keys)

    async def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the review process with content validation."""
        try:
            df = data.copy()
            total_rounds = len(self.workflow_schema)

            for review_round, review_task in enumerate(self.workflow_schema):
                round_id = review_task["round"]
                self._log(f"\nStarting review round {round_id} ({review_round + 1}/{total_rounds})...")

                reviewers = (
                    review_task["reviewers"]
                    if isinstance(review_task["reviewers"], list)
                    else [review_task["reviewers"]]
                )
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                filter_func = review_task.get("filter", lambda x: True)

                # Apply filter and get eligible rows
                mask = df.apply(filter_func, axis=1)
                if not mask.any():
                    self._log(f"Skipping review round {round_id} - no eligible rows")
                    continue

                self._log(f"Processing {mask.sum()} eligible rows")

                # Create input items with content tracking
                input_items = []
                input_hashes = []
                eligible_indices = []

                for idx in df[mask].index:
                    row = df.loc[idx]
                    input_text, content_hash = self._format_input_text(row, inputs)

                    # Add metadata header
                    input_text = (
                        f"Review Task ID: {round_id}-{idx}\n" f"Content Hash: {content_hash}\n\n" f"{input_text}"
                    )

                    input_items.append(input_text)
                    input_hashes.append(content_hash)
                    eligible_indices.append(idx)

                # Process each reviewer
                for reviewer in reviewers:
                    output_col = f"round-{round_id}_{reviewer.name}_output"
                    score_col = f"round-{round_id}_{reviewer.name}_score"
                    reasoning_col = f"round-{round_id}_{reviewer.name}_reasoning"

                    # Initialize the output column if it doesn't exist
                    if output_col not in df.columns:
                        df[output_col] = None
                    if score_col not in df.columns:
                        df[score_col] = None
                    if reasoning_col not in df.columns:
                        df[reasoning_col] = None

                    # Get reviewer outputs with metadata
                    outputs, review_cost = await reviewer.review_items(
                        input_items,
                        {
                            "round": round_id,
                            "reviewer_name": reviewer.name,
                        },
                    )
                    self.reviewer_costs[(round_id, reviewer.name)] = review_cost

                    # Verify output count
                    if len(outputs) != len(eligible_indices):
                        raise ReviewWorkflowError(
                            f"Reviewer {reviewer.name} returned {len(outputs)} outputs "
                            f"for {len(eligible_indices)} inputs"
                        )

                    # Process outputs with content validation
                    processed_outputs = []
                    processed_scores = []
                    processed_reasoning = []

                    for output, expected_hash in zip(outputs, input_hashes):
                        try:
                            if isinstance(output, dict):
                                processed_output = output
                            else:
                                processed_output = json.loads(output)

                            # Add content hash to output for validation
                            processed_output["_content_hash"] = expected_hash
                            processed_outputs.append(processed_output)

                            if "score" in processed_output:
                                processed_scores.append(processed_output["score"])

                            if "reasoning" in processed_output:
                                processed_reasoning.append(processed_output["reasoning"])

                        except Exception as e:
                            self._log(f"Warning: Error processing output: {e}")
                            processed_outputs.append({"reasoning": None, "score": None, "_content_hash": expected_hash})

                    # Update dataframe with validated outputs
                    output_dict = dict(zip(eligible_indices, processed_outputs))
                    df.loc[eligible_indices, output_col] = pd.Series(output_dict)

                    score_dict = dict(zip(eligible_indices, processed_scores))
                    df.loc[eligible_indices, score_col] = pd.Series(score_dict)

                    reasoning_dict = dict(zip(eligible_indices, processed_reasoning))
                    df.loc[eligible_indices, reasoning_col] = pd.Series(reasoning_dict)

            return df

        except Exception as e:
            raise ReviewWorkflowError(f"Error running workflow: {e}")

    def _log(self, x):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(x)

    def get_total_cost(self) -> float:
        """Return the total cost of the review process."""
        return sum(self.reviewer_costs.values())

```

## lattereview/generic_prompts/review_prompt.txt

```txt
Review the input item below and evaluate it against the following criteria:

Scoring task: <<${scoring_task}$>>

Input item: <<${item}$>>

The possible scores for you to choose from are: ${score_set}$.

Your scoring should be based on the following rules: <<${scoring_rules}$>>

If you are highly uncertain about what score to return, return a score of "0". 

${reasoning}$

${examples}$

```

## lattereview/providers/base_provider.py

```py
"""Base class for all API providers with consistent error handling and type hints."""

from typing import Optional, Any, List, Dict, Union
import pydantic
from tokencost import calculate_prompt_cost, calculate_completion_cost


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class ClientCreationError(ProviderError):
    """Raised when client creation fails."""

    pass


class ResponseError(ProviderError):
    """Raised when getting a response fails."""

    pass


class InvalidResponseFormatError(ProviderError):
    """Raised when response format is invalid."""

    pass


class ClientNotInitializedError(ProviderError):
    """Raised when client is not initialized."""

    pass


class BaseProvider(pydantic.BaseModel):
    provider: str = "DefaultProvider"
    client: Optional[Any] = None
    api_key: Optional[str] = None
    model: str = "default-model"
    system_prompt: str = "You are a helpful assistant."
    response_format: Optional[Dict[str, Any]] = None
    last_response: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def create_client(self) -> Any:
        """Create and initialize the client for the provider."""
        raise NotImplementedError("Subclasses must implement create_client")

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for the provider."""
        raise NotImplementedError("Subclasses must implement set_response_format")

    async def get_response(
        self,
        messages: Union[str, List[str]],
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> tuple[Any, Dict[str, float]]:
        """Get a response from the provider."""
        raise NotImplementedError("Subclasses must implement get_response")

    async def get_json_response(
        self,
        messages: Union[str, List[str]],
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> tuple[Any, Dict[str, float]]:
        """Get a JSON-formatted response from the provider."""
        raise NotImplementedError("Subclasses must implement get_json_response")

    def _prepare_message_list(
        self, message: str, message_list: Optional[List[Dict[str, str]]] = None, system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare the list of messages to be sent to the provider."""
        raise NotImplementedError("Subclasses must implement _prepare_message_list")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from the provider."""
        raise NotImplementedError("Subclasses must implement _fetch_response")

    async def _fetch_json_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON-formatted response from the provider."""
        raise NotImplementedError("Subclasses must implement _fetch_json_response")

    def _extract_content(self, response: Any) -> Any:
        """Extract content from the provider's response."""
        raise NotImplementedError("Subclasses must implement _extract_content")

    def _get_cost(self, input_messages: List[str], completion_text: str) -> Dict[str, float]:
        """Calculate the cost of a prompt completion."""
        try:
            input_cost = calculate_prompt_cost(input_messages, self.model)
            output_cost = calculate_completion_cost(completion_text, self.model)
            return {
                "input_cost": float(input_cost),
                "output_cost": float(output_cost),
                "total_cost": float(input_cost + output_cost),
            }
        except Exception as e:
            raise ProviderError(f"Error calculating costs: {str(e)}")

```

## lattereview/providers/openai_provider.py

```py
"""OpenAI API provider implementation with comprehensive error handling and type safety."""

from typing import Optional, List, Dict, Any, Tuple
import os
from pydantic import BaseModel, create_model
import openai
from .base_provider import BaseProvider, ProviderError, ClientCreationError, ResponseError


class OpenAIProvider(BaseProvider):
    provider: str = "OpenAI"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    client: Optional[openai.AsyncOpenAI] = None
    model: str = "gpt-4o-mini"
    response_format_class: Optional[BaseModel] = None

    def __init__(self, **data: Any) -> None:
        """Initialize the OpenAI provider with error handling."""
        super().__init__(**data)
        try:
            self.client = self.create_client()
        except Exception as e:
            raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for JSON responses."""
        try:
            if not isinstance(response_format, dict):
                raise ValueError("Response format must be a dictionary")
            self.response_format = response_format
            fields = {key: (value, ...) for key, value in response_format.items()}
            self.response_format_class = create_model("ResponseFormat", **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def create_client(self) -> openai.AsyncOpenAI:
        """Create and return the OpenAI client."""
        gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if not self.api_key:
            raise ClientCreationError("OPENAI_API_KEY environment variable is not set")
        try:
            if "gemini" not in self.model.lower():
                return openai.AsyncOpenAI(api_key=self.api_key)
            self.api_key = os.getenv("GEMINI_API_KEY", self.api_key)
            return openai.AsyncOpenAI(api_key=self.api_key, base_url=gemini_base_url)
        except Exception as e:
            raise ClientCreationError(f"Failed to create OpenAI client: {str(e)}")

    async def get_response(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a response from OpenAI."""
        try:
            message_list = self._prepare_message_list(messages, message_list)
            response = await self._fetch_response(message_list, kwargs)
            txt_response = self._extract_content(response)
            cost = self._get_cost(input_messages=messages, completion_text=txt_response)
            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting response: {str(e)}")

    async def get_json_response(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a JSON response from OpenAI."""
        try:
            if not self.response_format_class:
                raise ValueError("Response format is not set")
            message_list = self._prepare_message_list(messages, message_list)
            response = await self._fetch_json_response(message_list, kwargs)
            txt_response = self._extract_content(response)
            cost = self._get_cost(input_messages=messages, completion_text=txt_response)
            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting JSON response: {str(e)}")

    def _prepare_message_list(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare the message list for the API call."""
        try:
            if message_list:
                message_list.append({"role": "user", "content": messages})
            else:
                message_list = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": messages},
                ]
            return message_list
        except Exception as e:
            raise ProviderError(f"Error preparing message list: {str(e)}")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from OpenAI."""
        try:
            return await self.client.chat.completions.create(model=self.model, messages=message_list, **(kwargs or {}))
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def _fetch_json_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON response from OpenAI."""
        try:
            return await self.client.beta.chat.completions.parse(
                model=self.model, messages=message_list, response_format=self.response_format_class, **(kwargs or {})
            )
        except Exception as e:
            raise ResponseError(f"Error fetching JSON response: {str(e)}")

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response."""
        try:
            if not response:
                raise ValueError("Empty response received")
            self.last_response = response
            return response.choices[0].message.content
        except Exception as e:
            raise ResponseError(f"Error extracting content: {str(e)}")

```

## lattereview/providers/ollama_provider.py

```py
"""Ollama API provider implementation using AsyncClient with comprehensive error handling and type safety."""

from typing import Optional, List, Dict, Any, Union, Tuple, AsyncGenerator
import json
from ollama import AsyncClient
from pydantic import BaseModel, create_model
from .base_provider import BaseProvider, ProviderError, ClientCreationError, ResponseError


class OllamaProvider(BaseProvider):
    provider: str = "Ollama"
    client: Optional[AsyncClient] = None
    model: str = "llama3.2-vision:latest"  # Default model
    response_format_class: Optional[BaseModel] = None
    invalid_keywords: List[str] = ["temperature", "max_tokens"]
    host: str = "http://localhost:11434"  # Default Ollama API endpoint

    def __init__(self, **data: Any) -> None:
        """Initialize the Ollama provider with error handling."""
        super().__init__(**data)
        try:
            self.client = self.create_client()
        except Exception as e:
            raise ClientCreationError(f"Failed to initialize Ollama: {str(e)}")

    def _clean_kwargs(self, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Remove invalid keywords from kwargs."""
        if kwargs is None:
            return {}

        cleaned_kwargs = kwargs.copy()
        for keyword in self.invalid_keywords:
            cleaned_kwargs.pop(keyword, None)
        return cleaned_kwargs

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for JSON responses."""
        try:
            if not isinstance(response_format, dict):
                raise ValueError("Response format must be a dictionary")
            self.response_format = response_format
            fields = {key: (value, ...) for key, value in response_format.items()}
            self.response_format_class = create_model("ResponseFormat", **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def create_client(self) -> AsyncClient:
        """Create and return the Ollama AsyncClient."""
        try:
            return AsyncClient(host=self.host)
        except Exception as e:
            raise ClientCreationError(f"Failed to create Ollama client: {str(e)}")

    async def get_response(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, stream: bool = False, **kwargs: Any
    ) -> Union[Tuple[Any, Dict[str, float]], AsyncGenerator[str, None]]:
        """Get a response from Ollama, with optional streaming support."""
        try:
            message_list = self._prepare_message_list(messages, message_list)

            if stream:
                return self._stream_response(message_list, kwargs)
            else:
                response = await self._fetch_response(message_list, kwargs)
                txt_response = self._extract_content(response)
                cost = {
                    "input_cost": 0,
                    "output_cost": 0,
                    "total_cost": 0,
                }  # Ollama models are local and therefore free.
                return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting response: {str(e)}")

    async def get_json_response(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a JSON response from Ollama using the defined schema."""
        try:
            if not self.response_format_class:
                raise ValueError("Response format is not set")

            message_list = self._prepare_message_list(messages, message_list)

            # Update system message to request JSON output
            if message_list and message_list[0]["role"] == "system":
                schema_str = json.dumps(self.response_format_class.model_json_schema(), indent=2)
                message_list[0]["content"] = (
                    f"{message_list[0]['content']}\n\n"
                    f"Please provide your response as a JSON object following this schema:\n{schema_str}"
                )

            # Set format parameter to 'json'
            cleaned_kwargs = self._clean_kwargs(kwargs)
            cleaned_kwargs["format"] = "json"

            response = await self._fetch_response(message_list, cleaned_kwargs)
            txt_response = self._extract_content(response)
            # try:
            #     # Validate response against schema
            #     validated_response = self.response_format_class.model_validate_json(txt_response)
            #     txt_response = validated_response.model_dump()
            # except Exception as e:
            #     raise ResponseError(f"Response validation failed: {str(e)}\nResponse: {txt_response}")
            cost = {"input_cost": 0, "output_cost": 0, "total_cost": 0}  # Ollama models are local and therefore free.
            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting JSON response: {str(e)}")

    def _prepare_message_list(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare the message list for the API call."""
        try:
            if message_list:
                message_list.append({"role": "user", "content": messages})
            else:
                message_list = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": messages},
                ]
            return message_list
        except Exception as e:
            raise ProviderError(f"Error preparing message list: {str(e)}")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from Ollama using AsyncClient."""
        try:
            if not self.client:
                raise ValueError("Client not initialized")

            cleaned_kwargs = self._clean_kwargs(kwargs)
            response = await self.client.chat(model=self.model, messages=message_list, **cleaned_kwargs)
            return response
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def _stream_response(
        self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream the response from Ollama."""
        try:
            if not self.client:
                raise ValueError("Client not initialized")

            cleaned_kwargs = self._clean_kwargs(kwargs)
            cleaned_kwargs["stream"] = True

            async for part in self.client.chat(model=self.model, messages=message_list, **cleaned_kwargs):
                yield part.message.content
        except Exception as e:
            raise ResponseError(f"Error streaming response: {str(e)}")

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response."""
        try:
            if not response:
                raise ValueError("Empty response received")
            self.last_response = response
            return response.message.content
        except Exception as e:
            raise ResponseError(f"Error extracting content: {str(e)}")

    async def close(self) -> None:
        """Close the client session."""
        if self.client:
            await self.client.aclose()

```

## lattereview/providers/litellm_provider.py

```py
"""LiteLLM API provider implementation with comprehensive error handling and type safety."""

from typing import Optional, List, Dict, Any, Union, Tuple, Type
import json
from pydantic import BaseModel, create_model
import litellm
from litellm import acompletion, completion_cost
from .base_provider import BaseProvider, ProviderError, ResponseError, InvalidResponseFormatError

litellm.drop_params = True  # Drop unsupported parameters from the API
litellm.enable_json_schema_validation = True  # Enable client-side JSON schema validation


class LiteLLMProvider(BaseProvider):
    provider: str = "LiteLLM"
    model: str = "gpt-4o-mini"
    custom_llm_provider: Optional[str] = None
    response_format_class: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None

    def __init__(self, custom_llm_provider: Optional[str] = None, **data: Any) -> None:
        """Initialize the LiteLLM provider."""
        data_with_provider = {**data}
        if custom_llm_provider:
            data_with_provider["custom_llm_provider"] = custom_llm_provider

        super().__init__(**data_with_provider)

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for JSON responses."""
        try:
            if not response_format:
                raise InvalidResponseFormatError("Response format cannot be empty")
            if not isinstance(response_format, dict):
                raise InvalidResponseFormatError("Response format must be a dictionary")
            self.response_format = response_format
            fields = {key: (value, ...) for key, value in response_format.items()}
            self.response_format_class = create_model("ResponseFormat", **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def _prepare_message_list(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare the message list for the API call."""
        try:
            if message_list:
                message_list.append({"role": "user", "content": messages})
            else:
                message_list = [
                    {"role": "system", "content": system_message or self.system_prompt},
                    {"role": "user", "content": messages},
                ]
            return message_list
        except Exception as e:
            raise ProviderError(f"Error preparing message list: {str(e)}")

    async def _fetch_response(self, message_list: List[Dict[str, str]], kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch the raw response from LiteLLM."""
        try:
            response = await acompletion(
                model=self.model, messages=message_list, custom_llm_provider=self.custom_llm_provider, **(kwargs or {})
            )
            return response
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def get_response(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a response from LiteLLM."""
        try:
            message_list = self._prepare_message_list(messages, message_list)
            response = await self._fetch_response(message_list, kwargs)
            txt_response = self._extract_content(response)
            cost = completion_cost(completion_response=response)

            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting response: {str(e)}")

    async def get_json_response(
        self, messages: str, message_list: Optional[List[Dict[str, str]]] = None, **kwargs: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Get a JSON response from LiteLLM using the defined schema."""
        try:
            if not self.response_format_class:
                raise ValueError("Response format is not set")

            message_list = self._prepare_message_list(messages, message_list)

            # Pass response format directly to acompletion
            kwargs["response_format"] = self.response_format_class

            response = await self._fetch_response(message_list, kwargs)
            txt_response = self._extract_content(response)

            # Parse the response as JSON if it's a string
            if isinstance(txt_response, str):
                txt_response = json.loads(txt_response)

            cost = completion_cost(completion_response=response)

            return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting JSON response: {str(e)}")

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response, handling both direct content and tool calls."""
        try:
            if not response:
                raise ValueError("Empty response received")

            self.last_response = response
            response_message = response.choices[0].message

            # Check for direct content first
            if response_message.content is not None:
                return response_message.content

            # Check for tool calls if content is None
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "json_tool_call":
                        return tool_call.function.arguments

            raise ValueError("No content or valid tool calls found in response")

        except Exception as e:
            raise ResponseError(f"Error extracting content: {str(e)}")

```

## lattereview/agents/base_agent.py

```py
"""Base agent class with consistent error handling and type safety."""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field

DEFAULT_CONCURRENT_REQUESTS = 20


class ReasoningType(Enum):
    """Enumeration for reasoning types."""

    NONE = "none"
    BRIEF = "brief"
    LONG = "long"
    COT = "cot"


class AgentError(Exception):
    """Base exception for agent-related errors."""

    pass


class BaseAgent(BaseModel):
    response_format: Dict[str, Any]
    provider: Optional[Any] = None
    model_args: Dict[str, Any] = Field(default_factory=dict)
    max_concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS
    name: str = "BaseAgent"
    backstory: str = "a generic base agent"
    input_description: str = "article title/abstract"
    examples: Union[str, List[Union[str, Dict[str, Any]]]] = None
    reasoning: ReasoningType = ReasoningType.BRIEF
    system_prompt: Optional[str] = None
    item_prompt: Optional[str] = None
    cost_so_far: float = 0
    memory: List[Dict[str, Any]] = []
    identity: Dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        """Initialize the base agent with error handling."""
        try:
            super().__init__(**data)
            if isinstance(self.reasoning, str):
                self.reasoning = ReasoningType(self.reasoning.lower())
            if self.reasoning == ReasoningType.NONE:
                self.response_format.pop("reasoning", None)
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")

    def setup(self) -> None:
        """Setup the agent before use."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        try:
            return self._clean_text(
                f"""
                Your name is <<{self.name}>> and you are <<{self.backstory}>>.
                Your task is to review input itmes with the following description: <<{self.input_description}>>.
                Your final output should have the following keys: \
                    {", ".join(f"{k} ({v})" for k, v in self.response_format.items())}.
                """
            )
        except Exception as e:
            raise AgentError(f"Error building system prompt: {str(e)}")

    def build_item_prompt(self, base_prompt: str, item_dict: Dict[str, Any]) -> str:
        """Build the item prompt with variable substitution."""
        try:
            prompt = base_prompt
            if "examples" in item_dict:
                item_dict["examples"] = self.process_examples(item_dict["examples"])
            if "reasoning" in item_dict:
                item_dict["reasoning"] = self.process_reasoning(item_dict["reasoning"])

            for key, value in item_dict.items():
                if value is not None:
                    prompt = prompt.replace(f"${{{key}}}$", str(value))
                else:
                    prompt = prompt.replace(f"${{{key}}}$", "")

            return self._clean_text(prompt)
        except Exception as e:
            raise AgentError(f"Error building item prompt: {str(e)}")

    def process_reasoning(self, reasoning: Union[str, ReasoningType]) -> str:
        """Process the reasoning type into a prompt string."""
        try:
            if isinstance(reasoning, str):
                reasoning = ReasoningType(reasoning.lower())

            reasoning_map = {
                ReasoningType.NONE: "",
                ReasoningType.BRIEF:
                    "You must also provide a brief (1 sentence) reasoning for your scoring. First reason then score!",
                ReasoningType.LONG:
                    "You must also provide a detailed reasoning for your scoring. First reason then score!",
                ReasoningType.COT:
                    "You must also provide a reasoning for your scoring . Think step by step in your reasoning. \
                        First reason then score!",
            }

            return self._clean_text(reasoning_map.get(reasoning, ""))
        except Exception as e:
            raise AgentError(f"Error processing reasoning: {str(e)}")

    def process_examples(self, examples: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]) -> str:
        """Process examples into a formatted string."""
        try:
            if not examples:
                return ""

            if not isinstance(examples, list):
                examples = [examples]

            examples_str = []
            for example in examples:
                if isinstance(example, dict):
                    examples_str.append("***" + "".join(f"{k}: {v}\n" for k, v in example.items()))
                elif isinstance(example, str):
                    examples_str.append("***" + example)
                else:
                    raise ValueError(f"Invalid example type: {type(example)}")

            return self._clean_text(
                "<<Here is one or more examples of the performance you are expected to have: \n"
                + "".join(examples_str)
                + ">>"
            )
        except Exception as e:
            raise AgentError(f"Error processing examples: {str(e)}")

    def reset_memory(self) -> None:
        """Reset the agent's memory and cost tracking."""
        try:
            self.memory = []
            self.cost_so_far = 0
            self.identity = {}
        except Exception as e:
            raise AgentError(f"Error resetting memory: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Remove extra spaces and blank lines from text."""
        try:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return " ".join(" ".join(line.split()) for line in lines)
        except Exception as e:
            raise AgentError(f"Error cleaning text: {str(e)}")

    async def review_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Review a list of items asynchronously."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def review_item(self, item: str) -> Dict[str, Any]:
        """Review a single item asynchronously."""
        raise NotImplementedError("This method must be implemented by subclasses.")

```

## lattereview/agents/scoring_reviewer.py

```py
"""Reviewer agent implementation with consistent error handling and type safety."""

import asyncio
from pathlib import Path
import datetime
from typing import List, Dict, Any, Optional
from pydantic import Field
from .base_agent import BaseAgent, AgentError, ReasoningType
from tqdm.asyncio import tqdm
import warnings

DEFAULT_MAX_RETRIES = 3


class ScoringReviewer(BaseAgent):
    response_format: Dict[str, Any] = {
        "reasoning": str,
        "score": int,
    }
    scoring_task: Optional[str] = None
    score_set: List[int] = [1, 2]
    scoring_rules: str = "Your scores should follow the defined schema."
    generic_item_prompt: Optional[str] = Field(default=None)
    reasoning: ReasoningType = ReasoningType.BRIEF
    max_retries: int = DEFAULT_MAX_RETRIES

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            assert self.reasoning != ReasoningType.NONE, "Reasoning type cannot be 'none' for ScoreReviewer"
            assert (
                0 not in self.score_set
            ), "Score set must not contain 0. This value is reserved for uncertain scorings / errors."
            prompt_path = Path(__file__).parent.parent / "generic_prompts" / "review_prompt.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Review prompt template not found at {prompt_path}")
            self.generic_item_prompt = prompt_path.read_text(encoding="utf-8")
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")

    def setup(self) -> None:
        """Build the agent's identity and configure the provider."""
        try:
            self.system_prompt = self.build_system_prompt()
            self.score_set = str(self.score_set)
            keys_to_replace = ["scoring_task", "score_set", "scoring_rules", "reasoning", "examples"]

            self.item_prompt = self.build_item_prompt(
                self.generic_item_prompt, {key: getattr(self, key) for key in keys_to_replace}
            )

            self.identity = {
                "system_prompt": self.system_prompt,
                "item_prompt": self.item_prompt,
                "model_args": self.model_args,
            }

            if not self.provider:
                raise AgentError("Provider not initialized")

            self.provider.set_response_format(self.response_format)
            self.provider.system_prompt = self.system_prompt
        except Exception as e:
            raise AgentError(f"Error in setup: {str(e)}")

    async def review_items(self, items: List[str], tqdm_keywords: dict = None) -> List[Dict[str, Any]]:
        """Review a list of items asynchronously with concurrency control and progress bar."""
        try:
            self.setup()
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def limited_review_item(item: str, index: int) -> tuple[int, Dict[str, Any], Dict[str, float]]:
                async with semaphore:
                    response, cost = await self.review_item(item)
                    return index, response, cost

            # Building the tqdm desc
            if tqdm_keywords:
                tqdm_desc = f"""{[f'{k}: {v}' for k, v in tqdm_keywords.items()]} - \
                    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            else:
                tqdm_desc = f"Reviewing {len(items)} items - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Create tasks with indices
            tasks = [limited_review_item(item, i) for i, item in enumerate(items)]

            # Collect results with indices
            responses_costs = []
            async for result in tqdm(asyncio.as_completed(tasks), total=len(items), desc=tqdm_desc):
                responses_costs.append(await result)

            # Sort by original index and separate response and cost
            responses_costs.sort(key=lambda x: x[0])  # Sort by index
            results = []

            for i, response, cost in responses_costs:
                if isinstance(cost, dict):
                    cost = cost["total_cost"]
                self.cost_so_far += cost
                results.append(response)
                self.memory.append(
                    {
                        "identity": self.identity,
                        "item": items[i],
                        "response": response,
                        "cost": cost,
                        "model_args": self.model_args,
                    }
                )

            return results, cost
        except Exception as e:
            raise AgentError(f"Error reviewing items: {str(e)}")

    async def review_item(self, item: str) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Review a single item asynchronously with error handling."""
        num_tried = 0
        while num_tried < self.max_retries:
            try:
                item_prompt = self.build_item_prompt(self.item_prompt, {"item": item})
                response, cost = await self.provider.get_json_response(item_prompt, **self.model_args)
                return response, cost
            except Exception as e:
                warnings.warn(f"Error reviewing item: {str(e)}. Retrying {num_tried}/{self.max_retries}")
        raise AgentError("Error reviewing item!")

```

