## collect_scripts.py

```py
"""Script to collect and organize code files into a markdown document."""
from pathlib import Path
from typing import List, Tuple, Set

def gather_code_files(
    root_dir: Path,
    extensions: Set[str],
    exclude_files: Set[str],
    exclude_folders: Set[str]
) -> Tuple[List[Path], List[Path]]:
    """Gather code files while respecting exclusion rules."""
    try:
        code_files: List[Path] = []
        excluded_files_found: List[Path] = []
        
        for file_path in root_dir.rglob('*'):
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

def write_to_markdown(
    code_files: List[Path],
    excluded_files: List[Path],
    output_file: Path
) -> None:
    """Write collected files to a markdown document."""
    try:
        with output_file.open('w', encoding='utf-8') as md_file:
            for file_path in code_files:
                relative_path = file_path.relative_to(file_path.cwd())
                md_file.write(f"## {relative_path}\n\n")
                md_file.write("```" + file_path.suffix.lstrip('.') + "\n")
                md_file.write(file_path.read_text(encoding='utf-8'))
                md_file.write("\n```\n\n")
    except Exception as e:
        raise RuntimeError(f"Error writing markdown file: {str(e)}")

def create_markdown(
    root_dir: Path,
    extensions: Set[str],
    exclude_files: Set[str],
    exclude_folders: Set[str],
    output_file: Path = Path('code_base.md')
) -> None:
    """Create a markdown file containing all code files."""
    try:
        code_files, excluded_files = gather_code_files(root_dir, extensions, exclude_files, exclude_folders)
        write_to_markdown(code_files, excluded_files, output_file)
        print(f"Markdown file '{output_file}' created with {len(code_files)} code files and {len(excluded_files)} excluded files.")
    except Exception as e:
        raise RuntimeError(f"Error creating markdown: {str(e)}")

if __name__ == "__main__":
    root_directory = Path(__file__).parent
    extensions_to_look_for = {'.py', '.ipynb', '.txt'}
    exclude_files_list = {'.env', '__init__.py', 'init.py'}
    exclude_folders_list = {'venv'}
    
    create_markdown(root_directory, extensions_to_look_for, exclude_files_list, exclude_folders_list)
```

## test/test.ipynb

```ipynb
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Failed to update token costs. Using static costs.\n",
      "/Users/pouria/Documents/Coding/LatteReview/venv/lib/python3.9/site-packages/tokencost/constants.py:69: RuntimeWarning: coroutine 'update_token_costs' was never awaited\n",
      "  logger.error(\"Failed to update token costs. Using static costs.\")\n",
      "RuntimeWarning: Enable tracemalloc to get the object allocation traceback\n",
      "/Users/pouria/Documents/Coding/LatteReview/venv/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
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
   "execution_count": 2,
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
   "execution_count": 3,
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
     "execution_count": 3,
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
   "execution_count": 6,
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
     "execution_count": 6,
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
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('The capital of France is Paris.',\n",
       " {'input_cost': 0, 'output_cost': 0, 'total_cost': 0})"
      ]
     },
     "execution_count": 5,
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
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "response: <Response [200 OK]>\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "('The capital of France is Paris.', 3.464e-05)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# litellm_provider = LiteLLMProvider(model=\"gpt-4o-mini\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"claude-3-5-sonnet-20240620\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\")\n",
    "litellm_provider = LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\")\n",
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
   "execution_count": 34,
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
      "integration of large vision language models for efficient post-disaster damage assessment and reporting\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Reviewing 3 items - 2024-12-14 20:02:35: 100%|██████████| 3/3 [00:01<00:00,  1.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      " Outputs:\n",
      "{'reasoning': 'The article title is relevant and specific, meeting the criteria.', 'score': 2}\n",
      "{'reasoning': 'The article title is relevant and meets the criteria.', 'score': 2}\n",
      "{'reasoning': 'The article title is relevant to post-disaster damage assessment and reporting, thus meeting the criteria.', 'score': 2}\n",
      "\n",
      "Costs:\n",
      "\n",
      "2.13e-05\n",
      "2.0549999999999998e-05\n",
      "2.3024999999999997e-05\n",
      "\n",
      "Total cost:\n",
      "\n",
      "2.3024999999999997e-05\n"
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
    "    # provider=LiteLLMProvider(model=\"claude-3-5-sonnet-20240620\"),\n",
    "    # provider=LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\"),\n",
    "    provider=LiteLLMProvider(model=\"gemini/gemini-1.5-flash\"),\n",
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
    "print(\"Inputs:\\n\\n\", '\\n'.join(text_list[:3]))\n",
    "\n",
    "# Dummy review\n",
    "results, total_cost = asyncio.run(agent.review_items(text_list[:3]))\n",
    "print(\"\\n\\n Outputs:\")\n",
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
   "execution_count": 38,
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
   "execution_count": 39,
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
   "execution_count": 41,
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
      "['round: A', 'reviewer_name: Pouria'] - 2024-12-14 20:04:00: 100%|██████████| 10/10 [00:01<00:00,  8.81it/s]\n",
      "['round: A', 'reviewer_name: Bardia'] - 2024-12-14 20:04:01: 100%|██████████| 10/10 [00:01<00:00,  7.58it/s]"
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
      "0.000183375\n",
      "\n",
      "Detailed cost:\n",
      "{('A', 'Pouria'): 7.3425e-05, ('A', 'Bardia'): 0.00010995}\n"
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
       "      <td>174</td>\n",
       "      <td>Semantic Information in MC: Chemotaxis Beyond ...</td>\n",
       "      <td>Brand, L.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>The recently emerged molecular communication (...</td>\n",
       "      <td>{'reasoning': 'This article focuses on molecul...</td>\n",
       "      <td>2</td>\n",
       "      <td>This article focuses on molecular communicatio...</td>\n",
       "      <td>{'reasoning': 'The paper discusses molecular c...</td>\n",
       "      <td>2</td>\n",
       "      <td>The paper discusses molecular communication (M...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>389</td>\n",
       "      <td>Multi-Pose Fusion for Sparse-View CT Reconstru...</td>\n",
       "      <td>Yang, D.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>CT imaging works by reconstructing an object o...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article discusses a novel a...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses a novel algorithm for CT...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>199</td>\n",
       "      <td>Norm Enforcement with a Soft Touch: Faster Eme...</td>\n",
       "      <td>Tzeng, S.-T.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>A multiagent system is a society of autonomous...</td>\n",
       "      <td>{'reasoning': 'The provided abstract does not ...</td>\n",
       "      <td>2</td>\n",
       "      <td>The provided abstract does not discuss large l...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>345</td>\n",
       "      <td>Arena-Web - A Web-based Development and Benchm...</td>\n",
       "      <td>Kästner, L.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>In recent years, mobile robot navigation appro...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The article discusses a platfor...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses a platform for robot nav...</td>\n",
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
       "      <td>{'reasoning': 'The article discusses decision-...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses decision-making in econo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>616</td>\n",
       "      <td>Impact of tumor-parenchyma biomechanics on liv...</td>\n",
       "      <td>Wang, Y.</td>\n",
       "      <td>bioRxiv</td>\n",
       "      <td>2020</td>\n",
       "      <td>Colorectal carcinoma (CRC) and other cancers o...</td>\n",
       "      <td>{'reasoning': 'The article focuses on a multi-...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on a multi-model approach ...</td>\n",
       "      <td>{'reasoning': 'The article focuses on biomecha...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on biomechanical interacti...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>79</td>\n",
       "      <td>PersLLM: A Personified Training Approach for L...</td>\n",
       "      <td>Zeng, Z.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Large language models (LLMs) demonstrate human...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention m...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention medical imaging ...</td>\n",
       "      <td>{'reasoning': 'The article discusses personali...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses personality development ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>239</td>\n",
       "      <td>A MULTI-AGENT REINFORCEMENT LEARNING FRAMEWORK...</td>\n",
       "      <td>Sharma, D.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>Human immunodeficiency virus (HIV) is a major ...</td>\n",
       "      <td>{'reasoning': 'The article focuses on a multi-...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on a multi-agent reinforce...</td>\n",
       "      <td>{'reasoning': 'The article discusses a multi-a...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses a multi-agent reinforcem...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>426</td>\n",
       "      <td>Human-in-the-loop online multi-agent approach ...</td>\n",
       "      <td>Bravo-Rocca, G.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>Increasing a ML model accuracy is not enough, ...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention t...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention the application ...</td>\n",
       "      <td>{'reasoning': 'The article discusses a multi-a...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses a multi-agent approach t...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>160</td>\n",
       "      <td>Modeling the Spread of COVID-19 in University ...</td>\n",
       "      <td>Herrmann, J.W.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Mathematical and simulation models are often u...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article discusses mathemati...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses mathematical modeling of...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID                                              Title       1st author  \\\n",
       "0  174  Semantic Information in MC: Chemotaxis Beyond ...        Brand, L.   \n",
       "1  389  Multi-Pose Fusion for Sparse-View CT Reconstru...         Yang, D.   \n",
       "2  199  Norm Enforcement with a Soft Touch: Faster Eme...     Tzeng, S.-T.   \n",
       "3  345  Arena-Web - A Web-based Development and Benchm...      Kästner, L.   \n",
       "4  524  Learning from zero: how to make consumption-sa...          Shi, R.   \n",
       "5  616  Impact of tumor-parenchyma biomechanics on liv...         Wang, Y.   \n",
       "6   79  PersLLM: A Personified Training Approach for L...         Zeng, Z.   \n",
       "7  239  A MULTI-AGENT REINFORCEMENT LEARNING FRAMEWORK...       Sharma, D.   \n",
       "8  426  Human-in-the-loop online multi-agent approach ...  Bravo-Rocca, G.   \n",
       "9  160  Modeling the Spread of COVID-19 in University ...   Herrmann, J.W.   \n",
       "\n",
       "      repo  year                                           abstract  \\\n",
       "0    arXiv  2024  The recently emerged molecular communication (...   \n",
       "1    arXiv  2022  CT imaging works by reconstructing an object o...   \n",
       "2    arXiv  2024  A multiagent system is a society of autonomous...   \n",
       "3    arXiv  2023  In recent years, mobile robot navigation appro...   \n",
       "4    arXiv  2021  This exercise proposes a learning mechanism to...   \n",
       "5  bioRxiv  2020  Colorectal carcinoma (CRC) and other cancers o...   \n",
       "6    arXiv  2024  Large language models (LLMs) demonstrate human...   \n",
       "7    arXiv  2023  Human immunodeficiency virus (HIV) is a major ...   \n",
       "8    arXiv  2022  Increasing a ML model accuracy is not enough, ...   \n",
       "9    arXiv  2024  Mathematical and simulation models are often u...   \n",
       "\n",
       "                               round-A_Pouria_output round-A_Pouria_score  \\\n",
       "0  {'reasoning': 'This article focuses on molecul...                    2   \n",
       "1  {'reasoning': 'The article does not discuss la...                    2   \n",
       "2  {'reasoning': 'The provided abstract does not ...                    2   \n",
       "3  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "4  {'reasoning': 'The article does not discuss la...                    2   \n",
       "5  {'reasoning': 'The article focuses on a multi-...                    2   \n",
       "6  {'reasoning': 'The abstract does not mention m...                    2   \n",
       "7  {'reasoning': 'The article focuses on a multi-...                    2   \n",
       "8  {'reasoning': 'The abstract does not mention t...                    2   \n",
       "9  {'reasoning': 'The article does not discuss la...                    2   \n",
       "\n",
       "                            round-A_Pouria_reasoning  \\\n",
       "0  This article focuses on molecular communicatio...   \n",
       "1  The article does not discuss large language mo...   \n",
       "2  The provided abstract does not discuss large l...   \n",
       "3  The abstract does not mention large language m...   \n",
       "4  The article does not discuss large language mo...   \n",
       "5  The article focuses on a multi-model approach ...   \n",
       "6  The abstract does not mention medical imaging ...   \n",
       "7  The article focuses on a multi-agent reinforce...   \n",
       "8  The abstract does not mention the application ...   \n",
       "9  The article does not discuss large language mo...   \n",
       "\n",
       "                               round-A_Bardia_output round-A_Bardia_score  \\\n",
       "0  {'reasoning': 'The paper discusses molecular c...                    2   \n",
       "1  {'reasoning': 'The article discusses a novel a...                    2   \n",
       "2  {'reasoning': 'The article does not discuss la...                    2   \n",
       "3  {'reasoning': 'The article discusses a platfor...                    2   \n",
       "4  {'reasoning': 'The article discusses decision-...                    2   \n",
       "5  {'reasoning': 'The article focuses on biomecha...                    2   \n",
       "6  {'reasoning': 'The article discusses personali...                    2   \n",
       "7  {'reasoning': 'The article discusses a multi-a...                    2   \n",
       "8  {'reasoning': 'The article discusses a multi-a...                    2   \n",
       "9  {'reasoning': 'The article discusses mathemati...                    2   \n",
       "\n",
       "                            round-A_Bardia_reasoning  \n",
       "0  The paper discusses molecular communication (M...  \n",
       "1  The article discusses a novel algorithm for CT...  \n",
       "2  The article does not discuss large language mo...  \n",
       "3  The article discusses a platform for robot nav...  \n",
       "4  The article discusses decision-making in econo...  \n",
       "5  The article focuses on biomechanical interacti...  \n",
       "6  The article discusses personality development ...  \n",
       "7  The article discusses a multi-agent reinforcem...  \n",
       "8  The article discusses a multi-agent approach t...  \n",
       "9  The article discusses mathematical modeling of...  "
      ]
     },
     "execution_count": 41,
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
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "        Title: Semantic Information in MC: Chemotaxis Beyond Shannon\n",
      "        Abstract: The recently emerged molecular communication (MC) paradigm intends to leverage communication engineering tools for the design of synthetic chemical communication systems. These systems are envisioned to operate at nanoscale and in biological environments, such as the human body, and catalyze the emergence of revolutionary applications in the context of early disease monitoring and drug targeting. Despite the abundance of theoretical (and recently also experimental) MC system designs proposed over the past years, some fundamental questions remain unresolved, hindering the breakthrough of MC in real-world applications. One of these questions is: What can be a useful measure of information in the context of MC applications? While most existing works on MC build upon the concept of syntactic information as introduced by Shannon, in this paper, we explore the framework of semantic information as introduced by Kolchinsky and Wolpert for the information-theoretic analysis of a natural MC system, namely bacterial chemotaxis. Exploiting computational agent-based modeling (ABM), we are able to quantify, for the first time, the amount of information that the considered chemotactic bacterium (CB) utilizes to adapt to and survive in a dynamic environment. In other words, we show how the flow of information between the environment and the CB is related to the effectiveness of communication. Effectiveness here refers to the adaptation of the CB to the dynamic environment in order to ensure survival. Our analysis reveals that it highly depends on the environmental conditions how much information the CB can effectively utilize for improving their survival chances. Encouraged by our results, we envision that the proposed semantic information framework can open new avenues for the development of theoretical and experimental MC system designs for future nanoscale applications.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: This article focuses on molecular communication and bacterial chemotaxis, which are not related to large language models or medical imaging.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The paper discusses molecular communication (MC) and focuses on semantic information in the context of bacterial chemotaxis, which is unrelated to large language models or medical imaging data.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Multi-Pose Fusion for Sparse-View CT Reconstruction Using Consensus Equilibrium\n",
      "        Abstract: CT imaging works by reconstructing an object of interest from a collection of projections. Traditional methods such as filtered-back projection (FBP) work on projection images acquired around a fixed rotation axis. However, for some CT problems, it is desirable to perform a joint reconstruction from projection data acquired from multiple rotation axes. In this paper, we present Multi-Pose Fusion, a novel algorithm that performs a joint tomographic reconstruction from CT scans acquired from multiple poses of a single object, where each pose has a distinct rotation axis. Our approach uses multi-agent consensus equilibrium (MACE), an extension of plug-and-play, as a framework for integrating projection data from different poses. We apply our method on simulated data and demonstrate that Multi-Pose Fusion can achieve a better reconstruction result than single pose reconstruction.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  It focuses on a novel algorithm for CT reconstruction from multiple poses using a multi-agent consensus equilibrium approach. This is a different area of research than AI agents using large language models for medical image analysis. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses a novel algorithm for CT reconstruction but does not focus on large language models or AI agents applied to medical imaging data, therefore it does not meet the criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Norm Enforcement with a Soft Touch: Faster Emergence, Happier Agents\n",
      "        Abstract: A multiagent system is a society of autonomous agents whose interactions can be regulated via social norms. In general, the norms of a society are not hardcoded but emerge from the agents’ interactions. Specifically, how the agents in a society react to each other’s behavior and respond to the reactions of others determines which norms emerge in the society. We think of these reactions by an agent to the satisfactory or unsatisfactory behaviors of another agent as communications from the first agent to the second agent. Understanding these communications is a kind of social intelligence: these communications provide natural drivers for norm emergence by pushing agents toward certain behaviors, which can become established as norms. Whereas it is well-known that sanctioning can lead to the emergence of norms, we posit that a broader kind of social intelligence can prove more effective in promoting cooperation in a multiagent system. Accordingly, we develop Nest, a framework that models social intelligence via a wider variety of communications and understanding of them than in previous work. To evaluate Nest, we develop a simulated pandemic environment and conduct simulation experiments to compare Nest with baselines considering a combination of three kinds of social communication: sanction, tell, and hint. We find that societies formed of Nest agents achieve norms faster. Moreover, Nest agents effectively avoid undesirable consequences, which are negative sanctions and deviation from goals, and yield higher satisfaction for themselves than baseline agents despite requiring only an equivalent amount of information.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The provided abstract does not discuss large language models, AI agents, or medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data, focusing instead on social norms in multiagent systems.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Arena-Web - A Web-based Development and Benchmarking Platform for Autonomous Navigation Approaches\n",
      "        Abstract: In recent years, mobile robot navigation approaches have become increasingly important due to various application areas ranging from healthcare to warehouse logistics. In particular, Deep Reinforcement Learning approaches have gained popularity for robot navigation but are not easily accessible to non-experts and complex to develop. In recent years, efforts have been made to make these sophisticated approaches accessible to a wider audience. In this paper, we present Arena-Web, a web-based development and evaluation suite for developing, training, and testing DRL-based navigation planners for various robotic platforms and scenarios. The interface is designed to be intuitive and engaging to appeal to non-experts and make the technology accessible to a wider audience. With Arena-Web and its interface, training and developing Deep Reinforcement Learning agents is simplified and made easy without a single line of code. The web-app is free to use and openly available under the link stated in the supplementary materials.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models, AI agents, or medical imaging.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses a platform for robot navigation using Deep Reinforcement Learning, which is unrelated to large language models or medical imaging data.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Learning from zero: how to make consumption-saving decisions in a stochastic environment with an AI algorithm\n",
      "        Abstract: This exercise proposes a learning mechanism to model economic agent’s decision-making process using an actor-critic structure in the literature of artificial intelligence It is motivated by the psychology literature of learning through reinforcing good or bad decisions In a model of an environment, to learn to make decisions, this AI agent needs to interact with its environment and make explorative actions Each action in a given state brings a reward signal to the agent These interactive experience is saved in the agent’s memory, which is then used to update its subjective belief of the world The agent’s decision-making strategy is formed and adjusted based on this evolving subjective belief This agent does not only take an action that it knows would bring a high reward, it also explores other possibilities This is the process of taking explorative actions, and it ensures that the agent notices changes in its environment and adapt its subjective belief and decisions accordingly Through a model of stochastic optimal growth, I illustrate that the economic agent under this proposed learning structure is adaptive to changes in an underlying stochastic process of the economy AI agents can differ in their levels of exploration, which leads to different experience in the same environment This reflects on to their different learning behaviours and welfare obtained The chosen economic structure possesses the fundamental decision making problems of macroeconomic models, i.e., how to make consumption-saving decisions in a lifetime, and it can be generalised to other decision-making processes and economic models.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  The abstract describes an AI agent used for economic modeling, not medical imaging. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses decision-making in economic contexts using AI algorithms, but it does not focus on large language models or medical imaging data, making it irrelevant to the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Impact of tumor-parenchyma biomechanics on liver metastatic progression: A multi-model approach\n",
      "        Abstract: Colorectal carcinoma (CRC) and other cancers often metastasize to the liver in later stages of the disease, contributing significantly to patient death. While the biomechanical properties of the liver parenchyma (normal liver tissue) are known to affect primary and metastatic tumor growth in liver tissues, the role of these properties in driving or inhibiting metastatic inception remains poorly understood. This study uses a multi-model approach to study the effect of tumor-parenchyma biomechanical interactions on metastatic seeding and growth; the framework also allows investigating the impact of changing tissue biomechanics on tumor dormancy and “reawakening.” We employ a detailed poroviscoelastic (PVE) model to study how tumor metastases deform the surrounding tissue, induce pressure increases, and alter interstitial fluid flow in and near the metastases. Results from these short-time simulations in detailed single hepatic lobules motivate constitutive relations and biological hypotheses for use in an agent-based model of metastatic seeding and growth in centimeter-scale tissue over months-long time scales. We find that biomechanical tumor-parenchyma interactions on shorter time scales (adhesion, repulsion, and elastic tissue deformation over minutes) and longer time scales (plastic tissue relaxation over hours) may play a key role in tumor cell seeding and growth within liver tissue. These interactions may arrest the growth of micrometastases in a dormant state and can prevent newly arriving cancer cells from establishing successful metastatic foci. Moreover, the simulations indicate ways in which dormant tumors can “reawaken” after changes in parenchymal tissue mechanical properties, as may arise during aging or following acute liver illness or injury. We conclude that the proposed modeling approach can yield insight into the role of tumor-parenchyma biomechanics in promoting liver metastatic growth, with the longer term goal of identifying conditions to clinically arrest and reverse the course of late-stage cancer.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article focuses on a multi-model approach to study the effect of tumor-parenchyma biomechanics on liver metastasis.  It uses a poroviscoelastic model and an agent-based model, but these are not large language models applied to medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on biomechanical interactions in liver metastatic progression rather than the application of large language models in medical imaging, therefore it does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: PersLLM: A Personified Training Approach for Large Language Models\n",
      "        Abstract: Large language models (LLMs) demonstrate human-like intelligence, making them useful in social simulations, human-machine interactions, and multi-agent systems. However, their lack of distinct personalities, such as ingratiating behaviors, inconsistent opinions, and uniform responses, limits their practical applications. Developing personalities in LLMs is essential to unlock their potential. Current methods often conduct stylized training or prompt engineering to simulate personalities, capturing only superficial styles rather than core traits, leading to instability. This study introduces PersLLM, integrating psychology-based principles of personality—social practice, consistency, and dynamic development—into a comprehensive training approach. By constructing personified data and training models, we embed personality traits into the model parameters, enhancing induction resistance, promoting consistency, and supporting dynamic evolution. Single-agent evaluations show our superiority in aligning reference responses. Multi-agent evaluations highlight improvements in opinion consistency and collaborative creativity. Human-agent interaction evaluations indicate enhancements in interactive experiences, underscoring the practical implications of our research.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention medical imaging data or any application of LLMs to medical imaging.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses personality development in large language models but does not focus on their application to medical imaging data, thus it does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: A MULTI-AGENT REINFORCEMENT LEARNING FRAMEWORK FOR EVALUATING THE U.S. ‘ENDING THE HIV EPIDEMIC’ PLAN\n",
      "        Abstract: Human immunodeficiency virus (HIV) is a major public health concern in the United States, with about 1.2 million people living with HIV and 35,000 newly infected each year. There are considerable geographical disparities in HIV burden and care access across the U.S. The 2019’Ending the HIV Epidemic (EHE)’ initiative aims to reduce new infections by 90% by 2030, by improving coverage of diagnoses, treatment, and prevention interventions and prioritizing jurisdictions with high HIV prevalence. Identifying optimal scale-up of intervention combinations will help inform resource allocation. Existing HIV decision analytic models either evaluate specific cities or the overall national population, thus overlooking jurisdictional interactions or differences. In this paper, we propose a multi-agent reinforcement learning (MARL) model, that enables jurisdiction-specific decision analyses but in an environment with cross-jurisdictional epidemiological interactions. In experimental analyses, conducted on jurisdictions within California and Florida, optimal policies from MARL were significantly different than those generated from single-agent RL, highlighting the influence of jurisdictional variations and interactions. By using comprehensive modeling of HIV and formulations of state space, action space, and reward functions, this work helps demonstrate the strengths and applicability of MARL for informing public health policies, and provides a framework for expanding to the national-level to inform the EHE.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article focuses on a multi-agent reinforcement learning framework for evaluating the U.S. 'Ending the HIV Epidemic' plan.  It does not involve large language models or medical imaging data. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses a multi-agent reinforcement learning framework for HIV interventions, which does not involve large language models or medical imaging data, hence it does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Human-in-the-loop online multi-agent approach to increase trustworthiness in ML models through trust scores and data augmentation\n",
      "        Abstract: Increasing a ML model accuracy is not enough, we must also increase its trustworthiness. This is an important step for building resilient AI systems for safety-critical applications such as automotive, finance, and healthcare. For that purpose, we propose a multi-agent system that combines both machine and human agents. In this system, a checker agent calculates a trust score of each instance (which penalizes overconfidence and overcautiousness in predictions) using an agreement-based method and ranks it; then an improver agent filters the anomalous instances based on a human rule-based procedure (which is considered safe), gets the human labels, applies geometric data augmentation, and retrains with the augmented data using transfer learning. We evaluate the system on corrupted versions of the MNIST and FashionMNIST datasets. We get an improvement in accuracy and trust score with just few additional labels compared to a baseline approach.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention the application of large language models or AI agents to medical imaging data.  The focus is on a multi-agent system for improving the trustworthiness of machine learning models in general, using data augmentation and human-in-the-loop methods. While relevant to AI and trustworthiness, it does not meet the specified criteria of using large language models in medical imaging.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses a multi-agent approach to improve trustworthiness in ML models but does not focus on large language models or their application to medical imaging data specifically.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Modeling the Spread of COVID-19 in University Communities\n",
      "        Abstract: Mathematical and simulation models are often used to predict the spread of a disease and estimate the impact of public health interventions, and many such models have been developed and used during the COVID-19 pandemic. This paper describes a study that systematically compared models for a university community, which has a much smaller but more connected population than a state or nation. We developed a stochastic agent-based model, a deterministic compartment model, and a model based on ordinary differential equations. All three models represented the disease progression with the same susceptible-exposed-infectious-recovered (SEIR) model. We created a baseline scenario for a population of 14,000 students and faculty and eleven other scenarios for combinations of interventions such as regular testing, contact tracing, quarantine, isolation, moving courses online, mask wearing, improving ventilation, and vaccination. Where possible, our study used parameter values from other epidemiological studies and incorporated data about COVID-19 testing in College Park, Maryland, but the study was designed to compare modeling approaches to each other using a synthetic population, so comparisons with data about actual cases were not relevant. For each scenario we used the models to estimate the number of persons who become infected over a period of 119 days (17 weeks in a semester). We evaluated the models by comparing their predictions and evaluating their parsimony and computational effort. The agent-based model (ABM) and the deterministic compartment model (DCM) had similar results with cyclic flow of persons to and from quarantine, but the model based on ordinary differential equations failed to capture these dynamics. The ABM’s computation time was much greater than the other two models’ computation time. The DCM captured some of the dynamics that were present in the ABM’s predictions and, like those from the ABM, clearly showed the importance of testing and moving classes on-line.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  It focuses on mathematical and simulation models for predicting the spread of COVID-19, using agent-based modeling, deterministic compartment models, and ordinary differential equations. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses mathematical modeling of COVID-19 spread, which does not involve large language models or AI agents applied to medical imaging data, thus it does not meet the criteria.\n",
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
from tqdm.auto import tqdm
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
                reviewers = review_task["reviewers"] if isinstance(review_task["reviewers"], list) else [review_task["reviewers"]]
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
                
                reviewers = review_task["reviewers"] if isinstance(review_task["reviewers"], list) else [review_task["reviewers"]]
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
                        f"Review Task ID: {round_id}-{idx}\n"
                        f"Content Hash: {content_hash}\n\n"
                        f"{input_text}"
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
                        }
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
                            processed_outputs.append({
                                "reasoning": None,
                                "score": None,
                                "_content_hash": expected_hash
                            })
                    
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
        system_message: Optional[str] = None
    ) -> tuple[Any, Dict[str, float]]:
        """Get a response from the provider."""
        raise NotImplementedError("Subclasses must implement get_response")

    async def get_json_response(
        self, 
        messages: Union[str, List[str]], 
        message_list: Optional[List[Dict[str, str]]] = None, 
        system_message: Optional[str] = None
    ) -> tuple[Any, Dict[str, float]]:
        """Get a JSON-formatted response from the provider."""
        raise NotImplementedError("Subclasses must implement get_json_response")

    def _prepare_message_list(
        self, 
        message: str, 
        message_list: Optional[List[Dict[str, str]]] = None, 
        system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare the list of messages to be sent to the provider."""
        raise NotImplementedError("Subclasses must implement _prepare_message_list")

    async def _fetch_response(
        self, 
        message_list: List[Dict[str, str]], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the raw response from the provider."""
        raise NotImplementedError("Subclasses must implement _fetch_response")

    async def _fetch_json_response(
        self, 
        message_list: List[Dict[str, str]], 
        kwargs: Optional[Dict[str, Any]] = None
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
                "total_cost": float(input_cost + output_cost)
            }
        except Exception as e:
            raise ProviderError(f"Error calculating costs: {str(e)}")
```

## lattereview/providers/openai_provider.py

```py
"""OpenAI API provider implementation with comprehensive error handling and type safety."""
from typing import Optional, List, Dict, Any, Union, Tuple
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
            self.response_format_class = create_model('ResponseFormat', **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def create_client(self) -> openai.AsyncOpenAI:
        """Create and return the OpenAI client."""
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
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
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
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
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
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

    async def _fetch_response(
        self,
        message_list: List[Dict[str, str]],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the raw response from OpenAI."""
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=message_list,
                **(kwargs or {})
            )
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def _fetch_json_response(
        self,
        message_list: List[Dict[str, str]],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the JSON response from OpenAI."""
        try:
            return await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=message_list,
                response_format=self.response_format_class,
                **(kwargs or {})
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
import asyncio
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
            self.response_format_class = create_model('ResponseFormat', **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def create_client(self) -> AsyncClient:
        """Create and return the Ollama AsyncClient."""
        try:
            return AsyncClient(host=self.host)
        except Exception as e:
            raise ClientCreationError(f"Failed to create Ollama client: {str(e)}")

    async def get_response(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Tuple[Any, Dict[str, float]], AsyncGenerator[str, None]]:
        """Get a response from Ollama, with optional streaming support."""
        try:
            message_list = self._prepare_message_list(messages, message_list)
            
            if stream:
                return self._stream_response(message_list, kwargs)
            else:
                response = await self._fetch_response(message_list, kwargs)
                txt_response = self._extract_content(response)
                cost = {'input_cost': 0, 'output_cost': 0, 'total_cost': 0}  # Ollama models are local and therefore free.
                return txt_response, cost
        except Exception as e:
            raise ResponseError(f"Error getting response: {str(e)}")

    async def get_json_response(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
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
            cleaned_kwargs['format'] = 'json'
            
            response = await self._fetch_response(message_list, cleaned_kwargs)
            txt_response = self._extract_content(response)
            # try:
            #     # Validate response against schema
            #     validated_response = self.response_format_class.model_validate_json(txt_response)
            # except Exception as e:
            #     raise ResponseError(f"Response validation failed: {str(e)}\nResponse: {txt_response}")
            # txt_response = validated_response.model_dump()
            cost = {'input_cost': 0, 'output_cost': 0, 'total_cost': 0}  # Ollama models are local and therefore free.
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

    async def _fetch_response(
        self,
        message_list: List[Dict[str, str]],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the raw response from Ollama using AsyncClient."""
        try:
            if not self.client:
                raise ValueError("Client not initialized")
            
            cleaned_kwargs = self._clean_kwargs(kwargs)
            response = await self.client.chat(
                model=self.model,
                messages=message_list,
                **cleaned_kwargs
            )
            return response
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def _stream_response(
        self,
        message_list: List[Dict[str, str]],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream the response from Ollama."""
        try:
            if not self.client:
                raise ValueError("Client not initialized")
            
            cleaned_kwargs = self._clean_kwargs(kwargs)
            cleaned_kwargs["stream"] = True
            
            async for part in self.client.chat(
                model=self.model,
                messages=message_list,
                **cleaned_kwargs
            ):
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
from litellm import acompletion, completion_cost, get_supported_openai_params, supports_response_schema
from .base_provider import BaseProvider, ProviderError, ResponseError

litellm.drop_params = True  # Drop unsupported parameters from the API
litellm.set_verbose = False # Disable verbose mode
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
            data_with_provider['custom_llm_provider'] = custom_llm_provider
        
        super().__init__(**data_with_provider)

    def set_response_format(self, response_format: Dict[str, Any]) -> None:
        """Set the response format for JSON responses."""
        try:
            if not isinstance(response_format, dict):
                raise ValueError("Response format must be a dictionary")
            self.response_format = response_format
            fields = {key: (value, ...) for key, value in response_format.items()}
            self.response_format_class = create_model('ResponseFormat', **fields)
        except Exception as e:
            raise ProviderError(f"Error setting response format: {str(e)}")

    def _prepare_message_list(
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        system_message: Optional[str] = None
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

    async def _fetch_response(
        self,
        message_list: List[Dict[str, str]],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fetch the raw response from LiteLLM."""
        try:
            response = await acompletion(
                model=self.model,
                messages=message_list,
                custom_llm_provider=self.custom_llm_provider,
                **(kwargs or {})
            )
            return response
        except Exception as e:
            raise ResponseError(f"Error fetching response: {str(e)}")

    async def get_response(
        self, 
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
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
        self,
        messages: str,
        message_list: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
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
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == 'json_tool_call':
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
    max_concurrent_requests: int = 20
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
            return self._clean_text(f"""
                Your name is <<{self.name}>> and you are <<{self.backstory}>>.
                Your task is to review input itmes with the following description: <<{self.input_description}>>.
                Your final output should have the following keys: 
                {", ".join(f"{k} ({v})" for k, v in self.response_format.items())}.
                """)
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
                    prompt = prompt.replace(f'${{{key}}}$', str(value))
                else:
                    prompt = prompt.replace(f'${{{key}}}$', '')
            
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
                ReasoningType.BRIEF: "You must also provide a brief (1 sentence) reasoning for your scoring. First reason then score!",
                ReasoningType.LONG: "You must also provide a detailed reasoning for your scoring. First reason then score!",
                ReasoningType.COT: "You must also provide a reasoning for your scoring . Think step by step in your reasoning. First reason then score!"
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
            
            return self._clean_text("<<Here is one or more examples of the performance you are expected to have: \n" + 
                                  "".join(examples_str)+">>")
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
            return ' '.join(' '.join(line.split()) for line in lines)
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
    num_repeat_task: int = 3

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            assert self.reasoning != ReasoningType.NONE, "Reasoning type cannot be 'none' for ScoreReviewer"
            assert 0 not in self.score_set, "Score set must not contain 0. This value is reserved for uncertain scorings / errors."
            prompt_path = Path(__file__).parent.parent / "generic_prompts" / "review_prompt.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Review prompt template not found at {prompt_path}")
            self.generic_item_prompt = prompt_path.read_text(encoding='utf-8')
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")

    def setup(self) -> None:
        """Build the agent's identity and configure the provider."""
        try:
            self.system_prompt = self.build_system_prompt()
            self.score_set = str(self.score_set)
            keys_to_replace = ['scoring_task', 'score_set', 
                             'scoring_rules', 'reasoning', 'examples']
            
            self.item_prompt = self.build_item_prompt(
                self.generic_item_prompt,
                {key: getattr(self, key) for key in keys_to_replace}
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
                tqdm_desc = f"""{[f'{k}: {v}' for k, v in tqdm_keywords.items()]} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            else:
                tqdm_desc = f"Reviewing {len(items)} items - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Create tasks with indices
            tasks = [limited_review_item(item, i) for i, item in enumerate(items)]
            
            # Collect results with indices
            responses_costs = []
            async for result in tqdm(
                asyncio.as_completed(tasks),
                total=len(items),
                desc=tqdm_desc
            ):
                responses_costs.append(await result)

            # Sort by original index and separate response and cost
            responses_costs.sort(key=lambda x: x[0])  # Sort by index
            results = []
            
            for i, response, cost in responses_costs:
                if isinstance(cost, dict):
                    cost = cost["total_cost"]
                self.cost_so_far += cost
                results.append(response)
                self.memory.append({
                    'identity': self.identity,
                    'item': items[i],
                    'response': response,
                    'cost': cost,
                    'model_args': self.model_args
                })

            return results, cost
        except Exception as e:
            raise AgentError(f"Error reviewing items: {str(e)}")

    async def review_item(self, item: str) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Review a single item asynchronously with error handling."""
        num_tried = 0
        try:
            item_prompt = self.build_item_prompt(self.item_prompt, {'item': item})
            response, cost = await self.provider.get_json_response(
                item_prompt,
                **self.model_args
            )
            return response, cost
        except Exception as e:
            if num_tried < self.num_repeat_task:
                num_tried += 1
                warnings.warn(f"Error reviewing item: {str(e)}. Retrying {num_tried}/{self.num_repeat_task}")
                return await self.review_item(item)
            raise AgentError(f"Error reviewing item: {str(e)}")
```

