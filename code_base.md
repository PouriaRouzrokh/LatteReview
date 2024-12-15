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

## test/score_review_test.ipynb

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
   "execution_count": 4,
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
     "execution_count": 4,
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
       "('The capital of France is Paris!',\n",
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
   "execution_count": 6,
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
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# litellm_provider = LiteLLMProvider(model=\"gpt-4o-mini\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"claude-3-5-sonnet-20240620\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"groq/llama-3.3-70b-versatile\")\n",
    "# litellm_provider = LiteLLMProvider(model=\"ollama/llama3.2-vision:latest\")\n",
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
   "execution_count": 7,
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
      "Reviewing 3 items - 2024-12-14 21:22:53: 100%|██████████| 3/3 [00:09<00:00,  3.05s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Outputs:\n",
      "{'reasoning': 'The title clearly indicates a policy-focused study that addresses the balance between disease containment and social impacts through quarantine segmentation, which is a relevant and well-defined research topic.', 'score': 2}\n",
      "{'reasoning': 'The title clearly indicates a framework for protein engineering that leverages large language models and multimodal AutoML, which demonstrates a clear focus on protein engineering applications.', 'score': 2}\n",
      "{'reasoning': 'The title effectively indicates the use of vision-language models for post-disaster assessment, which is a clear and relevant application of AI technology for disaster management.', 'score': 2}\n",
      "\n",
      "Costs:\n",
      "\n",
      "0.003057\n",
      "0.0029760000000000003\n",
      "0.00294\n",
      "\n",
      "Total cost:\n",
      "\n",
      "0.00294\n"
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
   "execution_count": 8,
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
   "execution_count": 9,
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
   "execution_count": 10,
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
      "['round: A', 'reviewer_name: Pouria'] - 2024-12-14 21:23:02: 100%|██████████| 10/10 [00:01<00:00,  8.94it/s]\n",
      "['round: A', 'reviewer_name: Bardia'] - 2024-12-14 21:23:03: 100%|██████████| 10/10 [00:09<00:00,  1.04it/s]"
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
      "0.0001284\n",
      "\n",
      "Detailed cost:\n",
      "{('A', 'Pouria'): 4.785e-05, ('A', 'Bardia'): 8.055e-05}\n"
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
       "      <td>229</td>\n",
       "      <td>Towards a Surgeon-in-the-Loop Ophthalmic Robot...</td>\n",
       "      <td>Gomaa, A.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>Robot-assisted surgical systems have demonstra...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention t...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention the use of large...</td>\n",
       "      <td>{'reasoning': 'The article focuses on a surgeo...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on a surgeon-in-the-loop a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>430</td>\n",
       "      <td>Computational model for tumor response to adop...</td>\n",
       "      <td>Luque, L.M.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>One of the barriers to the development of effe...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>272</td>\n",
       "      <td>Using a library of chemical reactions to fit s...</td>\n",
       "      <td>Burrage, P.M.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>In this paper we introduce a new method based ...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'This paper discusses methods re...</td>\n",
       "      <td>2</td>\n",
       "      <td>This paper discusses methods related to chemic...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>38</td>\n",
       "      <td>Talk, Listen, Connect: Navigating Empathy in H...</td>\n",
       "      <td>Roshanaei, M.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Social interactions promote well-being, yet ch...</td>\n",
       "      <td>{'reasoning': 'The article focuses on AI agent...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on AI agents applied to me...</td>\n",
       "      <td>{'reasoning': 'The article focuses on empathy ...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on empathy in human-AI int...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>185</td>\n",
       "      <td>From multiplicity of infection to force of inf...</td>\n",
       "      <td>Zhan, Q.</td>\n",
       "      <td>medRxiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>High multiplicity of infection or MOI, the num...</td>\n",
       "      <td>{'reasoning': 'The article focuses on applying...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on applying queuing theory...</td>\n",
       "      <td>{'reasoning': 'The article discusses the appli...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses the application of queui...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>123</td>\n",
       "      <td>Can large language models understand uncommon ...</td>\n",
       "      <td>Wu, J.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Large language models (LLMs) like ChatGPT have...</td>\n",
       "      <td>{'reasoning': 'The abstract mentions large lan...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract mentions large language models (L...</td>\n",
       "      <td>{'reasoning': 'The article discusses large lan...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses large language models (L...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>352</td>\n",
       "      <td>Intrinsic Motivation in Model-based Reinforcem...</td>\n",
       "      <td>Latyshev, A.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>The reinforcement learning research area conta...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The article focuses on reinforc...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on reinforcement learning ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>349</td>\n",
       "      <td>The impact of vaccination frequency on COVID-1...</td>\n",
       "      <td>Stoddard, M.</td>\n",
       "      <td>medRxiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>While the rapid deployment of SARS-CoV-2 vacci...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article discusses vaccinati...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses vaccination frequency's ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>585</td>\n",
       "      <td>Biomechanic posture stabilisation via iterativ...</td>\n",
       "      <td>Hossny, M.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2020</td>\n",
       "      <td>It is not until we become senior citizens do w...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>715</td>\n",
       "      <td>Integrating episodic memory into a reinforceme...</td>\n",
       "      <td>Young, K.J.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2018</td>\n",
       "      <td>Episodic memory is a psychology term which ref...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The article discusses a reinfor...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses a reinforcement learning...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID                                              Title     1st author  \\\n",
       "0  229  Towards a Surgeon-in-the-Loop Ophthalmic Robot...      Gomaa, A.   \n",
       "1  430  Computational model for tumor response to adop...    Luque, L.M.   \n",
       "2  272  Using a library of chemical reactions to fit s...  Burrage, P.M.   \n",
       "3   38  Talk, Listen, Connect: Navigating Empathy in H...  Roshanaei, M.   \n",
       "4  185  From multiplicity of infection to force of inf...       Zhan, Q.   \n",
       "5  123  Can large language models understand uncommon ...         Wu, J.   \n",
       "6  352  Intrinsic Motivation in Model-based Reinforcem...   Latyshev, A.   \n",
       "7  349  The impact of vaccination frequency on COVID-1...   Stoddard, M.   \n",
       "8  585  Biomechanic posture stabilisation via iterativ...     Hossny, M.   \n",
       "9  715  Integrating episodic memory into a reinforceme...    Young, K.J.   \n",
       "\n",
       "      repo  year                                           abstract  \\\n",
       "0    arXiv  2023  Robot-assisted surgical systems have demonstra...   \n",
       "1    arXiv  2022  One of the barriers to the development of effe...   \n",
       "2    arXiv  2023  In this paper we introduce a new method based ...   \n",
       "3    arXiv  2024  Social interactions promote well-being, yet ch...   \n",
       "4  medRxiv  2024  High multiplicity of infection or MOI, the num...   \n",
       "5    arXiv  2024  Large language models (LLMs) like ChatGPT have...   \n",
       "6    arXiv  2023  The reinforcement learning research area conta...   \n",
       "7  medRxiv  2023  While the rapid deployment of SARS-CoV-2 vacci...   \n",
       "8    arXiv  2020  It is not until we become senior citizens do w...   \n",
       "9    arXiv  2018  Episodic memory is a psychology term which ref...   \n",
       "\n",
       "                               round-A_Pouria_output round-A_Pouria_score  \\\n",
       "0  {'reasoning': 'The abstract does not mention t...                    2   \n",
       "1  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "2  {'reasoning': 'The article does not discuss la...                    2   \n",
       "3  {'reasoning': 'The article focuses on AI agent...                    2   \n",
       "4  {'reasoning': 'The article focuses on applying...                    2   \n",
       "5  {'reasoning': 'The abstract mentions large lan...                    2   \n",
       "6  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "7  {'reasoning': 'The article does not discuss la...                    2   \n",
       "8  {'reasoning': 'The article does not discuss la...                    2   \n",
       "9  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "\n",
       "                            round-A_Pouria_reasoning  \\\n",
       "0  The abstract does not mention the use of large...   \n",
       "1  The abstract does not mention large language m...   \n",
       "2  The article does not discuss large language mo...   \n",
       "3  The article focuses on AI agents applied to me...   \n",
       "4  The article focuses on applying queuing theory...   \n",
       "5  The abstract mentions large language models (L...   \n",
       "6  The abstract does not mention large language m...   \n",
       "7  The article does not discuss large language mo...   \n",
       "8  The article does not discuss large language mo...   \n",
       "9  The abstract does not mention large language m...   \n",
       "\n",
       "                               round-A_Bardia_output round-A_Bardia_score  \\\n",
       "0  {'reasoning': 'The article focuses on a surgeo...                    2   \n",
       "1  {'reasoning': 'The article does not discuss la...                    2   \n",
       "2  {'reasoning': 'This paper discusses methods re...                    2   \n",
       "3  {'reasoning': 'The article focuses on empathy ...                    2   \n",
       "4  {'reasoning': 'The article discusses the appli...                    2   \n",
       "5  {'reasoning': 'The article discusses large lan...                    2   \n",
       "6  {'reasoning': 'The article focuses on reinforc...                    2   \n",
       "7  {'reasoning': 'The article discusses vaccinati...                    2   \n",
       "8  {'reasoning': 'The article does not discuss la...                    2   \n",
       "9  {'reasoning': 'The article discusses a reinfor...                    2   \n",
       "\n",
       "                            round-A_Bardia_reasoning  \n",
       "0  The article focuses on a surgeon-in-the-loop a...  \n",
       "1  The article does not discuss large language mo...  \n",
       "2  This paper discusses methods related to chemic...  \n",
       "3  The article focuses on empathy in human-AI int...  \n",
       "4  The article discusses the application of queui...  \n",
       "5  The article discusses large language models (L...  \n",
       "6  The article focuses on reinforcement learning ...  \n",
       "7  The article discusses vaccination frequency's ...  \n",
       "8  The article does not discuss large language mo...  \n",
       "9  The article discusses a reinforcement learning...  "
      ]
     },
     "execution_count": 10,
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
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "        Title: Towards a Surgeon-in-the-Loop Ophthalmic Robotic Apprentice using Reinforcement and Imitation Learning\n",
      "        Abstract: Robot-assisted surgical systems have demonstrated significant potential in enhancing surgical precision and minimizing human errors. However, existing systems cannot accommodate individual surgeons’ unique preferences and requirements. Additionally, they primarily focus on general surgeries (e.g., laparoscopy) and are unsuitable for highly precise microsurgeries, such as ophthalmic procedures. Thus, we propose an image-guided approach for surgeon-centered autonomous agents that can adapt to the individual surgeon’s skill level and preferred surgical techniques during ophthalmic cataract surgery. Our approach trains reinforcement and imitation learning agents simultaneously using curriculum learning approaches guided by image data to perform all tasks of the incision phase of cataract surgery. By integrating the surgeon’s actions and preferences into the training process, our approach enables the robot to implicitly learn and adapt to the individual surgeon’s unique techniques through surgeon-in-the-loop demonstrations. This results in a more intuitive and personalized surgical experience for the surgeon while ensuring consistent performance for the autonomous robotic apprentice. We define and evaluate the effectiveness of our approach in a simulated environment using our proposed metrics and highlight the trade-off between a generic agent and a surgeon-centered adapted agent. Finally, our approach has the potential to extend to other ophthalmic and microsurgical procedures, opening the door to a new generation of surgeon-in-the-loop autonomous surgical robots. We provide an open-source simulation framework for future development and reproducibility at https://github.com/amrgomaaelhady/CataractAdaptSurgRobot.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention the use of large language models or AI agents in the context of medical imaging.  The focus is on reinforcement and imitation learning for robot-assisted surgery, using image data for training, but not in the way specified by the criteria. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on a surgeon-in-the-loop approach for robotic ophthalmic surgery and discusses reinforcement and imitation learning, but it does not specifically address large language models or their application to medical imaging data.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Computational model for tumor response to adoptive cell transfer therapy\n",
      "        Abstract: One of the barriers to the development of effective adoptive cell transfer therapies (ACT), specifically for genetically engineered T-cell receptors (TCRs), and chimeric antigen receptor (CAR) T-cells, is target antigen heterogeneity. It is thought that intratumor heterogeneity is one of the leading determinants of therapeutic resistance and treatment failure. While understanding antigen heterogeneity is important for effective therapeutics, a good therapy strategy could enhance the therapy efficiency. In this work we introduce an agent-based model to rationalize the outcomes of two types of ACT therapies over heterogeneous tumors: antigen specific ACT therapy and multi-antigen recognition ACT therapy. We found that one dose of antigen specific ACT therapy should be expected to reduce the tumor size as well as its growth rate, however it may not be enough to completely eliminate it. A second dose also reduced the tumor size as well as the tumor growth rate, but, due to the intratumor heterogeneity, it turned out to be less effective than the previous dose. Moreover, an interesting emergent phenomenon results from the simulations, namely the formation of a shield-like structure of cells with low oncoprotein expression. This shield turns out to protect cells with high oncoprotein expression. On the other hand, our studies suggest that the earlier the multi-antigen recognition ACT therapy is applied, the more efficient it turns. In fact, it could completely eliminate the tumor. Based on our results, it is clear that a proper therapeutic strategy could enhance the therapies outcomes. In that direction, our computational approach provides a framework to model treatment combinations in different scenarios and explore the characteristics of successful and unsuccessful treatments.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models or AI agents.  The research uses an agent-based model, but this is a computational modeling technique distinct from large language models used in AI. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article does not discuss large language models or AI agents applied to medical imaging data; it focuses on computational modeling of tumor response to cell therapy.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Using a library of chemical reactions to fit systems of ordinary differential equations to agent-based models: a machine learning approach\n",
      "        Abstract: In this paper we introduce a new method based on a library of chemical reactions for constructing a system of ordinary differential equations from stochastic simulations arising from an agent-based model. The advantage of this approach is that this library respects any coupling between systems components, whereas the SINDy algorithm (introduced by Brunton, Proctor and Kutz) treats the individual components as decoupled from one another. Another advantage of our approach is that we can use a non-negative least squares algorithm to find the non-negative rate constants in a very robust, stable and simple manner. We illustrate our ideas on an agent-based model of tumour growth on a 2D lattice.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: This paper discusses methods related to chemical reactions and ordinary differential equations, which do not pertain to large language models or medical imaging data.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Talk, Listen, Connect: Navigating Empathy in Human-AI Interactions\n",
      "        Abstract: Social interactions promote well-being, yet challenges like geographic distance and mental health conditions can limit in-person engagement. Advances in AI agents are transferring communication, particularly in mental health, where AI chatbots provide accessible, non-judgmental support. However, a key challenge is how effectively these systems can express empathy, which is crucial in human-centered design. Current research highlights a gap in understanding how AI can authentically convey empathy, particularly as issues like anxiety, depression, and loneliness increase. Our research focuses on this gap by comparing empathy expression in human-human versus human-AI interactions. Using personal narratives and statistical analysis, we examine empathy levels elicited by humans and AI, including GPT-4o and fine-tuned versions of the model. This work aims to enhance the authenticity of AI-driven empathy, contributing to the future design of more reliable and effective mental health support systems that foster meaningful social interactions.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article focuses on AI agents applied to mental health support, using large language models like GPT-4.  It does not, however, discuss the application of AI agents to medical imaging data. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on empathy in human-AI interactions rather than the application of large language models in medical imaging data, which does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: From multiplicity of infection to force of infection for sparsely sampled Plasmodium falciparum populations at high transmission\n",
      "        Abstract: High multiplicity of infection or MOI, the number of genetically distinct parasite strains co-infecting a single human host, characterizes infectious diseases including falciparum malaria at high transmission. It accompanies high asymptomatic Plasmodium falciparum prevalence despite high exposure, creating a large transmission reservoir challenging intervention. High MOI and asymptomatic prevalence are enabled by immune evasion of the parasite achieved via vast antigenic diversity. Force of infection or FOI, the number of new infections acquired by an individual host over a given time interval, is the dynamic sister quantity of MOI, and a key epidemiological parameter for monitoring the impact of antimalarial interventions and assessing vaccine or drug efficacy in clinical trials. FOI remains difficult, expensive, and labor-intensive to accurately measure, especially in high-transmission regions, whether directly via cohort studies or indirectly via the fitting of epidemiological models to repeated cross-sectional surveys. We propose here the application of queuing theory to obtain FOI on the basis of MOI, in the form of either a two-moment approximation method or Little’s law. We illustrate these methods with MOI estimates obtained under sparse sampling schemes with the recently proposed \"varcoding\" method, based on sequences of the var multigene family encoding for the major variant surface antigen of the blood stage of malaria infection. The methods are evaluated with simulation output from a stochastic agent-based model, and are applied to an interrupted time-series study from Bongo District in northern Ghana before and immediately after a three-round transient indoor residual spraying (IRS) intervention. We incorporate into the sampling of the simulation output, limitations representative of those encountered in the collection of field data, including under-sampling of var genes, missing data, and usage of antimalarial drug treatment. We address these limitations in MOI estimates with a Bayesian framework and an imputation bootstrap approach. We demonstrate that both proposed methods give good and consistent FOI estimates across various simulated scenarios. Their application to the field surveys shows a pronounced reduction in annual FOI during intervention, of more than 70%. The proposed approach should be applicable to the many geographical locations where cohort or cross-sectional studies with regular and frequent sampling are lacking but single-time-point surveys under sparse sampling schemes are available, and for MOI estimates obtained in different ways. They should also be relevant to other pathogens of humans, wildlife and livestock whose immune evasion strategies are based on large antigenic variation resulting in high multiplicity of infection.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article focuses on applying queuing theory and statistical methods to estimate force of infection from multiplicity of infection data in malaria, which is not related to large language models or medical imaging.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses the application of queuing theory and epidemiological modeling to malaria transmission dynamics, but does not involve large language models-based AI agents or medical imaging data, thus it does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Can large language models understand uncommon meanings of common words?\n",
      "        Abstract: Large language models (LLMs) like ChatGPT have shown significant advancements across diverse natural language understanding (NLU) tasks, including intelligent dialogue and autonomous agents. Yet, lacking widely acknowledged testing mechanisms, answering ‘whether LLMs are stochastic parrots or genuinely comprehend the world’ remains unclear, fostering numerous studies and sparking heated debates. Prevailing research mainly focuses on surface-level NLU, neglecting fine-grained explorations. However, such explorations are crucial for understanding their unique comprehension mechanisms, aligning with human cognition, and finally enhancing LLMs’ general NLU capacities. To address this gap, our study delves into LLMs’ nuanced semantic comprehension capabilities, particularly regarding common words with uncommon meanings. The idea stems from foundational principles of human communication within psychology, which underscore accurate shared understandings of word semantics. Specifically, this paper presents the innovative construction of a Lexical Semantic Comprehension (LeSC) dataset with novel evaluation metrics, the first benchmark encompassing both fine-grained and cross-lingual dimensions. Introducing models of both open-source and closed-source, varied scales and architectures, our extensive empirical experiments demonstrate the inferior performance of existing models in this basic lexical-meaning understanding task. Notably, even the state-of-the-art LLMs GPT-4 and GPT-3.5 lag behind 16-year-old humans by 3.9% and 22.3%, respectively. Additionally, multiple advanced prompting techniques and retrieval-augmented generation are also introduced to help alleviate this trouble, yet limitations persist. By highlighting the above critical shortcomings, this research motivates further investigation and offers novel insights for developing more intelligent LLMs. The resources are available at https://github.com/jinyangwu/LeSC.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract mentions large language models (LLMs), but it does not discuss their application to medical imaging data.  The focus is on the models' understanding of nuanced word meanings, a topic in natural language processing, not medical image analysis. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses large language models (LLMs) but does not address their application to medical imaging data; therefore, it does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Intrinsic Motivation in Model-based Reinforcement Learning: A Brief Review\n",
      "        Abstract: The reinforcement learning research area contains a wide range of methods for solving the problems of intelligent agent control. Despite the progress that has been made, the task of creating a highly autonomous agent is still a significant challenge. One potential solution to this problem is intrinsic motivation, a concept derived from developmental psychology. This review considers the existing methods for determining intrinsic motivation based on the world model obtained by the agent. We propose a systematic approach to current research in this field, which consists of three categories of methods, distinguished by the way they utilize a world model in the agent’s components: complementary intrinsic reward, exploration policy, and intrinsically motivated goals. The proposed unified framework describes the architecture of agents using a world model and intrinsic motivation to improve learning. The potential for developing new techniques in this area of research is also examined.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models, AI agents, or medical imaging. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article focuses on reinforcement learning and intrinsic motivation rather than large language models or their application to medical imaging, making it irrelevant to the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: The impact of vaccination frequency on COVID-19 public health outcomes: A model-based analysis\n",
      "        Abstract: While the rapid deployment of SARS-CoV-2 vaccines had a significant impact on the ongoing COVID-19 pandemic, rapid viral immune evasion and waning neutralizing antibody titers have degraded vaccine efficacy. Nevertheless, vaccine manufacturers and public health authorities have a number of levers at their disposal to maximize the benefits of vaccination. Here, we use an agent-based modeling framework coupled with the outputs of a population pharmacokinetic model to examine the impact of boosting frequency and durability of vaccinal response on vaccine efficacy. Our work suggests that repeated dosing at frequent intervals (multiple times a year) may offset the degradation of vaccine efficacy, preserving their utility in managing the ongoing pandemic. Our work relies on assumptions about antibody accumulation and the tolerability of repeated vaccine doses. Given the practical significance of potential improvements in vaccinal utility, clinical research to better understand the effects of repeated vaccination would be highly impactful. These findings are particularly relevant as public health authorities worldwide seek to reduce the frequency of boosters to once a year or less. Our work suggests practical recommendations for vaccine manufacturers and public health authorities and draws attention to the possibility that better outcomes for SARS-CoV-2 public health remain within reach.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  The abstract clearly focuses on agent-based modeling in the context of vaccine efficacy and COVID-19, which is unrelated to medical imaging or large language models. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses vaccination frequency's impact on COVID-19 outcomes, which is unrelated to large language models or medical imaging data.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Biomechanic posture stabilisation via iterative training of multi-policy deep reinforcement learning agents\n",
      "        Abstract: It is not until we become senior citizens do we recognise how much we took maintaining a simple standing posture for granted. It is truly fascinating to observe the magnitude of control the human brain exercises, in real time, to activate and deactivate the lower body muscles and solve a multi-link 3D inverted pendulum problem in order to maintain a stable standing posture. This realisation is even more apparent when training an artificial intelligence (AI) agent to maintain a standing posture of a digital musculoskeletal avatar due to the error propagation problem. In this work we address the error propagation problem by introducing an iterative training procedure for deep reinforcement learning which allows the agent to learn a finite set of actions and how to coordinate between them in order to achieve a stable standing posture. The proposed training approach allowed the agent to increase standing duration from 4 seconds using the traditional training method to 348 seconds using the proposed method. The proposed training method allowed the agent to generalise and accommodate perception and actuation noise for almost 108 seconds.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  The abstract describes using deep reinforcement learning to train an AI agent to maintain a standing posture in a digital musculoskeletal avatar. This is not related to medical imaging or large language models.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article does not discuss large language models or their application to medical imaging data, but rather focuses on deep reinforcement learning for biomechanical posture, which is unrelated to the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Integrating episodic memory into a reinforcement learning agent using reservoir sampling\n",
      "        Abstract: Episodic memory is a psychology term which refers to the ability to recall specific events from the past. We suggest one advantage of this particular type of memory is the ability to easily assign credit to a specific state when remembered information is found to be useful. Inspired by this idea, and the increasing popularity of external memory mechanisms to handle long-term dependencies in deep learning systems, we propose a novel algorithm which uses a reservoir sampling procedure to maintain an external memory consisting of a fixed number of past states. The algorithm allows a deep reinforcement learning agent to learn online to preferentially remember those states which are found to be useful to recall later on. Critically this method allows for efficient online computation of gradient estimates with respect to the write process of the external memory. Thus unlike most prior mechanisms for external memory it is feasible to use in an online reinforcement learning setting.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models, AI agents, or medical imaging data. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses a reinforcement learning agent and episodic memory but does not mention large language models or their application to medical imaging data, making it irrelevant to the specified criteria.\n",
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
            #     txt_response = validated_response.model_dump()
            # except Exception as e:
            #     raise ResponseError(f"Response validation failed: {str(e)}\nResponse: {txt_response}")
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
            data_with_provider['custom_llm_provider'] = custom_llm_provider
        
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
        while num_tried < self.max_retries:
            try:
                item_prompt = self.build_item_prompt(self.item_prompt, {'item': item})
                response, cost = await self.provider.get_json_response(
                    item_prompt,
                    **self.model_args
                )
                return response, cost
            except Exception as e:
                warnings.warn(f"Error reviewing item: {str(e)}. Retrying {num_tried}/{self.num_repeat_task}")
        raise AgentError(f"Error reviewing item: {str(e)}")
```

