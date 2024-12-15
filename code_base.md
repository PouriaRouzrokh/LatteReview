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
      "Reviewing 3 items - 2024-12-14 20:37:42: 100%|██████████| 3/3 [00:06<00:00,  2.04s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Outputs:\n",
      "{'reasoning': 'The title clearly indicates a policy-focused study that addresses the balance between disease containment and social impacts through quarantine segmentation, which is a relevant and well-defined research topic.', 'score': 2}\n",
      "{'reasoning': 'The title clearly indicates a framework for protein engineering that leverages large language models and multimodal AutoML, suggesting a novel and relevant contribution to the field.', 'score': 2}\n",
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
      "['round: A', 'reviewer_name: Pouria'] - 2024-12-14 20:09:54: 100%|██████████| 10/10 [00:01<00:00,  6.03it/s]\n",
      "['round: A', 'reviewer_name: Bardia'] - 2024-12-14 20:09:55: 100%|██████████| 10/10 [00:01<00:00,  6.74it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Starting review round B (2/2)...\n",
      "Processing 2 eligible rows\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "['round: B', 'reviewer_name: Brad'] - 2024-12-14 20:09:57: 100%|██████████| 2/2 [00:02<00:00,  1.41s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total cost: \n",
      "0.00421395\n",
      "\n",
      "Detailed cost:\n",
      "{('A', 'Pouria'): 6.555e-05, ('A', 'Bardia'): 0.0001059, ('B', 'Brad'): 0.0040425}\n"
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
       "      <th>round-B_Brad_output</th>\n",
       "      <th>round-B_Brad_score</th>\n",
       "      <th>round-B_Brad_reasoning</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>72</td>\n",
       "      <td>Patient-centered data science: an integrative ...</td>\n",
       "      <td>Amoei, M.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>This study proposes a novel, integrative frame...</td>\n",
       "      <td>{'reasoning': 'The abstract explicitly mention...</td>\n",
       "      <td>1</td>\n",
       "      <td>The abstract explicitly mentions the use of la...</td>\n",
       "      <td>{'reasoning': 'The article discusses the use o...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses the use of large languag...</td>\n",
       "      <td>{'reasoning': 'Upon examining the abstract, it...</td>\n",
       "      <td>2</td>\n",
       "      <td>Upon examining the abstract, it does mention t...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>583</td>\n",
       "      <td>Surrogate assisted methods for the parameteris...</td>\n",
       "      <td>Perumal, R.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2020</td>\n",
       "      <td>Parameter calibration is a major challenge in ...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article discusses surrogate...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses surrogate assisted metho...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>614</td>\n",
       "      <td>In silico trial to test COVID-19 candidate vac...</td>\n",
       "      <td>Russo, G.</td>\n",
       "      <td>bioRxiv</td>\n",
       "      <td>2020</td>\n",
       "      <td>SARS-CoV-2 is a severe respiratory infection t...</td>\n",
       "      <td>{'reasoning': 'The article focuses on an in si...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article focuses on an in silico trial plat...</td>\n",
       "      <td>{'reasoning': 'The article discusses an in sil...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses an in silico platform fo...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>213</td>\n",
       "      <td>Natural Language Programming in Medicine: Admi...</td>\n",
       "      <td>Vaid, A.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Background Generative Large Language Models (L...</td>\n",
       "      <td>{'reasoning': 'The article title and abstract ...</td>\n",
       "      <td>1</td>\n",
       "      <td>The article title and abstract clearly indicat...</td>\n",
       "      <td>{'reasoning': 'The article discusses the appli...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses the application of large...</td>\n",
       "      <td>{'reasoning': 'The core distinction in the rev...</td>\n",
       "      <td>1</td>\n",
       "      <td>The core distinction in the reviews by Pouria ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>443</td>\n",
       "      <td>First passage time analysis of spatial mutatio...</td>\n",
       "      <td>Haughey, M.J.</td>\n",
       "      <td>bioRxiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>The footprint left by early cancer dynamics on...</td>\n",
       "      <td>{'reasoning': 'This article focuses on colorec...</td>\n",
       "      <td>2</td>\n",
       "      <td>This article focuses on colorectal cancer and ...</td>\n",
       "      <td>{'reasoning': 'The article discusses spatial m...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses spatial mutation pattern...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>413</td>\n",
       "      <td>Innovations in Integrating Machine Learning an...</td>\n",
       "      <td>Sivakumar, N.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>Abbreviations, acronyms: ABM, agent-based mode...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The title and abstract do not m...</td>\n",
       "      <td>2</td>\n",
       "      <td>The title and abstract do not mention large la...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>358</td>\n",
       "      <td>TEAMWORK UNDER EXTREME UNCERTAINTY: AI FOR POK...</td>\n",
       "      <td>Sarantinos, N.R.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>The highest grossing media franchise of all ti...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'This article focuses on AI agen...</td>\n",
       "      <td>2</td>\n",
       "      <td>This article focuses on AI agents for video ga...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>370</td>\n",
       "      <td>Generating synthetic data with a mechanism-bas...</td>\n",
       "      <td>Cockrell, C.</td>\n",
       "      <td>bioRxiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>Machine learning (ML) and Artificial Intellige...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The article discusses generatin...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses generating synthetic dat...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>708</td>\n",
       "      <td>Inferring the ground truth through crowdsourcing</td>\n",
       "      <td>Char, J.P.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2018</td>\n",
       "      <td>—Universally valid ground truth is almost impo...</td>\n",
       "      <td>{'reasoning': 'The abstract does not mention l...</td>\n",
       "      <td>2</td>\n",
       "      <td>The abstract does not mention large language m...</td>\n",
       "      <td>{'reasoning': 'The article discusses crowdsour...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses crowdsourcing for inferr...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>742</td>\n",
       "      <td>Verification &amp;Validation of Agent Based Simula...</td>\n",
       "      <td>Niazi, M.A.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2017</td>\n",
       "      <td>Agent Based Models are very popular in a numbe...</td>\n",
       "      <td>{'reasoning': 'The article does not discuss la...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article does not discuss large language mo...</td>\n",
       "      <td>{'reasoning': 'The article discusses agent-bas...</td>\n",
       "      <td>2</td>\n",
       "      <td>The article discusses agent-based models and t...</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID                                              Title        1st author  \\\n",
       "0   72  Patient-centered data science: an integrative ...         Amoei, M.   \n",
       "1  583  Surrogate assisted methods for the parameteris...       Perumal, R.   \n",
       "2  614  In silico trial to test COVID-19 candidate vac...         Russo, G.   \n",
       "3  213  Natural Language Programming in Medicine: Admi...          Vaid, A.   \n",
       "4  443  First passage time analysis of spatial mutatio...     Haughey, M.J.   \n",
       "5  413  Innovations in Integrating Machine Learning an...     Sivakumar, N.   \n",
       "6  358  TEAMWORK UNDER EXTREME UNCERTAINTY: AI FOR POK...  Sarantinos, N.R.   \n",
       "7  370  Generating synthetic data with a mechanism-bas...      Cockrell, C.   \n",
       "8  708   Inferring the ground truth through crowdsourcing        Char, J.P.   \n",
       "9  742  Verification &Validation of Agent Based Simula...       Niazi, M.A.   \n",
       "\n",
       "      repo  year                                           abstract  \\\n",
       "0    arXiv  2024  This study proposes a novel, integrative frame...   \n",
       "1    arXiv  2020  Parameter calibration is a major challenge in ...   \n",
       "2  bioRxiv  2020  SARS-CoV-2 is a severe respiratory infection t...   \n",
       "3    arXiv  2024  Background Generative Large Language Models (L...   \n",
       "4  bioRxiv  2022  The footprint left by early cancer dynamics on...   \n",
       "5    arXiv  2022  Abbreviations, acronyms: ABM, agent-based mode...   \n",
       "6    arXiv  2022  The highest grossing media franchise of all ti...   \n",
       "7  bioRxiv  2022  Machine learning (ML) and Artificial Intellige...   \n",
       "8    arXiv  2018  —Universally valid ground truth is almost impo...   \n",
       "9    arXiv  2017  Agent Based Models are very popular in a numbe...   \n",
       "\n",
       "                               round-A_Pouria_output round-A_Pouria_score  \\\n",
       "0  {'reasoning': 'The abstract explicitly mention...                    1   \n",
       "1  {'reasoning': 'The article does not discuss la...                    2   \n",
       "2  {'reasoning': 'The article focuses on an in si...                    2   \n",
       "3  {'reasoning': 'The article title and abstract ...                    1   \n",
       "4  {'reasoning': 'This article focuses on colorec...                    2   \n",
       "5  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "6  {'reasoning': 'The article does not discuss la...                    2   \n",
       "7  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "8  {'reasoning': 'The abstract does not mention l...                    2   \n",
       "9  {'reasoning': 'The article does not discuss la...                    2   \n",
       "\n",
       "                            round-A_Pouria_reasoning  \\\n",
       "0  The abstract explicitly mentions the use of la...   \n",
       "1  The article does not discuss large language mo...   \n",
       "2  The article focuses on an in silico trial plat...   \n",
       "3  The article title and abstract clearly indicat...   \n",
       "4  This article focuses on colorectal cancer and ...   \n",
       "5  The abstract does not mention large language m...   \n",
       "6  The article does not discuss large language mo...   \n",
       "7  The abstract does not mention large language m...   \n",
       "8  The abstract does not mention large language m...   \n",
       "9  The article does not discuss large language mo...   \n",
       "\n",
       "                               round-A_Bardia_output round-A_Bardia_score  \\\n",
       "0  {'reasoning': 'The article discusses the use o...                    2   \n",
       "1  {'reasoning': 'The article discusses surrogate...                    2   \n",
       "2  {'reasoning': 'The article discusses an in sil...                    2   \n",
       "3  {'reasoning': 'The article discusses the appli...                    2   \n",
       "4  {'reasoning': 'The article discusses spatial m...                    2   \n",
       "5  {'reasoning': 'The title and abstract do not m...                    2   \n",
       "6  {'reasoning': 'This article focuses on AI agen...                    2   \n",
       "7  {'reasoning': 'The article discusses generatin...                    2   \n",
       "8  {'reasoning': 'The article discusses crowdsour...                    2   \n",
       "9  {'reasoning': 'The article discusses agent-bas...                    2   \n",
       "\n",
       "                            round-A_Bardia_reasoning  \\\n",
       "0  The article discusses the use of large languag...   \n",
       "1  The article discusses surrogate assisted metho...   \n",
       "2  The article discusses an in silico platform fo...   \n",
       "3  The article discusses the application of large...   \n",
       "4  The article discusses spatial mutation pattern...   \n",
       "5  The title and abstract do not mention large la...   \n",
       "6  This article focuses on AI agents for video ga...   \n",
       "7  The article discusses generating synthetic dat...   \n",
       "8  The article discusses crowdsourcing for inferr...   \n",
       "9  The article discusses agent-based models and t...   \n",
       "\n",
       "                                 round-B_Brad_output round-B_Brad_score  \\\n",
       "0  {'reasoning': 'Upon examining the abstract, it...                  2   \n",
       "1                                               None               None   \n",
       "2                                               None               None   \n",
       "3  {'reasoning': 'The core distinction in the rev...                  1   \n",
       "4                                               None               None   \n",
       "5                                               None               None   \n",
       "6                                               None               None   \n",
       "7                                               None               None   \n",
       "8                                               None               None   \n",
       "9                                               None               None   \n",
       "\n",
       "                              round-B_Brad_reasoning  \n",
       "0  Upon examining the abstract, it does mention t...  \n",
       "1                                               None  \n",
       "2                                               None  \n",
       "3  The core distinction in the reviews by Pouria ...  \n",
       "4                                               None  \n",
       "5                                               None  \n",
       "6                                               None  \n",
       "7                                               None  \n",
       "8                                               None  \n",
       "9                                               None  "
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
      "        Title: Patient-centered data science: an integrative framework for evaluating and predicting clinical outcomes in the digital health era\n",
      "        Abstract: This study proposes a novel, integrative framework for patient-centered data science in the digital health era. We developed a multidimensional model that combines traditional clinical data with patient-reported outcomes, social determinants of health, and multi-omic data to create comprehensive digital patient representations. Our framework employs a multi-agent artificial intelligence approach, utilizing various machine learning techniques including large language models, to analyze complex, longitudinal datasets. The model aims to optimize multiple patient outcomes simultaneously while addressing biases and ensuring generalizability. We demonstrate how this framework can be implemented to create a learning healthcare system that continuously refines strategies for optimal patient care. This approach has the potential to significantly improve the translation of digital health innovations into real-world clinical benefits, addressing current limitations in AI-driven healthcare models.\n",
      "        Pouria's score: 1\n",
      "        Pouria's reasoning: The abstract explicitly mentions the use of large language models (LLMs) within a multi-agent AI approach for analyzing complex, longitudinal datasets in healthcare.  While not exclusively focused on medical imaging, the broad application of LLMs to healthcare data, including the potential for use with medical imaging data, makes this relevant. Therefore, it meets the criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses the use of large language models within a broader patient-centered data science framework, but it does not specifically focus on the application of these models to medical imaging data, which is a key aspect of the criteria.\n",
      "        Brad's score: 2\n",
      "        Brad's reasoning: Upon examining the abstract, it does mention the use of large language models within a comprehensive AI approach aimed at healthcare data analysis. However, the abstract does not specifically highlight the direct application of these models to medical imaging data. The focus is instead on a broader integrative framework that combines various types of data including clinical, patient-reported, social determinants, and multi-omic data. Pouria's reasoning points out the potential relevance to medical imaging data without confirming its specific application, while Bardia correctly identifies the lack of a specific focus on medical imaging within the abstract. Therefore, Bardia's interpretation is more aligned with the criteria, as medical imaging data involvement is not explicitly covered.\n",
      "        \n",
      "\n",
      "        Title: Surrogate assisted methods for the parameterisation of agent-based models\n",
      "        Abstract: Parameter calibration is a major challenge in agent-based modelling and simulation (ABMS). As the complexity of agent-based models (ABMs) increase, the number of parameters required to be calibrated grows. This leads to the ABMS equivalent of the “curse of dimensionality”. We propose an ABMS framework which facilitates the effective integration of different sampling methods and surrogate models (SMs) in order to evaluate how these strategies affect parameter calibration and exploration. We show that surrogate assisted methods perform better than the standard sampling methods. In addition, we show that the XGBoost and Decision Tree SMs are most optimal overall with regards to our analysis.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  The abstract focuses on parameter calibration in agent-based modeling and simulation, a different area of research.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses surrogate assisted methods in agent-based modeling but does not address large language models or their application to medical imaging data, which makes it irrelevant to the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: In silico trial to test COVID-19 candidate vaccines: A case study with UISS platform\n",
      "        Abstract: SARS-CoV-2 is a severe respiratory infection that infects humans. Its outburst entitled it as a pandemic emergence. To get a grip on this outbreak, specific preventive and therapeutic interventions are urgently needed. It must be said that, until now, there are no existing vaccines for coronaviruses. To promptly and rapidly respond to pandemic events, the application of in silico trials can be used for designing and testing medicines against SARS-CoV-2 and speed-up the vaccine discovery pipeline, predicting any therapeutic failure and minimizing undesired effects. Here, we present an in silico platform that showed to be in very good agreement with the latest literature in predicting SARS-CoV-2 dynamics and related immune system host response. Moreover, it has been used to predict the outcome of one of the latest suggested approach to design an effective vaccine, based on monoclonal antibody. Universal Immune System Simulator (UISS) in silico platform is potentially ready to be used as an in silico trial platform to predict the outcome of vaccination strategy against SARS-CoV-2.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article focuses on an in silico trial platform for testing COVID-19 vaccines, which is not directly related to large language models-based AI agents applied to medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses an in silico platform for testing vaccines against COVID-19, which does not involve large language models or medical imaging data, thus it does not meet the criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Natural Language Programming in Medicine: Administering Evidence Based Clinical Workflows with Autonomous Agents Powered by Generative Large Language Models\n",
      "        Abstract: Background Generative Large Language Models (LLMs) have emerged as versatile tools in healthcare, demonstrating the ability to regurgitate clinical knowledge and pass medical licensing exams. Despite their promise, they have been largely treated as slow, imperfect information retrieval tools and face limitations such as data staleness, resource intensity, and manufacturing incorrect text - reducing their applicability to dynamic healthcare settings. Methods This study explored the functionality of both proprietary and open-source LLMs to act as autonomous agents within a simulated tertiary care medical center. Real-world clinical cases across multiple specialties were structured into JSON files and presented to agents for solution using the resources available to a human physician. Agents were created using LLMs in combination with natural language prompts, tools with real-world interactions, and standard programming techniques. The technique of Retrieval Augmented Generation was used to provide agents with updated context whenever appropriate. Expert clinicians collected and evaluated model responses across several performance metrics including correctness of the final answer, judicious use of tools, guideline conformity, and resistance to hallucinations. Findings Agents showed varied performance across specialties, with proprietary models (e.g., GPT-4) generally outperforming open-source models. The use of Retrieval Augmented Generation (RAG) improved guideline adherence and contextually relevant responses for the best performing model. Interpretation LLMs can effectively function as autonomous agents in healthcare by leveraging their generative capabilities and integrating with real-world data. The study highlights the potential of LLMs to enhance decision-making in clinical settings through tailored prompts and retrieval tools. However, the variability in model performance and the necessity for ongoing manual evaluation suggest that further refinements in LLM technology and operational protocols are needed to optimize their utility in healthcare.\n",
      "        Pouria's score: 1\n",
      "        Pouria's reasoning: The article title and abstract clearly indicate that the study focuses on large language models (LLMs) as autonomous agents in a simulated medical setting.  The methods section explicitly mentions using LLMs to process structured clinical cases presented in JSON format.  The findings discuss the performance of these LLM-based agents in various medical specialties. Therefore, the article directly addresses the application of LLMs to medical data, meeting the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses the application of large language models (LLMs) as autonomous agents in a healthcare setting, but it does not specifically address their use with medical imaging data.\n",
      "        Brad's score: 1\n",
      "        Brad's reasoning: The core distinction in the reviews by Pouria and Bardia is on whether the article addresses large language models in the context of medical imaging specifically, as opposed to healthcare applications more generally. Pouria focuses on the fact that LLMs are discussed as autonomous agents in a medical setting, which aligns with the criteria, but not specifically on medical imaging. Bardia accurately points out that the article does not specifically address medical imaging data, which is a critical element of the criteria provided. Given the task's requirement for the study to focus on the application of large language models to medical imaging data, Bardia's reasoning is more aligned with the outlined criteria. Therefore, the correct score aligns with Bardia's interpretation that the article does not meet the required criteria for addressing LLMs in the context of medical imaging data specifically.\n",
      "        \n",
      "\n",
      "        Title: First passage time analysis of spatial mutation patterns reveals evolutionary dynamics of pre-existing resistance in colorectal cancer\n",
      "        Abstract: The footprint left by early cancer dynamics on the spatial arrangement of tumour cells is poorly understood, and yet could encode information about how therapy resistant sub-clones grew within the expanding tumour. Novel methods of quantifying spatial tumour data at the cellular scale are required to link evolutionary dynamics to the resulting spatial architecture of the tumour. Here, we propose a framework using first passage times of random walks to quantify the complex spatial patterns of tumour cell population mixing. First, using a toy model of cell mixing we demonstrate how first passage time statistics can distinguish between different pattern structures. We then apply our method to simulated patterns of wild-type and mutated tumour cell population mixing, generated using an agent-based model of expanding tumours, to explore how first passage times reflect mutant cell replicative advantage, time of emergence and strength of cell pushing. Finally, we analyse experimentally measured patterns of genetic point mutations in human colorectal cancer, and estimate parameters of early sub-clonal dynamics using our spatial computational model. We uncover a wide range of mutant cell replicative advantages and timings, with the majority of sampled tumours consistent with boundary driven growth or short-range cell pushing. By analysing multiple sub-sampled regions in a small number of samples, we explore how the distribution of inferred dynamics could inform about the initial mutational event. Our results demonstrate the efficacy of first passage time analysis as a new methodology for quantifying cell mixing patterns in vivo, and suggest that patterns of sub-clonal mixing can provide insights into early cancer dynamics.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: This article focuses on colorectal cancer and uses an agent-based model to simulate tumor growth.  It does not involve large language models or AI agents applied to medical imaging data. Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses spatial mutation patterns in colorectal cancer and their dynamics rather than large language models applied to medical imaging data, therefore it does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Innovations in Integrating Machine Learning and Agent-Based Modeling of Biomedical Systems\n",
      "        Abstract: Abbreviations, acronyms: ABM, agent-based model; AI, artificial intelligence; ANN, artificial neural network; BN, Bayesian network; CA, cellular automata; DNN, deep neural network; DR, diabetic retinopathy; EHR, electronic health record; FCM, fuzzy cognitive map or fuzzy c-means (depending upon context); GA, genetic algorithm; ML, machine learning; OOD, out-of-distribution; PDE, partial differential equation; RL, reinforcement learning; RWE, real-world evidence; SA, sensitivity analysis;\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models or AI agents.  The title mentions machine learning, but this is a very broad field and does not necessarily imply the use of large language models in medical imaging. Therefore, the article does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The title and abstract do not mention large language models or their application to medical imaging data, focusing instead on general machine learning and agent-based modeling concepts.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: TEAMWORK UNDER EXTREME UNCERTAINTY: AI FOR POKÉMON RANKS 33RD IN THE WORLD\n",
      "        Abstract: The highest grossing media franchise of all times, with over $90 billion in total revenue, is Pokémon. The video games belong to the class of Japanese Role Playing Games (J-RPG). Developing a powerful AI agent for these games is very hard because they present big challenges to MinMax, Monte Carlo Tree Search and statistical Machine Learning, as they are vastly different from the well explored in AI literature games. An AI agent for one of these games means significant progress in AI agents for the entire class. Further, the key principles of such work can hopefully inspire approaches to several domains that require excellent teamwork under conditions of extreme uncertainty, including managing a team of doctors, robots or employees in an ever changing environment, like a pandemic stricken region or a war-zone. In this paper we first explain the mechanics of the game and we perform a game analysis. We continue by proposing unique AI algorithms based on our understanding that the two biggest challenges in the game are keeping a balanced team and dealing with three sources of uncertainty. Later on, we describe why evaluating the performance of such agents is challenging and we present the results of our approach. Our AI agent performed significantly better than all previous attempts and peaked at the 33rd place in the world, in one of the most popular battle formats, while running on only 4 single socket servers.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: This article focuses on AI agents for video games rather than on large language models applied to medical imaging data, thus not meeting the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Generating synthetic data with a mechanism-based Critical Illness Digital Twin: Demonstration for Post Traumatic Acute Respiratory Distress Syndrome\n",
      "        Abstract: Machine learning (ML) and Artificial Intelligence (AI) approaches are increasingly applied to predicting the development of sepsis and multiple organ failure. While there has been success in demonstrating the clinical utility of such systems in terms of affecting various outcomes, there are fundamental challenges to the ML/AI approach in terms of improving the actual predictive performance and future robustness of such systems. Given that one of the primary proposed avenues for improving algorithmic performance is the addition of molecular/biomarker/genetic features to the data used to train these systems, the overall sparsity of such available data suggests the need to generate synthetic data to aid in training, as has been the case in numerous other ML/AI tasks, such as image recognition/generation and text analysis/generation. We propose the need to generate synthetic molecular/mediator time series data coincides with the advent of the concept of medical digital twins, specifically related to interpretations of medical digital twins that hew closely to the original description and use of industrial digital twins, which involve simulating multiple individual twins from a common computational model specification. Herein we present an example of generating synthetic time series data of a panel of pro- and anti-inflammatory cytokines using the Critical Illness Digital Twin (CIDT) regarding the development of post-traumatic acute respiratory distress syndrome.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models or AI agents applied to medical imaging data.  The focus is on generating synthetic data for sepsis and multiple organ failure prediction using a digital twin approach, which is a different area of AI in medicine.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses generating synthetic data and the use of digital twins in healthcare rather than focusing on large language models applied to medical imaging data, which does not meet the specified criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Inferring the ground truth through crowdsourcing\n",
      "        Abstract: —Universally valid ground truth is almost impossible to obtain or would come at a very high cost. For supervised learning without universally valid ground truth, a recommended approach is applying crowdsourcing: Gathering a large data set annotated by multiple individuals of varying possibly expertise levels and inferring the ground truth data to be used as labels to train the classifier. Nevertheless, due to the sensitivity of the problem at hand (e.g. mitosis detection in breast cancer histology images), the obtained data needs verification and proper assessment before being used for classifier training. Even in the context of organic computing systems, an indisputable ground truth might not always exist. Therefore, it should be inferred through the aggregation and verification of the local knowledge of each autonomous agent.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The abstract does not mention large language models or AI agents, nor does it discuss their application to medical imaging data.  Therefore, it does not meet the specified criteria.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses crowdsourcing for inferring ground truth in supervised learning but does not mention large language models or AI agents applied to medical imaging data, so it does not meet the criteria.\n",
      "        Brad's score: None\n",
      "        Brad's reasoning: None\n",
      "        \n",
      "\n",
      "        Title: Verification &Validation of Agent Based Simulations using the VOMAS (Virtual Overlay Multi-agent System) approach\n",
      "        Abstract: Agent Based Models are very popular in a number of different areas. For example, they have been used in a range of domains ranging from modeling of tumor growth, immune systems, molecules to models of social networks, crowds and computer and mobile self-organizing networks. One reason for their success is their intuitiveness and similarity to human cognition. However, with this power of abstraction, in spite of being easily applicable to such a wide number of domains, it is hard to validate agent-based models. In addition, building valid and credible simulations is not just a challenging task but also a crucial exercise to ensure that what we are modeling is, at some level of abstraction, a model of our conceptual system; the system that we have in mind. In this paper, we address this important area of validation of agent based models by presenting a novel technique which has broad applicability and can be applied to all kinds of agent-based models. We present a framework, where a virtual overlay multi-agent system can be used to validate simulation models. In addition, since agent-based models have been typically growing, in parallel, in multiple domains, to cater for all of these, we present a new single validation technique applicable to all agent based models. Our technique, which allows for the validation of agent based simulations uses VOMAS: a Virtual Overlay Multi-agent System. This overlay multi-agent system can comprise various types of agents, which form an overlay on top of the agent based simulation model that needs to be validated. Other than being able to watch and log, each of these agents contains clearly defined constraints, which, if violated, can be logged in real time. To demonstrate its effectiveness, we show its broad applicability in a wide variety of simulation models ranging from social sciences to computer networks in spatial and non-spatial conceptual models.\n",
      "        Pouria's score: 2\n",
      "        Pouria's reasoning: The article does not discuss large language models-based AI agents applied to medical imaging data.  The abstract focuses on agent-based modeling and simulation validation, a different topic.\n",
      "        Bardia's score: 2\n",
      "        Bardia's reasoning: The article discusses agent-based models and their validation techniques but does not focus on large language models or their application to medical imaging data.\n",
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
from litellm import acompletion, completion_cost, get_supported_openai_params, supports_response_schema
from .base_provider import BaseProvider, ProviderError, ResponseError

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
        while num_tried < self.num_repeat_task:
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

