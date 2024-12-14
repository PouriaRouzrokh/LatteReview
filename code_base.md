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
   "execution_count": 25,
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
    "import pandas as pd\n",
    "\n",
    "sys.path.append('../')\n",
    "from lattereview.providers.openai_provider import OpenAIProvider\n",
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
   "execution_count": null,
   "metadata": {},
   "outputs": [],
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
   "execution_count": 4,
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
     "execution_count": 4,
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
    "Testing the OpenAI provider:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('The capital of France is Paris.',\n",
       " {'input_cost': 1.05e-06, 'output_cost': 4.2e-06, 'total_cost': 5.25e-06})"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "openanai_provider = OpenAIProvider(model=\"gpt-4o-mini\")\n",
    "question = \"What is the capital of France?\"\n",
    "asyncio.run(openanai_provider.get_response(question, temperature=0.9))"
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
   "execution_count": 6,
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
      "integration of large vision language models for efficient post-disaster damage assessment and reporting\n",
      "\n",
      "Outputs:\n",
      "\n",
      " {\"score\":2,\"reasoning\":\"The segmentized quarantine policy effectively addresses the balance between controlling infectious disease spread and minimizing social costs, demonstrating a thoughtful approach to public health management.\"}\n",
      "{\"score\":2,\"reasoning\":\"The article presents a comprehensive framework for integrating large language models into protein engineering, which aligns well with current trends in multimodal automl, thus meeting the criteria effectively.\"}\n",
      "{\"score\":2,\"reasoning\":\"The integration of large vision language models for post-disaster damage assessment is a relevant and innovative approach that meets the criteria for effective reporting and analysis.\"}\n",
      "\n",
      "Costs:\n",
      "\n",
      "{'input_cost': 2.13e-05, 'output_cost': 2.22e-05, 'total_cost': 4.35e-05}\n",
      "{'input_cost': 2.145e-05, 'output_cost': 2.52e-05, 'total_cost': 4.665e-05}\n",
      "{'input_cost': 2.07e-05, 'output_cost': 2.28e-05, 'total_cost': 4.35e-05}\n",
      "\n",
      "Total cost:\n",
      "\n",
      "4.35e-05\n"
     ]
    }
   ],
   "source": [
    "agent = ScoringReviewer(\n",
    "    provider=OpenAIProvider(model=\"gpt-4o-mini\"),\n",
    "    name=\"Pouria\",\n",
    "    backstory=\"an expert reviewer and researcher!\",\n",
    "    input_description = \"article title\",\n",
    "    temperature=0.1,\n",
    "    reasoning = \"brief\",\n",
    "    max_tokens=100,\n",
    "    review_criteria=\"Look for articles that certainly do not employ any AI or machine learning agents\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules='Your scores should be either 1 or 2. 1 means that the paper does not meet the criteria, and 2 means that the paper meets the criteria.',\n",
    ")\n",
    "\n",
    "\n",
    "# Dummy input\n",
    "text_list = data.Title.str.lower().tolist()\n",
    "print(\"Inputs:\\n\\n\", '\\n'.join(text_list[:3]))\n",
    "\n",
    "# Dummy review\n",
    "results, total_cost = asyncio.run(agent.review_items(text_list[:3]))\n",
    "print(\"\\nOutputs:\\n\\n\", '\\n'.join(results))\n",
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
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "pouria = ScoringReviewer(\n",
    "    provider=OpenAIProvider(model=\"gpt-4o-mini\"),\n",
    "    name=\"Pouria\",\n",
    "    backstory=\"You are a junior radiologist with many years of background in statistcis and data science, who are famous among your colleagues for your systematic thinking, organizaton of thoughts, and being conservative\",\n",
    "    input_description = \"tilte and abstract of scientific articles\",\n",
    "    temperature=0.1,\n",
    "    reasoning = \"brief\",\n",
    "    max_tokens=100,\n",
    "    scoring_task=\"Look for articles that disucss large languange models-based AI agents applied to medical imaging data\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules='Your scores should be either 1 or 2. 1 means that the paper does not meet the criteria, and 2 means that the paper meets the criteria.',\n",
    ")\n",
    "\n",
    "bardia = ScoringReviewer(\n",
    "    provider=OpenAIProvider(model=\"gpt-4o-mini\"),\n",
    "    name=\"Bardia\",\n",
    "    backstory=\"You are an expert in data science with a background in developing ML models for healthcare, who are famous among your colleagues for your creativity and out of the box thinking\",\n",
    "    input_description = \"tilte and abstract of scientific articles\",\n",
    "    temperature=0.7,\n",
    "    reasoning = \"brief\",\n",
    "    max_tokens=100,\n",
    "    scoring_task=\"Look for articles that disucss large languange models-based AI agents applied to medical imaging data\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules='Your scores should be either 1 or 2. 1 means that the paper does not meet the criteria, and 2 means that the paper meets the criteria.',\n",
    ")\n",
    "\n",
    "brad = ScoringReviewer(\n",
    "    provider=OpenAIProvider(model=\"gpt-4o\"),\n",
    "    name=\"Brad\",\n",
    "    backstory=\"You are a senior radiologist with a PhD in computer science and years of experience as the director of a DL lab focused on developing ML models for radiology and healthcare\",\n",
    "    input_description = \"tilte and abstract of scientific articles\",\n",
    "    temperature=0.4,\n",
    "    reasoning = \"cot\",\n",
    "    max_tokens=100,\n",
    "    scoring_task=\"\"\"Pouria and Bardia have Looked for articles that disucss large languange models-based AI agents applied to medical imaging data. \n",
    "                       They scored an article 1 if they thought it does not meet this criteria, 2 if they thought it meets the criteria, 0 if they were uncertain of scoring.\n",
    "                       You will receive an article they have had different opinions about, as well as each of their scores and their reasoning for that score. Read their reviews and determine who you agree with. \n",
    "                    \"\"\",\n",
    "    score_set=[1, 2],\n",
    "    scoring_rules=\"\"\"Your scores should be either 1 or 2. \n",
    "                     1 means that you agree with Pouria, 2 means that you agree with Bardia, \n",
    "                  \"\"\",\n",
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
   "execution_count": 37,
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
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Starting review round A (1/2)...\n",
      "Reviewers: ['Pouria', 'Bardia']\n",
      "Input data: ['Title', 'abstract']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                       \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of eligible rows for review: 10\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                        \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Starting review round B (2/2)...\n",
      "Reviewers: ['Brad']\n",
      "Input data: ['Title', 'abstract', 'round-A_Pouria_output', 'round-A_Bardia_output']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                       \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of eligible rows for review: 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                        "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total cost: \n",
      "0.0029861\n",
      "\n",
      "Detailed cost:\n",
      "{('A', 'Pouria'): 9.495e-05, ('A', 'Bardia'): 9.615e-05, ('B', 'Brad'): 0.002795}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r"
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
       "      <th>round-A_Bardia_output</th>\n",
       "      <th>round-B_Brad_output</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>85</td>\n",
       "      <td>An Autonomous GIS Agent Framework for Geospati...</td>\n",
       "      <td>Ning, H.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2024</td>\n",
       "      <td>Powered by the emerging large language models ...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>281</td>\n",
       "      <td>Probabilistic Phase Labeling and Lattice Refin...</td>\n",
       "      <td>Chang, M.-C.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>X-ray diffraction (XRD) is an essential techni...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>504</td>\n",
       "      <td>Deep Multiagent Reinforcement Learning: Challe...</td>\n",
       "      <td>Wong, A.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2021</td>\n",
       "      <td>This paper surveys the field of deep multiagen...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>622</td>\n",
       "      <td>Universal masking is urgent in the COVID-19 pa...</td>\n",
       "      <td>Kai, D.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2020</td>\n",
       "      <td>We present two models for the COVID-19 pandemi...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article focuses...</td>\n",
       "      <td>{'score': 0, 'reasoning': 'The article focuses...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article provide...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>721</td>\n",
       "      <td>Hypoxia increases the tempo of evolution in gl...</td>\n",
       "      <td>Grimes, D.R.</td>\n",
       "      <td>bioRxiv</td>\n",
       "      <td>2018</td>\n",
       "      <td>Background: Low oxygen in tumours have long be...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>489</td>\n",
       "      <td>Agent-based investigation of the impact of low...</td>\n",
       "      <td>Krauland, M.G.</td>\n",
       "      <td>medRxiv</td>\n",
       "      <td>2021</td>\n",
       "      <td>Introduction Interventions to curb the spread ...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>419</td>\n",
       "      <td>A GENERAL FRAMEWORK FOR OPTIMISING COST-EFFECT...</td>\n",
       "      <td>Nguyen, Q.D.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2022</td>\n",
       "      <td>The COVID-19 pandemic created enormous public ...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article focuses...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>562</td>\n",
       "      <td>Open problems in cooperative AI</td>\n",
       "      <td>Dafoe, A.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2020</td>\n",
       "      <td>Problems of cooperation—in which agents seek w...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>253</td>\n",
       "      <td>Self-Supervised Neuron Segmentation with Multi...</td>\n",
       "      <td>Chen, Y.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2023</td>\n",
       "      <td>The performance of existing supervised neuron ...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>578</td>\n",
       "      <td>Who are the 'silent spreaders'?: Contact traci...</td>\n",
       "      <td>Hu, Y.</td>\n",
       "      <td>arXiv</td>\n",
       "      <td>2020</td>\n",
       "      <td>The COVID-19 epidemic has swept the world for ...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article focuses...</td>\n",
       "      <td>{'score': 1, 'reasoning': 'The article discuss...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    ID                                              Title      1st author  \\\n",
       "0   85  An Autonomous GIS Agent Framework for Geospati...        Ning, H.   \n",
       "1  281  Probabilistic Phase Labeling and Lattice Refin...    Chang, M.-C.   \n",
       "2  504  Deep Multiagent Reinforcement Learning: Challe...        Wong, A.   \n",
       "3  622  Universal masking is urgent in the COVID-19 pa...         Kai, D.   \n",
       "4  721  Hypoxia increases the tempo of evolution in gl...    Grimes, D.R.   \n",
       "5  489  Agent-based investigation of the impact of low...  Krauland, M.G.   \n",
       "6  419  A GENERAL FRAMEWORK FOR OPTIMISING COST-EFFECT...    Nguyen, Q.D.   \n",
       "7  562                    Open problems in cooperative AI       Dafoe, A.   \n",
       "8  253  Self-Supervised Neuron Segmentation with Multi...        Chen, Y.   \n",
       "9  578  Who are the 'silent spreaders'?: Contact traci...          Hu, Y.   \n",
       "\n",
       "      repo  year                                           abstract  \\\n",
       "0    arXiv  2024  Powered by the emerging large language models ...   \n",
       "1    arXiv  2023  X-ray diffraction (XRD) is an essential techni...   \n",
       "2    arXiv  2021  This paper surveys the field of deep multiagen...   \n",
       "3    arXiv  2020  We present two models for the COVID-19 pandemi...   \n",
       "4  bioRxiv  2018  Background: Low oxygen in tumours have long be...   \n",
       "5  medRxiv  2021  Introduction Interventions to curb the spread ...   \n",
       "6    arXiv  2022  The COVID-19 pandemic created enormous public ...   \n",
       "7    arXiv  2020  Problems of cooperation—in which agents seek w...   \n",
       "8    arXiv  2023  The performance of existing supervised neuron ...   \n",
       "9    arXiv  2020  The COVID-19 epidemic has swept the world for ...   \n",
       "\n",
       "                               round-A_Pouria_output  \\\n",
       "0  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "1  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "2  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "3  {'score': 1, 'reasoning': 'The article focuses...   \n",
       "4  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "5  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "6  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "7  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "8  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "9  {'score': 1, 'reasoning': 'The article focuses...   \n",
       "\n",
       "                               round-A_Bardia_output  \\\n",
       "0  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "1  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "2  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "3  {'score': 0, 'reasoning': 'The article focuses...   \n",
       "4  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "5  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "6  {'score': 1, 'reasoning': 'The article focuses...   \n",
       "7  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "8  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "9  {'score': 1, 'reasoning': 'The article discuss...   \n",
       "\n",
       "                                 round-B_Brad_output  \n",
       "0                                                NaN  \n",
       "1                                                NaN  \n",
       "2                                                NaN  \n",
       "3  {'score': 1, 'reasoning': 'The article provide...  \n",
       "4                                                NaN  \n",
       "5                                                NaN  \n",
       "6                                                NaN  \n",
       "7                                                NaN  \n",
       "8                                                NaN  \n",
       "9                                                NaN  "
      ]
     },
     "execution_count": 38,
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
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "        Title: An Autonomous GIS Agent Framework for Geospatial Data Retrieval\n",
      "        Abstract: Powered by the emerging large language models (LLMs), autonomous geographic information systems (GIS) agents have the potential to accomplish spatial analyses and cartographic tasks. However, a research gap exists to support fully autonomous GIS agents: how to enable agents to discover and download the necessary data for geospatial analyses. This study proposes an autonomous GIS agent framework capable of retrieving required geospatial data by generating, executing, and debugging programs. The framework utilizes the LLM as the decision-maker, selects the appropriate data source (s) from a pre-defined source list, and fetches the data from the chosen source. Each data source has a handbook that records the metadata and technical details for data retrieval. The proposed framework is designed in a plug-and-play style to ensure flexibility and extensibility. Human users or autonomous data scrawlers can add new data sources by adding new handbooks. We developed a prototype agent based on the framework, released as a QGIS plugin (GeoData Retrieve Agent) and a Python program. Experiment results demonstrate its capability of retrieving data from various sources including OpenStreetMap, administrative boundaries and demographic data from the US Census Bureau, satellite basemaps from ESRI World Imagery, global digital elevation model (DEM) from OpenTopography.org, weather data from a commercial provider, the COVID-19 cases from the NYTimes GitHub. Our study is among the first attempts to develop an autonomous geospatial data retrieval agent.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses an autonomous GIS agent framework utilizing large language models, but it does not specifically address applications to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses autonomous GIS agents using LLMs but does not focus on the application of these models specifically to medical imaging data, which is the main criterion required.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: Probabilistic Phase Labeling and Lattice Refinement for Autonomous Material Research\n",
      "        Abstract: X-ray diffraction (XRD) is an essential technique to determine a material's crystal structure in high-throughput experimentation, and has recently been incorporated in artificially intelligent agents in autonomous scientific discovery processes. However, rapid, automated and reliable analysis method of XRD data matching the incoming data rate remains a major challenge. To address these issues, we present CrystalShift, an efficient algorithm for probabilistic XRD phase labeling that employs symmetry-constrained pseudo-refinement optimization, best-first tree search, and Bayesian model comparison to estimate probabilities for phase combinations without requiring phase space information or training. We demonstrate that CrystalShift provides robust probability estimates, outperforming existing methods on synthetic and experimental datasets, and can be readily integrated into high-throughput experimental workflows. In addition to efficient phase-mapping, CrystalShift offers quantitative insights into materials' structural parameters, which facilitate both expert evaluation and AI-based modeling of the phase space, ultimately accelerating materials identification and discovery.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses an algorithm for X-ray diffraction data analysis in materials science, which does not relate to large language models or medical imaging.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses an algorithm for analyzing X-ray diffraction data in materials research, which does not involve large language models or medical imaging data.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: Deep Multiagent Reinforcement Learning: Challenges and Directions\n",
      "        Abstract: This paper surveys the field of deep multiagent reinforcement learning. The combination of deep neural networks with reinforcement learning has gained increased traction in recent years and is slowly shifting the focus from single-agent to multiagent environments. Dealing with multiple agents is inherently more complex as (a) the future rewards depend on multiple players’ joint actions and (b) the computational complexity increases. We present the most common multiagent problem representations and their main challenges, and identify five research areas that address one or more of these challenges: centralised training and decentralised execution, opponent modelling, communication, efficient coordination, and reward shaping. We find that many computational studies rely on unrealistic assumptions or are not generalisable to other settings; they struggle to overcome the curse of dimensionality or nonstationarity. Approaches from psychology and sociology capture promising relevant behaviours, such as communication and coordination, to help agents achieve better performance in multiagent settings. We suggest that, for multiagent reinforcement learning to be successful, future research should address these challenges with an interdisciplinary approach to open up new possibilities in multiagent reinforcement learning.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses deep multiagent reinforcement learning but does not specifically address large language models or their application to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses multiagent reinforcement learning in general and does not specifically address large language models or their application to medical imaging data.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: Universal masking is urgent in the COVID-19 pandemic: SEIR and agent based models, empirical validation, policy recommendations\n",
      "        Abstract: We present two models for the COVID-19 pandemic predicting the impact of universal face mask wearing upon the spread of the SARS-CoV-2 virusone employing a stochastic dynamic network based compartmental SEIR (susceptible-exposed-infectious-recovered) approach, and the other employing individual ABM (agentbased modelling) Monte Carlo simulationindicating (1) significant impact under (near) universal masking when at least 80% of a population is wearing masks, versus minimal impact when only 50% or less of the population is wearing masks, and (2) significant impact when universal masking is adopted early, by Day 50 of a regional outbreak, versus minimal impact when universal masking is adopted late. These effects hold even at the lower filtering rates of homemade masks. To validate these theoretical models, we compare their predictions against a new empirical data set we have collected that includes whether regions have universal masking cultures or policies, their daily case growth rates, and their percentage reduction from peak daily case growth rates. Results show a near perfect correlation between early universal masking and successful suppression of daily case growth rates and/or reduction from peak daily case growth rates, as predicted by our theoretical simulations. Taken in tandem, our theoretical models and empirical results argue for urgent implementation of universal masking in regions that have not yet adopted it as policy or.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article focuses on modeling the impact of universal masking during the COVID-19 pandemic and does not discuss large language models or AI agents applied to medical imaging data.'}\n",
      "        Bardia's review: {'score': 0, 'reasoning': 'The article focuses on COVID-19 modeling and universal masking, without discussing large language models or their application to medical imaging data.'}\n",
      "        Brad's review: {'score': 1, 'reasoning': \"The article provided is focused on modeling the impact of universal masking during the COVID-19 pandemic using SEIR and agent-based models. It does not mention or discuss large language models or AI agents applied to medical imaging data, which is the criteria for scoring. Pouria's reasoning is accurate in identifying that the article does not meet the specific criteria of discussing large language models or AI agents in the context of medical imaging. Bardia's score of 0 suggests uncertainty, but the article's content clearly does not align with the specified criteria, making Pouria's score of 1 more appropriate.\"}\n",
      "        \n",
      "\n",
      "        Title: Hypoxia increases the tempo of evolution in glioblastoma\n",
      "        Abstract: Background: Low oxygen in tumours have long been associated with poor prognosis and metastatic disease, precise reasons for which remain poorly understood. Somatic evolution drives cancer progression and treatment resistance. This process is fuelled not only by genetic and epigenetic mutation, but by selection resulting from the interactions between tumour cells, normal cells and physical microenvironment. The ecological habitat tumour cells inhabit influences evolutionary dynamics but impact on tempo of evolution is less clear. Methods: We explored this complex dialogue with a combined clinical-theoretical approach. Using an agent-based-model, we simulated proliferative hierarchy under heterogeneous oxygen availability. Predictions were compared against clinical data derived from histology samples taken from glioblastoma patients, stained to elucidate areas of hypoxia / necrosis, and p53 expression heterogeneity. Results: Simulation results indicate cell division in hypoxic environments is effectively upregulated, and that low-oxygen niches provide new avenues for tumour cells to spread. Analysis of human data indicates cell division isn't decreased in low-oxygen regions, despite evidence of significant physiological stress. This is consistent with simulation, suggesting hypoxia is a crucible that effectively warping evolutionary velocity, making deleterious mutations more likely than in well-oxygenated regions. Conclusions: Results suggest hypoxic regions alter evolutionary tempo, driving mutations which fuel tumour heterogeneity..\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses glioblastoma and hypoxia but does not mention large language models or AI agents applied to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses tumor evolution in glioblastoma and hypoxia, but it does not mention large language models or AI agents applied to medical imaging data.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: Agent-based investigation of the impact of low rates of influenza on next season influenza infections\n",
      "        Abstract: Introduction Interventions to curb the spread of SARS-CoV-2 during the 2020-21 influenza season essentially eliminated influenza during that season. Given waning antibody titers over time, future residual population immunity against influenza will be reduced. The implication for the subsequent 2021-22 influenza season is unknown. Methods We used an agent-based model of influenza implemented in the FRED (Framework for Reconstructing Epidemiological Dynamics) simulation platform to estimate cases and hospitalization over two succeeding influenza seasons. The model uses a synthetic population to represent an actual population, and individual interactions in workplaces, school, households and neighborhoods. The impact of reduced residual immunity was estimated as a consequence of increased protective measures (e.g., social distancing and school closure) in the first season. The impact was contrasted by the level of similarity (cross-immunity) between influenza strains over the seasons. Results When the second season strains were dissimilar to the first season (have a low level of cross immunity), a low first season has limited impact on second season cases. When a high level of cross-immunity exists between strains in the 2 seasons, the first season has a much greater impact on the second season. In both cases this is modified by the transmissibility of strains in the 2 seasons. In the context of the 2021-22 season, the worst case scenario is a highly transmissible strain causing increased cases and hospitalizations over average influenza seasons, with a possible significant increase in cases in some scenarios. The most likely overall scenario for 2021-22 is a more modest increase in flu cases over an average season. Discussion Given the light 2020-21 season, we found that a large, compensatory second season might occur in 2021-22, depending on cross-immunity from past infection and transmissibility of strains. Furthermore, we found that enhanced vaccine coverage could reduce this high, compensatory season. Young children may be especially at risk in 2021-22 since very young children were unlikely to have had any exposure to infection and most immunity in that age group would be from vaccination, which wanes quickly.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses an agent-based model for influenza infections but does not mention large language models or AI agents applied to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses an agent-based model for influenza but does not mention large language models or their application to medical imaging data.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: A GENERAL FRAMEWORK FOR OPTIMISING COST-EFFECTIVENESS OF PANDEMIC RESPONSE UNDER PARTIAL INTERVENTION MEASURES\n",
      "        Abstract: The COVID-19 pandemic created enormous public health and socioeconomic challenges. The health effects of vaccination and non-pharmaceutical interventions (NPIs) were often contrasted with significant social and economic costs. We describe a general framework aimed to derive adaptive cost-effective interventions, adequate for both recent and emerging pandemic threats. We also quantify the net health benefits and propose a reinforcement learning approach to optimise adaptive NPIs. The approach utilises an agent-based model simulating pandemic responses in Australia, and accounts for a heterogeneous population with variable levels of compliance fluctuating over time and across individuals. Our analysis shows that a significant net health benefit may be attained by adaptive NPIs formed by partial social distancing measures, coupled with moderate levels of the society’s willingness to pay for health gains (health losses averted). We demonstrate that a socially acceptable balance between health effects and incurred economic costs is achievable over a long term, despite possible early setbacks.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses pandemic response and cost-effectiveness but does not address large language models or AI agents applied to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article focuses on pandemic response and cost-effectiveness rather than discussing large language models-based AI agents applied to medical imaging data.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: Open problems in cooperative AI\n",
      "        Abstract: Problems of cooperation—in which agents seek ways to jointly improve their welfare—are ubiquitous and important. They can be found at scales ranging from our daily routines—such as driving on highways, scheduling meetings, and working collaboratively—to our global challenges—such as peace, commerce, and pandemic preparedness. Arguably, the success of the human species is rooted in our ability to cooperate. Since machines powered by artificial intelligence are playing an ever greater role in our lives, it will be important to equip them with the capabilities necessary to cooperate and to foster cooperation. We see an opportunity for the field of artificial intelligence to explicitly focus effort on this class of problems, which we term Cooperative AI. The objective of this research would be to study the many aspects of the problems of cooperation and to innovate in AI to contribute to solving these problems. Central goals include building machine agents with the capabilities needed for cooperation, building tools to foster cooperation in populations of (machine and/or human) agents, and otherwise conducting AI research for insight relevant to problems of cooperation. This research integrates ongoing work on multi-agent systems, game theory and social choice, human-machine interaction and alignment, natural-language processing, and the construction of social tools and platforms. However, Cooperative AI is not the union of these existing areas, but rather an independent bet about the productivity of specific kinds of conversations that involve these and other areas. We see opportunity to more explicitly focus on the problem of cooperation, to construct unified theory and vocabulary, and to build bridges with adjacent communities working on cooperation, including in the natural, social, and behavioural sciences. Conversations on Cooperative AI can be organized in part in terms of the dimensions of cooperative opportunities. These include the strategic context, the extent of common versus conflicting interest, the kinds of entities who are cooperating, and whether researchers take the perspective of an individual or of a social planner. Conversations can also be focused on key capabilities necessary for cooperation, such as understanding, communication, cooperative commitments, and cooperative institutions. Finally, research should study the potential downsides of cooperative capabilities—such as exclusion and coercion—and how to channel cooperative capabilities to best improve human welfare. This research would connect AI research to the broader scientific enterprise studying the problem of cooperation, and to the broader social effort to solve cooperation problems. This conversation will continue at: www.cooperativeAI.com\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses cooperative AI in general without specifically addressing large language models or their application to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses cooperative AI but does not specifically address large language models or their application to medical imaging data.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: Self-Supervised Neuron Segmentation with Multi-Agent Reinforcement Learning\n",
      "        Abstract: The performance of existing supervised neuron segmentation methods is highly dependent on the number of accurate annotations, especially when applied to large scale electron microscopy (EM) data. By extracting semantic information from unlabeled data, self-supervised methods can improve the performance of downstream tasks, among which the mask image model (MIM) has been widely used due to its simplicity and effectiveness in recovering original information from masked images. However, due to the high degree of structural locality in EM images, as well as the existence of considerable noise, many voxels contain little discriminative information, making MIM pretraining inefficient on the neuron segmentation task. To overcome this challenge, we propose a decision-based MIM that utilizes reinforcement learning (RL) to automatically search for optimal image masking ratio and masking strategy. Due to the vast exploration space, using single-agent RL for voxel prediction is impractical. Therefore, we treat each input patch as an agent with a shared behavior policy, allowing for multi-agent collaboration. Furthermore, this multi-agent model can capture dependencies between voxels, which is beneficial for the downstream segmentation task. Experiments conducted on representative EM datasets demonstrate that our approach has a significant advantage over alternative self-supervised methods on the task of neuron segmentation. Code is available at https://github.com/ydchen0806/dbMiM.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article discusses neuron segmentation using reinforcement learning but does not focus on large language models or their application to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses segmentation methods in the context of electron microscopy data but does not focus on large language models or AI agents applied to medical imaging data.'}\n",
      "        Brad's review: nan\n",
      "        \n",
      "\n",
      "        Title: Who are the 'silent spreaders'?: Contact tracing in spatio-temporal memory models\n",
      "        Abstract: The COVID-19 epidemic has swept the world for over a year. However, a large number of infectious asymptomatic COVID-19 cases (ACCs) are still making the breaking up of the transmission chains very difficult. Efforts by epidemiological researchers in many countries have thrown light on the clinical features of ACCs, but there is still a lack of practical approaches to detect ACCs so as to help contain the pandemic. To address the issue of ACCs, this paper presents a neural network model called Spatio-Temporal Episodic Memory for COVID-19 (STEM-COVID) to identify ACCs from contact tracing data. Based on the fusion Adaptive Resonance Theory (ART), the model encodes a collective spatio-temporal episodic memory of individuals and incorporates an effective mechanism of parallel searches for ACCs. Specifically, the episodic traces of the identified positive cases are used to map out the episodic traces of suspected ACCs using a weighted evidence pooling method. To evaluate the efficacy of STEM-COVID, a realistic agent based simulation model for COVID-19 spreading is implemented based on the recent epidemiological findings on ACCs. The experiments based on rigorous simulation scenarios, manifesting the current situation of COVID-19 spread, show that the STEM-COVID model with weighted evidence pooling has a higher level of accuracy and efficiency for identifying ACCs when compared with several baselines. Moreover, the model displays strong robustness against noisy data and different ACC proportions, which partially reflects the effect of breakthrough infections after vaccination on the virus transmission.\n",
      "        Pouria's review: {'score': 1, 'reasoning': 'The article focuses on a neural network model for contact tracing related to COVID-19 and does not discuss large language models or their application to medical imaging data.'}\n",
      "        Bardia's review: {'score': 1, 'reasoning': 'The article discusses a neural network model for identifying asymptomatic COVID-19 cases using contact tracing data, but it does not involve large language models or medical imaging data.'}\n",
      "        Brad's review: nan\n",
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
    "        Pouria's review: {row[\"round-A_Pouria_output\"]}\n",
    "        Bardia's review: {row[\"round-A_Bardia_output\"]}\n",
    "        Brad's review: {row[\"round-B_Brad_output\"]}\n",
    "        \"\"\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'identity': {'system_prompt': \"Your name is <<Brad>> and you are <<You are a senior radiologist with a PhD in computer science and years of experience as the director of a DL lab focused on developing ML models for radiology and healthcare>>. Your task is to review input itmes with the following description: <<tilte and abstract of scientific articles>>. Your final output should have the following keys: score (<class 'int'>), reasoning (<class 'str'>).\",\n",
       "   'item_prompt': 'Review the input item below and evaluate it against the following criteria: Scoring task: <<Pouria and Bardia have Looked for articles that disucss large languange models-based AI agents applied to medical imaging data. They scored an article 1 if they thought it does not meet this criteria, 2 if they thought it meets the criteria, 0 if they were uncertain of scoring. You will receive an article they have had different opinions about, as well as each of their scores and their reasoning for that score. Read their reviews and determine who you agree with. >> Input item: <<${item}$>> Now choose your score from the following values: [0, 1, 2]. Your scoring should be based on the following rules: <<Your scores should be either 1 or 2. 1 means that you agree with Pouria, 2 means that you agree with Bardia, >> If you are highly in doubt about what to score, score \"0\". You must also provide a reasoning for your scoring . Think step by step in your reasoning. First reason then score!',\n",
       "   'temperature': 0.4,\n",
       "   'max_tokens': 100},\n",
       "  'item': \"Title: Universal masking is urgent in the COVID-19 pandemic: SEIR and agent based models, empirical validation, policy recommendations abstract: We present two models for the COVID-19 pandemic predicting the impact of universal face mask wearing upon the spread of the SARS-CoV-2 virusone employing a stochastic dynamic network based compartmental SEIR (susceptible-exposed-infectious-recovered) approach, and the other employing individual ABM (agentbased modelling) Monte Carlo simulationindicating (1) significant impact under (near) universal masking when at least 80% of a population is wearing masks, versus minimal impact when only 50% or less of the population is wearing masks, and (2) significant impact when universal masking is adopted early, by Day 50 of a regional outbreak, versus minimal impact when universal masking is adopted late. These effects hold even at the lower filtering rates of homemade masks. To validate these theoretical models, we compare their predictions against a new empirical data set we have collected that includes whether regions have universal masking cultures or policies, their daily case growth rates, and their percentage reduction from peak daily case growth rates. Results show a near perfect correlation between early universal masking and successful suppression of daily case growth rates and/or reduction from peak daily case growth rates, as predicted by our theoretical simulations. Taken in tandem, our theoretical models and empirical results argue for urgent implementation of universal masking in regions that have not yet adopted it as policy or. round-A_Pouria_output: {'score': 1, 'reasoning': 'The article focuses on modeling the impact of universal masking during the COVID-19 pandemic and does not discuss large language models or AI agents applied to medical imaging data.'} round-A_Bardia_output: {'score': 0, 'reasoning': 'The article focuses on COVID-19 modeling and universal masking, without discussing large language models or their application to medical imaging data.'}\",\n",
       "  'response': '{\"score\":1,\"reasoning\":\"The article provided is focused on modeling the impact of universal masking during the COVID-19 pandemic using SEIR and agent-based models. It does not mention or discuss large language models or AI agents applied to medical imaging data, which is the criteria for scoring. Pouria\\'s reasoning is accurate in identifying that the article does not meet the specific criteria of discussing large language models or AI agents in the context of medical imaging. Bardia\\'s score of 0 suggests uncertainty, but the article\\'s content clearly does not align with the specified criteria, making Pouria\\'s score of 1 more appropriate.\"}',\n",
       "  'cost': {'input_cost': 0.001525,\n",
       "   'output_cost': 0.00127,\n",
       "   'total_cost': 0.002795}}]"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "brad.memory"
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

from .agents.scoring_reviewer import ScoringReviewer

class ReviewWorkflowError(Exception):
    """Base exception for workflow-related errors."""
    pass

class ReviewWorkflow(pydantic.BaseModel):
    workflow_schema: List[Dict[str, Any]]
    memory: List[Dict] = list()
    reviewer_costs: Dict = dict()
    total_cost: float = 0.0

    def __post_init__(self, __context):
        """Initialize after Pydantic model initialization."""
        try:
            for review_task in self.workflow_schema:
                round_id = review_task["round"]
                reviewers = review_task["reviewers"] if isinstance(review_task["reviewers"], list) else [review_task["reviewers"]]
                reviewer_names = [f"round-{round_id}_{reviewer.name}" for reviewer in reviewers]
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                initial_inputs = [col for col in inputs if "_output_" not in col]
                
                for reviewer in reviewers:
                    if not isinstance(reviewer, ScoringReviewer):
                        raise ReviewWorkflowError(f"Invalid reviewer: {reviewer}")
                
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

    async def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the review process."""
        try:            
            df = data.copy()
            total_rounds = len(self.workflow_schema)
            
            for review_round, review_task in enumerate(self.workflow_schema):
                round_id = review_task["round"]
                print(f"\nStarting review round {round_id} ({review_round + 1}/{total_rounds})...")
                print(f"Reviewers: {[reviewer.name for reviewer in review_task['reviewers']]}")
                print(f"Input data: {review_task['inputs']}")
                
                reviewers = review_task["reviewers"] if isinstance(review_task["reviewers"], list) else [review_task["reviewers"]]
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                filter_func = review_task.get("filter", lambda x: True)
                
                # Pre-process data with progress bar
                output_cols = [col for col in inputs if "_output_" in col]
                for col in output_cols:
                    mask = df[col].notna()
                    if mask.any():
                        df.loc[mask, col] = df.loc[mask, col].apply(
                            lambda x: x if isinstance(x, dict) else json.loads(x) if isinstance(x, str) else {"score": None}
                        )
                    else:
                        df[col] = df[col].apply(lambda x: {"score": None})
                
                # Create input items with progress bar
                input_text = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating inputs", leave=False):
                    text = " ".join(f"{input_col}: {str(row[input_col])}" for input_col in inputs)
                    input_text.append(text)
                df["input_item"] = input_text
                
                # Apply filter with progress bar
                try:
                    mask = []
                    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering", leave=False):
                        mask.append(filter_func(row))
                    mask = pd.Series(mask, index=df.index)
                    eligible_rows = mask.sum()
                    print(f"Number of eligible rows for review: {eligible_rows}")
                    
                    if (eligible_rows == 0):
                        print(f"Skipping review round {round_id} - no eligible rows")
                        continue
                        
                except Exception as e:
                    raise ReviewWorkflowError(f"Error applying filter in round {round_id}: {e}")
                
                df_filtered = df[mask].copy()
                input_items = df_filtered["input_item"].tolist()
                
                # Create progress bar for reviewers
                reviewer_outputs = []
                
                for reviewer in tqdm(reviewers, desc="Reviewers", leave=False):
                    outputs, review_cost = await reviewer.review_items(input_items)
                    reviewer_outputs.append(outputs)
                    self.reviewer_costs[(round_id, reviewer.name)] = review_cost

                # Add reviewer outputs with round prefix
                for reviewer, outputs in zip(reviewers, reviewer_outputs):
                    output_col = f"round-{round_id}_{reviewer.name}_output"
                    outputs = [json.loads(x) for x in outputs]
                    df_filtered[output_col] = outputs
                
                # Merge results back
                merge_cols = list(set(inputs) - set([col for col in inputs if "_output" in col]))
                if not merge_cols:
                    df_filtered.index = df[mask].index
                    for reviewer in reviewers:
                        output_col = f"round-{round_id}_{reviewer.name}_output"
                        df.loc[mask, output_col] = df_filtered[output_col]
                else:
                    output_cols = [f"round-{round_id}_{r.name}_output" for r in reviewers]
                    df = pd.merge(
                        df,
                        df_filtered[merge_cols + output_cols],
                        how="left",
                        on=merge_cols
                    )
                
                # Clean up temporary columns
                if "input_item" in df.columns:
                    df = df.drop("input_item", axis=1)
                
            return df
        except Exception as e:
            raise ReviewWorkflowError(f"Error running workflow: {e}")
        
    def get_total_cost(self) -> int:
        """Return the total cost of the review process."""
        return sum(self.reviewer_costs.values())

```

## lattereview/generic_prompts/review_prompt.txt

```txt
Review the input item below and evaluate it against the following criteria:

Scoring task: <<${scoring_task}$>>

Input item: <<${item}$>>

Now choose your score from the following values: ${score_set}$.

Your scoring should be based on the following rules: <<${scoring_rules}$>>

If you are highly in doubt about what to score, score "0". 

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
        if not self.api_key:
            raise ClientCreationError("OPENAI_API_KEY environment variable is not set")
        try:
            return openai.AsyncOpenAI(api_key=self.api_key)
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

## lattereview/agents/base_agent.py

```py
"""Base agent class with consistent error handling and type safety."""
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel

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
    max_concurrent_requests: int = 20
    name: str = "BaseAgent"
    backstory: str = "a generic base agent"
    input_description: str = "article title/abstract"
    examples: Union[str, List[Union[str, Dict[str, Any]]]] = None
    reasoning: ReasoningType = ReasoningType.BRIEF
    system_prompt: Optional[str] = None
    item_prompt: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 150
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
from typing import List, Dict, Any, Optional
from pydantic import Field
from .base_agent import BaseAgent, AgentError

class ScoringReviewer(BaseAgent):
    response_format: Dict[str, Any] = {
        "score": int,
        "reasoning": str,
    }
    scoring_task: Optional[str] = None
    score_set: List[int] = [1, 2]
    scoring_rules: str = "Your scores should follow the defined schema."
    generic_item_prompt: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            assert 0 not in self.score_set, "Score set must not contain 0. This value is reserved for uncertain scorings / errors."
            self.score_set.insert(0, 0)
            prompt_path = Path(__file__).parent.parent / "generic_prompts" / "review_prompt.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Review prompt template not found at {prompt_path}")
            self.generic_item_prompt = prompt_path.read_text(encoding='utf-8')
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initalizing agent: {str(e)}")

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
                'temperature': self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            if not self.provider:
                raise AgentError("Provider not initialized")
            
            self.provider.set_response_format(self.response_format)
            self.provider.system_prompt = self.system_prompt
        except Exception as e:
            raise AgentError(f"Error in setup: {str(e)}")

    async def review_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Review a list of items asynchronously with concurrency control."""
        try:
            self.setup()
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def limited_review_item(item: str) -> tuple[Dict[str, Any], Dict[str, float]]:
                async with semaphore:
                    return await self.review_item(item)

            responses_costs = await asyncio.gather(
                *(limited_review_item(item) for item in items),
                return_exceptions=True
            )

            # Handle any exceptions from the gathered tasks
            results = []
            for item, response_cost in zip(items, responses_costs):
                if isinstance(response_cost, Exception):
                    print(f"Error processing item: {item}, Error: {str(response_cost)}")
                    continue
                
                response, cost = response_cost
                self.cost_so_far += cost["total_cost"]
                results.append(response)
                self.memory.append({
                    'identity': self.identity,
                    'item': item,
                    'response': response,
                    'cost': cost
                })

            return results, cost["total_cost"]
        except Exception as e:
            raise AgentError(f"Error reviewing items: {str(e)}")

    async def review_item(self, item: str) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Review a single item asynchronously with error handling."""
        try:
            item_prompt = self.build_item_prompt(self.item_prompt, {'item': item})
            response, cost = await self.provider.get_json_response(
                item_prompt,
                temperature=self.temperature
            )
            return response, cost
        except Exception as e:
            raise AgentError(f"Error reviewing item: {str(e)}")
```

