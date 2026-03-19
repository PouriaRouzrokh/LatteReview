#!/usr/bin/env python
"""Generate evaluation notebooks for v1 vs v2 comparison.

Models: openai:gpt-5.4-mini, google-gla:gemini-3.1-flash-lite-preview
Structure: Task-by-task (each search criteria runs non-agentic + agentic, then metrics)
"""

import nbformat as nbf


def make_notebook(cells_data):
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for cell_type, source in cells_data:
        if cell_type == "md":
            nb["cells"].append(nbf.v4.new_markdown_cell(source))
        else:
            nb["cells"].append(nbf.v4.new_code_cell(source))
    return nb


# =============================================================================
# CUSTOM EVALUATION NOTEBOOK
# =============================================================================

def build_custom_notebook():
    cells = []

    # ── Title ──
    cells.append(("md", """# Custom Evaluation: v1 vs v2 Non-Agentic vs v2 Agentic

This notebook replicates the custom evaluation from v1 using the v2 agentic framework.

We compare three conditions:
1. **v1 Baseline** — original results with older models (gemini-1.5-flash, gpt-4o-mini, gpt-4o)
2. **v2 Non-Agentic** — same workflow with latest models (gemini-3.1-flash, gpt-5.4-mini), max_iterations=1
3. **v2 Agentic** — latest models + agentic skills (search, memory, flagging), max_iterations=15

All conditions use **identical prompts** — only the agentic capabilities differ.

**Execution order**: Each search task runs all 3 conditions before moving to the next."""))

    # ── Setup ──
    cells.append(("md", "## Setup"))
    cells.append(("code", """%reload_ext autoreload
%autoreload 2

from dotenv import load_dotenv
load_dotenv(dotenv_path='../.env')

import sys
sys.path.append('../')"""))

    cells.append(("code", """import glob
import json
import shutil
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, f1_score
)

from pydantic import BaseModel, Field
from lattereview.agentic import AgenticReviewer, AgenticWorkflow"""))

    # ── Data ──
    cells.append(("md", "## Data & Search Criteria"))
    cells.append(("code", """data = pd.read_csv('custom_data/custom_data.csv')
print(f"Loaded {len(data)} articles")

# Ground truth labels
data["search1"] = data["modality"].isin(["ct", "xr, ct"]).astype(int)
data["search2"] = (
    (data["search1"] == 1) &
    (data["deep_learning"] == 1) &
    (data["task"].str.contains("classification")) &
    (data["cardiovascular"] == 0)
).astype(int)
data["search3"] = (
    (data["search2"] == 1) &
    (data["clinical_application"] == "diagnosis") &
    (data["external_validation"] == 1)
).astype(int)

for s in ["search1", "search2", "search3"]:
    pos = data[s].sum()
    print(f"{s}: {pos} positive / {len(data)} total ({pos/len(data)*100:.1f}% relevant)")"""))

    # ── Output Model ──
    cells.append(("md", "## Output Model & Shared Infrastructure"))
    cells.append(("code", """class TitleAbstractOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the evaluation.")
    evaluation: int = Field(
        ge=1, le=5,
        description="Evaluation score: 1=absolutely exclude, 2=better to exclude, "
                    "3=not sure, 4=better to include, 5=absolutely include."
    )
    certainty: int = Field(ge=0, le=100, description="Confidence in the evaluation (0-100).")

# ── Prompt templates (IDENTICAL for agentic and non-agentic) ──

SYSTEM_PROMPT = (
    "You are an expert systematic reviewer. Your task is to evaluate whether "
    "studies should be included or excluded based on their title and abstract. "
    "Be thorough and apply the criteria consistently."
)

TASK_PROMPT_TEMPLATE = (
    "**Review the title and abstract below and evaluate whether they should be "
    "included based on the following inclusion and exclusion criteria (if any).**\\n"
    "**Note that the study should be included only and only if it meets ALL "
    "inclusion criteria and NONE of the exclusion criteria.**\\n\\n---\\n\\n"
    "**Input item:**\\n<<${item}$>>\\n\\n---\\n\\n"
    "**Inclusion criteria:**\\nINCLUSION_CRITERIA_PLACEHOLDER\\n\\n"
    "**Exclusion criteria:**\\nEXCLUSION_CRITERIA_PLACEHOLDER\\n\\n---\\n\\n"
    "**Instructions**\\n\\n"
    "1. Output your evaluation as an integer between 1 and 5, where:\\n"
    "   - 1 means absolutely to exclude.\\n"
    "   - 2 means better to exclude.\\n"
    "   - 3 Not sure if to include or exclude.\\n"
    "   - 4 means better to include.\\n"
    "   - 5 means absolutely to include.\\n"
    "2. Report your certainty level as a value between **0** (not certain at all) "
    "and **100** (completely certain).\\n"
    "3. Provide your reasoning before assigning a decision."
)

SENIOR_SYSTEM_PROMPT = (
    "You are a senior expert systematic reviewer. Two junior reviewers have already "
    "reviewed this article and their evaluations are included in the input. They may "
    "disagree. Use your expertise to make the final determination. "
    "Be thorough and apply the criteria consistently."
)

def create_reviewers(inclusion_criteria, exclusion_criteria, agentic=False):
    task_prompt = (
        TASK_PROMPT_TEMPLATE
        .replace("INCLUSION_CRITERIA_PLACEHOLDER", str(inclusion_criteria))
        .replace("EXCLUSION_CRITERIA_PLACEHOLDER", str(exclusion_criteria))
    )

    common_agentic = dict(
        agentic_effort="high",
        skills=["searching-duckduckgo"],  # memory + flagging are now default agentic skills
    )
    # Gemini needs more iterations (uses tools more aggressively)
    gemini_kwargs = {**common_agentic, "max_iterations": 30} if agentic else {"max_iterations": 1}
    gpt_kwargs = {**common_agentic, "max_iterations": 15} if agentic else {"max_iterations": 1}

    Agent1 = AgenticReviewer(
        name="Agent1",
        model="google-gla:gemini-3.1-flash-lite-preview",
        backstory="a PhD researcher",
        system_prompt=SYSTEM_PROMPT,
        task_prompt=task_prompt,
        output_type=TitleAbstractOutput,
        max_concurrent_requests=20 if not agentic else 10,
        model_settings={"temperature": 0.1},
        **gemini_kwargs,
    )
    Agent2 = AgenticReviewer(
        name="Agent2",
        model="openai:gpt-5.4-mini",
        backstory="a PhD researcher",
        system_prompt=SYSTEM_PROMPT,
        task_prompt=task_prompt,
        output_type=TitleAbstractOutput,
        max_concurrent_requests=20 if not agentic else 10,
        model_settings={"temperature": 0.1},
        **gpt_kwargs,
    )
    Agent3 = AgenticReviewer(
        name="Agent3",
        model="openai:gpt-5.4-mini",
        backstory="a senior MD-PhD researcher with years of experience in systematic reviews",
        system_prompt=SENIOR_SYSTEM_PROMPT,
        task_prompt=task_prompt,
        output_type=TitleAbstractOutput,
        max_concurrent_requests=20 if not agentic else 10,
        model_settings={"temperature": 0.1},
        **gpt_kwargs,
    )
    return Agent1, Agent2, Agent3

def split_dataframe(df, chunk_size):
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

async def run_evaluation(search_name, inclusion_criteria, exclusion_criteria, df, agentic=False, working_dir_base=None):
    mode = "agentic" if agentic else "non_agentic"
    print(f"\\n{'='*60}")
    print(f"{search_name} — {mode}")
    print(f"{'='*60}")

    sub_dfs = split_dataframe(df, 1000)
    all_results = []
    total_cost = 0.0

    for chunk_idx, sub_df in enumerate(sub_dfs):
        Agent1, Agent2, Agent3 = create_reviewers(inclusion_criteria, exclusion_criteria, agentic=agentic)

        def filter_func(row):
            try:
                score1 = int(row["round-A_Agent1_output"]["evaluation"])
                score2 = int(row["round-A_Agent2_output"]["evaluation"])
            except (TypeError, KeyError, ValueError):
                return False
            if score1 != score2:
                if score1 >= 4 and score2 >= 4:
                    return False
                if score1 >= 3 or score2 >= 3:
                    return True
            elif score1 == score2 == 3:
                return True
            return False

        working_dir = None
        if working_dir_base and agentic:
            working_dir = Path(working_dir_base) / f"{search_name}_{mode}_chunk{chunk_idx}"
            if working_dir.exists():
                shutil.rmtree(working_dir)

        workflow = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [Agent1, Agent2], "text_inputs": ["title", "abstract"]},
                {"round": "B", "reviewers": [Agent3],
                 "text_inputs": ["title", "abstract", "round-A_Agent1_output", "round-A_Agent2_output"],
                 "filter": filter_func},
            ],
            working_dir=working_dir,
            verbose=True,
        )

        result = await workflow(sub_df)
        all_results.append(result)
        total_cost += workflow.total_cost

    unified_df = pd.concat(all_results)
    print(f"\\n{search_name} ({mode}) complete. Cost: ${total_cost:.4f}")
    return unified_df, total_cost

def get_score(row):
    if "round-B_Agent3_output" in row and pd.notna(row.get("round-B_Agent3_evaluation")):
        try:
            return int(row["round-B_Agent3_evaluation"])
        except (TypeError, ValueError):
            pass
    try:
        score1 = int(row["round-A_Agent1_evaluation"])
        score2 = int(row["round-A_Agent2_evaluation"])
        return (score1 + score2) / 2
    except (TypeError, ValueError):
        return 1  # default to exclude on error

def evaluate_metrics(ground_truth, predictions, threshold_sensitive=1.5, threshold_balanced=3.0, threshold_specific=4.5):
    def classify(pred, threshold):
        return [1 if p >= threshold else 0 for p in pred]
    num_articles = len(ground_truth)
    percentage_relevant = sum(ground_truth) / len(ground_truth) * 100
    metrics = {}
    for label, threshold in [('sensitive', threshold_sensitive), ('balanced', threshold_balanced), ('specific', threshold_specific)]:
        pred = classify(predictions, threshold)
        tn, fp, fn, tp = confusion_matrix(ground_truth, pred, labels=[0, 1]).ravel()
        metrics[f'accuracy_{label}'] = accuracy_score(ground_truth, pred)
        metrics[f'precision_{label}'] = precision_score(ground_truth, pred, zero_division=0)
        metrics[f'recall_{label}'] = recall_score(ground_truth, pred, zero_division=0)
        metrics[f'specificity_{label}'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'f1_{label}'] = f1_score(ground_truth, pred, zero_division=0)
    metrics['auc'] = roc_auc_score(ground_truth, predictions) if len(set(ground_truth)) > 1 else float('nan')
    return {'num_articles': num_articles, 'percentage_relevant': percentage_relevant, **metrics}

def print_metrics(name, gt, scores):
    m = evaluate_metrics(gt, scores)
    print(f"{name:<25} AUC={m['auc']:.3f}  F1(bal)={m['f1_balanced']:.3f}  "
          f"Recall(bal)={m['recall_balanced']:.3f}  Prec(bal)={m['precision_balanced']:.3f}  "
          f"F1(sens)={m['f1_sensitive']:.3f}  Recall(sens)={m['recall_sensitive']:.3f}")
    return m

print("All infrastructure defined. Models: gemini-3.1-flash-lite-preview + gpt-5.4-mini")"""))

    # ── Load v1 ──
    cells.append(("md", "## Load v1 Baseline"))
    cells.append(("code", """v1_raw_data = {}
for csv_file in sorted(glob.glob("custom_data/search*_reviewed.csv")):
    df_v1 = pd.read_csv(csv_file)
    name = csv_file.split("/")[-1].split("_reviewed.csv")[0]
    labels = df_v1[name].apply(int).tolist()
    def get_v1_score(row):
        if "round-B_Agent3_output" in row and pd.notna(row.get("round-B_Agent3_evaluation")):
            return int(row["round-B_Agent3_evaluation"])
        score1 = int(row["round-A_Agent1_evaluation"])
        score2 = int(row["round-A_Agent2_evaluation"])
        return (score1 + score2) / 2
    scores = df_v1.apply(get_v1_score, axis=1).tolist()
    v1_raw_data[name] = (labels, scores)
    print(f"v1 {name}: {len(labels)} items")

non_agentic_results = {}
agentic_results = {}"""))

    # ── Task 1: search1 ──
    cells.append(("md", """## Task 1: search1 (easiest — CT modality only)

**Inclusion**: CT scans | **Exclusion**: No PET scans | **71.3% relevant**"""))
    cells.append(("code", """sc1_inc = {1: "The study must involve CT scans. If multiple modalities are involved, CT scans should be among them."}
sc1_exc = {1: "The study must not include PET scans as one of its modalities."}

# Non-agentic
r_na, _ = await run_evaluation("search1", sc1_inc, sc1_exc, data.copy(), agentic=False)
r_na.to_csv("custom_data/search1_v2_nonagentic.csv", index=False)
non_agentic_results["search1"] = r_na

# Agentic
r_ag, _ = await run_evaluation("search1", sc1_inc, sc1_exc, data.copy(), agentic=True, working_dir_base="custom_data/agentic_runs")
r_ag.to_csv("custom_data/search1_v2_agentic.csv", index=False)
agentic_results["search1"] = r_ag

# Metrics
print("\\n" + "=" * 70)
print("search1 RESULTS")
print("=" * 70)
v1_gt, v1_sc = v1_raw_data["search1"]
na_gt = data["search1"].tolist()
print_metrics("v1 (old models)", v1_gt, v1_sc)
print_metrics("v2 Non-Agentic", na_gt, r_na.apply(get_score, axis=1).tolist())
print_metrics("v2 Agentic", na_gt, r_ag.apply(get_score, axis=1).tolist())"""))

    # ── Task 2: search2 ──
    cells.append(("md", """## Task 2: search2 (moderate — CT + deep learning classifier, no cardio)

**Inclusion**: CT + DL classifier | **Exclusion**: No PET, no cardiovascular | **37.5% relevant**"""))
    cells.append(("code", """sc2_inc = {1: "The study must involve CT scans. If multiple modalities are involved, CT scans should be among them.",
           2: "The study should introduce, develop, or discuss a deep learning-based classifier."}
sc2_exc = {1: "The study must not include PET scans as one of its modalities.",
           2: "Studies focusing on cardiovascular organs or lung vasculature must be excluded."}

r_na, _ = await run_evaluation("search2", sc2_inc, sc2_exc, data.copy(), agentic=False)
r_na.to_csv("custom_data/search2_v2_nonagentic.csv", index=False)
non_agentic_results["search2"] = r_na

r_ag, _ = await run_evaluation("search2", sc2_inc, sc2_exc, data.copy(), agentic=True, working_dir_base="custom_data/agentic_runs")
r_ag.to_csv("custom_data/search2_v2_agentic.csv", index=False)
agentic_results["search2"] = r_ag

print("\\n" + "=" * 70)
print("search2 RESULTS")
print("=" * 70)
v1_gt, v1_sc = v1_raw_data["search2"]
na_gt = data["search2"].tolist()
print_metrics("v1 (old models)", v1_gt, v1_sc)
print_metrics("v2 Non-Agentic", na_gt, r_na.apply(get_score, axis=1).tolist())
print_metrics("v2 Agentic", na_gt, r_ag.apply(get_score, axis=1).tolist())"""))

    # ── Task 3: search3 ──
    cells.append(("md", """## Task 3: search3 (hardest — CT + DL + diagnosis + external validation)

**Inclusion**: CT + DL + diagnosis | **Exclusion**: No PET, no cardio, must have external validation | **5.6% relevant**

This is the hardest task — external validation info is often not in the abstract. Agentic search should help."""))
    cells.append(("code", """sc3_inc = {1: "The study must involve CT scans. If multiple modalities are involved, CT scans should be among them.",
           2: "The study should introduce, develop, or discuss a deep learning-based classifier.",
           3: "The clinical application or model in the study must focus on diagnosis."}
sc3_exc = {1: "The study must not include PET scans as one of its modalities.",
           2: "Studies focusing on cardiovascular organs or lung vasculature must be excluded.",
           3: "All studies that did not have an external test set in addition to their internal test set must be excluded."}

r_na, _ = await run_evaluation("search3", sc3_inc, sc3_exc, data.copy(), agentic=False)
r_na.to_csv("custom_data/search3_v2_nonagentic.csv", index=False)
non_agentic_results["search3"] = r_na

r_ag, _ = await run_evaluation("search3", sc3_inc, sc3_exc, data.copy(), agentic=True, working_dir_base="custom_data/agentic_runs")
r_ag.to_csv("custom_data/search3_v2_agentic.csv", index=False)
agentic_results["search3"] = r_ag

print("\\n" + "=" * 70)
print("search3 RESULTS")
print("=" * 70)
v1_gt, v1_sc = v1_raw_data["search3"]
na_gt = data["search3"].tolist()
print_metrics("v1 (old models)", v1_gt, v1_sc)
print_metrics("v2 Non-Agentic", na_gt, r_na.apply(get_score, axis=1).tolist())
print_metrics("v2 Agentic", na_gt, r_ag.apply(get_score, axis=1).tolist())"""))

    # ── Final comparison plots ──
    cells.append(("md", "## Final Comparison Plots"))
    cells.append(("code", """# Collect all metrics
all_metrics = {}

all_metrics["v1 (old models)"] = {}
for name, (labels, scores) in v1_raw_data.items():
    m = evaluate_metrics(labels, scores)
    m['_raw'] = (labels, scores)
    all_metrics["v1 (old models)"][name] = m

all_metrics["v2 Non-Agentic"] = {}
for name in ["search1", "search2", "search3"]:
    gt = data[name].tolist()
    sc = non_agentic_results[name].apply(get_score, axis=1).tolist()
    m = evaluate_metrics(gt, sc)
    m['_raw'] = (gt, sc)
    all_metrics["v2 Non-Agentic"][name] = m

all_metrics["v2 Agentic"] = {}
for name in ["search1", "search2", "search3"]:
    gt = data[name].tolist()
    sc = agentic_results[name].apply(get_score, axis=1).tolist()
    m = evaluate_metrics(gt, sc)
    m['_raw'] = (gt, sc)
    all_metrics["v2 Agentic"][name] = m

conditions = list(all_metrics.keys())
datasets = ["search1", "search2", "search3"]
colors_c = ['#4C72B0', '#DD8452', '#55A868']
x = np.arange(len(datasets))
n_c = len(conditions)
width = 0.8 / n_c"""))

    cells.append(("code", """# ── AUC Comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
for i, cond in enumerate(conditions):
    vals = [all_metrics[cond][ds]['auc'] for ds in datasets]
    bars = ax.bar(x + i*width - (n_c-1)*width/2, vals, width, label=cond, color=colors_c[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=9)
ax.set_ylabel('AUC'); ax.set_title('AUC Comparison'); ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.legend(); ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()"""))

    cells.append(("code", """# ── F1 (Balanced) Comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
for i, cond in enumerate(conditions):
    vals = [all_metrics[cond][ds]['f1_balanced'] for ds in datasets]
    bars = ax.bar(x + i*width - (n_c-1)*width/2, vals, width, label=cond, color=colors_c[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=9)
ax.set_ylabel('F1 Score'); ax.set_title('F1 Score (Balanced Threshold)'); ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.legend(); ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()"""))

    cells.append(("code", """# ── ROC Curves ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ds_idx, ds in enumerate(datasets):
    ax = axes[ds_idx]
    ax.plot([0,1], [0,1], '--', color='gray', alpha=0.5)
    for i, cond in enumerate(conditions):
        gt, sc = all_metrics[cond][ds]['_raw']
        if len(set(gt)) > 1:
            fpr, tpr, _ = roc_curve(gt, sc)
            auc_v = roc_auc_score(gt, sc)
            ax.plot(fpr, tpr, color=colors_c[i], label=f'{cond} ({auc_v:.3f})')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(ds)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
plt.suptitle('ROC Curves', fontsize=14); plt.tight_layout(); plt.show()"""))

    cells.append(("code", """# ── Recall & Precision (all thresholds) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_idx, mode in enumerate(['sensitive', 'balanced', 'specific']):
    ax = axes[ax_idx]
    for i, cond in enumerate(conditions):
        f1s = [all_metrics[cond][ds][f'f1_{mode}'] for ds in datasets]
        ax.bar(x + i*width - (n_c-1)*width/2, f1s, width, label=cond, color=colors_c[i], alpha=0.85)
    ax.set_ylabel('F1 Score'); ax.set_title(f'{mode.capitalize()} Mode')
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3)
    if ax_idx == 0: ax.legend(fontsize=8)
plt.suptitle('F1 Score Across Thresholds', fontsize=14); plt.tight_layout(); plt.show()"""))

    cells.append(("code", """# ── Summary Table ──
rows = []
for cond in conditions:
    for ds in datasets:
        m = all_metrics[cond][ds]
        rows.append({
            'Condition': cond, 'Dataset': ds, 'AUC': m['auc'],
            'F1 (Sensitive)': m['f1_sensitive'], 'Recall (Sensitive)': m['recall_sensitive'],
            'F1 (Balanced)': m['f1_balanced'], 'Recall (Balanced)': m['recall_balanced'],
            'Precision (Balanced)': m['precision_balanced'],
            'F1 (Specific)': m['f1_specific'],
        })
pd.DataFrame(rows).round(3)"""))

    # ── Action logs ──
    cells.append(("md", "## Action Log Analysis"))
    cells.append(("code", """base = Path("custom_data/agentic_runs")
for name in ["search1", "search2", "search3"]:
    reqs, tools, tokens = [], [], []
    for chunk_dir in sorted(base.glob(f"{name}_agentic_chunk*")):
        for agent_dir in sorted(chunk_dir.rglob("logs")):
            for lf in agent_dir.glob("*.jsonl"):
                for line in lf.read_text().strip().split("\\n"):
                    if not line: continue
                    e = json.loads(line)
                    if e.get("action_type") == "review_complete":
                        d = e.get("details", {})
                        if d.get("requests"): reqs.append(d["requests"])
                        if d.get("tool_calls"): tools.append(d["tool_calls"])
                        if d.get("total_tokens"): tokens.append(d["total_tokens"])
    if reqs:
        print(f"{name}: {len(reqs)} items, avg {np.mean(reqs):.1f} iters, "
              f"{np.mean(tools):.1f} tool calls, {np.mean(tokens):.0f} tokens")"""))

    return make_notebook(cells)


# =============================================================================
# SYNERGY EVALUATION NOTEBOOK
# =============================================================================

def build_synergy_notebook():
    cells = []

    cells.append(("md", """# Synergy Evaluation: v1 vs v2 Non-Agentic vs v2 Agentic

Replicates the Synergy dataset evaluation using v2. Same 3-way comparison as custom evaluation.
Models: `gemini-3.1-flash-lite-preview` + `gpt-5.4-mini`"""))

    cells.append(("md", "## Setup"))
    cells.append(("code", """%reload_ext autoreload
%autoreload 2
from dotenv import load_dotenv
load_dotenv(dotenv_path='../.env')
import sys
sys.path.append('../')"""))

    cells.append(("code", """import glob, json, shutil, warnings, pickle
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve, f1_score)
from pydantic import BaseModel, Field
from lattereview.agentic import AgenticReviewer, AgenticWorkflow"""))

    # Reuse the same infrastructure cell (models, prompts, workflow runner, metrics)
    cells.append(("md", "## Infrastructure (models, prompts, workflow)"))
    cells.append(("code", """class TitleAbstractOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the evaluation.")
    evaluation: int = Field(ge=1, le=5, description="Evaluation score: 1=absolutely exclude, 2=better to exclude, 3=not sure, 4=better to include, 5=absolutely include.")
    certainty: int = Field(ge=0, le=100, description="Confidence in the evaluation (0-100).")

SYSTEM_PROMPT = (
    "You are an expert systematic reviewer. Your task is to evaluate whether "
    "studies should be included or excluded based on their title and abstract. "
    "Be thorough and apply the criteria consistently."
)
TASK_PROMPT_TEMPLATE = (
    "**Review the title and abstract below and evaluate whether they should be "
    "included based on the following inclusion and exclusion criteria (if any).**\\n"
    "**Note that the study should be included only and only if it meets ALL "
    "inclusion criteria and NONE of the exclusion criteria.**\\n\\n---\\n\\n"
    "**Input item:**\\n<<${item}$>>\\n\\n---\\n\\n"
    "**Inclusion criteria:**\\nINCLUSION_CRITERIA_PLACEHOLDER\\n\\n"
    "**Exclusion criteria:**\\nEXCLUSION_CRITERIA_PLACEHOLDER\\n\\n---\\n\\n"
    "**Instructions**\\n\\n"
    "1. Output your evaluation as an integer between 1 and 5, where:\\n"
    "   - 1 means absolutely to exclude.\\n   - 2 means better to exclude.\\n"
    "   - 3 Not sure if to include or exclude.\\n   - 4 means better to include.\\n"
    "   - 5 means absolutely to include.\\n"
    "2. Report your certainty level between **0** and **100**.\\n"
    "3. Provide your reasoning before assigning a decision."
)
SENIOR_SYSTEM_PROMPT = (
    "You are a senior expert systematic reviewer. Two junior reviewers have already "
    "reviewed this article. Use your expertise to make the final determination."
)

def create_reviewers(inclusion_criteria, exclusion_criteria, agentic=False):
    task_prompt = TASK_PROMPT_TEMPLATE.replace("INCLUSION_CRITERIA_PLACEHOLDER", str(inclusion_criteria)).replace("EXCLUSION_CRITERIA_PLACEHOLDER", str(exclusion_criteria))
    ca = dict(agentic_effort="high", skills=["searching-duckduckgo"])
    gak = {**ca, "max_iterations": 30} if agentic else {"max_iterations": 1}
    oak = {**ca, "max_iterations": 15} if agentic else {"max_iterations": 1}
    A1 = AgenticReviewer(name="Agent1", model="google-gla:gemini-3.1-flash-lite-preview", backstory="a PhD researcher", system_prompt=SYSTEM_PROMPT, task_prompt=task_prompt, output_type=TitleAbstractOutput, max_concurrent_requests=20 if not agentic else 10, model_settings={"temperature": 0.1}, **gak)
    A2 = AgenticReviewer(name="Agent2", model="openai:gpt-5.4-mini", backstory="a PhD researcher", system_prompt=SYSTEM_PROMPT, task_prompt=task_prompt, output_type=TitleAbstractOutput, max_concurrent_requests=20 if not agentic else 10, model_settings={"temperature": 0.1}, **oak)
    A3 = AgenticReviewer(name="Agent3", model="openai:gpt-5.4-mini", backstory="a senior MD-PhD researcher", system_prompt=SENIOR_SYSTEM_PROMPT, task_prompt=task_prompt, output_type=TitleAbstractOutput, max_concurrent_requests=20 if not agentic else 10, model_settings={"temperature": 0.1}, **oak)
    return A1, A2, A3

def split_dataframe(df, n):
    return [df.iloc[i:i+n] for i in range(0, len(df), n)]

async def run_evaluation(name, inc, exc, df, agentic=False, wdir=None):
    mode = "agentic" if agentic else "non_agentic"
    print(f"\\n{'='*60}\\n{name} — {mode}\\n{'='*60}")
    chunks = split_dataframe(df, 1000)
    results, cost = [], 0.0
    for ci, sub in enumerate(chunks):
        A1, A2, A3 = create_reviewers(inc, exc, agentic)
        def ff(row):
            try:
                s1, s2 = int(row["round-A_Agent1_output"]["evaluation"]), int(row["round-A_Agent2_output"]["evaluation"])
            except (TypeError, KeyError, ValueError):
                return False
            if s1 != s2:
                if s1 >= 4 and s2 >= 4: return False
                if s1 >= 3 or s2 >= 3: return True
            elif s1 == s2 == 3: return True
            return False
        wd = Path(wdir)/f"{name}_{mode}_chunk{ci}" if wdir and agentic else None
        if wd and wd.exists(): shutil.rmtree(wd)
        wf = AgenticWorkflow(workflow_schema=[
            {"round": "A", "reviewers": [A1, A2], "text_inputs": ["title", "abstract"]},
            {"round": "B", "reviewers": [A3], "text_inputs": ["title", "abstract", "round-A_Agent1_output", "round-A_Agent2_output"], "filter": ff},
        ], working_dir=wd, verbose=True)
        r = await wf(sub); results.append(r); cost += wf.total_cost
    return pd.concat(results), cost

def get_score(row):
    if "round-B_Agent3_output" in row and pd.notna(row.get("round-B_Agent3_evaluation")):
        try: return int(row["round-B_Agent3_evaluation"])
        except (TypeError, ValueError): pass
    try: return (int(row["round-A_Agent1_evaluation"]) + int(row["round-A_Agent2_evaluation"])) / 2
    except (TypeError, ValueError): return 1

def evaluate_metrics(gt, preds, ts=1.5, tb=3.0, tp=4.5):
    cl = lambda p, t: [1 if x >= t else 0 for x in p]
    m = {}
    for l, t in [('sensitive', ts), ('balanced', tb), ('specific', tp)]:
        p = cl(preds, t); tn, fp, fn, tp_ = confusion_matrix(gt, p, labels=[0,1]).ravel()
        m[f'accuracy_{l}'] = accuracy_score(gt, p); m[f'precision_{l}'] = precision_score(gt, p, zero_division=0)
        m[f'recall_{l}'] = recall_score(gt, p, zero_division=0); m[f'f1_{l}'] = f1_score(gt, p, zero_division=0)
        m[f'specificity_{l}'] = tn/(tn+fp) if (tn+fp)>0 else 0
    m['auc'] = roc_auc_score(gt, preds) if len(set(gt))>1 else float('nan')
    return m

def pm(name, gt, sc):
    m = evaluate_metrics(gt, sc)
    print(f"{name:<25} AUC={m['auc']:.3f}  F1(bal)={m['f1_balanced']:.3f}  Rec(bal)={m['recall_balanced']:.3f}  Prec(bal)={m['precision_balanced']:.3f}")
    return m

print("Infrastructure ready. Models: gemini-3.1-flash-lite + gpt-5.4-mini")"""))

    cells.append(("md", "## Data & Criteria"))
    cells.append(("code", """with open('synergy_data/all_review_jobs.pickle', 'rb') as f:
    all_review_jobs = pickle.load(f)
for name, df in all_review_jobs:
    print(f"{name}: {len(df)} items, {df['label_included'].sum()} positive")

criteria = {
    "appenzeller-herzog_2019": {"inc": "-Patients with Wilson's Disease of any age or stage\\n-Study drug has to be one of four established therapies, namely DPen, trientine, TTM or Zn.\\n-Control could be placebo, no treatment or any other treatment\\n-Prospective or retrospective studies\\n-Randomized, non-randomized controlled trials and comparative observational studies", "exc": "-Animal studies, case reports, case series, cross-sectional studies, before-after studies, reviews, letters, abstract-only publications, editorials, diagnostic or other testing studies and non-controlled studies"},
    "donners_2021": {"inc": "Emicizumab studies providing (1) data on humans, (2) original PK data or modeled PK data or PK/PD relationships, and (3) access to the abstract and full text in English.", "exc": "Not specified"},
    "jeyaraman_2020": {"inc": "Population: Patients with knee osteoarthritis. Intervention: MSC therapy. Comparator: Usual care. Outcomes: VAS, WOMAC, Lysholm, WORMS, KOOS, adverse events. Study Design: RCTs", "exc": "Observational studies without comparator group, animal studies, reviews"},
    "meijboom_2021": {"inc": "Studies involving transitioning from TNF-alpha inhibitor originator to biosimilar, with retransition data, original research, baseline characteristics, English.", "exc": "Not specified"},
    "muthu_2021": {"inc": "RCT with 1:1 parallel two-arm design, related to spine surgery, with dichotomous primary or secondary outcome.", "exc": "Non-human studies, continuous variable outcomes without clinical success criteria, studies without statistically significant outcomes"},
    "oud_2018": {"inc": "RCTs on DBT, MBT, TFP or ST for adults with BPD, including individual psychotherapy, 16+ weeks duration.", "exc": "Studies with <66% BPD participants, incomplete versions of specialized treatment"},
}"""))

    cells.append(("md", "## Load v1 Baseline"))
    cells.append(("code", """v1_raw = {}
for f in sorted(glob.glob("synergy_data/*_reviewed.csv")):
    df_v1 = pd.read_csv(f)
    name = f.split("/")[-1].replace("_reviewed.csv", "")
    gt = df_v1["label_included"].apply(int).tolist()
    def gvs(row):
        if "round-B_Agent3_output" in row and pd.notna(row.get("round-B_Agent3_evaluation")):
            return int(row["round-B_Agent3_evaluation"])
        return (int(row["round-A_Agent1_evaluation"]) + int(row["round-A_Agent2_evaluation"])) / 2
    sc = df_v1.apply(gvs, axis=1).tolist()
    v1_raw[name] = (gt, sc)
    print(f"v1 {name}: {len(gt)} items")

na_results = {}
ag_results = {}"""))

    # Task-by-task for each synergy dataset
    for ds_name in ["Donners_2021", "Meijboom_2021", "Oud_2018", "Jeyaraman_2020", "Muthu_2021", "Appenzeller-Herzog_2019"]:
        key = ds_name.lower()
        cells.append(("md", f"## {ds_name}"))
        cells.append(("code", f"""rn = "{ds_name}"
df = [d for n, d in all_review_jobs if n == rn][0]
c = criteria["{key}"]

r_na, _ = await run_evaluation(rn, c["inc"], c["exc"], df.copy(), agentic=False)
r_na.to_csv(f"synergy_data/{{rn}}_v2_nonagentic.csv", index=False)
na_results[rn] = r_na

r_ag, _ = await run_evaluation(rn, c["inc"], c["exc"], df.copy(), agentic=True, wdir="synergy_data/agentic_runs")
r_ag.to_csv(f"synergy_data/{{rn}}_v2_agentic.csv", index=False)
ag_results[rn] = r_ag

print("\\n" + "="*60 + f"\\n{{rn}} RESULTS\\n" + "="*60)
v1g, v1s = v1_raw[rn]
gt = df["label_included"].apply(int).tolist()
pm("v1 (old models)", v1g, v1s)
pm("v2 Non-Agentic", gt, r_na.apply(get_score, axis=1).tolist())
pm("v2 Agentic", gt, r_ag.apply(get_score, axis=1).tolist())"""))

    # Final plots
    cells.append(("md", "## Final Comparison Plots"))
    cells.append(("code", """all_m = {}
all_m["v1 (old models)"] = {n: {**evaluate_metrics(g, s), '_raw': (g, s)} for n, (g, s) in v1_raw.items()}
all_m["v2 Non-Agentic"] = {}
all_m["v2 Agentic"] = {}
for rn, df_na in na_results.items():
    df_orig = [d for n, d in all_review_jobs if n == rn][0]
    gt = df_orig["label_included"].apply(int).tolist()
    sc = df_na.apply(get_score, axis=1).tolist()
    all_m["v2 Non-Agentic"][rn] = {**evaluate_metrics(gt, sc), '_raw': (gt, sc)}
for rn, df_ag in ag_results.items():
    df_orig = [d for n, d in all_review_jobs if n == rn][0]
    gt = df_orig["label_included"].apply(int).tolist()
    sc = df_ag.apply(get_score, axis=1).tolist()
    all_m["v2 Agentic"][rn] = {**evaluate_metrics(gt, sc), '_raw': (gt, sc)}

conds = list(all_m.keys()); dsets = list(all_m[conds[0]].keys())
colors_c = ['#4C72B0', '#DD8452', '#55A868']; x = np.arange(len(dsets)); nc = len(conds); w = 0.8/nc

fig, ax = plt.subplots(figsize=(14, 5))
for i, c in enumerate(conds):
    v = [all_m[c][d]['auc'] for d in dsets]
    bars = ax.bar(x+i*w-(nc-1)*w/2, v, w, label=c, color=colors_c[i], alpha=0.85)
    for b, val in zip(bars, v):
        if not np.isnan(val): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{val:.2f}', ha='center', fontsize=7)
ax.set_ylabel('AUC'); ax.set_title('Synergy: AUC Comparison'); ax.set_xticks(x)
ax.set_xticklabels(dsets, rotation=30, ha='right'); ax.legend(); ax.set_ylim(0,1.1); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
for i, c in enumerate(conds):
    v = [all_m[c][d]['f1_balanced'] for d in dsets]
    bars = ax.bar(x+i*w-(nc-1)*w/2, v, w, label=c, color=colors_c[i], alpha=0.85)
    for b, val in zip(bars, v): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{val:.2f}', ha='center', fontsize=7)
ax.set_ylabel('F1'); ax.set_title('Synergy: F1 (Balanced)'); ax.set_xticks(x)
ax.set_xticklabels(dsets, rotation=30, ha='right'); ax.legend(); ax.set_ylim(0,1.1); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()"""))

    cells.append(("code", """# Summary table
rows = []
for c in conds:
    for d in dsets:
        m = all_m[c][d]
        rows.append({'Condition': c, 'Dataset': d, 'AUC': m['auc'], 'F1(bal)': m['f1_balanced'],
                     'Recall(bal)': m['recall_balanced'], 'Prec(bal)': m['precision_balanced'],
                     'F1(sens)': m['f1_sensitive'], 'Recall(sens)': m['recall_sensitive']})
pd.DataFrame(rows).round(3)"""))

    return make_notebook(cells)


if __name__ == "__main__":
    custom_nb = build_custom_notebook()
    with open("evaluation/agentic_custom_evaluation.ipynb", "w") as f:
        nbf.write(custom_nb, f)
    print("Created: evaluation/agentic_custom_evaluation.ipynb")

    synergy_nb = build_synergy_notebook()
    with open("evaluation/agentic_synergy_evaluation.ipynb", "w") as f:
        nbf.write(synergy_nb, f)
    print("Created: evaluation/agentic_synergy_evaluation.ipynb")
