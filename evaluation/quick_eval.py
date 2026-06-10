"""Quick evaluation: 10 items per search, agentic vs non-agentic."""

import asyncio
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from lattereview.agentic import AgenticReviewer, AgenticWorkflow


# ── Output model ──
class TitleAbstractOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the evaluation.")
    evaluation: int = Field(
        ge=1, le=5,
        description="Evaluation score: 1=absolutely exclude, 2=better to exclude, "
                    "3=not sure, 4=better to include, 5=absolutely include.",
    )
    certainty: int = Field(ge=0, le=100, description="Confidence in the evaluation (0-100).")


# ── Prompts (identical to notebook) ──
SYSTEM_PROMPT = (
    "You are an expert systematic reviewer. Your task is to evaluate whether "
    "studies should be included or excluded based on their title and abstract. "
    "Be thorough and apply the criteria consistently."
)

TASK_PROMPT_TEMPLATE = (
    "**Review the title and abstract below and evaluate whether they should be "
    "included based on the following inclusion and exclusion criteria (if any).**\n"
    "**Note that the study should be included only and only if it meets ALL "
    "inclusion criteria and NONE of the exclusion criteria.**\n\n---\n\n"
    "**Input item:**\n<<${item}$>>\n\n---\n\n"
    "**Inclusion criteria:**\nINCLUSION_CRITERIA_PLACEHOLDER\n\n"
    "**Exclusion criteria:**\nEXCLUSION_CRITERIA_PLACEHOLDER\n\n---\n\n"
    "**Instructions**\n\n"
    "1. Output your evaluation as an integer between 1 and 5, where:\n"
    "   - 1 means absolutely to exclude.\n"
    "   - 2 means better to exclude.\n"
    "   - 3 Not sure if to include or exclude.\n"
    "   - 4 means better to include.\n"
    "   - 5 means absolutely to include.\n"
    "2. Report your certainty level as a value between **0** (not certain at all) "
    "and **100** (completely certain).\n"
    "3. Provide your reasoning before assigning a decision."
)

SENIOR_SYSTEM_PROMPT = (
    "You are a senior expert systematic reviewer. Two junior reviewers have already "
    "reviewed this article and their evaluations are included in the input. They may "
    "disagree. Use your expertise to make the final determination. "
    "Be thorough and apply the criteria consistently."
)


# ── Search criteria ──
SEARCHES = {
    "search1": {
        "inc": {1: "The study must involve CT scans. If multiple modalities are involved, CT scans should be among them."},
        "exc": {1: "The study must not include PET scans as one of its modalities."},
    },
    "search2": {
        "inc": {
            1: "The study must involve CT scans. If multiple modalities are involved, CT scans should be among them.",
            2: "The study should introduce, develop, or discuss a deep learning-based classifier.",
        },
        "exc": {
            1: "The study must not include PET scans as one of its modalities.",
            2: "Studies focusing on cardiovascular organs or lung vasculature must be excluded.",
        },
    },
    "search3": {
        "inc": {
            1: "The study must involve CT scans. If multiple modalities are involved, CT scans should be among them.",
            2: "The study should introduce, develop, or discuss a deep learning-based classifier.",
            3: "The clinical application or model in the study must focus on diagnosis.",
        },
        "exc": {
            1: "The study must not include PET scans as one of its modalities.",
            2: "Studies focusing on cardiovascular organs or lung vasculature must be excluded.",
            3: "All studies that did not have an external test set in addition to their internal test set must be excluded.",
        },
    },
}


def create_reviewers(inclusion_criteria, exclusion_criteria, agentic=False):
    task_prompt = (
        TASK_PROMPT_TEMPLATE
        .replace("INCLUSION_CRITERIA_PLACEHOLDER", str(inclusion_criteria))
        .replace("EXCLUSION_CRITERIA_PLACEHOLDER", str(exclusion_criteria))
    )

    gemini_agentic_kwargs = dict(
        max_iterations=30,
        agentic_effort="high",
        skills=["searching-duckduckgo"],
    ) if agentic else dict(max_iterations=1)

    gpt_agentic_kwargs = dict(
        max_iterations=15,
        agentic_effort="high",
        skills=["searching-duckduckgo"],
    ) if agentic else dict(max_iterations=1)

    Agent1 = AgenticReviewer(
        name="Agent1",
        model="google-gla:gemini-3.1-flash-lite-preview",
        backstory="a PhD researcher",
        system_prompt=SYSTEM_PROMPT,
        task_prompt=task_prompt,
        output_type=TitleAbstractOutput,
        max_concurrent_requests=20 if not agentic else 10,
        model_settings={"temperature": 0.1},
        **gemini_agentic_kwargs,
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
        **gpt_agentic_kwargs,
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
        **gpt_agentic_kwargs,
    )
    return Agent1, Agent2, Agent3


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
        return 1


def print_metrics(name, gt, scores):
    pred_bal = [1 if s >= 3.0 else 0 for s in scores]
    pred_sens = [1 if s >= 1.5 else 0 for s in scores]
    auc = roc_auc_score(gt, scores) if len(set(gt)) > 1 else float("nan")
    f1_bal = f1_score(gt, pred_bal, zero_division=0)
    f1_sens = f1_score(gt, pred_sens, zero_division=0)
    recall_bal = sum(1 for g, p in zip(gt, pred_bal) if g == 1 and p == 1) / max(sum(gt), 1)
    recall_sens = sum(1 for g, p in zip(gt, pred_sens) if g == 1 and p == 1) / max(sum(gt), 1)
    print(f"  {name:<20} AUC={auc:.3f}  F1(bal)={f1_bal:.3f}  Recall(bal)={recall_bal:.3f}  F1(sens)={f1_sens:.3f}  Recall(sens)={recall_sens:.3f}")


async def run_search(search_name, data, n_sample=10):
    """Run agentic + non-agentic on n_sample items for one search."""
    cfg = SEARCHES[search_name]
    gt_col = search_name

    # Sample: half positive, half negative (stratified)
    pos = data[data[gt_col] == 1]
    neg = data[data[gt_col] == 0]
    n_pos = min(n_sample // 2, len(pos))
    n_neg = n_sample - n_pos
    sample = pd.concat([
        pos.sample(n=n_pos, random_state=42),
        neg.sample(n=n_neg, random_state=42),
    ]).reset_index(drop=True)

    gt = sample[gt_col].tolist()
    print(f"\n{'='*60}")
    print(f"{search_name}: {n_sample} items ({sum(gt)} pos, {n_sample - sum(gt)} neg)")
    print(f"{'='*60}")

    results = {}
    for mode_name, agentic in [("non-agentic", False), ("agentic", True)]:
        Agent1, Agent2, Agent3 = create_reviewers(cfg["inc"], cfg["exc"], agentic=agentic)

        def filter_func(row):
            try:
                s1 = int(row["round-A_Agent1_output"]["evaluation"])
                s2 = int(row["round-A_Agent2_output"]["evaluation"])
            except (TypeError, KeyError, ValueError):
                return False
            if s1 != s2:
                if s1 >= 4 and s2 >= 4:
                    return False
                if s1 >= 3 or s2 >= 3:
                    return True
            elif s1 == s2 == 3:
                return True
            return False

        working_dir = None
        if agentic:
            working_dir = Path(f"/tmp/quick_eval_{search_name}_{mode_name}")
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

        result_df = await workflow(sample.copy())
        scores = result_df.apply(get_score, axis=1).tolist()
        results[mode_name] = scores
        print(f"  {mode_name} cost: ${workflow.total_cost:.4f}")

    # Print comparison
    print(f"\n  --- {search_name} metrics ---")
    print_metrics("Non-Agentic", gt, results["non-agentic"])
    print_metrics("Agentic", gt, results["agentic"])

    return gt, results


async def main():
    data = pd.read_csv(Path(__file__).resolve().parent / "custom_data" / "custom_data.csv")

    # Ground truth labels
    data["search1"] = data["modality"].isin(["ct", "xr, ct"]).astype(int)
    data["search2"] = (
        (data["search1"] == 1)
        & (data["deep_learning"] == 1)
        & (data["task"].str.contains("classification"))
        & (data["cardiovascular"] == 0)
    ).astype(int)
    data["search3"] = (
        (data["search2"] == 1)
        & (data["clinical_application"] == "diagnosis")
        & (data["external_validation"] == 1)
    ).astype(int)

    print(f"Loaded {len(data)} articles")
    for s in ["search1", "search2", "search3"]:
        print(f"  {s}: {data[s].sum()} positive / {len(data)} total")

    all_results = {}
    for search_name in ["search1", "search2", "search3"]:
        gt, results = await run_search(search_name, data, n_sample=10)
        all_results[search_name] = (gt, results)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for search_name in ["search1", "search2", "search3"]:
        gt, results = all_results[search_name]
        print(f"\n{search_name}:")
        print_metrics("Non-Agentic", gt, results["non-agentic"])
        print_metrics("Agentic", gt, results["agentic"])


if __name__ == "__main__":
    asyncio.run(main())
