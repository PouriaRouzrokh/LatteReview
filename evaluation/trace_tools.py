"""Trace exactly which tools GPT and Gemini call per item."""

import asyncio
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from lattereview.agentic import AgenticReviewer
from lattereview.agentic.deps import ReviewDeps
from lattereview.agentic.prompts import build_system_prompt, build_task_prompt


class TitleAbstractOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the evaluation.")
    evaluation: int = Field(ge=1, le=5, description="Evaluation score: 1-5.")
    certainty: int = Field(ge=0, le=100, description="Confidence 0-100.")


SYSTEM_PROMPT = (
    "You are an expert systematic reviewer. Your task is to evaluate whether "
    "studies should be included or excluded based on their title and abstract. "
    "Be thorough and apply the criteria consistently."
)

sc3_inc = {
    1: "The study must involve CT scans. If multiple modalities are involved, CT scans should be among them.",
    2: "The study should introduce, develop, or discuss a deep learning-based classifier.",
    3: "The clinical application or model in the study must focus on diagnosis.",
}
sc3_exc = {
    1: "The study must not include PET scans as one of its modalities.",
    2: "Studies focusing on cardiovascular organs or lung vasculature must be excluded.",
    3: "All studies that did not have an external test set in addition to their internal test set must be excluded.",
}

TASK_PROMPT_TEMPLATE = (
    "**Review the title and abstract below and evaluate whether they should be "
    "included based on the following inclusion and exclusion criteria (if any).**\n"
    "**Note that the study should be included only and only if it meets ALL "
    "inclusion criteria and NONE of the exclusion criteria.**\n\n---\n\n"
    "**Input item:**\n<<${item}$>>\n\n---\n\n"
    f"**Inclusion criteria:**\n{sc3_inc}\n\n"
    f"**Exclusion criteria:**\n{sc3_exc}\n\n---\n\n"
    "**Instructions**\n\n"
    "1. Output your evaluation as an integer between 1 and 5.\n"
    "2. Report your certainty level 0-100.\n"
    "3. Provide your reasoning before assigning a decision."
)


async def trace_single_item(model_str, model_label, item_text, item_id):
    """Run one item and trace all tool calls from message history."""
    r = AgenticReviewer(
        name=f"Tracer_{model_label}",
        model=model_str,
        backstory="a PhD researcher",
        system_prompt=SYSTEM_PROMPT,
        task_prompt=TASK_PROMPT_TEMPLATE,
        output_type=TitleAbstractOutput,
        max_iterations=30,
        agentic_effort="high",
        skills=["searching-duckduckgo"],
        model_settings={"temperature": 0.1},
    )

    # Run review to get the result object with message history
    # We need to go lower-level to capture messages
    skill_toolsets, _descs = r._setup_skills()
    system_prompt_str = r._build_system_prompt()
    agent = r._build_agent(system_prompt_str, skill_toolsets)
    user_prompt = build_task_prompt(r.task_prompt, item_text)

    deps = ReviewDeps(
        item_id=item_id,
        item_text=item_text,
        agent_name=r.name,
        round_id="A",
        max_iterations=r.max_iterations,
        agentic_effort=r.agentic_effort,
    )

    result = await agent.run(
        user_prompt,
        deps=deps,
        usage_limits=UsageLimits(request_limit=30),
    )

    # Trace tool calls from message history
    tool_calls = []
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            part_kind = getattr(part, "part_kind", "")
            if part_kind == "tool-call":
                tool_name = getattr(part, "tool_name", "?")
                # Get args summary
                args = getattr(part, "args", {})
                if isinstance(args, dict):
                    args_summary = {k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v) for k, v in args.items()}
                else:
                    args_summary = str(args)[:100]
                tool_calls.append((tool_name, args_summary))
            elif part_kind == "tool-return":
                pass  # we already have the call

    output = result.output
    print(f"\n{'='*70}")
    print(f"{model_label} on item {item_id}")
    print(f"  Score: {output.evaluation}  Certainty: {output.certainty}")
    print(f"  Tool calls ({len(tool_calls)}):")
    for i, (name, args) in enumerate(tool_calls):
        print(f"    {i+1}. {name}({args})")
    print(f"  Reasoning: {output.reasoning[:200]}...")
    return output, tool_calls


async def main():
    data = pd.read_csv(Path(__file__).resolve().parent / "custom_data" / "custom_data.csv")
    data["search3"] = (
        data["modality"].isin(["ct", "xr, ct"]).astype(int)
        & (data["deep_learning"] == 1)
        & data["task"].str.contains("classification")
        & (data["cardiovascular"] == 0)
        & (data["clinical_application"] == "diagnosis")
        & (data["external_validation"] == 1)
    ).astype(int)

    # Pick 3 items: 1 positive, 1 negative-but-tricky, 1 clear negative
    pos = data[data["search3"] == 1].sample(n=1, random_state=42)
    neg = data[data["search3"] == 0].sample(n=2, random_state=42)
    sample = pd.concat([pos, neg]).reset_index(drop=True)

    for idx, row in sample.iterrows():
        item_text = f"Title: {row['title']}\nAbstract: {row['abstract']}"
        gt = row["search3"]
        print(f"\n{'#'*70}")
        print(f"Item {idx} (ground truth: {'INCLUDE' if gt == 1 else 'EXCLUDE'})")
        print(f"Title: {row['title'][:100]}...")

        # Run GPT
        await trace_single_item("openai:gpt-5.4-mini", "GPT-5.4-mini", item_text, f"item-{idx}")

        # Run Gemini
        await trace_single_item("google-gla:gemini-3.1-flash-lite-preview", "Gemini-flash-lite", item_text, f"item-{idx}")


if __name__ == "__main__":
    asyncio.run(main())
