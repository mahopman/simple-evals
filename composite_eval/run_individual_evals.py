import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# Relative imports from the simple-evals package
from ..mmlu_eval import MMLUEval
from ..math_eval import MathEval
from ..gpqa_eval import GPQAEval
from ..drop_eval import DropEval
from ..mgsm_eval import MGSMEval
from ..humaneval_eval import HumanEval
from ..simpleqa_eval import SimpleQAEval
from ..browsecomp_eval import BrowseCompEval
from ..healthbench_eval import HealthBenchEval
from ..sampler.chat_completion_sampler import ChatCompletionSampler, OPENAI_SYSTEM_MESSAGE_API, OPENAI_SYSTEM_MESSAGE_CHATGPT
from ..sampler.o_chat_completion_sampler import OChatCompletionSampler
from ..sampler.responses_sampler import ResponsesSampler
from ..sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from ..types import Eval, SamplerBase


def build_sampler(model: str) -> SamplerBase:
    reasoning_effort = os.getenv("REASONING_EFFORT")
    # Use appropriate samplers for o-series models
    # - o1 family via o_chat_completion_sampler
    # - o3, o4-mini via responses sampler with reasoning
    # - o3-mini via o_chat_completion_sampler
    model_lower = model.lower()
    # Claude models
    if model_lower.startswith("claude-"):
        return ClaudeCompletionSampler(model=model, system_message=os.getenv("CLAUDE_SYSTEM_MESSAGE", CLAUDE_SYSTEM_MESSAGE_LMSYS))
    # ChatGPT named model
    if model_lower == "chatgpt-4o-latest":
        return ChatCompletionSampler(model=model, system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT, max_tokens=2048)
    if model_lower.startswith("o1-pro"):
        return ResponsesSampler(model=model, reasoning_model=True, reasoning_effort=reasoning_effort)
    if model_lower.startswith("o1"):
        return OChatCompletionSampler(model=model, reasoning_effort=reasoning_effort)
    if model_lower == "o3" or model_lower.startswith("o3-"):
        return ResponsesSampler(model=model, reasoning_model=True, reasoning_effort=reasoning_effort)
    if model_lower == "o4-mini" or model_lower.startswith("o4-mini"):
        return ResponsesSampler(model=model, reasoning_model=True, reasoning_effort=reasoning_effort)
    if model_lower.startswith("o3-mini"):
        return OChatCompletionSampler(model=model, reasoning_effort=reasoning_effort)
    # default chat completion sampler for non o-series
    return ChatCompletionSampler(model=model)


def build_grader_sampler() -> SamplerBase:
    # Use a strong, stable model for grading by default; allow override via GRADER_MODEL
    grader_model = os.getenv("GRADER_MODEL", "gpt-4.1-2025-04-14")
    if grader_model.lower().startswith("o"):
        return build_sampler(grader_model)
    return ChatCompletionSampler(
        model=grader_model,
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )


def maybe_limit_eval(eval_obj: Eval, max_examples: Optional[int]) -> Eval:
    # Best-effort limiting for heavy evals if user passed --max-examples
    if max_examples is None:
        return eval_obj
    # Where supported by constructors, we prefer to pass limits there. Here we rely on defaults already set
    # in individual evals; most do not expose a direct setter, so we skip mutation.
    return eval_obj


def make_eval_constructors(grader_sampler: SamplerBase, max_examples: Optional[int]) -> List[Tuple[str, Callable[[SamplerBase], Eval]]]:
    # Each entry returns a freshly constructed Eval instance when called (so per-model runs are independent)
    return [
        (
            "MMLU",
            lambda _sampler: maybe_limit_eval(MMLUEval(), max_examples),
        ),
        (
            "MATH",
            lambda _sampler: maybe_limit_eval(
                MathEval(equality_checker=grader_sampler),
                max_examples,
            ),
        ),
        (
            "GPQA",
            lambda _sampler: maybe_limit_eval(GPQAEval(), max_examples),
        ),
        (
            "DROP",
            lambda _sampler: maybe_limit_eval(DropEval(), max_examples),
        ),
        (
            "MGSM",
            lambda _sampler: maybe_limit_eval(MGSMEval(), max_examples),
        ),
        (
            "HumanEval",
            lambda _sampler: maybe_limit_eval(HumanEval(), max_examples),
        ),
        (
            "SimpleQA",
            lambda _sampler: maybe_limit_eval(SimpleQAEval(grader_model=grader_sampler), max_examples),
        ),
        (
            "BrowseComp",
            lambda _sampler: maybe_limit_eval(BrowseCompEval(grader_model=grader_sampler), max_examples),
        ),
        (
            "HealthBench",
            lambda _sampler: maybe_limit_eval(HealthBenchEval(grader_model=grader_sampler), max_examples),
        ),
    ]


def run_single_eval(eval_builder: Callable[[SamplerBase], Eval], sampler: SamplerBase) -> Optional[float]:
    try:
        eval_obj = eval_builder(sampler)
        result = eval_obj(sampler)
        return result.score
    except Exception as e:
        print(f"Eval failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run all simple-evals benchmarks for two models and output a CSV.")
    parser.add_argument("--current_model", type=str, default=os.getenv("CURRENT_MODEL", "o4-mini"))
    parser.add_argument("--next_gen_model", type=str, default=os.getenv("NEXT_GEN_MODEL", "claude-opus-4-1-20250805"))
    parser.add_argument("--output_csv", type=str, default="composite_eval/results.csv")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap for examples per eval (best-effort; some evals may ignore).",
    )
    args = parser.parse_args()

    current_model_name = args.current_model
    next_gen_model_name = args.next_gen_model

    grader_sampler = build_grader_sampler()
    eval_constructors = make_eval_constructors(grader_sampler, args.max_examples)

    # Prepare result table
    rows: List[Dict[str, object]] = []

    # Map column labels to model ids
    model_specs = {
        "current_model": current_model_name,
        "next_gen_model": next_gen_model_name,
    }

    for eval_name, eval_builder in eval_constructors:
        result_row: Dict[str, object] = {"benchmark": eval_name}
        for col_label, model_id in model_specs.items():
            print(f"Running {eval_name} with {col_label} = {model_id} ...")
            sampler = build_sampler(model_id)
            score = run_single_eval(eval_builder, sampler)
            result_row[col_label] = score
            print(f"Finished {eval_name} with {col_label}: score={score}")
        rows.append(result_row)

    df = pd.DataFrame(rows).set_index("benchmark")[["current_model", "next_gen_model"]]
    df.to_csv(args.output_csv)
    print("\nSaved results to:", args.output_csv)
    print("\nResults:\n", df)


if __name__ == "__main__":
    main()


