# Overview
This repository contains a lightweight library for evaluating language models.

## Background

Evals are sensitive to prompting, and there's significant variation in the formulations used in recent publications and libraries.
Some use few-shot prompts or role playing prompts ("You are an expert software programmer...").
These approaches are carryovers from evaluating *base models* (rather than instruction/chat-tuned models) and from models that were worse at following instructions.

For this library, we are emphasizing the *zero-shot, chain-of-thought* setting, with simple instructions like "Solve the following multiple choice problem". We believe that this prompting technique is a better reflection of the models' performance in realistic usage.

## Evals

This repository currently contains the following evals:

- MMLU: Measuring Massive Multitask Language Understanding — [arXiv](https://arxiv.org/abs/2009.03300), [dataset](https://github.com/hendrycks/test), [MIT License](https://github.com/hendrycks/test/blob/master/LICENSE)
- MATH: Measuring Mathematical Problem Solving With the MATH Dataset — [arXiv](https://arxiv.org/abs/2103.03874), [dataset](https://github.com/hendrycks/math)
- GPQA: A Graduate-Level Google-Proof Q&A Benchmark — [arXiv](https://arxiv.org/abs/2311.12022), [dataset](https://github.com/idavidrein/gpqa/), [MIT License](https://github.com/idavidrein/gpqa/blob/main/LICENSE)
- DROP: Discrete Reasoning Over Paragraphs — [arXiv](https://arxiv.org/abs/1903.00161), [dataset](https://allenai.org/data/drop), [Apache 2.0](https://github.com/allenai/allennlp-models/blob/main/LICENSE)
- MGSM: Multilingual Grade School Math — [arXiv](https://arxiv.org/abs/2210.03057), [dataset](https://github.com/google-research/url-nlp), [CC-BY 4.0](https://github.com/google-research/url-nlp/blob/main/LICENSE)
- HumanEval: Evaluating LLMs Trained on Code — [arXiv](https://arxiv.org/abs/2107.03374), [dataset](https://github.com/openai/human-eval), [MIT License](https://github.com/openai/human-eval/blob/master/LICENSE)
- SimpleQA: Measuring short-form factuality — [overview](https://openai.com/index/introducing-simpleqa), [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)
- BrowseComp: A simple yet challenging benchmark for browsing agents — [overview](https://openai.com/index/browsecomp), [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)
- HealthBench: Evaluating LLMs towards improved human health — [overview](https://openai.com/index/healthbench), [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)

Datasets needed for these evals are vendored under `data/simple-evals/...` (no separate download required for the default subsets used here).

## Samplers

Implemented sampling interfaces:

- OpenAI — [docs](https://platform.openai.com/docs/overview)
- Anthropic Claude — [API](https://www.anthropic.com/api)
- xAI Grok — [site](https://x.ai)

Make sure to set the `*_API_KEY` environment variables before using these APIs.

## Setup

Use Python 3.10+ and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the evals

Use the CLI in `cli.py`:

```bash
python cli.py --list-models
```

Run a specific eval/model (override example count if desired):

```bash
# MMLU with GPT-4.1
python cli.py --eval mmlu --model gpt-4.1 --examples 200

# MATH with o3 (reasoning model), with repeats
python cli.py --eval math --model o3 --n-repeats 10

# Multiple evals and models at once (comma-separated)
python cli.py --eval mmlu,math --model gpt-4.1,gpt-4o

# Use Grok or Claude
python cli.py --eval simpleqa --model grok-4-0709
python cli.py --eval mmlu --model claude-3-7-sonnet-20250219

# Quick smoke test (small samples)
python cli.py --debug
```

If `--eval` is omitted, a default suite runs: `mmlu, math, gpqa, mgsm, drop, humaneval, simpleqa, browsecomp, healthbench, healthbench_hard, healthbench_consensus, healthbench_meta`.

### Output

For each (eval, model) pair, three files are written under `/tmp/` with a timestamp:

- HTML report: `/tmp/<eval>_<model>_<YYYYMMDD_HHMMSS>[ _DEBUG].html`
- Metrics JSON: `/tmp/<eval>_<model>_<YYYYMMDD_HHMMSS>[ _DEBUG].json`
- Full results JSON: `/tmp/<eval>_<model>_<YYYYMMDD_HHMMSS>[ _DEBUG]_allresults.json`

At the end, a merged markdown table of metrics is printed to stdout.

## Notes

- OpenAI o-series models do not accept a system prompt (handled internally by the samplers).
- Reasoning models (e.g., `o3`, `o4-mini`) can optionally set `reasoning_effort` in the OpenAI Responses API path.
- Some evals support repeats/threads: HealthBench and HealthBenchMeta respect `--n-repeats` and `--n-threads`.
