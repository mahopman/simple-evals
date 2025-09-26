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

## Notes

- OpenAI o-series models do not accept a system prompt (handled internally by the samplers).
- Reasoning models (e.g., `o3`, `o4-mini`) can optionally set `reasoning_effort` in the OpenAI Responses API path.
- Some evals support repeats/threads: HealthBench and HealthBenchMeta respect `--n-repeats` and `--n-threads`.
