# Overview
This repository contains a lightweight library for evaluating language models.

## Background

Evals are sensitive to prompting, and there's significant variation in the formulations used in recent publications and libraries.
Some use few-shot prompts or role playing prompts ("You are an expert software programmer...").
These approaches are carryovers from evaluating *base models* (rather than instruction/chat-tuned models) and from models that were worse at following instructions.

For this library, we are emphasizing the *zero-shot, chain-of-thought* setting, with simple instructions like "Solve the following multiple choice problem". We believe that this prompting technique is a better reflection of the models' performance in realistic usage.

## Evals

This repository currently contains the following evals:

- MMLU: Measuring Massive Multitask Language Understanding, reference: https://arxiv.org/abs/2009.03300, https://github.com/hendrycks/test, [MIT License](https://github.com/hendrycks/test/blob/master/LICENSE)
- MATH: Measuring Mathematical Problem Solving With the MATH Dataset, reference: https://arxiv.org/abs/2103.03874, https://github.com/hendrycks/math, [MIT License](https://github.com/idavidrein/gpqa/blob/main/LICENSE)
- GPQA: A Graduate-Level Google-Proof Q&A Benchmark, reference: https://arxiv.org/abs/2311.12022, https://github.com/idavidrein/gpqa/,  [MIT License](https://github.com/idavidrein/gpqa/blob/main/LICENSE)
- DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs, reference: https://arxiv.org/abs/1903.00161, https://allenai.org/data/drop, [Apache License 2.0](https://github.com/allenai/allennlp-models/blob/main/LICENSE)
- MGSM: Multilingual Grade School Math Benchmark (MGSM), Language Models are Multilingual Chain-of-Thought Reasoners, reference: https://arxiv.org/abs/2210.03057, https://github.com/google-research/url-nlp, [Creative Commons Attribution 4.0 International Public License (CC-BY)](https://github.com/google-research/url-nlp/blob/main/LICENSE)
- HumanEval: Evaluating Large Language Models Trained on Code, reference https://arxiv.org/abs/2107.03374, https://github.com/openai/human-eval, [MIT License](https://github.com/openai/human-eval/blob/master/LICENSE)
- SimpleQA: Measuring short-form factuality in large language models, reference: https://openai.com/index/introducing-simpleqa, [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)
- BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents, reference: https://openai.com/index/browsecomp, [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)
- HealthBench: Evaluating Large Language Models Towards Improved Human Health, reference: https://openai.com/index/healthbench, [MIT License](https://github.com/openai/simple-evals/blob/main/LICENSE)

## Samplers

We have implemented sampling interfaces for the following language model APIs:

- OpenAI: https://platform.openai.com/docs/overview
- Claude: https://www.anthropic.com/api

Make sure to set the `*_API_KEY` environment variables before using these APIs.

## Setup

Due to the optional dependencies, we're not providing a unified setup mechanism. Instead, we're providing instructions for each eval and sampler.

For [HumanEval](https://github.com/openai/human-eval/) (python programming)
```bash
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

For the [OpenAI API](https://pypi.org/project/openai/):
```bash
pip install openai
```

For the [Anthropic API](https://docs.anthropic.com/claude/docs/quickstart-guide):
```bash
pip install anthropic
```

## Running the evals
```bash
python -m simple-evals.simple_evals --list-models
```
This will list all the models that you can evaluate.

To run the evaluations, you can use the following command:
```bash
python -m simple-evals.simple_evals --model <model_name> --examples <num_examples>
```


## Notes

[^1]:chatgpt system message: "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
[^2]:assistant system message in [OpenAI API doc](https://platform.openai.com/docs/api-reference/introduction): "You are a helpful assistant." .
[^3]:claude-3 empty system message: suggested by Anthropic API doc, and we have done limited experiments due to [rate limit](https://docs.anthropic.com/claude/reference/rate-limits) issues, but we welcome PRs with alternative choices.
[^4]:claude-3 lmsys system message: system message in LMSYS [Fast-chat open source code](https://github.com/lm-sys/FastChat/blob/7899355ebe32117fdae83985cf8ee476d2f4243f/fastchat/conversation.py#L894): "The assistant is Claude, created by Anthropic. The current date is {{currentDateTime}}. Claude's knowledge base was last updated ... ". We have done limited experiments due to [rate limit](https://docs.anthropic.com/claude/reference/rate-limits) issues, but we welcome PRs with alternative choices.
[^5]:We believe these evals are saturated for our newer models, but are reporting them for completeness.
[^6]:For newer models (anything on or after o1) we evaluate on [MATH-500](https://github.com/openai/prm800k/tree/main/prm800k/math_splits), which is a newer, IID version of MATH.
[^7]:o-series models do not support using a system prompt.
[^8]:Includes an answer regex tweak for GPQA benchmark.
[^9]:The default reasoning level for o3-mini is "medium".
[^10]:These results are with no tools enabled for o3 or o4-mini

