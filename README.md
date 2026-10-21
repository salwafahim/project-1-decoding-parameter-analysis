# Decoding Parameter and Sampling Analysis

## Project Overview
This project evaluates how Temperature, Top-K, and Top-P decoding parameters affect the outputs of Large Language Models (LLMs).

The experiments focus on three aspects:
- Output determinism
- Hallucination
- Output structure and formatting

## Models
- Llama 3.2 1B (`llama3.2:1b`)
- Qwen 2.5 1.5B (`qwen2.5:1.5b`)

Both models are run locally using Ollama because of hardware limitations (8 GB RAM, CPU only).

## Libraries
Python libraries used in this project:
- ollama
- pandas
- numpy
- matplotlib
- seaborn
- json
- difflib
- tqdm
- itertools

## Project Files
- `prompt.py` — Prompt set used for experiments
- `common.py` — Shared functions and settings
- `temperature.py` — Temperature experiments
- `top_k.py` — Top-K experiments
- `top_p.py` — Top-P experiments
- `analyze.py` — Results analysis and visualization

## Methodology
The project uses factual questions, repeated prompts, and JSON-format prompts. Each prompt is tested multiple times under different decoding settings to compare model outputs.

## Expected Outcomes
The experiments investigate how decoding parameters affect consistency, hallucinations, diversity, and formatting. The results are documented in the project report.

## Report
See the PDF report included in this repository for the methodology, results, graphs, and conclusion.
