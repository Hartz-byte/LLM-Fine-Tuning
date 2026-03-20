# Mistral-7B-Instruct-v0.2 Base Evaluation

This repository contains the baseline evaluation of the **Mistral-7B-Instruct-v0.2** model on a structured scientific research paper analysis task. The goal is to establish a performance benchmark for the base model before any fine-tuning, using the same instruction-output alignment and task style intended for the final dataset.

## Project Overview

The objective of this evaluation is to assess how well the base Mistral model can extract structured insights from research paper abstracts. This serves as a "ground truth" performance metric to later measure the effectiveness of fine-tuning.

### Key Task
- **Input**: Scientific Research Paper Abstract.
- **Goal**: Generate a structured analysis based on specific instruction keys.
- **Constraints**: Follow a strict schema and remain faithful to the source text.

## Dataset & Alignment

The evaluation is performed on a subset of the **Hartz-byte/scientific-research-papers** dataset, ensuring strict alignment with the fine-tuning data format.

### Data Structure
The dataset follows an **Instruction-Input-Output** format:

| Field | Description |
| :--- | :--- |
| **Instruction** | A natural language prompt defining the specific fields to extract (e.g., "Extract a summary and gaps"). |
| **Input** | The raw research paper abstract. |
| **Output Keys** | Structured fields derived dynamically from the task requirements. |

### Derived Output Fields
The model is evaluated on its ability to populate the following fields based on the provided instruction:
- `summary`: A concise overview of the research.
- `methods`: The technical approach or methodology used.
- `gaps`: Identified limitations or missing elements in the research.
- `evaluation`: Critical assessment or performance results mentioned in the abstract.
- `contradictions`: Identified conflicts or paradoxes within the findings.
- `comparison`: How the work relates to traditional or state-of-the-art approaches.

## Evaluation Methodology

The evaluation uses a **zero-shot** prompting technique with the Mistral instruction format:

```text
[INST]
You are analyzing a research paper abstract.

Task:
{instruction}

Return the answer using only the following fields:
- {field_1}
- {field_2}
...

Keep the response faithful to the abstract.
Do not invent unsupported claims.

Research Abstract:
{abstract}
[/INST]
```

### Metrics Collected
We use a combination of semantic and structural metrics:
- **ROUGE (1, 2, L, Lsum)**: Measures n-gram overlap with reference summaries.
- **BERTScore (F1)**: Captures semantic similarity using deep contextual embeddings.
- **Schema Field Recall**: Percentage of requested keys that the model actually produced.
- **Exact Schema Match**: Percentage of outputs that perfectly followed the requested key structure.

## Baseline Results

Below are the results from the evaluation on the test subset ($N=50$):

| Metric | Score |
| :--- | :--- |
| **ROUGE-1** | 0.5695 |
| **ROUGE-L** | 0.4385 |
| **BERTScore (F1)** | 0.9107 |
| **Schema Field Recall** | 94.28% |
| **Exact Schema Match** | 88.00% |

> [!NOTE]
> The base model shows strong semantic alignment (high BERTScore) but occasionally misses specific keys or follows a slightly different formatting, leading to the 88% Exact Schema Match. Fine-tuning is expected to push this to near 100%.

## Project Structure

- `mistral7b-baseline-evaluation.ipynb`: The core evaluation implementation notebook.
- `baseline_predictions.json`: Raw model outputs compared against references.
- `baseline_metrics.json`: Aggregated performance scores.
- `manual_review_samples.json`: A curated list of samples for qualitative inspection.
- `dataset-split/`: Directory containing the `train`, `val`, and `test` JSON splits used for consistency across experiments.

---
*Built for the LLM Fine-Tuning Pipeline*
