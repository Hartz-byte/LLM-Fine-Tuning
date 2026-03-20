# End-to-End LLM Fine-Tuning Pipeline for Structured Scientific Research Analysis

This repository contains a complete **end-to-end LLM project** for building a domain-specific structured analysis system over **scientific research paper abstracts**. The project covers the full lifecycle from automated data generation to parameter-efficient fine-tuning and quantitative evaluation.

The final outcome is a fine-tuned model that reliably transforms scientific abstracts into structured analytical outputs (Summary, Methods, Gaps, Evaluation, Comparison, and Contradictions).

---

## Project Overview

Large Language Models are strong at general text generation, but domain-specific structured extraction tasks often suffer from poor schema adherence and inconsistent formatting. This project addresses these challenges by:
- Creating a custom instruction-tuning dataset from raw scientific papers.
- Establishing a performance benchmark using the base **Mistral-7B-Instruct-v0.2**.
- Fine-tuning the model using **QLoRA** to specialize it for structured research analysis.
- Measuring measurable gains in both content quality and structural reliability.

---

## Technical Architecture

The project is divided into three distinct phases, each contained within its own module:

### 1. Dataset Generation Pipeline
A modular, automated system that fetches research papers and uses local LLMs to synthesize high-quality training pairs.
- **Source**: ArXiv API.
- **Engine**: Local Ollama instance.
- **Output**: 1,500+ instruction-aligned JSON samples.

### 2. Base Model Evaluation
A rigorous benchmarking of the un-tuned Mistral model on the specific task.
- **Technique**: Zero-shot prompting with Mistral-specific instruction tags.
- **Metrics**: ROUGE, BERTScore, and Schema Match Accuracy.
- **Kaggle Notebook**: [Baseline Evaluation](https://www.kaggle.com/code/hartzbyte/mistral7b-baseline-evaluation)


### 3. QLoRA Fine-Tuning
Parameter-efficient adaptation of the Mistral model to the custom dataset.
- **Method**: 4-bit Quantized Low-Rank Adaptation.
- **Hardware**: Optimized for consumer-grade GPUs (T4/P100).
- **Result**: A specialized adapter that achieves perfect schema compliance.
- **Kaggle Notebook**: [QLoRA Fine-Tuning](https://www.kaggle.com/code/hartzbyte/mistral7b-fine-tuning)


---

## Project Phases & Workflow

### Phase 1: Data Synthesis (`dataset-pipeline`)
The pipeline automates the ingestion, cleaning, and task assignment process. It uses a "Task Pool" strategy to randomly assign different analytical requirements to each paper, ensuring the model generalizes well across various research-related queries.

### Phase 2: Benchmarking (`Mistral-7B-Instruct-v0.2-Base-Evaluation`)
Before training, we establish a "ground truth" performance metric. This helps quantify exactly how much value the fine-tuning adds. We found that while the base model is semantically strong, it has a significant gap in structural reliability (88% exact schema match).

### Phase 3: Fine-Tuning (`Mistral7B-QLoRA-Fine-Tuning`)
Using the synthetic dataset, we fine-tune the model to learn the specific "flattened" response format. The QLoRA method allows us to train the model on 1,200+ samples in approximately 3 hours, significantly improving its task-specific performance.

---

## Key Results & Improvements

The fine-tuned model produced a clear performance lift across every metric, particularly in structural adherence.

| Metric | Base Model | Fine-Tuned Model | Gain |
| :--- | :--- | :--- | :--- |
| **ROUGE-1** | 0.5695 | 0.6630 | +0.0935 |
| **ROUGE-L** | 0.4385 | 0.5441 | +0.1056 |
| **BERTScore F1** | 0.9107 | 0.9307 | +0.0200 |
| **Schema Field Recall** | 94.28% | **100.00%** | +5.72% |
| **Exact Schema Match** | 88.00% | **100.00%** | +12.00% |

**Key Takeaway**: Fine-tuning converted a general-purpose model into a reliable extraction engine capable of perfect schema adherence on held-out data.

---

## Repository Structure

```text
.
├── dataset-pipeline/           # Data generation logic (ArXiv + Ollama)
├── Mistral-7B-Instruct-v0.2-Base-Evaluation/
│                               # Notebooks and baseline metrics
├── Mistral7B-QLoRA-Fine-Tuning/
│                               # Training configs, adapters, and results
└── README.md                   # Project overview (this file)
```

### Key Artifacts
- **Dataset**: `dataset-pipeline/data/final/dataset.json`
- **Baseline Report**: `Mistral-7B-Instruct-v0.2-Base-Evaluation/README.md`
- **Fine-Tuning Adapters**: `Mistral7B-QLoRA-Fine-Tuning/adapter/`
- **Performance Comparison**: `Mistral7B-QLoRA-Fine-Tuning/README.md`

---

## Technical Highlights
- **End-to-End Workflow**: Covers the full cycle from raw data to a specialized model.
- **Resource Efficient**: Uses QLoRA to make 7B model training accessible on minimal hardware.
- **Quantitative Driven**: Every decision is backed by metrics (ROUGE, BERTScore, Schema Match).
- **Practical Domain Adaptation**: Demonstrates how to adapt an LLM for scientific NLP tasks.

---

## Limitations & Future Work
- **Dataset Size**: The current dataset is relatively small (1.5k samples); scaling this could improve generalization.
- **Factuality**: While schema match is 100%, deeper human review for factual faithfulness is always recommended.
- **Future Work**: Plans to scale to more scientific domains, merge adapters for faster deployment, and compare against newer base models (Llama 3, Qwen 2, etc.).

---
*Built as part of a complete Structured LLM Adaptation Pipeline for Scientific Research Analysis.*
