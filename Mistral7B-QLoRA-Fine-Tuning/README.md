# QLoRA Fine-Tuning of Mistral-7B-Instruct-v0.2 for Structured Scientific Research Analysis

This repository contains the **parameter-efficient fine-tuning (QLoRA)** of **Mistral-7B-Instruct-v0.2** on a custom structured dataset for **scientific research abstract analysis**. The goal is to adapt the base instruction model to reliably generate schema-constrained outputs such as `summary`, `methods`, `gaps`, `evaluation`, `comparison`, and `contradictions` from research abstracts.

The project demonstrates a complete supervised fine-tuning workflow:
- Data formatting for chat-style supervised fine-tuning.
- 4-bit QLoRA adaptation on Mistral-7B-Instruct-v0.2.
- LoRA adapter checkpoint saving.
- Held-out test set evaluation.
- Quantitative comparison against the base model baseline.

---

## Project Objective

The objective of this fine-tuning stage is to improve the base model’s ability to:
- Follow **strict structured output schemas**.
- Remain **faithful to research abstracts**.
- Improve **field-level completeness**.
- Generate outputs that are **closer to gold references**.
- Reduce schema omissions seen in the base model.

This is a task-oriented fine-tuning setup for **instruction-following structured extraction and analysis** over scientific text.

---

## Task Definition

### Input
- Scientific research paper abstract.
- Instruction describing which fields to extract.

### Output
A structured textual response using only the requested fields, for example:
```text
summary:
- ...

methods:
- ...

gaps:
- ...
```

### Example Output Fields
Depending on the instruction, the model may be required to generate:
- `summary`
- `methods`
- `gaps`
- `evaluation`
- `comparison`
- `contradictions`

---

## Training Data

The fine-tuning dataset follows an Instruction-Input-Output schema:

| Field | Description |
| :--- | :--- |
| **instruction** | Natural language instruction describing the required analysis |
| **input** | Scientific research abstract |
| **output** | Structured JSON object containing task-specific fields |

### Dataset Split Used
- **Train**: 1193 usable samples
- **Validation**: 150 usable samples
- **Test**: 50 held-out samples

### Data Quality Handling
During formatting, the dataset was inspected for output type consistency. Initial inspection showed:
- `dict`: 1493
- `str`: 7

The formatting pipeline:
1. Accepted valid dictionary outputs.
2. Attempted JSON parsing for string outputs.
3. Skipped malformed plain-string outputs during SFT preparation.

**Final supervised fine-tuning data used:**
- **Train**: 1193
- **Validation**: 150

This ensured the model was trained only on structurally valid target examples.

---

## Supervised Fine-Tuning Format

Each sample was converted into a chat-style training example using the Mistral instruction template.

### Training Prompt Format
```text
<s> [INST] You are analyzing a scientific research abstract.

Task:
{instruction}

Return the answer using only these fields:
- {field_1}
- {field_2}
...

Keep the answer faithful to the abstract.
Do not invent unsupported claims.

Research Abstract:
{abstract} [/INST]
{assistant_response}</s>
```

### Target Formatting Strategy
The structured JSON output was flattened into a stable text format:
```text
summary:
- item 1
- item 2

methods:
- item 1
- item 2
```
This created a deterministic response style for supervised fine-tuning and improved schema consistency.

---

## Model and Fine-Tuning Setup

### Base Model
- **Model ID**: `mistralai/Mistral-7B-Instruct-v0.2`

### Fine-Tuning Method
- **Approach**: QLoRA
- **Quantization**: 4-bit NF4
- **PEFT Method**: LoRA adapters on attention and MLP projection layers

### Quantization Configuration
- 4-bit loading enabled
- `bnb_4bit_quant_type` = "nf4"
- `bnb_4bit_compute_dtype` = `torch.float16`
- Double quantization enabled

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Bias**: none
- **Task Type**: Causal Language Modeling

#### Target Modules
LoRA adapters were applied to:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

### Training Configuration
| Parameter | Value |
| :--- | :--- |
| Epochs | 2 |
| Learning Rate | 2e-4 |
| Per-device Train Batch Size | 1 |
| Per-device Eval Batch Size | 1 |
| Gradient Accumulation Steps | 8 |
| Max Sequence Length | 1024 |
| Optimizer | `paged_adamw_8bit` |
| Scheduler | `cosine` |
| Warmup Steps | 20 |
| FP16 | true |
| Seed | 42 |

### Training Runtime
- **Train Runtime**: 11,463.407 sec (~3h 10m)

---

## Training Results

### Final Trainer Output
| Epoch | Training Loss | Validation Loss |
| :--- | :--- | :--- |
| 1 | 1.252096 | 1.230647 |

### Interpretation
- Training completed successfully without catastrophic instability.
- Validation loss remained close to training loss, indicating healthy adaptation.
- The model learned the required response format and schema structure very effectively.

---

## Evaluation Methodology

After fine-tuning, the LoRA adapters were saved and reloaded on top of the base model for evaluation on the same held-out test set used in the baseline.

### Metrics Collected
- **ROUGE (1, 2, L, Lsum)**: Lexical overlap.
- **BERTScore (F1)**: Semantic similarity.
- **Schema Field Recall**: Fraction of requested fields actually generated.
- **Exact Schema Match**: Percentage of outputs where all requested fields were present.

---

## Fine-Tuned Evaluation Results

### Held-Out Test Set
- **Total test samples**: 50
- **Used**: 50
- **Skipped**: 0

### Fine-Tuned Metrics
| Metric | Score |
| :--- | :--- |
| **ROUGE-1** | 0.6630 |
| **ROUGE-2** | 0.4723 |
| **ROUGE-L** | 0.5441 |
| **ROUGE-Lsum** | 0.6132 |
| **BERTScore (F1)** | 0.9307 |
| **Schema Field Recall** | 1.0000 |
| **Exact Schema Match** | 1.0000 |

### Improvement Over the Base Model
The fine-tuned model produced a clear performance lift over the baseline.

| Metric | Base Model | Fine-Tuned Model | Gain |
| :--- | :--- | :--- | :--- |
| **ROUGE-1** | 0.5695 | 0.6630 | +0.0935 |
| **ROUGE-2** | 0.3577 | 0.4723 | +0.1146 |
| **ROUGE-L** | 0.4385 | 0.5441 | +0.1056 |
| **ROUGE-Lsum** | 0.5107 | 0.6132 | +0.1025 |
| **BERTScore F1** | 0.9107 | 0.9307 | +0.0200 |
| **Schema Field Recall** | 94.28% | 100.00% | +5.72% |
| **Exact Schema Match** | 88.00% | 100.00% | +12.00% |

---

## Key Takeaways
- The fine-tuned model is significantly better at schema adherence (100% vs 88%).
- It generates outputs that are closer to references both lexically and semantically.
- The model is more reliable for structured scientific abstract understanding tasks.

---

## Project Structure
- `adapter/`: Saved LoRA adapter weights and tokenizer files.
- `evaluation/finetuned_predictions.json`: Model predictions on the test set.
- `evaluation/finetuned_metrics.json`: Aggregated fine-tuned evaluation metrics.
- `evaluation/manual_review_samples.json`: Qualitative sample predictions.
- `train_result.json`: Training runtime, loss, and dataset stats.
- `trainer_output/`: HF Trainer checkpoints and logs.

---

## Technical Notes

### Why QLoRA?
QLoRA enabled efficient adaptation of a 7B model under constrained hardware by:
- Loading the base model in 4-bit.
- Training only low-rank adapter matrices.
- Reducing VRAM requirements while preserving strong performance.

### Limitations
- The dataset is relatively small for broad generalization.
- The test set contains only 50 samples.
- Some fields (like gaps) involve inference rather than strict extraction.

### Future Improvements
- Expand dataset size and task diversity.
- Add stricter hallucination controls for inferred fields.
- Evaluate on out-of-domain scientific abstracts.
- Compare against models like Llama, Qwen, or Gemma.

---

## Conclusion

This project successfully fine-tunes Mistral-7B-Instruct-v0.2 using QLoRA for structured scientific abstract analysis. The resulting model shows strong gains in both content quality and schema compliance, achieving perfect schema recall and match on the held-out test set.

---
*Built as part of a full LLM Fine-Tuning Pipeline*

