# LLM Dataset Generation Pipeline

This project is a modular, automated pipeline designed to generate high-quality, structured datasets for fine-tuning Large Language Models. It extracts information from scientific research papers and uses local LLMs to synthesize complex, instruction-aligned training data.

## Setup

1. Clone the repository to your local machine.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables by creating a `.env` file based on `example.env`:
   ```bash
   cp example.env .env
   ```
   Provide your local Ollama configuration and data paths.
5. Ensure Ollama is running locally with the desired model (e.g., mistral, llama3, or gemma) pulled.

## Architecture

The system follows a modular architecture to ensure flexibility and scalability:

- Ingestion: Responsible for fetching raw data from academic sources like ArXiv.
- Processing: Handles text cleaning, metadata extraction, and dataset formatting.
- LLM Integration: Manages communication with the local Ollama instance, supporting both standard text and structured JSON outputs.
- Pipeline Orchestration: Coordinates the flow from ingestion to final storage, featuring dynamic task selection and progressive data saving.
- Config: Centralized management of environment-based settings.

## Process

1. Ingestion: The system queries ArXiv for research papers related to a specific topic (e.g., Large Language Models).
2. Data Extraction: Basic metadata like titles and abstracts are extracted from the raw API response.
3. Pre-processing: Raw abstracts are cleaned by removing redundant whitespace and standardizing the text format.
4. Task Assignment: For each paper, a specialized task is randomly selected from a task pool (e.g., Detailed Breakdown, Critical Evaluation, Contradiction Detection).
5. Synthetic Generation: A local LLM processes the cleaned abstract based on the assigned task and a strict style guide.
6. Progressive Saving: Results are saved to a JSON file every few iterations to prevent data loss during long runs.
7. Final Formatting: A machine-friendly JSON dataset is compiled, ready for supervised fine-tuning.

## Extraction and Processing Details

The pipeline goes beyond simple summarization by extracting and synthesizing multiple technical facets:
- Summary: Concise and technical overviews of the research.
- Methodology: Extraction of specific algorithms, frameworks, and experimental setups.
- Research Gaps: Identification of limitations mentioned in the text or inferred through model reasoning.
- Critical Evaluation: Analytical assessment of the paper's claims and contributions.
- Contradiction Detection: Identifying internal inconsistencies or limitations within the study.
- Methodological Comparison: Comparing proposed techniques with traditional or state-of-the-art approaches in the field.

## Tools and AI Used

- Language: Python
- Data Sourcing: ArXiv API (via the `arxiv` library)
- Progress Monitoring: `tqdm`
- Environment Management: `python-dotenv`
- Local AI Engine: Ollama
- Models Supported: mistral, llama, gemma, qwen
- Output Format: JSON (enforced via prompt engineering and Ollama's structured output mode)

## Result and Format

The final product is a single `dataset.json` file. Each entry in the file is a JSON object with the following structure:

- instruction: A diverse, task-specific prompt (Instruction Shuffling).
- input: The cleaned research abstract.
- output: A structured JSON object containing task-specific keys (e.g., summary, methods, gaps) as defined by the instruction.

## Sample Result

```json
{
  "instruction": "Identify the key methodology and common research gaps in this paper.",
  "input": "Large language models (LLMs) have demonstrated impressive capabilities... [Abstract Text] ...",
  "output": {
    "methods": [
      "Mechanistic interpretability techniques",
      "Logit lens analysis",
      "Causal interventions",
      "Contrastive activation steering",
      "Behavioral steering vectors"
    ],
    "gaps": [
      "The phenomenon of lying by LLMs knowingly generating falsehoods to achieve an ulterior objective remains underexplored."
    ]
  }
}
```
