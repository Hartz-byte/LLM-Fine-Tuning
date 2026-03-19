import json
import random
from tqdm import tqdm

from ingestion.arxiv_fetcher import fetch_papers
from processing.extractor import extract_basic_info
from processing.cleaner import clean_text
from processing.formatter import format_dataset

from llm.local_llm import local_generate
from configs.config import MAX_PAPERS, SAVE_PATH

# Pool of tasks with specific instructions and expected fields
TASK_POOL = [
    {
        "instruction": "Summarize this research abstract in a concise paragraph.",
        "fields": ["summary"],
        "style": "paragraph",
        "length": "short"
    },
    {
        "instruction": "Identify the key methodology and common research gaps in this paper.",
        "fields": ["methods", "gaps"],
        "style": "bullets",
        "length": "long"
    },
    {
        "instruction": "Extract a summary, methods, and gaps for this research paper.",
        "fields": ["summary", "methods", "gaps"],
        "style": "bullets",
        "length": "long"
    },
    {
        "instruction": "Find contradictions in this research and identify its methodology.",
        "fields": ["contradictions", "methods"],
        "style": "paragraph",
        "length": "medium"
    },
    {
        "instruction": "Critically evaluate this research and its proposed methodology.",
        "fields": ["evaluation", "methods"],
        "style": "paragraph",
        "length": "long"
    },
    {
        "instruction": "Compare this method with traditional approaches in the field.",
        "fields": ["comparison", "summary"],
        "style": "paragraph",
        "length": "medium"
    },
    {
        "instruction": "What are the research gaps and contradictions in this text?",
        "fields": ["gaps", "contradictions"],
        "style": "bullets",
        "length": "medium"
    },
    {
        "instruction": "Give me a technical breakdown of this paper.",
        "fields": ["summary", "methods", "gaps", "evaluation"],
        "style": "bullets",
        "length": "long"
    }
]

def build_prompt(abstract, task):
    """Dynamic prompt builder that ensures output matches instruction."""
    fields_str = ", ".join(task["fields"])
    length_desc = f"Use {task['length']} length for each field."
    style_desc = f"Write in {task['style']} format."
    
    prompt = f"""
    You are a research assistant. Analyze the following abstract:
    
    TEXT:
    {abstract}
    
    TASK:
    Extract the following fields ONLY: {fields_str}.
    
    Style Guide:
    - {style_desc}
    - {length_desc}
    
    Output MUST be a valid JSON object with EXACTLY these keys: {fields_str}.
    Each key's value should be a string or a list of strings if appropriate.
    """
    return prompt


def run_pipeline():
    papers = fetch_papers(max_results=MAX_PAPERS)
    dataset = []

    print(f"Starting pipeline... Generating dynamic tasks and structured outputs.")

    for i, paper in enumerate(tqdm(papers)):
        try:
            extracted = extract_basic_info(paper)
            abstract = clean_text(extracted["abstract"])

            # 1. Select a random task
            task = random.choice(TASK_POOL)

            # 2. Build and query local LLM
            prompt = build_prompt(abstract, task)
            structured_output = local_generate(prompt, expect_json=True)

            # 3. Format dataset entry
            data = format_dataset(
                instruction=task["instruction"],
                input_text=abstract,
                output_text=structured_output
            )

            dataset.append(data)

            # Progressive saving every 5 items (more frequent now as extraction is heavier)
            if (i + 1) % 5 == 0:
                with open(SAVE_PATH, "w") as f:
                    json.dump(dataset, f, indent=2)

        except Exception as e:
            print(f"Error on paper {i}: {e}")
            continue

    # Final save
    with open(SAVE_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset complete. Total {len(dataset)} items saved to {SAVE_PATH}")


if __name__ == "__main__":
    run_pipeline()
