import json
import random
from tqdm import tqdm

from ingestion.arxiv_fetcher import fetch_papers
from processing.extractor import extract_basic_info
from processing.cleaner import clean_text
from processing.formatter import format_dataset

from llm.local_llm import local_summarize
from configs.config import MAX_PAPERS, SAVE_PATH

# Diverse instructions to make training more robust
INSTRUCTION_VARIANTS = [
    "Analyze research paper",
    "Summarize this research abstract",
    "Identify the methodology and research gaps in this paper",
    "Give me a technical breakdown of this paper including its methods and gaps",
    "Extract the core information from this research text",
    "Read this abstract and summarize it",
    "What are the key takeaways from this research?",
    "Analyze this paper's abstract and find its methodology",
    "Synthesize the summary, methods, and gaps for this research paper",
    "Break down this abstract for technical analysis"
]

# Simplified to Local Only
def smart_generate(text):
    return local_summarize(text)


def run_pipeline():
    papers = fetch_papers(max_results=MAX_PAPERS)
    dataset = []

    print(f"Starting pipeline... Progressive saving every 10 items. Using {len(INSTRUCTION_VARIANTS)} variants.")

    for i, paper in enumerate(tqdm(papers)):
        try:
            extracted = extract_basic_info(paper)
            abstract = clean_text(extracted["abstract"])

            # Integrated generation strategy
            structured_output = smart_generate(abstract)

            # Randomize instruction
            instruction = random.choice(INSTRUCTION_VARIANTS)

            # Format dataset
            data = format_dataset(
                instruction=instruction,
                input_text=abstract,
                output_text=structured_output
            )

            dataset.append(data)

            # Progressive saving every 10 items
            if (i + 1) % 10 == 0:
                with open(SAVE_PATH, "w") as f:
                    json.dump(dataset, f, indent=2)

        except Exception as e:
            print(f"Error on paper {i}: {e}")
            continue

    # Final save for all items
    with open(SAVE_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset complete. Total {len(dataset)} items saved to {SAVE_PATH}")


if __name__ == "__main__":
    run_pipeline()
