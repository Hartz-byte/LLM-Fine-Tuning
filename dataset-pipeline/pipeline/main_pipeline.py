import json
from tqdm import tqdm

from ingestion.arxiv_fetcher import fetch_papers
from processing.extractor import extract_basic_info
from processing.cleaner import clean_text
from processing.formatter import format_dataset

from llm.local_llm import local_summarize
from llm.api_llm import generate_structured_output

from configs.config import MAX_PAPERS, SAVE_PATH


def safe_generate(text):
    try:
        return generate_structured_output(text)
    except Exception as e:
        print("Gemini failed, fallback to local model:", e)
        return local_summarize(text)


def run_pipeline():
    papers = fetch_papers(max_results=MAX_PAPERS)
    dataset = []

    for paper in tqdm(papers):
        try:
            extracted = extract_basic_info(paper)
            
            abstract = clean_text(extracted["abstract"])

            # Step 1: Local LLM (cheap)
            summary = local_summarize(abstract)

            # Step 2: API LLM (high-quality)
            structured_output = safe_generate(abstract)

            # Format dataset
            data = format_dataset(
                instruction="Analyze research paper",
                input_text=abstract,
                output_text=structured_output
            )

            dataset.append(data)

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Save dataset
    with open(SAVE_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to {SAVE_PATH}")


if __name__ == "__main__":
    run_pipeline()
