import requests
from configs.config import OLLAMA_BASE_URL, OLLAMA_MODEL

def local_summarize(text):
    prompt = f"Summarize this in 3 bullet points:\n{text}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json().get("response", "")
