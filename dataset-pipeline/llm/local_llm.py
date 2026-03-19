import requests
from configs.config import OLLAMA_BASE_URL, OLLAMA_MODEL

def local_summarize(text):
    prompt = f"""
    Analyze this research abstract and extract:
    1. Summary
    2. Key Methods
    3. Research Gaps

    Text:
    {text}
    """

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json().get("response", "")
