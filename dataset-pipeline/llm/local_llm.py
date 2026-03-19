import requests
import json
from configs.config import OLLAMA_BASE_URL, OLLAMA_MODEL

def local_generate(prompt, expect_json=True):
    """Query local Ollama instance with optional JSON formatting."""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    if expect_json:
        payload["format"] = "json"

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60 # Increased timeout for larger extractions
        )
        
        result_text = response.json().get("response", "")
        
        if expect_json:
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # If LLM failed to return valid JSON despite the flag, return as is
                return result_text
        
        return result_text

    except Exception as e:
        print(f"Ollama Error: {e}")
        return {} if expect_json else ""
