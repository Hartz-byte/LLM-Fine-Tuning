import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

# List of reliable free models on OpenRouter
FREE_MODELS = [
    "google/gemma-3-4b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "openrouter/auto:free"
]

def openrouter_generate(text):
    last_error = "All free models exhausted"
    
    for model in FREE_MODELS:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Hartz-byte/LLM-Fine-Tuning",
                    "X-Title": "LLM Fine-Tuning Pipeline"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": f"""
                        Extract:
                        - Summary
                        - Methods
                        - Gaps

                        Text:
                        {text}
                        """}
                    ]
                },
                timeout=20  # Prevent hanging
            )
            
            # If rate-limited or other errors, try next model
            if response.status_code == 429:
                print(f"Model {model} rate-limited. Trying next...")
                continue
            
            if response.status_code != 200:
                print(f"Model {model} failed ({response.status_code}). Trying next...")
                continue

            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                print(f"Success with model: {model}")
                return data["choices"][0]["message"]["content"]
            else:
                continue

        except Exception as e:
            print(f"Error connecting to {model}: {e}")
            last_error = str(e)
            continue
    
    # If we get here, all models in the loop failed
    raise Exception(f"All OpenRouter models failed. Last error: {last_error}")
