import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

def openrouter_generate(text):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mistral-7b-instruct:free",
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
        }
    )

    return response.json()["choices"][0]["message"]["content"]
