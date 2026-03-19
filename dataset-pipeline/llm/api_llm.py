import google.generativeai as genai
from configs.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

def generate_structured_output(text):
    prompt = f"""
    Given this research abstract, generate:
    1. Summary
    2. Key Methods
    3. Research Gaps
    4. Limitations

    Keep output structured and concise.

    Text:
    {text}
    """

    response = model.generate_content(prompt)
    return response.text
