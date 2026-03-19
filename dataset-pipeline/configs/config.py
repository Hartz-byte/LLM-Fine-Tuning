import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ollama Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Pipeline Config
MAX_PAPERS = int(os.getenv("MAX_PAPERS", 50))
SAVE_PATH = os.getenv("DATA_PATH", "data/final/dataset.json")
