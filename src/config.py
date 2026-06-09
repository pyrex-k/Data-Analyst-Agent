import os

MODEL = "gemini-2.0-flash"
MAX_ITERATIONS = 20
OUTPUT_DIR = "outputs"

def get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return key
