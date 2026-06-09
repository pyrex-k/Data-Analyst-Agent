import os

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 20
OUTPUT_DIR = "outputs"

def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
    return key
