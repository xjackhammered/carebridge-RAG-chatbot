"""All settings live here, read from environment variables (or a .env file).

Why: models get deprecated (your Groq models did!), thresholds get tuned, paths
differ between your laptop and the VPS. If these are hardcoded across files, every
change is a code edit. In one place + env vars, it's a config change.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _csv(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


# --- Embeddings & vector store ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./data/docs")

# --- Chunking ---
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))
# If the best chunk scores below this, we refuse to answer instead of guessing.
# Don't guess this number: run `python -m eval.evaluate` and it will suggest one.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))

# --- LLM (Groq). Comma-separated, tried in order. Check Groq's deprecations page! ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODELS = _csv("LLM_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

# --- Sessions ---
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))  # messages kept per session

# --- Abuse protection (per client IP, /chat only) ---
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
