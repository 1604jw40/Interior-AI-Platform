import os


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
PERCEPTION_SCALE_UNIT = os.getenv("PERCEPTION_SCALE_UNIT", "m").strip().lower()
FIREBASE_STORAGE_BASE_URI = os.getenv(
    "FIREBASE_STORAGE_BASE_URI",
    "gs://interiorplatform-d58e0.firebasestorage.app",
).rstrip("/")
MAX_RAG_CANDIDATES = int(os.getenv("MAX_RAG_CANDIDATES", "5"))

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
