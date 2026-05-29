import os
from pathlib import Path


_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else Path.cwd()


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
PERCEPTION_SCALE_UNIT = os.getenv("PERCEPTION_SCALE_UNIT", "m").strip().lower()
FIREBASE_STORAGE_BASE_URI = os.getenv(
    "FIREBASE_STORAGE_BASE_URI",
    "gs://interiorplatform-d58e0.firebasestorage.app",
).rstrip("/")
FIREBASE_STORAGE_PREFIX = os.getenv("FIREBASE_STORAGE_PREFIX", "furniture_images").strip().strip("/")
MAX_RAG_CANDIDATES = int(os.getenv("MAX_RAG_CANDIDATES", "5"))
PRODUCT_RAW_DATA_DIR = Path(
    os.getenv(
        "PRODUCT_RAW_DATA_DIR",
        str(REPO_ROOT / "infrastructure" / "product_db" / "data" / "raw"),
    )
)
