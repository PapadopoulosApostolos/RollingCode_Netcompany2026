import os
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# CHROMADB CLIENT — Docker (HttpClient) or Local fallback
# ==========================================

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

# Local fallback path (used only if HttpClient fails)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma")


def _create_client():
    """
    Tries HttpClient first (Docker container).
    Falls back to PersistentClient for local development.
    """
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()  # Verify connection
        print(f"[ChromaDB] Connected to server at {CHROMA_HOST}:{CHROMA_PORT}")
        return client
    except Exception as e:
        print(f"[ChromaDB] Server not available ({e}). Falling back to local storage.")
        os.makedirs(DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=DB_PATH)
        print(f"[ChromaDB] Using PersistentClient at {DB_PATH}")
        return client


def get_chroma_collections():
    """
    Returns the two main collections: (historical_projects, domain_knowledge).
    Uses SentenceTransformer embeddings (computed client-side, works with both
    HttpClient and PersistentClient).
    """
    client = _create_client()

    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    projects_collection = client.get_or_create_collection(
        name="historical_projects",
        embedding_function=local_ef
    )

    experts_collection = client.get_or_create_collection(
        name="domain_knowledge",
        embedding_function=local_ef
    )

    return projects_collection, experts_collection