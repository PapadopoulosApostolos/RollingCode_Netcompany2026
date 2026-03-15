import os
import shutil
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# PATHS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "rag":
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
else:
    PROJECT_ROOT = CURRENT_DIR

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "docs")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "data", "chroma")

# Docker ChromaDB connection
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))


def _create_client():
    """HttpClient (Docker) with PersistentClient fallback."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        print(f"[ChromaDB] Connected to server at {CHROMA_HOST}:{CHROMA_PORT}")
        return client, "http"
    except Exception:
        print(f"[ChromaDB] Server not available. Using local PersistentClient.")
        os.makedirs(CHROMA_DIR, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_DIR), "local"


def _clear_collections(client, mode):
    """Clears existing data — API delete for Docker, rmtree for local."""
    if mode == "http":
        for name in ["langchain", "historical_projects", "domain_knowledge"]:
            try:
                client.delete_collection(name)
                print(f"   Deleted collection '{name}'")
            except Exception:
                pass  # Collection doesn't exist yet
    else:
        if os.path.exists(CHROMA_DIR):
            print("Deleting old local database...")
            for attempt in range(3):
                try:
                    shutil.rmtree(CHROMA_DIR)
                    break
                except PermissionError:
                    print(f"   Retry... ({attempt+1}/3)")
                    time.sleep(2)
            os.makedirs(CHROMA_DIR, exist_ok=True)


def build_vector_db():
    print(f"Looking for PDFs in: {DATA_DIR}")

    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    if not documents:
        print("No documents found. Make sure there are PDFs in data/docs.")
        return

    print(f"Loaded {len(documents)} pages from PDFs.")

    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    client, mode = _create_client()
    _clear_collections(client, mode)

    print("Creating ChromaDB collections...")
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    col_default = client.get_or_create_collection(name="langchain", embedding_function=local_ef)
    col_projects = client.get_or_create_collection(name="historical_projects", embedding_function=local_ef)
    col_experts = client.get_or_create_collection(name="domain_knowledge", embedding_function=local_ef)

    documents_list = [chunk.page_content for chunk in chunks]
    metadatas_list = [chunk.metadata for chunk in chunks]
    ids_list = [f"doc_{i}" for i in range(len(chunks))]

    print("   -> 1/3: Populating 'langchain'...")
    col_default.add(documents=documents_list, metadatas=metadatas_list, ids=ids_list)

    print("   -> 2/3: Populating 'historical_projects'...")
    col_projects.add(documents=documents_list, metadatas=metadatas_list, ids=ids_list)

    print("   -> 3/3: Populating 'domain_knowledge'...")
    col_experts.add(documents=documents_list, metadatas=metadatas_list, ids=ids_list)

    print("Database built successfully.")


if __name__ == "__main__":
    build_vector_db()