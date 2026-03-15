FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit config must be at WORKDIR/.streamlit/ for auto-detection.
# Project has it at app/.streamlit/config.toml — copy to correct path.
RUN cp -r app/.streamlit .streamlit 2>/dev/null || true

# Create data directory for SQLite (history.db)
RUN mkdir -p /app/data

EXPOSE 8501

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Seed ChromaDB knowledge base on startup (also downloads embedding model on first run),
# then launch Streamlit
CMD ["sh", "-c", "python app/backend/vector_store/seed_domain_knowledge.py && streamlit run app/frontend/ui.py --server.address=0.0.0.0 --server.port=8501"]
