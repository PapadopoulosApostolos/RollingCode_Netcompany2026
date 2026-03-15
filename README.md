
<h1 align="center">ArchitectAI</h1>

<p align="center">
  <strong>A multi-agent system that designs production-grade software architectures in under 60 seconds.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=flat&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/GPT--4o-Powered-412991?style=flat&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-RAG-FF6F00?style=flat" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white" />
</p>

---
 
# ArchitectAI — AI-Powered System Design Assistant

> **Netcompany Hackathon 2026** · Built by Team RollingCode

ArchitectAI is a multi-agent AI application that transforms a plain-text project description into a full enterprise-grade system architecture — complete with C4 Context Diagrams, Entity-Relationship Diagrams, a cloud cost estimate, security analysis, and a scalability roadmap. All in under a minute.

---

# The Problem

Designing a software architecture for a new project typically takes days to weeks — gathering requirements, consulting specialists, evaluating trade-offs, estimating costs, and producing documentation. Junior architects lack the breadth of experience; senior architects lack the time.

**ArchitectAI** compresses this entire process into a single interactive session. Describe your project in plain language, and a committee of six AI specialists collaborates to produce a complete architecture dossier — with diagrams, cost estimates, and a scaling roadmap.


| Output | Description |
|---|---|
| **C4 Context Diagram** | Visual system overview rendered with Mermaid.js |
| **ERD Diagram** | Full entity-relationship schema for your data model |
| **Tech Stack Summary** | Justified technology choices per layer |
| **Cost Estimate Table** | Line-item FinOps breakdown with monthly $ estimates |
| **Trade-off Analysis** | What the architecture gains and sacrifices |
| **Scaling Roadmap** | How to grow to 100× traffic |
| **Security Threat Model** | OWASP-aligned risks and mitigations |
| **Architecture Metrics** | Radar scores for Cost, Security, Performance, Scalability |

---

## How It Works — The Multi-Agent Pipeline

The application is built on **LangGraph**, a stateful graph framework for orchestrating AI agents. Each node in the graph is a specialized agent with a distinct role.

```
User Input
    │
    ▼
┌─────────────────────┐
│  Requirement Analyst│  ← Extracts structured requirements from free text + form data
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Initial Validator  │  ← Checks if input is complete; triggers Clarification Wizard if not
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     │            │
  [PASS]        [FAIL]
     │            │
     │     ┌──────▼───────┐
     │     │ Clarification│  ← Dynamic form wizard shown to user (stops graph until answered)
     │     │    Wizard    │
     │     └──────────────┘
     │            |
     ▼            ▼
┌─────────────────────┐
│    Memory (RAG)     │  ← Retrieves relevant context from ChromaDB vector store
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│  Technical Committee     │  ← 6 expert agents in 1 API call (Security, DB, AI, Deploy, Data, Enterprise)
│  (GPT-4o-mini + RAG)     │
└──────────┬───────────────┘
           │
           │
           │
           │ ┌─────────────────────┐
           │ │    Design critic    │  ← Critics the designs made by the system designer (GPT-4o)
           │ └──────┬────────┬─────┘
           │        │        │
           ▼        │        │
 ┌─────────────────────┐←────┘
 │   System Designer   │  ← Synthesizes all inputs into the final architecture dossier (GPT-4o)
 └──────────┬──────────┘
            │
            ▼
     Final Design
  (Diagrams + Reports)
```

### The Agents

**Requirement Analyst** — Uses GPT-4o-mini with structured output (Pydantic schema) to extract six core technical dimensions from the user's description: core functionality, scalability load, budget, security compliance, performance targets, and availability SLA.

**Initial Validator** — Rather than making freeform judgments, this agent selects from a curated pool of ~30 pre-defined question templates and presents only the most relevant ones as a dynamic wizard form. This prevents hallucinated form fields and ensures consistent UX.

**Memory Node (RAG)** — Queries a local ChromaDB vector store seeded with domain knowledge from OWASP and the AWS Well-Architected Framework. Uses `all-MiniLM-L6-v2` sentence embeddings for fast local retrieval (~30–80ms per query, no API call needed).

**Technical Committee** — Consolidates the perspectives of six domain experts into a single GPT-4o-mini call. Each expert is grounded by RAG-retrieved context from their domain: Security, Database, AI/ML, Deployment, Data Engineering, and Enterprise Architecture.

**System Designer** — The final synthesis step. Uses GPT-4o (full model) with a Pydantic-enforced output schema to generate the complete architecture dossier, including Mermaid diagram code, cost tables, and all analysis sections. Post-processing sanitizes Mermaid syntax to prevent rendering errors.

**Design Critic** — Runs parallel with the System Designer. Critics the designs made until they meet the requirements set by the
analyst and validator and meet a high enough standard.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Orchestration** | LangGraph |
| **LLMs** | OpenAI GPT-4o (designer) + GPT-4o-mini (analysts) |
| **Vector Store** | ChromaDB (local persistent) |
| **Embeddings** | `all-MiniLM-L6-v2` via SentenceTransformers |
| **Data Validation** | Pydantic v2 |
| **Charts** | Plotly |
| **Diagrams** | Mermaid.js |
| **Knowledge Base** | OWASP Top 10, AWS Well-Architected Framework (PDF ingestion) |

---

## Getting Started

### Prerequisites

- Python 3.14.3
- An OpenAI API key
- Docker Desktop

### 1. Clone & Install

```bash
git clone <repo-url>
cd <root-project-folder>
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="your_openai_api_key"
```

### 3. Build the Knowledge Base

Run the ingestion script once to process the PDF knowledge base and populate the ChromaDB vector store:

```bash
python ingest.py
```

This loads `data/docs/owasp.pdf` and `data/docs/AWS_Well-Architected_Framework.pdf`, splits them into chunks, embeds them locally, and persists them to `data/chroma/`.

### 4. Set up the Docker Containers

Run the following command on your terminal to set up the docker containers:

```bash
docker compose up --build -d
```
### 5. Launch the App

Through the Docker Desktop app, navigate to the Containers tab to see the newly created containers of the app. There you will see a container called ```app-1``` with a port number of ```8501:8501```. Click the port number. Alternatively follow the ```localhost``` URL provided by the terminal.

You will then be redirected to the front page of the app.

---

## Project Structure

```
netcompany-hackathon-2026/
├── app/
│   ├── artifacts/
│   │   └── (generated AI artifacts, exports, caches)
│   ├── backend/
│   │   ├── graph/
│   │   │   ├── nodes/
│   │   │   │   ├── analyst.py        # Requirement extraction agent
│   │   │   │   ├── validator.py      # Input validation + clarification wizard
│   │   │   │   ├── experts.py        # Technical committee (6 experts + RAG)
│   │   │   │   ├── designer.py       # Final architecture synthesis agent
│   │   │   │   ├── design_critic.py  # Post-design validation
│   │   │   │   └── memory.py         # RAG retrieval node
│   │   │   ├── __init__.py
│   │   │   ├── prompts.py            # Shared prompt templates
│   │   │   ├── state.py              # LangGraph shared state (TypedDict)
│   │   │   └── workflow.py           # Graph definition and routing logic
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py            # Pydantic schemas for structured outputs
│   │   └── vector_store/
│   │       ├── client.py             # ChromaDB client initialization
│   │       ├── history_db.py         # local history store logic
│   │       ├── operations.py         # RAG query helpers
│   │       ├── seed_domain_knowledge.py
│   │       └── db_data/
│   │           ├── chroma.sqlite3
│   │           └── (Chroma binary data directory)
│   └── frontend/
│       └── ui.py                     # Main Streamlit application
├── data/
│   ├── chroma/
│   │   ├── chroma.sqlite3
│   └── docs/                         # Ingested reference documents
├── ingest.py                         # Knowledge base ingestion pipeline
├── main.py                           # (likely app orchestration/entrypoint)
├── compose.yaml                      # Docker Compose config (app + chroma)
├── Dockerfile
├── README.md                         # Instructions and Overview
├── README.Docker.md                  
├── requirements.txt                  # Project Dependancies
├── .env                              # the file you will create with the API Key
└── .streamlit/
    └── config.toml                   # Streamlit UI theme/settings
```

---

## The Clarification Wizard

When the Initial Validator determines the user's description is missing critical information, the graph halts and presents a dynamic form — the **Clarification Wizard**.

Instead of asking freeform questions (which would produce inconsistent, hard-to-parse answers), the system uses a curated pool of ~30 pre-defined question templates across domains like Auth, Data Storage, Deployment, Real-Time, Payments, and Microservices.

The AI selects 6–8 of the most relevant templates for the specific project type (Web Application, Microservice Architecture, or Data/ML Pipeline). Once the user submits the form, the structured answers are injected back into the pipeline and the graph resumes — this time bypassing the validator to go straight to design.

---

## Project History

All generated architectures are persisted locally in SQLite Database. The sidebar displays your past projects with auto-generated titles derived from the input description. You can reload any previous design or delete entries directly from the UI.

---

## Configuration

**Adding knowledge to the RAG:** Place additional PDFs in `data/docs/` and re-run `python ingest.py`. The script will rebuild all three ChromaDB collections (`langchain`, `historical_projects`, `domain_knowledge`).

**Streamlit theme:** Configured in `app/.streamlit/config.toml`.

---

## Architecture Scoring

The final output includes a radar chart with four dimensions, scored 1–10 by the System Designer agent based on the actual architectural decisions made:

- **Cost Efficiency** — open-source vs. licensed, reserved vs. on-demand, right-sizing
- **Security Level** — encryption, WAF, DDoS protection, auth mechanism, compliance
- **Performance / Speed** — caching layers, CDN, DB indexing, async processing
- **Scalability** — horizontal scaling, statelessness, sharding readiness, queue decoupling

Scores are calibrated to the design — an enterprise-grade system should not score 5 across the board.

---

## Known Limitations

- Mermaid C4 diagrams use a locked "Trident Topology" template to prevent line intersections in the Dagre layout engine. Custom topologies may cause rendering issues.
- The ChromaDB vector store uses local SentenceTransformer embeddings. First-time startup may be slow while the model downloads.
- On Windows with OneDrive sync enabled, rebuilding the vector store may require pausing OneDrive to avoid file lock errors.

---

## Built at Netcompany Hackathon 2026

This project was created in a hackathon setting by Team RollingCode. The goal was to demonstrate how a LangGraph multi-agent pipeline, grounded in real domain knowledge via RAG, can produce structured, actionable engineering artifacts from a natural language description.

## Team

Built for the **Netcompany Hackathon 2026** by:

<!-- Add team member names/links here -->
- **Anastasios Papageorgiou**
- **Apostolos Papadopoulos**
- **Theodoros Katsanos**
- **Giannis Eleftherakos**
- **Dimos Katsarelis**
