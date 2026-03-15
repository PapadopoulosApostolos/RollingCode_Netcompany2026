import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.backend.graph.state import SystemDesignState

# ==========================================
# VECTOR STORE — Lazy-loaded module-level cache
# ==========================================
_experts_collection = None

def _get_experts_collection():
    """
    Φορτώνει το domain_knowledge collection ΜΙΑ φορά και το κρατάει στη μνήμη.
    Αποφεύγει re-init του PersistentClient σε κάθε query.
    """
    global _experts_collection
    if _experts_collection is None:
        try:
            from app.backend.vector_store.client import get_chroma_collections
            _, _experts_collection = get_chroma_collections()
            count = _experts_collection.count()
            print(f"   [RAG] domain_knowledge collection loaded ({count} documents)")
        except Exception as e:
            print(f"   [RAG] ChromaDB init failed: {e}")
            _experts_collection = None
    return _experts_collection


# ==========================================
# MOCK DATA — Fallback αν η βάση είναι άδεια ή αποτύχει
# ==========================================
MOCK_RAG_DATABASE = {
    "security": (
        "OWASP Top 10 Architecture Guidelines: Implement a Web Application Firewall (WAF) "
        "at the edge network. All Personally Identifiable Information (PII) must use "
        "AES-256 encryption at rest and TLS 1.3 in transit. Use Redis-backed Rate Limiting "
        "to mitigate DDoS and brute-force attacks. Authentication should rely on OAuth2/OIDC."
    ),
    "database": (
        "High-Concurrency Database Patterns: For highly relational and transactional (ACID) "
        "workloads, PostgreSQL with PgBouncer for connection pooling is mandatory. "
        "To absorb traffic spikes (e.g., 500+ concurrent users), implement a Redis caching layer "
        "for frequent read-queries (Cache-Aside pattern). Keep database isolation level to Read Committed."
    ),
    "ai_ml": (
        "AI Integration Best Practices: Decouple AI inference from the main request-response cycle. "
        "Use message brokers (RabbitMQ or Kafka) to queue heavy predictive tasks. "
        "For initial MVP, favor managed API-based AI models to reduce operational overhead, "
        "but design interfaces to allow swapping to local open-source models later."
    ),
    "deployment": (
        "Cloud Deployment Standards: Use container orchestration (Kubernetes or ECS) with "
        "multi-AZ deployments for high availability. Implement blue-green or canary deployment "
        "strategies. CI/CD via GitHub Actions or GitLab CI with automated testing gates."
    ),
    "data_engineering": (
        "Data Pipeline Architecture: Use event-driven ingestion with Kafka or AWS Kinesis. "
        "Store raw data in a data lake (S3/GCS) and transform via dbt or Apache Spark. "
        "Implement data quality checks at ingestion and transformation stages."
    ),
    "enterprise_architecture": (
        "Enterprise Integration Patterns: Use API Gateway for external consumers, "
        "service mesh (Istio/Linkerd) for internal communication. Implement circuit breakers "
        "and retry policies. Maintain a centralized service registry and config management."
    ),
}


# ==========================================
# RAG RETRIEVER — Real ChromaDB query + fallback
# ==========================================
def retrieve_domain_knowledge(query: str, domain: str) -> str:
    """
    Αναζητεί στο ChromaDB domain_knowledge collection.
    Αν δεν βρει αποτελέσματα ή αποτύχει, κάνει fallback στα mock data.
    
    Latency: ~30-80ms per query (local SentenceTransformer embeddings, no API call).
    """
    collection = _get_experts_collection()

    if collection is not None:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=3,
                where={"domain": domain} if domain else None,
            )

            docs = results.get("documents", [[]])[0]
            if docs and any(d.strip() for d in docs):
                combined = "\n".join(d.strip() for d in docs if d.strip())
                print(f"   [RAG] ChromaDB returned {len(docs)} docs for domain='{domain}'")
                return combined

            # Αν δεν βρέθηκε τίποτα ΜΕ φίλτρο domain, δοκίμασε χωρίς φίλτρο
            results_no_filter = collection.query(
                query_texts=[query],
                n_results=3,
            )
            docs_nf = results_no_filter.get("documents", [[]])[0]
            if docs_nf and any(d.strip() for d in docs_nf):
                combined = "\n".join(d.strip() for d in docs_nf if d.strip())
                print(f"   [RAG] ChromaDB returned {len(docs_nf)} docs (no domain filter)")
                return combined

        except Exception as e:
            print(f"   [RAG] ChromaDB query failed for domain='{domain}': {e}")

    # Fallback στα mock data
    fallback = MOCK_RAG_DATABASE.get(
        domain,
        "Implement standard cloud-native microservices patterns with high availability."
    )
    print(f"   [RAG] Fallback to mock data for domain='{domain}'")
    return fallback


# --- Βοηθητική συνάρτηση για το LLM (OpenAI) ---
def get_expert_llm():
    """Επιστρέφει το GPT-4o-mini για την Τεχνική Επιτροπή."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )


# ==========================================
# CONSOLIDATED NODE: THE TECHNICAL COMMITTEE
# ==========================================
def technical_committee_node(state: SystemDesignState):
    """
    Η Τεχνική Επιτροπή συνθέτει τις απόψεις 6 experts σε 1 API call.
    Χρησιμοποιεί real ChromaDB RAG με fallback σε mock data.
    """
    print("--- [NODE] Technical Committee: Ανάλυση (GPT-4o-mini + ChromaDB RAG) ---")
    requirements = state.get("requirements", {})
    req_str = json.dumps(requirements, ensure_ascii=False)

    # Κατασκευή queries από τα requirements
    core_func = requirements.get('core_functionality', 'core features')
    load = requirements.get('scalability_load', 'high traffic')
    security_needs = requirements.get('security_compliance', 'standard security')

    # RAG retrieval — 6 domains για τους 6 experts
    sec_ctx = retrieve_domain_knowledge(
        f"Security best practices for {core_func} with {security_needs}", "security"
    )
    db_ctx = retrieve_domain_knowledge(
        f"Database architecture for {load} with high concurrency", "database"
    )
    ai_ctx = retrieve_domain_knowledge(
        f"AI and ML integration patterns for {core_func}", "ai_ml"
    )
    deploy_ctx = retrieve_domain_knowledge(
        f"Cloud deployment strategy for {load}", "deployment"
    )
    data_ctx = retrieve_domain_knowledge(
        f"Data pipeline and engineering for {core_func}", "data_engineering"
    )
    enterprise_ctx = retrieve_domain_knowledge(
        f"Enterprise architecture patterns for {core_func}", "enterprise_architecture"
    )

    combined_practices = (
        f"[SECURITY CONTEXT]:\n{sec_ctx}\n\n"
        f"[DATABASE CONTEXT]:\n{db_ctx}\n\n"
        f"[AI/ML CONTEXT]:\n{ai_ctx}\n\n"
        f"[DEPLOYMENT CONTEXT]:\n{deploy_ctx}\n\n"
        f"[DATA ENGINEERING CONTEXT]:\n{data_ctx}\n\n"
        f"[ENTERPRISE ARCHITECTURE CONTEXT]:\n{enterprise_ctx}"
    )

    llm = get_expert_llm()
    system_prompt = """
    Είσαι η Τεχνική Επιτροπή Αρχιτεκτονικής. Παρέχεις ΜΟΝΟ ΤΕΧΝΙΚΑ ΔΕΔΟΜΕΝΑ.
    
    --- REQUIREMENTS ---
    {requirements}
    
    --- ΑΝΑΚΤΗΘΕΝΤΑ ΕΓΓΡΑΦΑ (RAG CONTEXT) ---
    Βασίσου ΣΕ ΑΥΤΑ για να δώσεις τις συμβουλές σου:
    {best_practices}
    
    ΑΠΑΙΤΗΣΕΙΣ ΜΕΓΙΣΤΗΣ ΤΑΧΥΤΗΤΑΣ (ΑΥΣΤΗΡΟ):
    1. ΑΠΑΓΟΡΕΥΕΤΑΙ Η ΦΛΥΑΡΙΑ και οι εισαγωγικοί χαιρετισμοί.
    2. Δώσε ΑΥΣΤΗΡΑ ΚΑΙ ΜΟΝΟ 6 bullet points (ένα για κάθε ρόλο).
    3. Κάθε bullet point πρέπει να είναι ΜΙΑ σύντομη πρόταση (5-12 λέξεις).
    
    Format: "- [Όνομα Ρόλου]: [Τεχνολογική επιλογή]"
    
    Ρόλοι:
    1. CyberSecurity Agent
    2. Database Agent
    3. AI Agent
    4. Deployment Agent
    5. Data Engineer
    6. Enterprise Architect
    """

    chain = ChatPromptTemplate.from_messages([("human", system_prompt)]) | llm
    result = chain.invoke({
        "requirements": req_str,
        "best_practices": combined_practices
    })

    return {"expert_opinions": [result.content]}