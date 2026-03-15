"""
Seed Script: Populates the ChromaDB 'domain_knowledge' collection
with enterprise-grade technical documents for each expert domain.

Usage:
    python -m app.backend.vector_store.seed_domain_knowledge

    Or from project root:
    python app/backend/vector_store/seed_domain_knowledge.py
"""
import os
import sys

# Fix path so imports work from any location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.backend.vector_store.client import get_chroma_collections

# ==========================================
# DOMAIN KNOWLEDGE DOCUMENTS
# ==========================================
# Κάθε document είναι ένα αυτόνομο τεχνικό κείμενο ετικεταρισμένο
# με domain metadata για filtered retrieval.

DOMAIN_DOCUMENTS = [
    # ──────────────────────────────────────
    # SECURITY (6 documents)
    # ──────────────────────────────────────
    {
        "id": "sec_owasp_top10",
        "domain": "security",
        "source": "OWASP Foundation",
        "text": (
            "OWASP Top 10 Architectural Mitigations: "
            "A01 Broken Access Control — enforce deny-by-default, implement RBAC at API gateway level. "
            "A02 Cryptographic Failures — use AES-256-GCM for data at rest, TLS 1.3 for transit, rotate keys via KMS. "
            "A03 Injection — use parameterized queries exclusively, validate all inputs with allowlist schemas. "
            "A07 Authentication Failures — implement MFA, use bcrypt/argon2 for password hashing, enforce session timeout."
        ),
    },
    {
        "id": "sec_zero_trust",
        "domain": "security",
        "source": "NIST SP 800-207",
        "text": (
            "Zero Trust Architecture Principles: Never trust, always verify. "
            "Authenticate every request regardless of network location. "
            "Use micro-segmentation to isolate services. Implement mutual TLS (mTLS) between microservices. "
            "Apply least-privilege access with short-lived tokens (JWT with 15-minute expiry). "
            "Deploy a centralized Policy Decision Point (PDP) for authorization decisions."
        ),
    },
    {
        "id": "sec_api_protection",
        "domain": "security",
        "source": "API Security Best Practices",
        "text": (
            "API Security Layer Design: Deploy a Web Application Firewall (WAF) at the edge (AWS WAF or Cloudflare). "
            "Implement rate limiting per API key (Redis-backed sliding window, 100 req/min default). "
            "Use OAuth 2.0 with PKCE for public clients, client_credentials for service-to-service. "
            "Log all API access to a SIEM (Splunk/ELK) with correlation IDs for audit trails. "
            "Validate request payloads against OpenAPI schemas before processing."
        ),
    },
    {
        "id": "sec_gdpr_compliance",
        "domain": "security",
        "source": "GDPR Technical Requirements",
        "text": (
            "GDPR-Compliant Architecture: Implement data classification (PII vs non-PII) at ingestion. "
            "Store PII in EU-region-only databases with encryption at rest. "
            "Build a consent management service with versioned consent records. "
            "Implement right-to-erasure via soft-delete with 30-day hard-delete cron jobs. "
            "Maintain data processing audit logs with immutable append-only storage."
        ),
    },
    {
        "id": "sec_container_security",
        "domain": "security",
        "source": "CIS Kubernetes Benchmark",
        "text": (
            "Container Security Hardening: Use minimal base images (distroless or Alpine). "
            "Scan images in CI/CD with Trivy or Snyk before deployment. "
            "Run containers as non-root with read-only filesystems. "
            "Implement Pod Security Standards (restricted profile) in Kubernetes. "
            "Use network policies to restrict pod-to-pod communication to explicit allowlists."
        ),
    },
    {
        "id": "sec_secrets_management",
        "domain": "security",
        "source": "HashiCorp Vault Patterns",
        "text": (
            "Secrets Management Architecture: Never store secrets in code, environment variables, or config files. "
            "Use a dedicated vault (HashiCorp Vault, AWS Secrets Manager). "
            "Implement dynamic secrets with automatic rotation (database credentials every 24h). "
            "Use envelope encryption: vault encrypts a data-encryption-key, application uses DEK for data. "
            "Audit all secret access with centralized logging."
        ),
    },

    # ──────────────────────────────────────
    # DATABASE (6 documents)
    # ──────────────────────────────────────
    {
        "id": "db_postgresql_patterns",
        "domain": "database",
        "source": "PostgreSQL Architecture Guide",
        "text": (
            "PostgreSQL Production Patterns: Use PgBouncer for connection pooling (transaction mode, 20-50 pool size). "
            "Implement read replicas for read-heavy workloads (80/20 read/write ratio). "
            "Use JSONB columns for semi-structured data instead of a separate NoSQL store. "
            "Partition large tables by date range (monthly partitions for time-series data). "
            "Set isolation level to Read Committed for OLTP, Repeatable Read for financial transactions."
        ),
    },
    {
        "id": "db_caching_strategies",
        "domain": "database",
        "source": "Caching Architecture Patterns",
        "text": (
            "Multi-Layer Caching Strategy: L1 — application-level in-memory cache (LRU, 1000 entries, 60s TTL). "
            "L2 — Redis cluster for shared cache across instances (Cache-Aside pattern). "
            "L3 — CDN edge cache for static assets and API responses (Cloudflare/CloudFront). "
            "Implement cache invalidation via event-driven pub/sub on data mutations. "
            "Use cache stampede protection with probabilistic early expiration or distributed locks."
        ),
    },
    {
        "id": "db_nosql_when",
        "domain": "database",
        "source": "Database Selection Decision Matrix",
        "text": (
            "Database Selection Criteria: Use PostgreSQL for transactional workloads with complex joins and ACID requirements. "
            "Use MongoDB for document-centric data with variable schemas and rapid prototyping. "
            "Use Redis for session storage, real-time leaderboards, and rate limiting counters. "
            "Use Elasticsearch for full-text search, log aggregation, and analytics dashboards. "
            "Use TimescaleDB or InfluxDB for time-series data (IoT sensors, metrics, logs)."
        ),
    },
    {
        "id": "db_migration_strategy",
        "domain": "database",
        "source": "Database Migration Patterns",
        "text": (
            "Schema Migration Best Practices: Use versioned migrations (Flyway, Alembic, or Prisma Migrate). "
            "Always make migrations backward-compatible — add columns as nullable, never rename in-place. "
            "Use expand-contract pattern for breaking changes: add new column, dual-write, backfill, switch reads, drop old. "
            "Test migrations against production-size datasets before deployment. "
            "Implement rollback scripts for every migration."
        ),
    },
    {
        "id": "db_sharding",
        "domain": "database",
        "source": "Horizontal Scaling Patterns",
        "text": (
            "Database Sharding Strategies: Use hash-based sharding on tenant_id for multi-tenant SaaS applications. "
            "Range-based sharding on created_at for time-series and log data. "
            "Implement a shard routing layer (application-level or proxy like Vitess/Citus). "
            "Design shard keys to minimize cross-shard queries. "
            "Keep a global metadata table mapping shard keys to physical database instances."
        ),
    },
    {
        "id": "db_backup_dr",
        "domain": "database",
        "source": "Disaster Recovery Standards",
        "text": (
            "Database Backup and DR: Implement automated daily full backups with point-in-time recovery (PITR). "
            "Store backups in a different region with cross-region replication. "
            "Target RPO < 1 hour for production databases (WAL archiving for PostgreSQL). "
            "Target RTO < 30 minutes with standby replicas (synchronous replication for critical data). "
            "Test disaster recovery procedures quarterly with documented runbooks."
        ),
    },

    # ──────────────────────────────────────
    # AI / ML (5 documents)
    # ──────────────────────────────────────
    {
        "id": "ai_integration_patterns",
        "domain": "ai_ml",
        "source": "AI Architecture Patterns",
        "text": (
            "AI Integration Architecture: Decouple AI inference from the request-response cycle using async queues. "
            "Use a model registry (MLflow, Weights & Biases) for versioned model deployment. "
            "Implement A/B testing infrastructure to compare model versions in production. "
            "Design a feature store (Feast or Tecton) for consistent feature serving between training and inference. "
            "Monitor model performance with drift detection (data drift and concept drift alerts)."
        ),
    },
    {
        "id": "ai_llm_patterns",
        "domain": "ai_ml",
        "source": "LLM Integration Guide",
        "text": (
            "LLM Application Patterns: Use managed APIs (OpenAI, Anthropic) for MVP to reduce operational overhead. "
            "Implement prompt templates with versioning and A/B testing. "
            "Add guardrails: input validation, output filtering, token budget limits. "
            "Use RAG (Retrieval-Augmented Generation) with vector databases for domain-specific knowledge. "
            "Design fallback chains: primary model -> secondary model -> cached response -> graceful error."
        ),
    },
    {
        "id": "ai_mlops",
        "domain": "ai_ml",
        "source": "MLOps Maturity Model",
        "text": (
            "MLOps Pipeline Design: Level 0 — manual training and deployment (acceptable for MVP). "
            "Level 1 — automated training pipeline with scheduled retraining (weekly/monthly). "
            "Level 2 — full CI/CD for ML with automated testing, validation, and canary deployment. "
            "Use containerized training environments for reproducibility. "
            "Implement data versioning (DVC or LakeFS) alongside code versioning."
        ),
    },
    {
        "id": "ai_cost_optimization",
        "domain": "ai_ml",
        "source": "AI Cost Management",
        "text": (
            "AI Infrastructure Cost Control: Use spot/preemptible instances for training workloads (60-90% savings). "
            "Implement request batching for inference APIs to maximize throughput per dollar. "
            "Cache frequent LLM responses with semantic similarity matching (Redis + embeddings). "
            "Use model distillation or quantization (INT8/INT4) for production inference. "
            "Set per-user and per-request token budgets with hard limits."
        ),
    },
    {
        "id": "ai_vector_db",
        "domain": "ai_ml",
        "source": "Vector Database Selection",
        "text": (
            "Vector Database Architecture: Use ChromaDB or SQLite-VSS for prototyping and small-scale (<100K vectors). "
            "Use Pinecone or Weaviate for managed production deployments with high availability. "
            "Use pgvector extension for PostgreSQL when you want to avoid a separate database. "
            "Implement hybrid search: combine vector similarity with keyword BM25 for better recall. "
            "Chunk documents at 200-500 tokens with 50-token overlap for optimal retrieval."
        ),
    },

    # ──────────────────────────────────────
    # DEPLOYMENT (5 documents)
    # ──────────────────────────────────────
    {
        "id": "deploy_kubernetes",
        "domain": "deployment",
        "source": "Kubernetes Production Guide",
        "text": (
            "Kubernetes Production Deployment: Use managed Kubernetes (EKS, GKE, AKS) to reduce operational burden. "
            "Implement Horizontal Pod Autoscaler (HPA) based on CPU/memory and custom metrics. "
            "Use multi-AZ node groups for high availability (minimum 3 nodes across 3 AZs). "
            "Implement resource requests and limits on all pods to prevent noisy-neighbor issues. "
            "Use Helm charts or Kustomize for templated, reproducible deployments."
        ),
    },
    {
        "id": "deploy_cicd",
        "domain": "deployment",
        "source": "CI/CD Pipeline Standards",
        "text": (
            "CI/CD Pipeline Architecture: Use trunk-based development with short-lived feature branches. "
            "Pipeline stages: lint -> unit test -> build -> integration test -> security scan -> deploy staging -> deploy prod. "
            "Implement automated rollback on failed health checks (readiness/liveness probes). "
            "Use blue-green deployments for zero-downtime releases, canary for high-risk changes. "
            "Gate production deploys on test coverage thresholds (minimum 80% line coverage)."
        ),
    },
    {
        "id": "deploy_observability",
        "domain": "deployment",
        "source": "Observability Stack Guide",
        "text": (
            "Observability Architecture (Three Pillars): "
            "Metrics — Prometheus + Grafana for system and application metrics with alerting rules. "
            "Logs — structured JSON logging with correlation IDs, shipped to ELK or Loki. "
            "Traces — OpenTelemetry instrumentation with Jaeger or Tempo for distributed tracing. "
            "Implement SLO-based alerting (error budget burn rate) instead of threshold-based alerts. "
            "Create runbooks for every alert with clear escalation procedures."
        ),
    },
    {
        "id": "deploy_serverless",
        "domain": "deployment",
        "source": "Serverless Architecture Patterns",
        "text": (
            "Serverless Decision Framework: Use serverless (Lambda/Cloud Functions) for event-driven, bursty workloads. "
            "Avoid serverless for latency-sensitive or long-running processes (>15 min). "
            "Combine serverless with containers: API Gateway + Lambda for lightweight endpoints, "
            "ECS/Fargate for heavy processing. "
            "Watch for cold start latency — use provisioned concurrency for critical paths. "
            "Implement structured logging as serverless functions lack traditional APM."
        ),
    },
    {
        "id": "deploy_infrastructure_as_code",
        "domain": "deployment",
        "source": "IaC Best Practices",
        "text": (
            "Infrastructure as Code Standards: Use Terraform for multi-cloud infrastructure provisioning. "
            "Organize Terraform code in modules: networking, compute, database, security. "
            "Use remote state backend (S3 + DynamoDB locking) for team collaboration. "
            "Implement drift detection with scheduled plan-only runs. "
            "Use Terragrunt or Terraform workspaces for environment separation (dev/staging/prod)."
        ),
    },

    # ──────────────────────────────────────
    # DATA ENGINEERING (4 documents)
    # ──────────────────────────────────────
    {
        "id": "data_pipeline_patterns",
        "domain": "data_engineering",
        "source": "Data Pipeline Architecture",
        "text": (
            "Data Pipeline Design: Use event-driven architecture with Kafka or AWS Kinesis for real-time ingestion. "
            "Implement ELT over ETL — load raw data into a data lake first, transform in-place with dbt. "
            "Use Apache Airflow or Dagster for orchestrating batch pipelines with DAG-based scheduling. "
            "Implement data quality checks (Great Expectations or dbt tests) at every pipeline stage. "
            "Design idempotent pipelines — re-running any stage should produce the same result."
        ),
    },
    {
        "id": "data_lake_architecture",
        "domain": "data_engineering",
        "source": "Data Lake Design Guide",
        "text": (
            "Data Lake Architecture: Organize storage in medallion layers — Bronze (raw), Silver (cleaned), Gold (aggregated). "
            "Use Parquet or Delta Lake format for columnar storage with schema evolution support. "
            "Implement partition pruning on date columns for efficient query execution. "
            "Use a data catalog (AWS Glue Catalog, Apache Hive Metastore) for schema discovery. "
            "Apply column-level encryption for sensitive fields within the data lake."
        ),
    },
    {
        "id": "data_streaming",
        "domain": "data_engineering",
        "source": "Real-Time Processing Patterns",
        "text": (
            "Real-Time Data Processing: Use Kafka Streams or Apache Flink for stream processing with exactly-once semantics. "
            "Implement windowed aggregations (tumbling, sliding, session windows) for real-time analytics. "
            "Use a dead-letter queue for messages that fail processing after retries. "
            "Design for late-arriving data with watermarks and allowed lateness thresholds. "
            "Implement backpressure mechanisms to handle traffic spikes without data loss."
        ),
    },
    {
        "id": "data_governance",
        "domain": "data_engineering",
        "source": "Data Governance Framework",
        "text": (
            "Data Governance Architecture: Implement data lineage tracking (Apache Atlas or OpenLineage). "
            "Define data ownership per domain with a data mesh approach for large organizations. "
            "Use schema registries (Confluent Schema Registry) for event schema evolution. "
            "Implement data retention policies with automated lifecycle management. "
            "Create a data access control layer with row-level and column-level security."
        ),
    },

    # ──────────────────────────────────────
    # ENTERPRISE ARCHITECTURE (4 documents)
    # ──────────────────────────────────────
    {
        "id": "ea_api_gateway",
        "domain": "enterprise_architecture",
        "source": "API Gateway Patterns",
        "text": (
            "API Gateway Architecture: Use Kong, AWS API Gateway, or Envoy as the single entry point. "
            "Implement request routing, rate limiting, and authentication at the gateway level. "
            "Use the Backend-for-Frontend (BFF) pattern — separate gateways for web, mobile, and third-party consumers. "
            "Implement request/response transformation at the gateway to decouple internal APIs from external contracts. "
            "Enable circuit breakers at the gateway to protect downstream services from cascading failures."
        ),
    },
    {
        "id": "ea_microservices",
        "domain": "enterprise_architecture",
        "source": "Microservices Architecture Guide",
        "text": (
            "Microservices Decomposition: Split by business domain (DDD bounded contexts), not by technical layer. "
            "Each service owns its database (database-per-service pattern). "
            "Use asynchronous communication (events) as default, synchronous (REST/gRPC) only when necessary. "
            "Implement the Saga pattern for distributed transactions (choreography for simple flows, orchestration for complex). "
            "Design for failure: every service call must have timeout, retry with backoff, and circuit breaker."
        ),
    },
    {
        "id": "ea_event_driven",
        "domain": "enterprise_architecture",
        "source": "Event-Driven Architecture",
        "text": (
            "Event-Driven Architecture Patterns: Use domain events for inter-service communication. "
            "Implement event sourcing for audit-critical domains (financial transactions, compliance). "
            "Use CQRS to separate read and write models for performance optimization. "
            "Design events as immutable facts with schema versioning (backward compatible). "
            "Implement an outbox pattern to ensure reliable event publishing with database transactions."
        ),
    },
    {
        "id": "ea_multi_tenancy",
        "domain": "enterprise_architecture",
        "source": "Multi-Tenancy Architecture",
        "text": (
            "Multi-Tenant Architecture Strategies: "
            "Silo model — separate infrastructure per tenant (highest isolation, highest cost). "
            "Pool model — shared infrastructure with tenant_id filtering (lowest cost, requires careful security). "
            "Bridge model — shared compute, separate databases per tenant (balanced approach). "
            "Implement tenant-aware middleware that injects tenant context into every request. "
            "Use row-level security (RLS) in PostgreSQL for pool-model data isolation."
        ),
    },
]


# ==========================================
# SEED FUNCTION
# ==========================================
def seed_domain_knowledge(force_reseed: bool = False):
    """
    Γεμίζει το domain_knowledge collection με τεχνικά documents.
    
    Args:
        force_reseed: Αν True, διαγράφει τα υπάρχοντα και ξανα-γεμίζει.
    """
    _, experts_collection = get_chroma_collections()

    existing_count = experts_collection.count()
    print(f"[SEED] domain_knowledge collection has {existing_count} documents.")

    if existing_count > 0 and not force_reseed:
        print("[SEED] Collection already populated. Use --force to reseed.")
        return

    if existing_count > 0 and force_reseed:
        # Διαγραφή υπαρχόντων
        existing_ids = experts_collection.get()["ids"]
        if existing_ids:
            experts_collection.delete(ids=existing_ids)
            print(f"[SEED] Deleted {len(existing_ids)} existing documents.")

    # Εισαγωγή documents
    ids = []
    documents = []
    metadatas = []

    for doc in DOMAIN_DOCUMENTS:
        ids.append(doc["id"])
        documents.append(doc["text"])
        metadatas.append({
            "domain": doc["domain"],
            "source": doc["source"],
        })

    experts_collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[SEED] Successfully added {len(ids)} documents across domains:")
    domain_counts = {}
    for doc in DOMAIN_DOCUMENTS:
        d = doc["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"       {domain}: {count} documents")


# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    force = "--force" in sys.argv
    seed_domain_knowledge(force_reseed=force)