from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ==========================================
# 0. ΜΟΝΤΕΛΟ ΔΟΜΗΜΕΝΗΣ ΦΟΡΜΑΣ (WIZARD FIELDS)
# ==========================================
class ClarificationField(BaseModel):
    """
    Αναπαριστά ένα πεδίο εισαγωγής στη φόρμα διευκρινίσεων.
    Ο τύπος (field_type) ορίζεται ΑΠΟΚΛΕΙΣΤΙΚΑ από τα templates,
    ΠΟΤΕ από το LLM — εξαλείφει 100% τα form-type bugs.
    """
    id: str = Field(..., description="Unique ID (π.χ. 'auth_method')")
    label: str = Field(..., description="Η ερώτηση στα Ελληνικά")
    field_type: Literal["checkbox", "select", "multi_select"] = Field(
        ..., 
        description=(
            "checkbox = Ναι/Όχι toggle. "
            "select = Dropdown μιας επιλογής. "
            "multi_select = Dropdown πολλαπλών επιλογών."
        )
    )
    options: List[str] = Field(
        default_factory=list,
        description="Επιλογές — κενό αν checkbox."
    )
    required: bool = Field(True, description="Υποχρεωτικό;")
    default_value: Optional[str] = Field(None, description="Προεπιλογή αν υπάρχει")


class ClarificationWizard(BaseModel):
    """Φόρμα χωρισμένη σε κατηγορίες προτεραιότητας."""
    essential: List[ClarificationField] = Field(
        default_factory=list, 
        description="Κρίσιμα πεδία — χωρίς αυτά δεν σχεδιάζεται η αρχιτεκτονική."
    )
    recommended: List[ClarificationField] = Field(
        default_factory=list, 
        description="Σημαντικά πεδία — βελτιστοποιούν trade-offs και κόστος."
    )
    optional: List[ClarificationField] = Field(
        default_factory=list, 
        description="Nice-to-have."
    )

# ==========================================
# 1. Radar Chart Metrics
# ==========================================
class ArchitectureMetrics(BaseModel):
    cost_efficiency: int = Field(
        ..., ge=1, le=10, 
        description="Score 1-10. Factor in: open-source vs licensed, reserved vs on-demand, right-sizing. 10 = very cheap."
    )
    security_level: int = Field(
        ..., ge=1, le=10, 
        description="Score 1-10. Factor in: encryption at rest/transit, WAF, DDoS, auth mechanism, compliance. 10 = military-grade."
    )
    performance_speed: int = Field(
        ..., ge=1, le=10, 
        description="Score 1-10. Factor in: caching layers, CDN, DB indexing, async processing, connection pooling. 10 = real-time."
    )
    scalability: int = Field(
        ..., ge=1, le=10, 
        description="Score 1-10. Factor in: horizontal scaling, statelessness, sharding readiness, queue decoupling. 10 = planet-scale."
    )

# ==========================================
# 2. Architecture Dossier
# ==========================================
class FinalArchitectureDesign(BaseModel):
    mermaid_c4_code: str = Field(
        ..., 
        description=(
            "COMPLETE Mermaid C4 Context diagram. Raw code, NO markdown fences. "
            "Line 1 MUST be 'C4Context' alone. "
            "ALLOWED elements ONLY: Person(), Person_Ext(), System(), SystemDb(), System_Ext(), Rel(). "
            "FORBIDDEN: System_Boundary, Container_Boundary, Enterprise_Boundary, curly braces. "
            "MINIMUM COMPLEXITY: at least 2 Person/Person_Ext, at least 5 System/SystemDb/System_Ext, "
            "at least 10 Rel() connections with descriptive English labels explaining data flow. "
            "Every node MUST have at least 1 Rel(). "
            "Use ONLY English ASCII. No commas inside string arguments — use 'and' or '/'."
        )
    )
    mermaid_erd_code: str = Field(
        ..., 
        description=(
            "COMPLETE Mermaid ERD diagram. Raw code, NO markdown fences. "
            "Line 1 MUST be 'erDiagram' alone. "
            "MINIMUM: 6 entities, each with 3-8 typed attributes using: uuid, string, int, float, boolean, datetime, text, enum. "
            "Mark PK and FK on relevant attributes. Include created_at/updated_at on main entities. "
            "MINIMUM: 7 relationships using ONLY valid Mermaid cardinality symbols: "
            "||--|| (one-to-one), ||--o{ (one-to-zero-or-many), ||--|{ (one-to-one-or-many). "
            "ALL relationship labels MUST be in double quotes: ENTITY_A ||--o{ ENTITY_B : \"label here\". "
            "Use ONLY English ASCII. No commas inside labels."
        )
    )
    tech_stack_summary: str = Field(
        ..., 
        description=(
            "5-8 bullet points in Greek. Each explains ONE technology and WHY it was chosen. "
            "Format each as: '- **TechName** — technical justification'. "
            "Cover at minimum: language/framework, primary DB, caching, cloud/hosting, CI/CD, monitoring."
        )
    )
    trade_off_analysis: str = Field(
        ..., 
        description=(
            "3-5 real architectural trade-offs in Greek. "
            "Each must state the dilemma, what we gain, and what we sacrifice."
        )
    )
    future_scaling_path: str = Field(
        ..., 
        description=(
            "Technical roadmap in Greek for scaling to 100x current load. "
            "Describe: DB sharding/read-replicas, CQRS, cache tiers, CDN, "
            "microservice extraction, async job queues, auto-scaling policies."
        )
    )
    implementation_roadmap: str = Field(
        default="N/A", 
        description="Always return 'N/A'."
    )
    cost_estimate_table: str = Field(
        ..., 
        description=(
            "Detailed Markdown table: "
            "| Resource Category | Service and Tier | Scaling Unit | Est. Monthly Cost ($) | Rationale |. "
            "Include rows for: Compute, Database, Caching, Object Storage, CDN/Bandwidth, "
            "Message Queue (if needed), Search (if needed), Monitoring/Logging, Backup/DR, Security (WAF/certs), CI/CD. "
            "Last row MUST be: | **TOTAL** | — | — | **$X** | — |. "
            "Use realistic AWS/Azure/GCP pricing for the chosen tier."
        )
    )
    metrics: ArchitectureMetrics = Field(
        ...,
        description="Realistic scores based on actual design choices. Do not default to 5 — justify each score."
    )

# ==========================================
# 3. Requirements Schema
# ==========================================
class RequirementsSchema(BaseModel):
    core_functionality: str = Field(
        ..., description="Περιγραφή επιχειρηματικού στόχου και βασικών λειτουργιών."
    )
    scalability_load: str = Field(
        ..., description="Πρόβλεψη χρηστών και δεδομένων (π.χ. 1M requests/day)."
    )
    budget_constraint: str = Field(
        ..., description="Οικονομικοί περιορισμοί (Low, Medium, Enterprise)."
    )
    security_compliance: str = Field(
        ..., description="Κανονιστικές απαιτήσεις (GDPR, HIPAA) και ασφάλεια."
    )
    performance_latency: str = Field(
        ..., description="Απαιτήσεις ταχύτητας (π.χ. <200ms TTFB)."
    )
    availability_sla: str = Field(
        ..., description="Διαθεσιμότητα (π.χ. 99.9% uptime, DR requirements)."
    )

# ==========================================
# 4. Validation Schema
# ==========================================
class ValidationSchema(BaseModel):
    """
    Χρησιμοποιείται ΜΟΝΟ για τον bypass check.
    Οι πραγματικές ερωτήσεις φτιάχνονται από τα templates.
    """
    is_valid: bool = Field(
        ..., description="True αν επαρκούν οι πληροφορίες."
    )
    feedback: str = Field(
        ..., description="Εξήγηση απόφασης."
    )
    needs_clarification: bool = Field(
        default=False, description="True αν χρειάζεται wizard."
    )

# ==========================================
# 5. NEW: Template Selection Schema (AI output)
# ==========================================
class TemplateSelection(BaseModel):
    """
    Το AI επιλέγει ΜΟΝΟ ποια templates ταιριάζουν.
    ΔΕΝ αποφασίζει form types, options, ή UI elements.
    """
    selected_essential_ids: List[str] = Field(
        ..., 
        description=(
            "4-5 template IDs from the pool for ESSENTIAL questions. "
            "Choose the ones that reveal the most important architectural decisions "
            "based on what is MISSING or AMBIGUOUS in the user's description."
        )
    )
    selected_recommended_ids: List[str] = Field(
        ..., 
        description=(
            "2-3 template IDs from the pool for RECOMMENDED questions. "
            "Choose domain-specific or optimization questions."
        )
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of why these questions were selected (for debugging)."
    )