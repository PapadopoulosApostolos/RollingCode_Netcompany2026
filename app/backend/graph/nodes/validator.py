import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.backend.graph.state import SystemDesignState
from app.backend.models.schemas import TemplateSelection

# ══════════════════════════════════════════════════════════════════
# QUESTION POOL — Κάθε template ορίζει ΑΚΡΙΒΩΣ τον τύπο φόρμας.
# Το AI επιλέγει ΜΟΝΟ ποια templates ταιριάζουν. Τίποτα άλλο.
# ══════════════════════════════════════════════════════════════════

QUESTION_POOL = [

    # ═══════════════════════════════════════
    # AUTH & ACCESS CONTROL
    # ═══════════════════════════════════════
    {
        "id": "auth_method",
        "label": "Ποια μέθοδο ταυτοποίησης χρηστών προτιμάτε;",
        "field_type": "select",
        "options": [
            "JWT Tokens (Stateless API)",
            "Session Cookies (Server-side)",
            "OAuth2 / Social Login (Google / GitHub)",
            "SAML / SSO (Enterprise)",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture"],
    },
    {
        "id": "role_complexity",
        "label": "Πόσο πολύπλοκο είναι το σύστημα δικαιωμάτων;",
        "field_type": "select",
        "options": [
            "Απλό — 2 ρόλοι (Admin / User)",
            "Μεσαίο — 3-5 ρόλοι με ξεχωριστά δικαιώματα",
            "Σύνθετο — Dynamic RBAC / ABAC permissions",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture"],
    },

    # ═══════════════════════════════════════
    # DATA & STORAGE
    # ═══════════════════════════════════════
    {
        "id": "data_sensitivity",
        "label": "Τι είδους ευαίσθητα δεδομένα θα αποθηκεύει το σύστημα;",
        "field_type": "multi_select",
        "options": [
            "Προσωπικά δεδομένα (PII / GDPR)",
            "Οικονομικά / Πληρωμές (PCI-DSS)",
            "Ιατρικά δεδομένα (HIPAA)",
            "Κρατικά / Κυβερνητικά",
            "Κανένα ιδιαίτερο",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture", "Data/ML Pipeline"],
    },
    {
        "id": "data_volume",
        "label": "Ποιος είναι ο αναμενόμενος όγκος δεδομένων σε 1 χρόνο;",
        "field_type": "select",
        "options": [
            "Μικρός (< 10 GB)",
            "Μεσαίος (10 GB - 500 GB)",
            "Μεγάλος (500 GB - 5 TB)",
            "Πολύ μεγάλος (> 5 TB)",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture", "Data/ML Pipeline"],
    },
    {
        "id": "file_upload_needs",
        "label": "Τι είδους αρχεία θα ανεβάζουν οι χρήστες;",
        "field_type": "select",
        "options": [
            "Κανένα — μόνο δεδομένα φόρμας",
            "Εικόνες / Avatars (< 10 MB ανά αρχείο)",
            "Έγγραφα (PDF / Excel / CSV)",
            "Βίντεο / Μεγάλα αρχεία (> 100 MB)",
            "System Recommendation",
        ],
        "category": "recommended",
        "tags": ["Web Application", "Data/ML Pipeline"],
    },

    # ═══════════════════════════════════════
    # INTEGRATIONS & APIS
    # ═══════════════════════════════════════
    {
        "id": "third_party_services",
        "label": "Ποιες εξωτερικές υπηρεσίες θα χρειαστεί να ενσωματωθούν;",
        "field_type": "multi_select",
        "options": [
            "Payment Gateway (Stripe / PayPal)",
            "Email Service (SendGrid / SES)",
            "SMS / Push Notifications",
            "Maps / Geolocation",
            "Analytics (Google / Mixpanel)",
            "Storage (S3 / Azure Blob)",
            "AI/ML APIs (OpenAI / HuggingFace)",
            "Κανένα εξωτερικό",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture", "Data/ML Pipeline"],
    },
    {
        "id": "api_consumers",
        "label": "Ποιοι θα καταναλώνουν τα APIs σας;",
        "field_type": "multi_select",
        "options": [
            "Web Frontend (SPA / SSR)",
            "Mobile App (iOS / Android)",
            "Τρίτα συστήματα (B2B Partners)",
            "Internal Microservices",
            "IoT Devices",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture"],
    },

    # ═══════════════════════════════════════
    # REAL-TIME & COMMUNICATION
    # ═══════════════════════════════════════
    {
        "id": "realtime_needs",
        "label": "Χρειάζεται real-time επικοινωνία ή live updates;",
        "field_type": "select",
        "options": [
            "Όχι — κλασσικό request/response",
            "Ναι — Live notifications (WebSockets)",
            "Ναι — Real-time chat / collaboration",
            "Ναι — Live dashboards / streaming data",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture"],
    },
    {
        "id": "notification_channels",
        "label": "Μέσω ποιων καναλιών θα ειδοποιούνται οι χρήστες;",
        "field_type": "multi_select",
        "options": [
            "Email",
            "SMS",
            "Push Notifications (Mobile)",
            "In-App Notifications",
            "Slack / Teams webhooks",
            "Κανένα",
        ],
        "category": "recommended",
        "tags": ["Web Application", "Microservice Architecture"],
    },

    # ═══════════════════════════════════════
    # PAYMENT & MONETIZATION
    # ═══════════════════════════════════════
    {
        "id": "payment_model",
        "label": "Πώς θα γίνονται οι πληρωμές / χρεώσεις;",
        "field_type": "select",
        "options": [
            "Δεν υπάρχουν πληρωμές",
            "Απλό checkout (One-time purchase)",
            "Συνδρομές (Recurring / SaaS billing)",
            "Marketplace (Split payments μεταξύ πωλητών)",
            "Freemium + Premium upgrades",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Web Application"],
    },

    # ═══════════════════════════════════════
    # DEPLOYMENT & INFRASTRUCTURE
    # ═══════════════════════════════════════
    {
        "id": "deployment_target",
        "label": "Πού θα γίνει η ανάπτυξη (deployment);",
        "field_type": "select",
        "options": [
            "Managed PaaS (Heroku / Railway / Render)",
            "AWS (ECS / EKS / Lambda)",
            "Azure (App Service / AKS)",
            "GCP (Cloud Run / GKE)",
            "On-Premise / Private Cloud",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Web Application", "Microservice Architecture", "Data/ML Pipeline"],
    },
    {
        "id": "multi_region",
        "label": "Χρειάζεται γεωγραφική κατανομή σε πολλαπλά regions;",
        "field_type": "checkbox",
        "options": [],
        "category": "recommended",
        "tags": ["Web Application", "Microservice Architecture"],
    },
    {
        "id": "ci_cd_complexity",
        "label": "Τι επίπεδο CI/CD αυτοματοποίησης χρειάζεστε;",
        "field_type": "select",
        "options": [
            "Βασικό (Manual deploy + Git push)",
            "Μεσαίο (Auto deploy on merge + staging env)",
            "Προχωρημένο (Blue/Green + Canary + Rollback)",
            "System Recommendation",
        ],
        "category": "recommended",
        "tags": ["Web Application", "Microservice Architecture", "Data/ML Pipeline"],
    },

    # ═══════════════════════════════════════
    # SEARCH & DISCOVERY
    # ═══════════════════════════════════════
    {
        "id": "search_complexity",
        "label": "Τι είδους αναζήτηση χρειάζεται η εφαρμογή;",
        "field_type": "select",
        "options": [
            "Απλό SQL LIKE / filtering",
            "Full-text search (Elasticsearch / Algolia)",
            "Faceted search με φίλτρα και sorting",
            "AI-powered semantic search",
            "Δεν χρειάζεται αναζήτηση",
            "System Recommendation",
        ],
        "category": "recommended",
        "tags": ["Web Application"],
    },

    # ═══════════════════════════════════════
    # DOMAIN-SPECIFIC: WEB
    # ═══════════════════════════════════════
    {
        "id": "multi_tenancy",
        "label": "Χρειάζεται multi-tenancy (πολλοί πελάτες σε 1 σύστημα);",
        "field_type": "select",
        "options": [
            "Όχι — single-tenant",
            "Shared DB με tenant_id column",
            "Ξεχωριστό schema ανά tenant",
            "Ξεχωριστή βάση ανά tenant",
            "System Recommendation",
        ],
        "category": "recommended",
        "tags": ["Web Application", "Microservice Architecture"],
    },
    {
        "id": "audit_logging",
        "label": "Χρειάζεται καταγραφή ενεργειών χρηστών (audit trail);",
        "field_type": "checkbox",
        "options": [],
        "category": "recommended",
        "tags": ["Web Application", "Microservice Architecture"],
    },
    {
        "id": "i18n_support",
        "label": "Χρειάζεται υποστήριξη πολλαπλών γλωσσών (i18n);",
        "field_type": "checkbox",
        "options": [],
        "category": "recommended",
        "tags": ["Web Application"],
    },
    {
        "id": "offline_support",
        "label": "Χρειάζεται offline λειτουργία (PWA / Service Workers);",
        "field_type": "checkbox",
        "options": [],
        "category": "recommended",
        "tags": ["Web Application"],
    },

    # ═══════════════════════════════════════
    # DOMAIN-SPECIFIC: MICROSERVICES
    # ═══════════════════════════════════════
    {
        "id": "service_communication",
        "label": "Πώς θα επικοινωνούν τα microservices μεταξύ τους;",
        "field_type": "select",
        "options": [
            "Synchronous REST / gRPC",
            "Asynchronous Events (Kafka / RabbitMQ)",
            "Hybrid (REST για queries + Events για commands)",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Microservice Architecture"],
    },
    {
        "id": "service_count",
        "label": "Πόσα ξεχωριστά microservices εκτιμάτε αρχικά;",
        "field_type": "select",
        "options": [
            "2-4 services (Mini microservices)",
            "5-10 services (Standard)",
            "10+ services (Large-scale)",
            "Ξεκίνα ως monolith → σπάσε αργότερα",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Microservice Architecture"],
    },

    # ═══════════════════════════════════════
    # DOMAIN-SPECIFIC: DATA / ML
    # ═══════════════════════════════════════
    {
        "id": "data_source_types",
        "label": "Από πού θα έρχονται τα δεδομένα;",
        "field_type": "multi_select",
        "options": [
            "REST / GraphQL APIs",
            "Βάσεις δεδομένων (CDC / Replication)",
            "Αρχεία (CSV / JSON / Parquet)",
            "Streaming (Kafka / Kinesis / MQTT)",
            "Web Scraping",
            "IoT Sensors",
        ],
        "category": "essential",
        "tags": ["Data/ML Pipeline"],
    },
    {
        "id": "ml_serving",
        "label": "Πώς θα σερβίρονται τα ML μοντέλα;",
        "field_type": "select",
        "options": [
            "Batch predictions (Scheduled jobs)",
            "Real-time API inference",
            "Edge deployment (Mobile / IoT)",
            "Δεν υπάρχει ML component",
            "System Recommendation",
        ],
        "category": "essential",
        "tags": ["Data/ML Pipeline"],
    },
    {
        "id": "data_quality",
        "label": "Τι επίπεδο data quality / validation χρειάζεστε;",
        "field_type": "select",
        "options": [
            "Βασικό (Schema validation μόνο)",
            "Μεσαίο (Data profiling + anomaly detection)",
            "Υψηλό (Great Expectations / dbt tests / SLA alerting)",
            "System Recommendation",
        ],
        "category": "recommended",
        "tags": ["Data/ML Pipeline"],
    },
    {
        "id": "pipeline_orchestration",
        "label": "Τι εργαλείο orchestration προτιμάτε για τα pipelines;",
        "field_type": "select",
        "options": [
            "Apache Airflow",
            "Prefect / Dagster",
            "AWS Step Functions / Glue",
            "Azure Data Factory",
            "Simple cron jobs",
            "System Recommendation",
        ],
        "category": "recommended",
        "tags": ["Data/ML Pipeline"],
    },
]


def _build_wizard_from_selection(selection_result, project_type):
    """
    Παίρνει τα selected IDs και φτιάχνει ClarificationWizard
    αντιγράφοντας τα templates ΑΥΤΟΥΣΙΑ (field_type, options κλπ).
    """
    # Index pool by ID
    pool_index = {q["id"]: q for q in QUESTION_POOL}
    
    essential_fields = []
    recommended_fields = []
    
    for qid in selection_result.selected_essential_ids:
        template = pool_index.get(qid)
        if template:
            essential_fields.append({
                "id": template["id"],
                "label": template["label"],
                "field_type": template["field_type"],
                "options": template["options"],
                "required": True,
            })
    
    for qid in selection_result.selected_recommended_ids:
        template = pool_index.get(qid)
        if template:
            recommended_fields.append({
                "id": template["id"],
                "label": template["label"],
                "field_type": template["field_type"],
                "options": template["options"],
                "required": False,
            })
    
    return {
        "essential": essential_fields,
        "recommended": recommended_fields,
        "optional": [],
    }


def _get_available_ids_for_type(project_type):
    """Φιλτράρει το pool βάσει project type."""
    return [
        q for q in QUESTION_POOL 
        if project_type in q.get("tags", [])
    ]


def initial_validator_node(state: SystemDesignState):
    """
    Template-based validator: Το AI επιλέγει 7-8 templates,
    ΔΕΝ αποφασίζει form types — αυτά είναι hardcoded στα templates.
    """
    print("--- [NODE] Initial Validator: Template-Based Question Selection ---")
    
    current_loop = state.get("loop_count", 0) + 1
    user_prompt = state.get("user_prompt", "")
    
    # ── BYPASS αν ο χρήστης ήδη απάντησε ──
    has_answered = "ΔΥΝΑΜΙΚΕΣ ΔΙΕΥΚΡΙΝΙΣΕΙΣ" in user_prompt or "ΕΠΙΠΛΕΟΝ ΔΙΕΥΚΡΙΝΙΣΕΙΣ" in user_prompt
    
    if has_answered:
        print("   -> [BYPASS] Οι διευκρινίσεις λήφθηκαν. Μετάβαση στο Design.")
        return {
            "validation_status": "PASS",
            "validator_feedback": "OK",
            "initial_validation": {
                "is_valid": True,
                "feedback": "OK",
                "needs_clarification": False,
                "wizard": {"essential": [], "recommended": [], "optional": []}
            },
            "loop_count": current_loop
        }
    
    # ── Εξαγωγή project type από user prompt ──
    project_type = "Web Application"  # default
    for ptype in ["Microservice Architecture", "Data/ML Pipeline", "Web Application"]:
        if ptype.lower() in user_prompt.lower() or ptype in user_prompt:
            project_type = ptype
            break
    
    # ── Φιλτράρισμα pool βάσει project type ──
    available_questions = _get_available_ids_for_type(project_type)
    
    # Δημιουργία readable pool summary (IDs + labels) για το AI
    pool_summary = "\n".join([
        f"  - id=\"{q['id']}\" | category={q['category']} | label=\"{q['label']}\""
        for q in available_questions
    ])
    
    # ── AI: Επέλεξε τα πιο σχετικά templates ──
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    ).with_structured_output(TemplateSelection)
    
    selection_prompt = f"""You are an Elite System Architect. Analyze the user's project description and select the most relevant questions.

USER DESCRIPTION:
{user_prompt[:2000]}

PROJECT TYPE: {project_type}

AVAILABLE QUESTION TEMPLATES (pick from these IDs only):
{pool_summary}

INSTRUCTIONS:
1. Select 4-5 IDs for selected_essential_ids — questions that reveal CRITICAL unknowns for the architecture.
2. Select 2-3 IDs for selected_recommended_ids — domain-specific optimization questions.
3. DO NOT select questions whose answer is ALREADY CLEAR from the user's description.
   For example, if the user already mentioned "JWT authentication", do NOT select "auth_method".
4. Prefer questions that will MOST IMPACT the database schema, infrastructure, and cost.
5. Return ONLY valid IDs from the list above."""

    try:
        selection = llm.invoke([{"role": "user", "content": selection_prompt}])
    except Exception as e:
        print(f"   -> [ERROR] Template selection failed: {e}")
        # Fallback: πρώτα 5 essential + 2 recommended
        essential_fallback = [q["id"] for q in available_questions if q["category"] == "essential"][:5]
        recommended_fallback = [q["id"] for q in available_questions if q["category"] == "recommended"][:2]
        
        selection = TemplateSelection(
            selected_essential_ids=essential_fallback,
            selected_recommended_ids=recommended_fallback,
            reasoning="Fallback selection due to AI error"
        )
    
    # ── Build wizard from selected templates ──
    wizard_data = _build_wizard_from_selection(selection, project_type)
    
    total_questions = len(wizard_data["essential"]) + len(wizard_data["recommended"])
    print(f"   -> Selected {total_questions} questions (E:{len(wizard_data['essential'])} R:{len(wizard_data['recommended'])})")
    print(f"   -> Reasoning: {selection.reasoning[:100]}")
    
    return {
        "validation_status": "FAIL",
        "validator_feedback": selection.reasoning,
        "initial_validation": {
            "is_valid": False,
            "feedback": selection.reasoning,
            "needs_clarification": True,
            "wizard": wizard_data
        },
        "loop_count": current_loop
    }


def final_validator_node(state: SystemDesignState):
    pass