import os
import json
import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.backend.graph.state import SystemDesignState
from app.backend.models.schemas import FinalArchitectureDesign

# ══════════════════════════════════════════════════════════════════
# SECTION 1: POST-PROCESSING & SANITIZATION UTILITIES
# ══════════════════════════════════════════════════════════════════

def _light_post_process(code: str) -> str:
    if not code:
        return ""
    code = code.replace("```mermaid", "").replace("```", "").strip()
    code = code.replace("\\n", "\n")
    code = code.replace("|||", "\n")
    code = re.sub(r'^(C4Context\s*)+', 'C4Context\n', code, flags=re.IGNORECASE)
    lines = code.split('\n')
    clean_lines = []
    for i, line in enumerate(lines):
        if 'C4Context' in line and i > 0:
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines).strip()


def _erd_post_process(code: str) -> str:
    code = _light_post_process(code)
    if not code:
        return ""
    code = re.sub(
        r'(\b(?:datetime|string|uuid|int|float|boolean|text|enum)\s+\w+)'
        r'([A-Z][A-Z_]+\s*\{)',
        r'\1\n}\n\2',
        code,
    )
    code = re.sub(r'\}([A-Z])', r'}\n\1', code)
    code = re.sub(r'\}\s*\n(\s*[A-Z][A-Z_]+ \{)', r'}\n\1', code)
    return code.strip()


def sanitize_markdown_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", "\n")
    text = text.replace("\n-", "\n\n-").replace("\n*", "\n\n*")
    return text.strip()


# ══════════════════════════════════════════════════════════════════
# SECTION 2: THE SYSTEM DESIGNER NODE
# ══════════════════════════════════════════════════════════════════

def system_designer_node(state: SystemDesignState) -> Dict[str, Any]:
    """
    Synthesizes the final architecture. Now retrieves past lessons from
    the self-learning memory (ChromaDB design_lessons collection) to
    avoid repeating mistakes from previous runs.
    """
    print("--- [NODE] System Designer: Synthesis (GPT-4o + Lessons from Memory) ---")

    requirements = state.get("requirements", {})
    expert_opinions = state.get("expert_opinions", [])
    opinions_text = "\n\n".join(expert_opinions)

    # ── Self-Learning: Retrieve past lessons ──────────────────────
    lessons_context = ""
    try:
        from app.backend.graph.nodes.design_critic import retrieve_lessons
        core_func = requirements.get("core_functionality", "system design")
        lessons_context = retrieve_lessons(
            f"Architecture lessons for {core_func}",
            n_results=5
        )
        if lessons_context:
            print(f"   [DESIGNER] Injecting {lessons_context.count(chr(10)) + 1} lessons from memory")
    except Exception as e:
        print(f"   [DESIGNER] Lesson retrieval skipped: {e}")

    # ── LLM Setup ─────────────────────────────────────────────────
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.15,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(FinalArchitectureDesign)

    req_json_str = json.dumps(requirements, ensure_ascii=False)

    # ── Lessons block (injected only if lessons exist) ────────────
    lessons_block = ""
    if lessons_context:
        lessons_block = (
            "\n═══════════════════════════════════════════════════════════════\n"
            "LESSONS FROM PREVIOUS DESIGNS (SELF-LEARNING MEMORY)\n"
            "═══════════════════════════════════════════════════════════════\n"
            "The following are mistakes found in previous architecture generations.\n"
            "You MUST avoid repeating these. Apply the corrective guidance:\n\n"
            f"{lessons_context}\n\n"
        )

    system_prompt = (
        "Είσαι ο Principal System Architect και Cloud FinOps Specialist.\n"
        "Σχεδίασε μια ΠΛΗΡΗ, ΑΝΑΛΥΤΙΚΗ και ΚΑΘΑΡΗ αρχιτεκτονική βάσει των παρακάτω δεδομένων.\n\n"
        "--- ΑΠΑΙΤΗΣΕΙΣ (REQUIREMENTS) ---\n"
        f"{req_json_str}\n\n"
        "--- ΕΙΣΗΓΗΣΕΙΣ EXPERTS (TECHNICAL COMMITTEE) ---\n"
        f"{opinions_text}\n\n"
        f"{lessons_block}"
        "═══════════════════════════════════════════════════════════════\n"
        "SECTION A — C4 CONTEXT DIAGRAM  (field: mermaid_c4_code)\n"
        "═══════════════════════════════════════════════════════════════\n"
        "ΣΤΟΧΟΣ: Ένα C4 Context diagram με ΜΗΔΕΝΙΚΕΣ ΔΙΑΣΤΑΥΡΩΣΕΙΣ (Zero Intersecting Lines).\n"
        "ΤΟ ΠΡΟΒΛΗΜΑ: Ο αλγόριθμος Dagre του Mermaid καταρρέει (spaghetti) αν του αφήσεις ελευθερία.\n"
        "Η ΑΠΟΛΥΤΗ ΛΥΣΗ: Θα χρησιμοποιήσεις το παρακάτω ΚΛΕΙΔΩΜΕΝΟ TEMPLATE ΤΡΙΑΙΝΑΣ (Trident Topology).\n\n"
        "ΚΑΝΟΝΕΣ ΣΥΝΤΑΞΗΣ (ΑΔΙΑΠΡΑΓΜΑΤΕΥΤΟΙ):\n"
        "1. Γραμμή 1: C4Context\n"
        "2. ΜΟΝΟ English ASCII. Απαγορεύονται τα Ελληνικά στο διάγραμμα.\n"
        "3. ΑΠΑΓΟΡΕΥΟΝΤΑΙ ΤΑ System_Boundary. \n"
        "4. ΜΗΝ βάζεις ```mermaid fences.\n\n"
        "TRIDENT TEMPLATE (0 INTERSECTIONS):\n"
        "ΠΡΕΠΕΙ ΝΑ ΑΝΤΙΓΡΑΨΕΙΣ ΤΟ ΠΑΡΑΚΑΤΩ TEMPLATE ΚΑΤΑ ΓΡΑΜΜΑ. \n"
        "Το μόνο που επιτρέπεται να αλλάξεις είναι τα ονόματα μέσα στα ΕΙΣΑΓΩΓΙΚΑ \"\".\n\n"
        "C4Context\n"
        "title [Τίτλος Συστήματος Εδώ]\n\n"
        "Person(user, \"[Όνομα Χρήστη]\", \"User of the system\")\n"
        "System(ui, \"[Όνομα UI/App]\", \"Frontend interface\")\n"
        "System(gateway, \"[API Gateway]\", \"Entry point\")\n\n"
        "System(auth_svc, \"[Όνομα Core Service 1]\", \"Microservice\")\n"
        "System(core_svc, \"[Όνομα Core Service 2]\", \"Microservice\")\n"
        "System(ext_svc, \"[Όνομα Integration Svc]\", \"Microservice\")\n\n"
        "SystemDb(auth_db, \"[Όνομα DB 1]\", \"Data Store\")\n"
        "SystemDb(core_db, \"[Όνομα DB 2]\", \"Data Store\")\n"
        "System_Ext(ext_api, \"[Όνομα External API]\", \"External Service\")\n\n"
        "Rel_D(user, ui, \"Uses\")\n"
        "Rel_D(ui, gateway, \"API Calls\")\n"
        "Rel_D(gateway, auth_svc, \"Routes\")\n"
        "Rel_D(gateway, core_svc, \"Routes\")\n"
        "Rel_D(gateway, ext_svc, \"Routes\")\n"
        "Rel_D(auth_svc, auth_db, \"Reads/Writes\")\n"
        "Rel_D(core_svc, core_db, \"Reads/Writes\")\n"
        "Rel_D(ext_svc, ext_api, \"API Calls\")\n\n"
        "ΑΥΣΤΗΡΗ ΕΝΤΟΛΗ: \n"
        "1. Χρησιμοποίησε ΑΚΡΙΒΩΣ ΑΥΤΕΣ τις 8 γραμμές Rel_D(). Η χρήση Rel_D() αναγκάζει τον κώδικα να χτιστεί ιεραρχικά.\n"
        "2. ΜΗΝ ΠΡΟΣΘΕΣΕΙΣ ΟΥΤΕ ΑΦΑΙΡΕΣΕΙΣ ΚΑΜΙΑ άλλη σύνδεση.\n"
        "3. Αν αλλάξεις έστω και μία σχέση, το γράφημα θα καταρρεύσει.\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "SECTION B — ERD DIAGRAM  (field: mermaid_erd_code)\n"
        "═══════════════════════════════════════════════════════════════\n"
        "ΣΤΟΧΟΣ: Ένα ΠΛΗΡΕΣ και σωστά μορφοποιημένο Entity-Relationship diagram.\n\n"
        "ΚΡΙΣΙΜΟΙ ΚΑΝΟΝΕΣ ΜΟΡΦΟΠΟΙΗΣΗΣ:\n"
        "1. Γραμμή 1 = \"erDiagram\" ΜΟΝΟ ΤΟΥ.\n"
        "2. ENTITY BLOCKS: Κάθε entity ΠΡΕΠΕΙ να βρίσκεται ΣΕ ΔΙΚΟ ΤΟΥ BLOCK και να υπάρχει ΚΕΝΗ ΓΡΑΜΜΗ πριν από κάθε οντότητα.\n"
        "   Σωστό:\n"
        "     erDiagram\n\n"
        "     USER {\n"
        "         uuid id PK\n"
        "         string email\n"
        "     }\n\n"
        "     PRODUCT {\n"
        "         uuid id PK\n"
        "         string title\n"
        "     }\n\n"
        "   ΛΑΘΟΣ (χωρίς κενή γραμμή ή χωρίς closing brace):\n"
        "     USER { uuid id PK string email }PRODUCT { ...\n\n"
        "3. ΚΑΡΔΙΝΑΛΙΤΗΤΑ — Επιτρέπονται ΜΟΝΟ αυτά τα σύμβολα:\n"
        "   ||--|| (one-to-one)\n"
        "   ||--o{ (one-to-zero-or-many)\n"
        "   ||--|{ (one-to-one-or-many)\n"
        "4. Labels: Τα labels των σχέσεων ΠΡΕΠΕΙ ΠΑΝΤΑ να είναι σε ΔΙΠΛΑ ΕΙΣΑΓΩΓΙΚΑ:\n"
        "   USER ||--o{ ORDER : \"places\"\n"
        "5. Attribute types: Επίτρεπτα είναι μόνο τα uuid, string, int, float, boolean, datetime, text, enum.\n"
        "6. ΜΟΝΟ English ASCII. Κανένα κόμμα μέσα σε labels σχέσεων.\n"
        "7. ΜΗΝ βάζεις ```mermaid fences.\n\n"
        "ΕΛΑΧΙΣΤΗ ΠΟΛΥΠΛΟΚΟΤΗΤΑ ΓΙΑ ΤΟ ERD:\n"
        "- Τουλάχιστον 5 έως 8 entities (αναλόγως το μέγεθος του project).\n"
        "- 3-8 attributes ανά entity (Πρέπει να περιλαμβάνει PK, FK και data fields).\n"
        "- Συμπερίληψη timestamp fields (created_at, updated_at) στους κύριους πίνακες.\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "SECTION C — COST TABLE  (field: cost_estimate_table)\n"
        "═══════════════════════════════════════════════════════════════\n"
        "ΣΤΟΧΟΣ: Δημιουργία ενός αναλυτικού, επαγγελματικού πίνακα εξόδων (FinOps).\n\n"
        "ΑΠΑΙΤΟΥΜΕΝΟ FORMAT (Αυστηρό Markdown Table):\n"
        "| Resource Category | Service and Tier | Scaling Unit | Est. Monthly Cost ($) | Rationale |\n"
        "|---|---|---|---|---|\n\n"
        "ΥΠΟΧΡΕΩΤΙΚΕΣ ΓΡΑΜΜΕΣ ΠΟΥ ΠΡΕΠΕΙ ΝΑ ΥΠΑΡΧΟΥΝ:\n"
        "- Compute (π.χ. EC2, ECS, EKS, App Service)\n"
        "- Primary Database (π.χ. RDS, Cloud SQL)\n"
        "- Caching (π.χ. ElastiCache)\n"
        "- Object Storage (π.χ. S3, Blob Storage)\n"
        "- CDN / Bandwidth (π.χ. CloudFront, Data Egress)\n"
        "- Message Queue (αν εφαρμόζεται στην αρχιτεκτονική)\n"
        "- Search Engine (αν χρειάζεται)\n"
        "- Monitoring / Logging (π.χ. CloudWatch, Datadog)\n"
        "- Backup / DR\n"
        "- Security (π.χ. WAF, GuardDuty)\n"
        "- CI/CD\n"
        "- **TOTAL** (Πρέπει ΠΑΝΤΑ να είναι η τελευταία γραμμή, με έντονα γράμματα)\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "SECTION D — ΥΠΟΛΟΙΠΑ ΠΕΔΙΑ (TEXT & METRICS)\n"
        "═══════════════════════════════════════════════════════════════\n\n"
        "tech_stack_summary:\n"
        "  Γράψε 5-8 bullets στα Ελληνικά. \n"
        "  Format: \"- **[Τεχνολογία]** — [Αιτιολόγηση]\"\n"
        "  Κάλυψε οπωσδήποτε: Γλώσσα προγραμματισμού/Framework, Βάση Δεδομένων, Caching, Cloud Provider, CI/CD, Monitoring.\n\n"
        "trade_off_analysis:\n"
        "  Γράψε 3-5 trade-offs στα Ελληνικά. Τι κερδίζουμε με αυτή την αρχιτεκτονική έναντι του τι θυσιάζουμε.\n\n"
        "future_scaling_path:\n"
        "  Περιέγραψε το roadmap στα Ελληνικά για κλιμάκωση 100x.\n\n"
        "implementation_roadmap: \"N/A\" (Αγνόησέ το, βάλε απλά \"N/A\").\n\n"
        "metrics: \n"
        "  Δώσε ρεαλιστικές βαθμολογίες από το 1 έως το 10. \n"
        "  ΜΗΝ βάζεις 5 σε όλα.\n\n"
        "ΓΕΝΙΚΟΙ ΚΑΝΟΝΕΣ:\n"
        "- Κείμενα: Αποκλειστικά Ελληνικά (εκτός από τους τεχνικούς όρους).\n"
        "- Diagrams: Αποκλειστικά Αγγλικά (ASCII).\n"
        "- ΠΟΤΕ και ΠΟΥΘΕΝΑ μην χρησιμοποιείς markdown fences (```mermaid).\n"
    )

    messages = [SystemMessage(content=system_prompt)]
    result = structured_llm.invoke(messages)
    final_design_dict = result.model_dump()

    # Post-processing
    final_design_dict["mermaid_c4_code"] = _light_post_process(final_design_dict.get("mermaid_c4_code", ""))
    final_design_dict["mermaid_erd_code"] = _erd_post_process(final_design_dict.get("mermaid_erd_code", ""))
    for field in ["tech_stack_summary", "trade_off_analysis", "future_scaling_path", "implementation_roadmap"]:
        final_design_dict[field] = sanitize_markdown_text(final_design_dict.get(field, ""))

    return {"final_design": final_design_dict}