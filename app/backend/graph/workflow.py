from langgraph.graph import StateGraph, END
from app.backend.graph.state import SystemDesignState

# Εισαγωγή των Nodes
from app.backend.graph.nodes.memory import fetch_past_projects_node
from app.backend.graph.nodes.analyst import requirement_analyst_node
from app.backend.graph.nodes.validator import initial_validator_node
from app.backend.graph.nodes.experts import technical_committee_node
from app.backend.graph.nodes.designer import system_designer_node
from app.backend.graph.nodes.design_critic import design_critic_node

# ==========================================
# 1. ΣΥΝΑΡΤΗΣΕΙΣ ΔΡΟΜΟΛΟΓΗΣΗΣ (ROUTING)
# ==========================================

def route_after_initial_validation(state: SystemDesignState):
    """
    Δυναμική Δρομολόγηση:
    Αποφασίζει αν η ροή θα συνεχίσει για επεξεργασία ή αν θα σταματήσει 
    για να ζητηθούν διευκρινίσεις από τον χρήστη (Dynamic Wizard).
    """
    val = state.get("initial_validation", {})
    status = state.get("validation_status", "FAIL")

    # 1. Κανονική ροή (Validation Pass):
    # Μεταφέρουμε τη ροή στο 'memory' (RAG) ΑΦΟΥ έχουμε έγκυρα requirements.
    if status == "PASS" or val.get("is_valid") is True:
        print("[ROUTE] Requirements validated. Proceeding to RAG retrieval...")
        return "memory"
    
    # 2. Clarification Mode (Validation Fail):
    # Το σύστημα σταματά (END) για να εμφανιστεί ο Wizard στο Streamlit UI.
    else:
        feedback = val.get('feedback') if val else state.get("validator_feedback", "Ελλιπή στοιχεία")
        print(f"[ROUTE] Clarification needed: {feedback}")
        return END

# ==========================================
# 2. ΧΤΙΣΙΜΟ ΤΟΥ ΓΡΑΦΗΜΑΤΟΣ
# ==========================================

workflow = StateGraph(SystemDesignState)

# Ορισμός Κόμβων
workflow.add_node("requirement_analyst", requirement_analyst_node)
workflow.add_node("initial_validator", initial_validator_node)
workflow.add_node("memory", fetch_past_projects_node)
workflow.add_node("technical_committee", technical_committee_node)
workflow.add_node("system_designer", system_designer_node)
workflow.add_node("design_critic", design_critic_node)  # Self-critique & consistency check

# --- ΚΑΘΟΡΙΣΜΟΣ ΡΟΗΣ (EDGES) ---

# Entry: Analyst πρώτος για μέγιστη αρχική ταχύτητα
workflow.set_entry_point("requirement_analyst")

# Analyst -> Validator
workflow.add_edge("requirement_analyst", "initial_validator")

# Δρομολόγηση μετά τον Validator:
# Αν PASS -> memory (RAG)
# Αν FAIL -> END (UI Wizard)
workflow.add_conditional_edges(
    "initial_validator",
    route_after_initial_validation
)

# Μετά την ανάκτηση από τη μνήμη (RAG), προχωράμε στην Τεχνική Επιτροπή
workflow.add_edge("memory", "technical_committee")

# Τεχνική Επιτροπή -> Designer
workflow.add_edge("technical_committee", "system_designer")

# Designer -> Design Critic (self-validation & auto-patching)
workflow.add_edge("system_designer", "design_critic")

# Design Critic -> END (παραδίδει validated + patched αρχιτεκτονική)
workflow.add_edge("design_critic", END)

# Compile της εφαρμογής
app_graph = workflow.compile()