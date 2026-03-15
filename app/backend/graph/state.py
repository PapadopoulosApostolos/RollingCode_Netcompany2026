import operator
from typing import TypedDict, Annotated, List, Dict, Any

class SystemDesignState(TypedDict):
    """
    Η Κεντρική Μνήμη (State) του LangGraph Workflow.
    """
    
    # --- 1. ΕΙΣΟΔΟΣ & ΜΝΗΜΗ (INPUT & CONTEXT) ---
    user_prompt: str
    
    # ΝΕΟ: Εδώ θα αποθηκεύονται τα δομημένα δεδομένα από τα drop-downs του UI (π.χ. Budget, Traffic)
    user_form_data: Dict[str, Any] 
    
    # RAG: Φέρνουμε τα "Top K" παρόμοια projects από το ChromaDB.
    historical_context: List[Dict[str, Any]]
    
    # --- 2. ΑΝΑΛΥΣΗ (REQUIREMENTS) ---
    # Εδώ θα μπει το δομημένο JSON που εξάγει ο Requirement Analyst.
    requirements: Dict[str, Any]
    
    
    # --- 3. ΠΑΡΑΛΛΗΛΗ ΣΚΕΨΗ (THE EXPERTS DEBATE) ---
    # REDUCER (operator.add): Όταν οι 6 Experts τρέχουν ταυτόχρονα, 
    # το LangGraph ΔΕΝ σβήνει την απάντηση του ενός με του άλλου.
    # Τις προσθέτει όλες σε αυτή την ενιαία λίστα!
    expert_opinions: Annotated[List[str], operator.add]
    
    
    # --- 4. ΕΛΕΓΧΟΣ & ΠΡΟΣΤΑΣΙΑ BUDGET (VALIDATION & SAFETY) ---
    validation_status: str  # Μπορεί να είναι: "PASS", "FAIL", ή "CLARIFY"
    validator_feedback: str # Το σχόλιο/παρατήρηση του Validator προς τους πράκτορες
    
    # ΝΕΑ ΚΛΕΙΔΙΑ (CRITICAL FIX): Χωρίς αυτά, το LangGraph διαγράφει σιωπηλά το "is_valid" και τον "wizard"!
    initial_validation: Dict[str, Any]
    final_validation: Dict[str, Any]
    
    # SAFETY VALVE: Προστατεύει το 10€ API Key. 
    # Αν φτάσει το 2, το σύστημα περνάει σε "Clarification Mode" και ζητάει βοήθεια από τον χρήστη.
    loop_count: int
    
    
    # --- 5. ΤΕΛΙΚΟ ΠΑΡΑΔΟΤΕΟ (STRUCTURED OUTPUT) ---
    # Το τεράστιο JSON που θα περιέχει το Mermaid C4 Diagram, το Cost Table και το Security Report.
    # Το Streamlit UI θα διαβάσει αυτό ακριβώς το πεδίο για να ζωγραφίσει την οθόνη.
    final_design: Dict[str, Any]