import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.backend.graph.state import SystemDesignState
from app.backend.models.schemas import RequirementsSchema

def requirement_analyst_node(state: SystemDesignState):
    """
    Κόμβος LangGraph: Υβριδική ανάλυση απαιτήσεων χρησιμοποιώντας GPT-4o-mini.
    Συνδυάζει δεδομένα από τη φόρμα του UI, το ελεύθερο κείμενο του χρήστη και το RAG context.
    """
    print("--- [NODE] Requirement Analyst: Εξαγωγή Απαιτήσεων (GPT-4o-mini) ---")
    
    # 1. Ανάκτηση δεδομένων από το State
    user_prompt = state.get("user_prompt", "")
    user_form_data = state.get("user_form_data", {})
    historical_context = state.get("historical_context", [])
    
    # Διαχείριση του Feedback Loop (αν ο Validator επέστρεψε παρατηρήσεις)
    validator_feedback = state.get("validator_feedback", "")
    loop_count = state.get("loop_count", 0)
    
    # 2. Αρχικοποίηση OpenAI Μοντέλου
    llm = ChatOpenAI(
        model="gpt-4.1", 
        temperature=0.1, 
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Σύνδεση με το δομημένο output (RequirementsSchema)
    structured_llm = llm.with_structured_output(RequirementsSchema)
    
    # Δημιουργία οδηγιών διόρθωσης αν βρισκόμαστε σε επόμενο γύρο (Loop)
    feedback_instruction = ""
    if loop_count > 0 and validator_feedback:
        print(f"   -> [RE-ITERATION] Ενσωμάτωση παρατηρήσεων ελέγχου (Γύρος {loop_count})")
        feedback_instruction = f"""
        ΚΡΙΣΙΜΟ: Η προηγούμενη ανάλυση απορρίφθηκε. 
        ΔΙΟΡΘΩΣΕ τα εξής σημεία σύμφωνα με τον ελεγκτή: "{validator_feedback}"
        """
    
    # 3. Το Αυστηρό System Prompt (Prompt Engineering)
    # FIX: Escape curly braces στο JSON — αλλιώς το ChatPromptTemplate τα 
    # ερμηνεύει ως template variables και σκάει KeyError: "core_functionality".
    system_prompt = """
    Είσαι ένας Senior Business Analyst και System Architect.
    Στόχος σου είναι να εξάγεις τα 6 βασικά τεχνικά χαρακτηριστικά (Requirements) του συστήματος.
    
    {feedback_instruction}
    
    ΠΗΓΕΣ ΔΕΔΟΜΕΝΩΝ (ΙΕΡΑΡΧΙΑ):
    1. ΔΕΔΟΜΕΝΑ ΦΟΡΜΑΣ (ΑΠΟΛΥΤΟΣ ΝΟΜΟΣ): 
    {user_form_data}
    *Οφείλεις να τηρήσεις τις επιλογές της φόρμας (Budget, Load, Security) ως αδιαπραγμάτευτες.*
    
    2. ΕΛΕΥΘΕΡΟ ΚΕΙΜΕΝΟ ΧΡΗΣΤΗ (ΟΡΑΜΑ):
    {user_prompt}
    *Χρησιμοποίησέ το για να εξάγεις τη λειτουργικότητα και τους στόχους.*
    
    3. ΙΣΤΟΡΙΚΟ ΠΛΑΙΣΙΟ (RAG CONTEXT):
    {context}
    *Βασίσου σε παρόμοια έργα για να συμπληρώσεις κενά στις τεχνικές προδιαγραφές.*
    
    ΟΔΗΓΙΕΣ:
    - Αν ένα στοιχείο δεν αναφέρεται, κάνε μια λογική τεχνική υπόθεση.
    - Μην φλυαρείς. Πήγαινε κατευθείαν στην τεχνική ουσία.
    - Απάντησε αποκλειστικά σε JSON format.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", system_prompt)
    ])
    
    chain = prompt | structured_llm
    
    # FIX: Escape curly braces μέσα στα JSON strings ώστε το 
    # ChatPromptTemplate να μην τα ερμηνεύει ως {variable_name}.
    form_data_str = (
        json.dumps(user_form_data, ensure_ascii=False)
        .replace("{", "{{").replace("}", "}}")
        if user_form_data 
        else "Δεν υπάρχουν δεδομένα φόρμας."
    )
    
    # 4. Εκτέλεση της αλυσίδας
    result = chain.invoke({
        "context": historical_context,
        "user_prompt": user_prompt,
        "user_form_data": form_data_str,
        "feedback_instruction": feedback_instruction
    })
    
    # 5. Ενημέρωση του State
    return {"requirements": result.model_dump()}