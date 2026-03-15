import time
from app.backend.graph.state import SystemDesignState

def fetch_past_projects_node(state: SystemDesignState):
    """
    Κόμβος Ανάκτησης Μνήμης (RAG Entry Point).
    MOCKED VERSION: Επιστρέφει hardcoded "παλιά projects" για απόλυτη ταχύτητα στο Hackathon.
    """
    print("--- [NODE] Memory: Ανάκτηση Ιστορικού Context (Mocked RAG) ---")
    
    # 1. Παίρνουμε το ελεύθερο κείμενο του χρήστη
    user_query = state.get("user_prompt", "")
    
    # 2. Αν ο χρήστης δεν έγραψε τίποτα, δεν ψάχνουμε
    if not user_query.strip():
        print("   -> Κενό prompt, παρακάμπτεται η αναζήτηση RAG.")
        return {"historical_context": []}
    
    try:
        # 3. Προσομοίωση αναζήτησης στη Vector DB (Smoke & Mirrors)
        # Η μικρή καθυστέρηση δίνει την ψευδαίσθηση "βαριάς" αναζήτησης στο τερματικό
        time.sleep(0.5)
        
        # Πειστικά, σκληρά κωδικοποιημένα μαθήματα από υποτιθέμενα παλαιότερα έργα
        mock_past_lessons = [
            "Lessons from 'Project Nexus' (High-Traffic E-commerce): Η χρήση Event-Driven αρχιτεκτονικής (Kafka) για την επεξεργασία παραγγελιών απέτρεψε την κατάρρευση της βάσης δεδομένων (PostgreSQL) κατά τη διάρκεια μεγάλων αιχμών κίνησης.",
            "Lessons from 'Project Titan' (B2C Platform): Η καθυστέρηση στην ενσωμάτωση Redis caching από την αρχή του έργου, οδήγησε σε υψηλό API latency. Το caching πρέπει να είναι βασικό συστατικό του αρχικού σχεδιασμού."
        ]
        
        print(f"   -> Βρέθηκαν {len(mock_past_lessons)} σχετικά έγγραφα από το παρελθόν.")
        
        # 4. Επιστρέφουμε τα αποτελέσματα για να μπουν στο State
        return {"historical_context": mock_past_lessons}
        
    except Exception as e:
        # Failsafe σε περίπτωση που κάτι πάει στραβά
        print(f"❌ Σφάλμα (Mocked RAG): {e}")
        return {"historical_context": ["Δεν βρέθηκε ιστορικό context."]}