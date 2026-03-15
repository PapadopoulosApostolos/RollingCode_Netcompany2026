from app.backend.vector_store.client import get_chroma_collections

# Φέρνουμε τα Collections από το client.py που γράψαμε πριν
projects_collection, experts_collection = get_chroma_collections()

# ==========================================
# 1. AUTO-SEEDING: Γεμίζουμε τη βάση αν είναι άδεια!
# ==========================================
def seed_database_if_empty():
    """Ελέγχει αν η βάση έχει δεδομένα. Αν όχι, βάζει τα 'Golden Mock Data'."""
    
    # --- Seeding για τα παλιά projects (Analyst) ---
    if projects_collection.count() == 0:
        print("🌱 [ChromaDB] Η βάση 'projects' είναι άδεια. Ξεκινάει το Seeding...")
        
        projects_collection.add(
            ids=["proj_1", "proj_2", "proj_3"],
            documents=[
                "Στο προηγούμενο e-commerce e-shop, η βάση κράσαρε στη Black Friday. ΜΑΘΗΜΑ: Σε B2C συστήματα με υψηλό traffic, απαιτείται υποχρεωτικά Caching (π.χ. Redis).",
                "Στο εσωτερικό εργαλείο HR (B2B), βάλαμε Kubernetes αλλά το budget ήταν μικρό και το project απέτυχε οικονομικά. ΜΑΘΗΜΑ: Για low budget εσωτερικά εργαλεία, απαιτείται Serverless ή απλό VPS.",
                "Στο video streaming app, τα κόστη δικτύου στο AWS εκτοξεύτηκαν. ΜΑΘΗΜΑ: Όταν το σύστημα σερβίρει media (βίντεο/εικόνες), απαιτείται χρήση CDN."
            ],
            metadatas=[{"type": "ecommerce"}, {"type": "internal_tool"}, {"type": "media"}]
        )
        
    # --- Seeding για τους Experts (Domain Knowledge) ---
    if experts_collection.count() == 0:
        print("🌱 [ChromaDB] Η βάση 'experts' είναι άδεια. Ξεκινάει το Seeding...")
        
        experts_collection.add(
            ids=["exp_sec_1", "exp_sec_2", "exp_db_1", "exp_ai_1", "exp_ai_2", "exp_ai_3"],
            documents=[
                "OWASP & NIST Guidelines: Συστήματα με PII (GDPR) απαιτούν AES-256 Encryption. Public APIs απαιτούν WAF (Web Application Firewall).",
                "Security Trade-offs: Το WAF κοστίζει. Για αυστηρό Low Budget, προτείνεται Cloudflare Free Tier ή fail2ban σε VPS.",
                "Database Rules: Σχεσιακά δεδομένα (Παραγγελίες, Χρήστες) = PostgreSQL/MySQL. Ευέλικτα έγγραφα ή Big Data = MongoDB/NoSQL.",
                # Τα 3 "Χρυσά" δεδομένα για τον AI/ML Expert:
                "AI Strategy - Hosting: Για γρήγορο MVP προτείνεται API (OpenAI). Για προστασία δεδομένων (Privacy) ή τεράστιο scale, προτείνονται Open-Source Models (π.χ. Llama 3) σε δικούς μας servers.",
                "AI Strategy - Knowledge: Για εισαγωγή εταιρικών δεδομένων χρησιμοποιούμε ΠΑΝΤΑ RAG. Το Fine-Tuning είναι μόνο για αλλαγή ύφους και είναι ακριβό.",
                "AI Strategy - Latency: Για Real-time εφαρμογές απαιτούνται μικρά μοντέλα (π.χ. gpt-4o-mini). Για πολύπλοκο reasoning χρησιμοποιούμε μεγάλα μοντέλα (gpt-4o)."
            ],
            metadatas=[
                {"domain": "security"}, 
                {"domain": "security"}, 
                {"domain": "database"}, 
                {"domain": "ai_ml"}, 
                {"domain": "ai_ml"}, 
                {"domain": "ai_ml"}
            ]
        )
        print("✅ [ChromaDB] Το Seeding ολοκληρώθηκε επιτυχώς!")

# ==========================================
# 2. ΟΙ ΣΥΝΑΡΤΗΣΕΙΣ ΑΝΑΖΗΤΗΣΗΣ (Το RAG μας)
# ==========================================

def retrieve_historical_projects(query: str, n_results: int = 2) -> list:
    """Ψάχνει στα παλιά projects. Το καλεί ο Analyst (μέσω του memory.py)."""
    # Φροντίζουμε η βάση να έχει δεδομένα πριν ψάξουμε
    seed_database_if_empty()
    
    results = projects_collection.query(
        query_texts=[query],
        n_results=n_results
    )
    # Επιστρέφουμε τα κείμενα
    return results["documents"][0] if results["documents"] else []


def retrieve_domain_knowledge(query: str, domain: str, n_results: int = 1) -> str:
    """
    Ψάχνει στα Expert Guidelines. Το καλούν οι Experts στο experts.py!
    Χρησιμοποιεί το 'where' filter για να φέρει π.χ. ΜΟΝΟ security έγγραφα.
    """
    seed_database_if_empty()
    
    results = experts_collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"domain": domain} # Το μαγικό φίλτρο του Multi-Agent RAG!
    )
    
    docs = results["documents"][0] if results["documents"] else []
    return " ".join(docs) if docs else "Δεν βρέθηκαν ειδικές πρακτικές. Βασίσου στη γενική σου γνώση."