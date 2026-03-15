import os
import json
import uuid
from datetime import datetime
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from app.backend.graph.state import SystemDesignState


# ==========================================
# STRUCTURED OUTPUT: Critique Result
# ==========================================
class DesignIssue(BaseModel):
    component: str = Field(..., description="Which part has the issue (e.g. 'ERD', 'Tech Stack', 'Cost Table')")
    severity: str = Field(..., description="'critical' or 'minor'")
    description: str = Field(..., description="What is wrong, in one sentence")
    fix_applied: str = Field(..., description="What was corrected, in one sentence")


class DesignCritiqueResult(BaseModel):
    is_consistent: bool = Field(..., description="True if the design is internally consistent after fixes")
    issues_found: List[DesignIssue] = Field(default_factory=list, description="List of issues found and fixed")
    patched_tech_stack_summary: str = Field(..., description="The corrected tech_stack_summary (copy verbatim if no changes)")
    patched_trade_off_analysis: str = Field(..., description="The corrected trade_off_analysis (copy verbatim if no changes)")
    patched_cost_estimate_table: str = Field(..., description="The corrected cost_estimate_table (copy verbatim if no changes)")
    consistency_score: int = Field(..., ge=1, le=10, description="Overall consistency score 1-10")


# ==========================================
# SELF-LEARNING: Store lessons in ChromaDB
# ==========================================
_lessons_collection = None

def _get_lessons_collection():
    """Lazy-loads the design_lessons collection from ChromaDB."""
    global _lessons_collection
    if _lessons_collection is None:
        try:
            from app.backend.vector_store.client import get_chroma_collections
            # We reuse the client but create a third collection for lessons
            import chromadb
            from chromadb.utils import embedding_functions

            # Get the client from existing infra
            projects_col, _ = get_chroma_collections()
            client = projects_col._client

            local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            _lessons_collection = client.get_or_create_collection(
                name="design_lessons",
                embedding_function=local_ef
            )
            count = _lessons_collection.count()
            print(f"   [LEARN] design_lessons collection loaded ({count} lessons)")
        except Exception as e:
            print(f"   [LEARN] Could not load lessons collection: {e}")
            _lessons_collection = None
    return _lessons_collection


def store_lessons(issues: List[DesignIssue], project_type: str):
    """
    Converts critic findings into reusable lessons and stores them in ChromaDB.
    Each lesson is a searchable document: "When designing X, avoid Y. Instead do Z."
    """
    collection = _get_lessons_collection()
    if collection is None or not issues:
        return 0

    stored = 0
    for issue in issues:
        # Only store critical issues — minor ones are noise
        if issue.severity != "critical":
            continue

        lesson_text = (
            f"Architecture lesson for {project_type}: "
            f"In the {issue.component} component, a common mistake is: {issue.description}. "
            f"The correct approach is: {issue.fix_applied}."
        )

        doc_id = f"lesson_{uuid.uuid4().hex[:8]}"

        try:
            collection.add(
                ids=[doc_id],
                documents=[lesson_text],
                metadatas=[{
                    "source": "design_critic",
                    "component": issue.component,
                    "severity": issue.severity,
                    "project_type": project_type,
                    "created_at": datetime.now().isoformat(),
                }],
            )
            stored += 1
            print(f"   [LEARN] Stored lesson: {issue.component} — {issue.description[:60]}...")
        except Exception as e:
            print(f"   [LEARN] Failed to store lesson: {e}")

    return stored


def retrieve_lessons(query: str, n_results: int = 5) -> str:
    """
    Retrieves past lessons relevant to the current design context.
    Returns a formatted string for injection into the Designer prompt.
    """
    collection = _get_lessons_collection()
    if collection is None:
        return ""

    try:
        count = collection.count()
        if count == 0:
            return ""

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""

        lessons_text = "\n".join(f"- {doc}" for doc in docs if doc.strip())
        print(f"   [LEARN] Retrieved {len(docs)} relevant lessons from memory")
        return lessons_text

    except Exception as e:
        print(f"   [LEARN] Lesson retrieval failed: {e}")
        return ""


def get_lessons_count() -> int:
    """Returns the total number of stored lessons (for UI display)."""
    collection = _get_lessons_collection()
    if collection is None:
        return 0
    try:
        return collection.count()
    except Exception:
        return 0


# ==========================================
# NODE: Design Critic (Self-Critique + Learning)
# ==========================================
def design_critic_node(state: SystemDesignState):
    """
    Post-design validation node. Checks, patches, and LEARNS from mistakes.
    
    Phase 1: Structural pre-checks (instant, no LLM)
    Phase 2: LLM consistency audit (gpt-4o-mini, ~1.5s)
    Phase 3: Store critical issues as lessons in ChromaDB (self-learning)
    """
    print("--- [NODE] Design Critic: Self-validation & learning ---")

    design = state.get("final_design", {})
    requirements = state.get("requirements", {})
    expert_opinions = state.get("expert_opinions", [])

    # ── PHASE 1: Structural pre-checks ──
    structural_issues = []

    c4_code = design.get("mermaid_c4_code", "")
    erd_code = design.get("mermaid_erd_code", "")
    cost_table = design.get("cost_estimate_table", "")
    metrics = design.get("metrics", {})

    if not c4_code or len(c4_code.strip()) < 30:
        structural_issues.append("C4 diagram is empty or too short")
    if not erd_code or len(erd_code.strip()) < 30:
        structural_issues.append("ERD diagram is empty or too short")
    if not cost_table or "TOTAL" not in cost_table.upper():
        structural_issues.append("Cost table is missing or has no TOTAL row")
    if metrics and all(v == 5 for v in [metrics.get("cost_efficiency"), metrics.get("security_level"),
                                         metrics.get("performance_speed"), metrics.get("scalability")]):
        structural_issues.append("All metrics are exactly 5 — likely default values, not justified")

    # ── PHASE 2: LLM-powered consistency critique ──
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    ).with_structured_output(DesignCritiqueResult)

    design_summary = {
        "tech_stack_summary": design.get("tech_stack_summary", ""),
        "trade_off_analysis": design.get("trade_off_analysis", ""),
        "cost_estimate_table": design.get("cost_estimate_table", ""),
        "mermaid_c4_code": c4_code[:500],
        "mermaid_erd_code": erd_code[:500],
        "metrics": metrics,
    }

    structural_notes = "\n".join(f"- {s}" for s in structural_issues) if structural_issues else "None"

    critique_prompt = (
        "You are a Senior Architecture Reviewer performing a consistency audit.\n\n"
        "REQUIREMENTS:\n"
        f"{json.dumps(requirements, ensure_ascii=False)[:800]}\n\n"
        "EXPERT RECOMMENDATIONS:\n"
        f"{'; '.join(expert_opinions)[:500]}\n\n"
        "GENERATED DESIGN:\n"
        f"{json.dumps(design_summary, ensure_ascii=False)}\n\n"
        "STRUCTURAL PRE-CHECK ISSUES:\n"
        f"{structural_notes}\n\n"
        "VALIDATION CHECKLIST:\n"
        "1. ERD <-> Tech Stack: If PostgreSQL is chosen, ERD should have relational entities. "
        "If MongoDB is chosen, document-style is expected.\n"
        "2. Cost Table: Every technology mentioned in tech_stack_summary should appear in "
        "cost_estimate_table. If Redis is in the stack, there must be a Caching row.\n"
        "3. Metrics <-> Design: security_level >= 7 requires encryption/WAF/auth mentioned in tech stack. "
        "scalability >= 7 requires horizontal scaling/caching mentioned.\n"
        "4. Trade-offs: Must reference actual technologies from the stack, not generic statements.\n"
        "5. Completeness: tech_stack_summary must have 5+ items. cost_estimate_table must have TOTAL row.\n\n"
        "INSTRUCTIONS:\n"
        "- Find all inconsistencies.\n"
        "- For each issue, apply a FIX directly in the patched fields.\n"
        "- If a field has no issues, copy it VERBATIM to the patched version.\n"
        "- Keep all content in the SAME LANGUAGE as the original.\n"
        "- Be strict but fair. Minor formatting issues are 'minor', missing components are 'critical'.\n"
    )

    try:
        result = llm.invoke([{"role": "user", "content": critique_prompt}])

        # Apply patches
        patched_design = design.copy()
        if result.patched_tech_stack_summary and result.patched_tech_stack_summary.strip():
            patched_design["tech_stack_summary"] = result.patched_tech_stack_summary
        if result.patched_trade_off_analysis and result.patched_trade_off_analysis.strip():
            patched_design["trade_off_analysis"] = result.patched_trade_off_analysis
        if result.patched_cost_estimate_table and result.patched_cost_estimate_table.strip():
            patched_design["cost_estimate_table"] = result.patched_cost_estimate_table

        n_issues = len(result.issues_found)
        n_critical = sum(1 for i in result.issues_found if i.severity == "critical")
        print(f"   [CRITIC] Found {n_issues} issues ({n_critical} critical). Consistency: {result.consistency_score}/10")

        # ── PHASE 3: Self-learning — store critical lessons ──
        project_type = requirements.get("core_functionality", "system")
        lessons_stored = store_lessons(result.issues_found, project_type)
        total_lessons = get_lessons_count()
        print(f"   [LEARN] Stored {lessons_stored} new lessons. Total in memory: {total_lessons}")

        return {
            "final_design": patched_design,
            "final_validation": {
                "is_consistent": result.is_consistent,
                "consistency_score": result.consistency_score,
                "issues_found": [i.model_dump() for i in result.issues_found],
                "structural_pre_checks": structural_issues,
                "lessons_stored": lessons_stored,
                "total_lessons": total_lessons,
            }
        }

    except Exception as e:
        print(f"   [CRITIC] Validation failed (non-blocking): {e}")
        return {
            "final_validation": {
                "is_consistent": True,
                "consistency_score": 0,
                "issues_found": [],
                "error": str(e),
                "lessons_stored": 0,
                "total_lessons": get_lessons_count(),
            }
        }