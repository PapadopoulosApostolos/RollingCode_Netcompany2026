from openai import OpenAI
import streamlit as st
import plotly.graph_objects as go
import os
import sys
import re
import json
import uuid
import base64
from datetime import datetime
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ==========================================
# 0. PYTHONPATH FIX
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# ==========================================
# 0.5. ENV VARS & SESSION STATE INIT
# ==========================================
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

# --- ΜΗΧΑΝΙΣΜΟΣ ΜΟΝΙΜΟΥ ΙΣΤΟΡΙΚΟΥ (SQLite) ---
from app.backend.vector_store.history_db import (
    load_history, save_project, update_project, delete_project,
    migrate_from_json, get_project_count
)

# One-time migration from old JSON file (if it exists)
_old_json_path = os.path.join(project_root, "projects_history.json")
migrate_from_json(_old_json_path)


def delete_history_entry(index):
    """Διαγράφει μια καταγραφή από το ιστορικό (session state + SQLite)."""
    history = st.session_state.get('project_history', [])
    if 0 <= index < len(history):
        project_id = history[index].get('id')
        if project_id:
            delete_project(project_id)
        history.pop(index)
        st.session_state['project_history'] = history
        current_idx = st.session_state.get('current_project_index', -1)
        if current_idx == index:
            st.session_state['step'] = 'select_type'
            st.session_state['final_design'] = None
            st.session_state['current_project_index'] = -1
        elif current_idx > index:
            st.session_state['current_project_index'] = current_idx - 1


def generate_project_title(prompt_text, project_type, max_chars=45):
    """
    Εξάγει ένα περιγραφικό τίτλο από το prompt χωρίς API call.
    Στρατηγική:
      1. Παίρνει το free-text block (ΓΕΝΙΚΗ ΠΕΡΙΓΡΑΦΗ) αν υπάρχει
      2. Καθαρίζει boilerplate λέξεις/headers
      3. Κρατάει τις πρώτες ουσιαστικές λέξεις
      4. Fallback: project_type + αύξων αριθμός
    """
    text = prompt_text or ""
    
    # 1. Εξαγωγή του free-text block μεταξύ ΓΕΝΙΚΗ ΠΕΡΙΓΡΑΦΗ και επόμενου section
    desc_match = re.search(
        r'---\s*ΓΕΝΙΚΗ ΠΕΡΙΓΡΑΦΗ\s*---\s*\n(.+?)(?:\n\s*---|$)',
        text, re.DOTALL
    )
    if desc_match:
        text = desc_match.group(1).strip()
    else:
        # Fallback: πάρε ό,τι είναι πριν τις ΣΤΑΤΙΚΕΣ ΠΡΟΔΙΑΓΡΑΦΕΣ
        parts = re.split(r'---\s*ΣΤΑΤΙΚ', text, maxsplit=1)
        text = parts[0].strip()
    
    # 2. Αφαίρεση boilerplate γραμμών (Τύπος Συστήματος:, headers, κενές)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Αγνόησε headers και metadata γραμμές
        if re.match(r'^(Τύπος Συστήματος|---|\*\*|##|#)', s):
            continue
        # Αγνόησε bullet-style γραμμές (είναι static answers)
        if re.match(r'^-\s+', s):
            continue
        cleaned_lines.append(s)
    
    text = " ".join(cleaned_lines).strip()
    
    # 3. Πάρε την πρώτη πρόταση (μέχρι τελεία, κόμμα ή παρένθεση)
    first_sentence = re.split(r'[.!;\n]', text)[0].strip()
    
    if not first_sentence or len(first_sentence) < 3:
        # Fallback: χρησιμοποίησε τον project_type
        return project_type or "Νέο Έργο"
    
    # 4. Truncate αν χρειαστεί, κόβοντας σε λέξη
    if len(first_sentence) > max_chars:
        truncated = first_sentence[:max_chars].rsplit(' ', 1)[0]
        first_sentence = truncated.rstrip(',.;:') + "…"
    
    return first_sentence


# State Machine της Εφαρμογής (select_type -> static_input -> dynamic_wizard -> results)
if 'step' not in st.session_state:
    st.session_state['step'] = 'select_type'
if 'project_type' not in st.session_state:
    st.session_state['project_type'] = None
if 'clarification_wizard' not in st.session_state:
    st.session_state['clarification_wizard'] = None
if 'working_prompt' not in st.session_state:
    st.session_state['working_prompt'] = ""
if 'final_design' not in st.session_state:
    st.session_state['final_design'] = None

# --- ΦΟΡΤΩΣΗ ΙΣΤΟΡΙΚΟΥ ΑΠΟ ΤΟ ΑΡΧΕΙΟ ---
if 'project_history' not in st.session_state:
    st.session_state['project_history'] = load_history()
if 'current_project_index' not in st.session_state:
    st.session_state['current_project_index'] = -1

# --- State για το Personal Context Chatbot & Review ---
if 'context_step' not in st.session_state:
    st.session_state['context_step'] = 'chat'
if 'context_conclusions' not in st.session_state:
    st.session_state['context_conclusions'] = []
if 'editing_index' not in st.session_state:
    st.session_state['editing_index'] = None
if 'context_chat_history' not in st.session_state:
    st.session_state['context_chat_history'] = [
        {"role": "assistant", "content": "Γεια σας. Είμαι εδώ για να καταγράψω τις εμπειρίες σας. Περιγράψτε μου ένα προηγούμενο έργο σας (π.χ. τι αφορούσε, τι τεχνολογίες χρησιμοποιήσατε και ποιες ήταν οι μεγαλύτερες προκλήσεις)."}
    ]

# ── NEW: State για User Iteration Chat ──
if 'iteration_chat_history' not in st.session_state:
    st.session_state['iteration_chat_history'] = []
# Flag used to show the user's message first, then generate the assistant response on the next rerun.
if 'iteration_pending_response' not in st.session_state:
    st.session_state['iteration_pending_response'] = False
# Flag to show banner when architecture was updated via chat
if 'design_just_updated' not in st.session_state:
    st.session_state['design_just_updated'] = False

# --- State για pending delete ---
if 'pending_delete_index' not in st.session_state:
    st.session_state['pending_delete_index'] = None

# ==========================================
# 1. LAZY LOADING BACKEND (OPTIMIZATION)
# ==========================================
@st.cache_resource
def get_app_graph():
    """Φορτώνει το backend μόνο όταν ζητηθεί και το κρατάει στη μνήμη."""
    from app.backend.graph.workflow import app_graph
    return app_graph

@st.cache_resource
def get_vector_db():
    """Φορτώνει τη σύνδεση με το ChromaDB."""
    try:
        from app.backend.vector_store.client import get_chroma_collections
        return get_chroma_collections()
    except Exception as e:
        print(f"[ChromaDB] Connection failed: {e}")
        return None, None

# Λεξικό με τις Στατικές Ερωτήσεις ανά τύπο έργου
STATIC_QUESTIONS = {
    "Web Application": [
        {"id": "audience", "label": "Ποιο είναι το βασικό κοινό (Target Audience);", "type": "select",
         "options": ["B2C (Καταναλωτές)", "B2B (Επιχειρήσεις)", "Internal (Υπάλληλοι)", "B2B2C (Marketplace)","Public Information / Blog"]},
        {"id": "frontend_framework", "label": "Προτίμηση Frontend Framework;", "type": "select",
         "options": ["React / Next.js", "Vue / Nuxt", "Angular", "Server-rendered (Django/Rails templates)", "System Recommendation"]},
        {"id": "seo", "label": "Είναι το SEO κρίσιμο για την εφαρμογή;", "type": "checkbox"},
        {"id": "user_generated_content", "label": "Θα υπάρχει user-generated content (reviews / posts / uploads);", "type": "checkbox"},
        {"id": "expected_concurrent", "label": "Μέγιστοι ταυτόχρονοι χρήστες (Peak);", "type": "select",
         "options": ["< 100", "100 - 1.000", "1.000 - 10.000", "> 10.000"]},
    ],
    "Microservice Architecture": [
        {"id": "protocol", "label": "Βασικό Πρωτόκολλο Επικοινωνίας;", "type": "select",
         "options": ["REST API", "gRPC", "Event-Driven (Kafka/RabbitMQ)", "GraphQL Federation"]},
        {"id": "db_strategy", "label": "Στρατηγική Βάσης Δεδομένων;", "type": "select",
         "options": ["Κοινή Βάση Δεδομένων (Shared)", "Βάση ανά Υπηρεσία (Database-per-service)", "Polyglot Persistence (μικτή)"]},
        {"id": "cloud_native", "label": "Απαιτείται πλήρης Cloud-Native (Kubernetes) ανάπτυξη;", "type": "checkbox"},
        {"id": "existing_infra", "label": "Υπάρχει ήδη υποδομή που πρέπει να ενσωματωθεί;", "type": "checkbox"},
        {"id": "team_size", "label": "Μέγεθος ομάδας ανάπτυξης;", "type": "select",
         "options": ["1-3 developers", "4-8 developers", "9+ developers (πολλαπλά teams)"]},
    ],
    "Data/ML Pipeline": [
        {"id": "ingestion", "label": "Ρυθμός εισαγωγής δεδομένων;", "type": "select",
         "options": ["Μαζική Επεξεργασία (Batch - Ανά ώρες/μέρες)", "Πραγματικός Χρόνος (Real-time / Streaming)", "Hybrid (Batch + Real-time)"]},
        {"id": "data_type", "label": "Κύριος τύπος δεδομένων;", "type": "select",
         "options": ["Δομημένα (SQL / Tables)", "Μη Δομημένα (JSON / Αρχεία / Εικόνες)", "Χρονοσειρές (Time-series / Logs)", "Μικτά"]},
        {"id": "ml_training", "label": "Θα περιλαμβάνει εκπαίδευση ML μοντέλων (Training);", "type": "checkbox"},
        {"id": "data_governance", "label": "Απαιτείται data lineage / data catalog;", "type": "checkbox"},
        {"id": "output_consumers", "label": "Ποιος καταναλώνει τα αποτελέσματα;", "type": "select",
         "options": ["Dashboards / BI (Tableau / Looker)", "APIs / Applications", "Data Scientists (Notebooks)", "Εξωτερικοί Partners / Exports"]},
    ]
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def _sanitize_mermaid(code: str) -> str:
    code = code.replace('```mermaid', '').replace('```', '').strip()
    code = code.replace('\\n', '\n')
    code = code.replace('|||', '\n')
    code = re.sub(r'(?<![-.])\|\|(?![-.])', '\n', code)
    NON_GRAPH_KEYWORDS = [
        'C4Context', 'C4Container', 'C4Component', 'C4Dynamic', 'C4Deployment',
        'erDiagram', 'sequenceDiagram', 'classDiagram', 'flowchart',
    ]
    for kw in NON_GRAPH_KEYWORDS:
        code = re.sub(re.escape(kw) + r'([^\s\n])', kw + r'\n\1', code)
    ALL_KWS = NON_GRAPH_KEYWORDS + ['graph']
    for kw in ALL_KWS:
        code = re.sub(
            r'(' + re.escape(kw) + r'[^\n]*\n)(?:' + re.escape(kw) + r'[^\n]*\n)+',
            r'\1', code,
        )
    code = re.sub(
        r'(?m)^(C4\w+|erDiagram|sequenceDiagram|classDiagram|flowchart|graph\s*\w*)'
        r'\s+[Dd]iagram\b', r'\1', code)
    code = re.sub(r'(?m)^\s*[Dd]iagram\s*$', '', code)
    SOLO_KEYWORDS = [
        'C4Context', 'C4Container', 'C4Component', 'C4Dynamic', 'C4Deployment',
        'erDiagram',
    ]
    for kw in SOLO_KEYWORDS:
        code = re.sub(
            r'(?m)^(' + re.escape(kw) + r')[ \t]+(\S)',
            r'\1\n\2',
            code,
        )
    GRAPH_DIRS = {'TD', 'LR', 'TB', 'RL', 'BT', 'UD', 'DT'}
    def _fix_graph(m):
        d, rest = m.group(1).strip(), m.group(2)
        if d.upper() in GRAPH_DIRS:
            return f'graph {d}\n{rest}' if rest.strip() else f'graph {d}'
        return f'graph TD\n{d}{rest}'
    code = re.sub(r'(?m)^graph\s+(\w+)(.*)', _fix_graph, code)
    if 'erDiagram' in code:
        def _card_sym(c):
            c = c.strip()
            if re.fullmatch(r'1\.\.1|1',            c): return '||'
            if re.fullmatch(r'0\.\.1',               c): return '|o'
            if re.fullmatch(r'1\.\.\*|1\.\.n',       c): return '|{'
            if re.fullmatch(r'0\.\.\*|0\.\.n|\*|n',  c): return 'o{'
            return c
        CARD_PAT = r'(\w+)\s+([\d\*n][\.0-9\*n]*)\s+[-\.]{2,}\s+([\d\*n][\.0-9\*n]*)\s+(\w+)'
        code = re.sub(CARD_PAT,
            lambda m: f'{m.group(1)} {_card_sym(m.group(2))}--{_card_sym(m.group(3))} {m.group(4)}',
            code)
        code = re.sub(
            r'(?m)^(\s*)(\w+)\s+([-\.]{2,})([|o{]{1,2})\s+(\w+)',
            r'\1\2 ||\3\4 \5', code,
        )
        def _quote_label(m):
            label = m.group(2).strip().strip('`\u201c\u201d\u2018\u2019\'"')
            return f'{m.group(1)}"{label}"'
        code = re.sub(
            r'([|o}]{1,2}[-\.]+[|o{]{1,2}\s+\w+\s*:[ \t]*)(?![ \t]*")([^\n"]+)',
            _quote_label, code,
        )
    C4_ELEMS = (r'(?:Person_Ext|PersonExt|Person|SystemExt|SystemDb|System'
                r'|ContainerDb|Container|Component|Rel_Back|BiRel|Rel|title)')
    code = re.sub(r'\)(' + C4_ELEMS + r')\(', r')\n\1(', code)
    code = re.sub(
        r'^\s*(?:Container_Boundary|System_Boundary|Enterprise_Boundary|Boundary)'
        r'\s*\([^)\n]*\)[ \t]*\{?[ \t]*',
        '', code, flags=re.MULTILINE,
    )
    # FIX: Remove orphaned } ONLY for non-ERD diagrams.
    if 'erDiagram' not in code:
        code = re.sub(r'^\s*\}\s*$', '', code, flags=re.MULTILINE)
    stripped = code.strip()
    known_starts = (
        'graph ', 'graph\n', 'flowchart ', 'erDiagram', 'sequenceDiagram',
        'classDiagram', 'C4Context', 'C4Container', 'C4Component',
        'C4Dynamic', 'C4Deployment',
    )
    if not any(stripped.startswith(k) for k in known_starts):
        if any(x in stripped for x in ('Person(', 'Container(', 'System(', 'Rel(')):
            code = 'C4Context\n' + stripped
        else:
            code = 'graph TD\n' + stripped
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()


def render_mermaid(code):
    if not code or not str(code).strip() or str(code).strip().lower() == "none":
        return st.info("Δεν υπάρχει διαθέσιμο διάγραμμα.")
    clean_code = _sanitize_mermaid(str(code))
    if len(clean_code) < 5:
        return st.info("Το παραγόμενο διάγραμμα ήταν κενό ή πολύ μικρό.")
    js_safe_code = clean_code.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    # Escape the raw code for safe HTML display inside the fallback
    _raw_escaped = clean_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{
                startOnLoad: false,
                theme: 'default',
                flowchart: {{ curve: 'linear' }},
                er: {{ diagramPadding: 20, layoutDirection: 'TB' }}
            }});
            window.onload = async function() {{
                const graphDefinition = `{js_safe_code}`;
                const container = document.getElementById('mermaid-container');
                try {{
                    const {{ svg }} = await mermaid.render('mermaid-svg', graphDefinition);
                    container.innerHTML = svg;
                }} catch (error) {{
                    container.innerHTML = `
                        <div style="padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
                            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                                        border-radius:10px;padding:16px 20px;margin-bottom:12px;">
                                <div style="font-size:13px;font-weight:600;color:#fbbf24;margin-bottom:6px;">
                                    Diagram rendering failed
                                </div>
                                <div style="font-size:12px;color:#94a3b8;line-height:1.5;">
                                    The AI generated syntax that Mermaid.js could not parse.
                                    The raw code is shown below for reference.
                                </div>
                            </div>
                            <pre style="background:#0f172a;border:1px solid rgba(255,255,255,0.08);
                                        border-radius:8px;padding:16px;font-size:12px;color:#94a3b8;
                                        font-family:'Fira Code',Consolas,monospace;
                                        overflow-x:auto;white-space:pre-wrap;line-height:1.6;
                                        max-height:400px;overflow-y:auto;">{_raw_escaped}</pre>
                        </div>`;
                }}
            }};
        </script>
    </head>
    <body style="background-color:white;margin:0;padding:20px;font-family:sans-serif;">
        <div id="mermaid-container" style="display:flex;justify-content:center;width:100%;">
            <span style="color:gray;">Rendering diagram...</span>
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=600, scrolling=True)
    with st.expander("Κώδικας Mermaid"):
        st.code(clean_code)


# ==========================================
# EXPORT DOSSIER — Markdown + HTML/PDF
# ==========================================

def generate_markdown_dossier(design: dict, project_type: str, prompt: str) -> str:
    """Compiles the full architecture dossier into a clean Markdown document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    m = design.get("metrics", {})

    md = f"""# Architecture Dossier
**Generated:** {now}  
**Project Type:** {project_type}

---

## 1. Requirements Summary

{prompt.strip() if prompt else "N/A"}

---

## 2. C4 Context Diagram

```mermaid
{design.get("mermaid_c4_code", "N/A")}
```

---

## 3. Entity-Relationship Diagram

```mermaid
{design.get("mermaid_erd_code", "N/A")}
```

---

## 4. Technology Stack

{design.get("tech_stack_summary", "N/A")}

---

## 5. Trade-off Analysis

{design.get("trade_off_analysis", "N/A")}

---

## 6. Scaling Roadmap

{design.get("future_scaling_path", "N/A")}

---

## 7. Architecture Metrics

| Metric | Score |
|--------|-------|
| Cost Efficiency | {m.get("cost_efficiency", "N/A")}/10 |
| Security Level | {m.get("security_level", "N/A")}/10 |
| Performance | {m.get("performance_speed", "N/A")}/10 |
| Scalability | {m.get("scalability", "N/A")}/10 |

---

## 8. Cost Estimation

{design.get("cost_estimate_table", "N/A")}

---

*Generated by Enterprise AI Architect*
"""
    return md


def generate_html_dossier(design: dict, project_type: str, prompt: str) -> str:
    """Generates a print-ready HTML document with rendered Mermaid diagrams and proper tables."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    m = design.get("metrics", {})

    def _md_to_html(text):
        """Converts markdown text to HTML (bold, bullets, tables)."""
        if not text:
            return ""
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        lines = text.split('\n')
        html_lines = []
        table_rows = []

        def _flush_table():
            if not table_rows:
                return
            html_lines.append('<table>')
            for ri, row in enumerate(table_rows):
                cells = [c.strip() for c in row.strip().strip('|').split('|')]
                # Skip separator rows (|---|---|)
                if all(set(c.strip()) <= set('-: ') for c in cells):
                    continue
                tag = 'th' if ri == 0 else 'td'
                html_lines.append('<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>')
            html_lines.append('</table>')
            table_rows.clear()

        for line in lines:
            s = line.strip()
            if s.startswith('|'):
                table_rows.append(s)
            else:
                _flush_table()
                if s.startswith('- '):
                    html_lines.append(f'<li>{s[2:]}</li>')
                elif s:
                    html_lines.append(f'<p>{s}</p>')
        _flush_table()

        result = '\n'.join(html_lines)
        result = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ul>{m.group(0)}</ul>', result)
        return result

    tech_stack = _md_to_html(design.get("tech_stack_summary", ""))
    trade_offs = _md_to_html(design.get("trade_off_analysis", ""))
    scaling = _md_to_html(design.get("future_scaling_path", ""))
    cost_table_html = _md_to_html(design.get("cost_estimate_table", "N/A"))

    # Escape mermaid code for safe JS embedding
    c4_code = design.get("mermaid_c4_code", "").replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    erd_code = design.get("mermaid_erd_code", "").replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Architecture Dossier — {project_type}</title>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: false, theme: 'default', flowchart: {{ curve: 'linear' }} }});

  async function renderDiagram(containerId, code) {{
    const el = document.getElementById(containerId);
    if (!code || code.trim().length < 10) {{
      el.innerHTML = '<p style="color:#94a3b8;font-style:italic;">No diagram available</p>';
      return;
    }}
    try {{
      const {{ svg }} = await mermaid.render(containerId + '-svg', code);
      el.innerHTML = svg;
    }} catch(e) {{
      el.innerHTML = '<pre style="color:#dc2626;font-size:12px;">Render error: ' + e.message + '</pre>';
    }}
  }}

  window.addEventListener('load', async () => {{
    await renderDiagram('c4-diagram', `{c4_code}`);
    await renderDiagram('erd-diagram', `{erd_code}`);
  }});
</script>
<style>
  @media print {{
    body {{ margin: 0; padding: 20px; }}
    .no-print {{ display: none; }}
    .page-break {{ page-break-before: always; }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 32px;
    color: #1e293b;
    line-height: 1.7;
    font-size: 14px;
  }}
  h1 {{
    font-size: 28px;
    font-weight: 700;
    border-bottom: 3px solid #2563eb;
    padding-bottom: 12px;
    margin-bottom: 8px;
  }}
  h2 {{
    font-size: 18px;
    font-weight: 600;
    color: #1e40af;
    margin-top: 36px;
    margin-bottom: 12px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
  }}
  .meta {{
    color: #64748b;
    font-size: 13px;
    margin-bottom: 28px;
  }}
  .diagram-container {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px;
    margin: 12px 0;
    min-height: 100px;
    display: flex;
    justify-content: center;
    overflow-x: auto;
  }}
  .diagram-container svg {{
    max-width: 100%;
    height: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 13px;
  }}
  th, td {{
    border: 1px solid #e2e8f0;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{
    background: #f1f5f9;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background: #f8fafc;
  }}
  ul {{ padding-left: 24px; }}
  li {{ margin-bottom: 6px; }}
  strong {{ color: #1e40af; }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
  }}
  .metric-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }}
  .metric-score {{
    font-size: 28px;
    font-weight: 700;
    color: #2563eb;
  }}
  .metric-label {{
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
  }}
  .print-btn {{
    position: fixed;
    top: 20px;
    right: 20px;
    background: #2563eb;
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    font-weight: 600;
    z-index: 100;
  }}
  .print-btn:hover {{ background: #1d4ed8; }}
</style>
</head>
<body>
<button class="print-btn no-print" onclick="window.print()">Save as PDF</button>

<h1>Architecture Dossier</h1>
<div class="meta">
  Generated: {now} &nbsp;|&nbsp; Project Type: {project_type}
</div>

<h2>1. Requirements Summary</h2>
{_md_to_html(prompt) if prompt else '<p>N/A</p>'}

<h2>2. C4 Context Diagram</h2>
<div class="diagram-container" id="c4-diagram">
  <p style="color:#94a3b8;">Rendering diagram...</p>
</div>

<h2>3. Entity-Relationship Diagram</h2>
<div class="diagram-container" id="erd-diagram">
  <p style="color:#94a3b8;">Rendering diagram...</p>
</div>

<div class="page-break"></div>

<h2>4. Technology Stack</h2>
{tech_stack}

<h2>5. Trade-off Analysis</h2>
{trade_offs}

<h2>6. Scaling Roadmap</h2>
{scaling}

<h2>7. Architecture Metrics</h2>
<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-score">{m.get("cost_efficiency", "?")}</div>
    <div class="metric-label">Cost Efficiency</div>
  </div>
  <div class="metric-card">
    <div class="metric-score">{m.get("security_level", "?")}</div>
    <div class="metric-label">Security</div>
  </div>
  <div class="metric-card">
    <div class="metric-score">{m.get("performance_speed", "?")}</div>
    <div class="metric-label">Performance</div>
  </div>
  <div class="metric-card">
    <div class="metric-score">{m.get("scalability", "?")}</div>
    <div class="metric-label">Scalability</div>
  </div>
</div>

<h2>8. Cost Estimation</h2>
{cost_table_html}

<hr style="margin-top:40px;border-color:#e2e8f0;">
<p style="color:#94a3b8;font-size:12px;text-align:center;">
  Generated by Enterprise AI Architect
</p>
</body>
</html>"""
    return html


def run_architecture_graph(prompt_text, budget, load, security):
    # Node-to-label mapping for live progress
    NODE_PROGRESS = {
        "requirement_analyst": (0.12, "Analyzing requirements..."),
        "initial_validator":   (0.25, "Validating completeness..."),
        "memory":              (0.40, "Retrieving knowledge base (RAG)..."),
        "technical_committee": (0.55, "Consulting expert committee..."),
        "system_designer":     (0.80, "Generating architecture dossier..."),
        "design_critic":       (0.95, "Self-critique & consistency check..."),
    }

    with st.status("Σχεδιασμός σε εξέλιξη...", expanded=True) as status:
        try:
            app_graph = get_app_graph()
        except Exception as e:
            st.error(f"Failed to load the architecture pipeline. Check your installation.\n\n`{e}`")
            return

        initial_state = {
            "user_prompt": prompt_text,
            "user_form_data": {"budget": budget, "load": load, "security": security},
            "loop_count": 0, "expert_opinions": [], "historical_context": []
        }

        progress_bar = st.progress(0, text="Initializing pipeline...")
        final_state = initial_state.copy()

        try:
            # Stream: LangGraph yields {node_name: state_update} after each node
            for event in app_graph.stream(initial_state):
                for node_name, state_update in event.items():
                    if node_name in NODE_PROGRESS:
                        pct, label = NODE_PROGRESS[node_name]
                        progress_bar.progress(pct, text=label)
                    if isinstance(state_update, dict):
                        final_state.update(state_update)
        except Exception as e:
            progress_bar.empty()
            err_msg = str(e)
            if "api_key" in err_msg.lower() or "authentication" in err_msg.lower() or "401" in err_msg:
                st.error("OpenAI API key is invalid or missing. Check your `.env` file.")
            elif "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                st.error("The request timed out. The OpenAI API may be slow. Please try again.")
            elif "rate limit" in err_msg.lower() or "429" in err_msg:
                st.error("API rate limit reached. Wait a minute and try again.")
            else:
                st.error(f"Architecture generation failed.\n\n`{err_msg}`")
            status.update(label="Generation failed.", state="error", expanded=False)
            return

        progress_bar.progress(1.0, text="Complete.")
        progress_bar.empty()

        # Process results
        init_val = final_state.get("initial_validation", {})
        final_design = final_state.get("final_design", {})
        has_design = bool(final_design)
        needs_clarification = init_val.get("needs_clarification", False)
        if needs_clarification or (init_val.get("is_valid") is False and not has_design):
            st.session_state['step'] = 'dynamic_wizard'
            st.session_state['clarification_wizard'] = init_val.get("wizard")
            status.update(label="Η 1η φάση ανάλυσης ολοκληρώθηκε.", state="complete", expanded=False)
            st.rerun()
        elif not has_design:
            st.session_state['step'] = 'dynamic_wizard'
            st.session_state['clarification_wizard'] = {
                "essential": [
                    {"id": "fb_goal", "label": "Ποιος είναι ο κύριος επιχειρηματικός στόχος αυτού του συστήματος;", "field_type": "text", "required": True},
                    {"id": "fb_users", "label": "Ποιοι είναι οι τελικοί χρήστες και τι ενέργειες κάνουν μέσα στην εφαρμογή;", "field_type": "text", "required": True}
                ],
                "recommended": [], "optional": []
            }
            status.update(label="Προετοιμασία ερωτήσεων εξειδίκευσης...", state="complete", expanded=False)
            st.rerun()
        else:
            st.session_state['step'] = 'results'
            st.session_state['final_design'] = final_design
            st.session_state['final_validation'] = final_state.get("final_validation", {})
            st.session_state['iteration_chat_history'] = []
            st.session_state['design_just_updated'] = False
            
            if not st.session_state['project_history'] or st.session_state['project_history'][-1].get('design') != final_design:
                title = generate_project_title(
                    st.session_state.get('working_prompt', ''),
                    st.session_state['project_type']
                )
                new_project = {
                    "title": title,
                    "type": st.session_state['project_type'],
                    "prompt": st.session_state.get('working_prompt', ''),
                    "design": final_design,
                    "chat_history": []
                }
                new_id = save_project(new_project)
                new_project["id"] = new_id
                st.session_state['project_history'].append(new_project)
            
            st.session_state['current_project_index'] = len(st.session_state['project_history']) - 1
            status.update(label="Ο σχεδιασμός ολοκληρώθηκε.", state="complete", expanded=False)
            st.rerun()

# ==========================================
# 2.5 MODAL GATHERING & REVIEW PERSONAL CONTEXT (RAG)
# ==========================================
class ExtractedConclusions(BaseModel):
    conclusions: List[str] = Field(
        ...,
        description="Μια λίστα με ξεχωριστά, αυστηρά τεχνικά συμπεράσματα."
    )


@st.dialog("Προσθήκη Προσωπικής Εμπειρίας", width="large")
def personal_context_modal():
    main_placeholder = st.empty()

    def render_ui():
        with main_placeholder.container():
            step = st.session_state.get('context_step', 'chat')
            if step == 'chat':
                st.markdown("Συνομιλήστε με τον Agent για να καταγράψει τις εμπειρίες σας. Όταν τελειώσετε, πατήστε **Εξαγωγή Συμπερασμάτων**.")
                st.divider()
                chat_container = st.container(height=200)
                with chat_container:
                    for msg in st.session_state['context_chat_history']:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                if prompt := st.chat_input("Περιγράψτε το έργο σας, τις δυσκολίες ή τις λύσεις που βρήκατε..."):
                    st.session_state['context_chat_history'].append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.spinner("Ο Agent επεξεργάζεται..."):
                            try:
                                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=os.environ.get("OPENAI_API_KEY"))
                                sys_prompt = {
                                    "role": "system",
                                    "content": (
                                        "Είσαι ένας Senior Technical Interviewer. Ο χρήστης σου περιγράφει εμπειρίες από παλιά IT projects. "
                                        "Ο στόχος σου είναι να αντλήσεις χρήσιμα αρχιτεκτονικά μαθήματα (lessons learned) και stack details. "
                                        "Κάνε 1 ή το πολύ 2 σύντομες, στοχευμένες ερωτήσεις κάθε φορά. Να είσαι φιλικός και σύντομος. (Μίλα Ελληνικά)."
                                    )
                                }
                                response = llm.invoke([sys_prompt] + st.session_state['context_chat_history'])
                                st.session_state['context_chat_history'].append({"role": "assistant", "content": response.content})
                            except Exception as e:
                                err_reply = f"Σφάλμα σύνδεσης με το AI: {type(e).__name__}"
                                st.session_state['context_chat_history'].append({"role": "assistant", "content": err_reply})
                        with st.chat_message("assistant"):
                            st.markdown(response.content)
                st.divider()
                def extract_conclusions_callback():
                    with st.spinner("Ανάλυση της συζήτησης και εξαγωγή δεδομένων..."):
                        chat_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state['context_chat_history']])
                        summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.environ.get("OPENAI_API_KEY")).with_structured_output(ExtractedConclusions)
                        summary_prompt = f"""
                        Διάβασε την παρακάτω τεχνική συζήτηση.
                        Εξήγαγε ΜΟΝΟ την ουσία σε μια μικρή λίστα από ανεξάρτητα συμπεράσματα.
                        ΑΥΣΤΗΡΟΙ ΚΑΝΟΝΕΣ ΠΟΙΟΤΗΤΑΣ:
                        1. ΑΠΑΓΟΡΕΥΕΤΑΙ ΡΗΤΑ η χρήση προσωπικών αντωνυμιών ή συναισθηματικών/υποκειμενικών εκφράσεων.
                        2. Γράψε τα συμπεράσματα ΕΝΤΕΛΩΣ ΑΠΡΟΣΩΠΑ και αυστηρά τεχνικά.
                        3. Κάθε συμπέρασμα να είναι μια ξεκάθαρη, επαγγελματική πρόταση-αξίωμα.
                        ΣΥΖΗΤΗΣΗ:
                        {chat_text}
                        """
                        extracted = summary_llm.invoke([{"role": "user", "content": summary_prompt}])
                        st.session_state['context_conclusions'] = extracted.conclusions
                        st.session_state['context_step'] = 'review'
                if len(st.session_state['context_chat_history']) > 1:
                    st.button("Εξαγωγή Συμπερασμάτων", on_click=extract_conclusions_callback, type="primary", use_container_width=True)
            elif step == 'review':
                st.markdown("### Επαλήθευση Τεχνικών Γεγονότων")
                st.markdown("<p style='font-size: 0.9rem; color: gray; margin-top:-10px;'>Το AI εξήγαγε τα παρακάτω αντικειμενικά συμπεράσματα. Αν κάποιο είναι ανακριβές, πατήστε <b>Διόρθωση</b>.</p>", unsafe_allow_html=True)
                def go_to_edit_callback(index):
                    st.session_state['editing_index'] = index
                    st.session_state['context_step'] = 'edit'
                for i, conc in enumerate(st.session_state['context_conclusions']):
                    with st.container(border=True):
                        col1, col2 = st.columns([5, 1], vertical_alignment="center")
                        col1.markdown(f"<span style='font-size:0.95rem;'>{conc}</span>", unsafe_allow_html=True)
                        col2.button("Διόρθωση", key=f"edit_btn_{i}", on_click=go_to_edit_callback, args=(i,), use_container_width=True)
                st.divider()
                if st.button("Οριστική Ενσωμάτωση στη Βάση Γνώσης", type="primary", use_container_width=True):
                    with st.spinner("Ενσωμάτωση στο ChromaDB..."):
                        try:
                            projects_col, _ = get_vector_db()
                            if projects_col is None:
                                st.error("Knowledge Base is not available. ChromaDB connection failed.")
                            else:
                                doc_id = "user_exp_" + uuid.uuid4().hex[:8]
                                final_knowledge = "\n".join([f"- {c}" for c in st.session_state['context_conclusions']])
                                projects_col.add(
                                    documents=[final_knowledge],
                                    metadatas=[{"source": "user_interview", "type": "lessons_learned"}],
                                    ids=[doc_id]
                                )
                                st.success("Η τεχνογνωσία επαληθεύτηκε και ενσωματώθηκε.")
                        except Exception as e:
                            st.error(f"Σφάλμα κατά την αποθήκευση: {e}")
                        st.session_state['context_step'] = 'chat'
                        st.session_state['context_conclusions'] = []
                        st.session_state['context_chat_history'] = [
                            {"role": "assistant", "content": "Γεια σας. Είμαι εδώ για να καταγράψω τις εμπειρίες σας. Περιγράψτε μου ένα προηγούμενο έργο σας."}
                        ]
                    st.rerun()
            elif step == 'edit':
                st.markdown("### Διόρθωση Δεδομένου")
                idx = st.session_state['editing_index']
                current_conc = st.session_state['context_conclusions'][idx]
                st.info(current_conc, )
                user_correction = st.text_input("Γράψτε τι συνέβη πραγματικά:", key="correction_input", placeholder="π.χ. Η καθυστέρηση δεν οφειλόταν στη βάση, αλλά στο API...")
                st.write("")
                col1, col2, _ = st.columns([1, 1, 2])
                def handle_cancel_callback():
                    st.session_state['context_step'] = 'review'
                    if 'correction_input' in st.session_state:
                        del st.session_state['correction_input']
                def handle_save_callback():
                    correction = st.session_state.get('correction_input', '')
                    if correction.strip():
                        with st.spinner("Αναδιατύπωση γεγονότος..."):
                            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.environ.get("OPENAI_API_KEY"))
                            rewrite_prompt = f"ΑΡΧΙΚΟ ΓΕΓΟΝΟΣ: '{current_conc}'\nΔΙΟΡΘΩΣΗ ΧΡΗΣΤΗ: '{correction}'\nΞαναγράψε το γεγονός ενσωματώνοντας τη διόρθωση. ΚΑΝΟΝΑΣ: Γράψε το ΕΝΤΕΛΩΣ ΑΠΡΟΣΩΠΑ και ΑΥΣΤΗΡΑ ΤΕΧΝΙΚΑ. Δώσε ΜΟΝΟ το νέο κείμενο."
                            new_conc = llm.invoke([{"role": "user", "content": rewrite_prompt}]).content
                            st.session_state['context_conclusions'][idx] = new_conc
                            st.session_state['context_step'] = 'review'
                            if 'correction_input' in st.session_state:
                                del st.session_state['correction_input']
                    else:
                        st.session_state['context_edit_error'] = "Παρακαλώ συμπληρώστε τη διόρθωση."
                col1.button("Ακύρωση", on_click=handle_cancel_callback, use_container_width=True)
                col2.button("Αποθήκευση", type="primary", on_click=handle_save_callback, use_container_width=True)
                if st.session_state.get('context_edit_error'):
                    st.error(st.session_state['context_edit_error'])
                    del st.session_state['context_edit_error']
    render_ui()


# ==========================================
# 2.7 USER ITERATION — BACKEND HELPERS
# ==========================================

class ChatFastResponse(BaseModel):
    reply: str = Field(description="Η απάντηση προς τον χρήστη (στα Ελληνικά).")
    needs_update: bool = Field(description="True ΜΟΝΟ αν ο χρήστης ζητά ρητά αλλαγή στην αρχιτεκτονική. False για απλές ερωτήσεις.")
    change_summary: str = Field(description="Αν needs_update=True, γράψε την τεχνική αλλαγή στα Αγγλικά. Αλλιώς άδειο string.")

def _get_iteration_response(design_dict, chat_history):
    print("\n--- [DEBUG] Ξεκινάει η _get_iteration_response ---")
    
    # 1. Χρησιμοποιούμε τον ΕΠΙΣΗΜΟ, καθαρό client του OpenAI (ΌΧΙ το LangChain)
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print("[DEBUG] Ο OpenAI client δημιουργήθηκε επιτυχώς.")
    except Exception as e:
        print(f"[DEBUG] Σφάλμα στο API Key: {e}")
        raise e

    design_summary = {
        "tech_stack": design_dict.get("tech_stack_summary", ""),
        "trade_offs": design_dict.get("trade_off_analysis", ""),
    }

    system_msg = {
        "role": "system",
        "content": f"""Είσαι ο Principal System Architect.
ΒΑΣΗ: {json.dumps(design_summary, ensure_ascii=False)}

ΚΑΝΟΝΕΣ:
1. Απάντα σε ερωτήσεις του χρήστη εξηγώντας τεχνικά.
2. ΑΝ ο χρήστης ζητήσει ρητά αλλαγή (π.χ. "άλλαξέ το"), ΠΡΕΠΕΙ να ξεκινήσεις την απάντησή σου ΜΟΝΟ με τη λέξη "ΑΛΛΑΓΗ:" και μετά να γράψεις στα Αγγλικά τι πρέπει να αλλάξει.
3. Μίλα Ελληνικά, σύντομα (έως 80 λέξεις)."""
    }

    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    messages = [system_msg] + recent_history
    
    print(f"[DEBUG] Στέλνω {len(messages)} μηνύματα στο gpt-4o-mini...")
    
    # 2. Απευθείας κλήση στο OpenAI με αυστηρό timeout (10 δευτερόλεπτα)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            timeout=10.0  # Αν αργήσει πάνω από 10 sec, θα πετάξει error αμέσως!
        )
        print("[DEBUG] Η απάντηση ήρθε επιτυχώς!")
        raw_text_response = response.choices[0].message.content
    except Exception as e:
        print(f"[DEBUG] ΣΦΑΛΜΑ κατά την επικοινωνία με το OpenAI: {e}")
        raise e
    
    # 3. Πακετάρισμα για το UI
    class ParsedResponse:
        def __init__(self, text):
            self.needs_update = text.strip().startswith("ΑΛΛΑΓΗ:")
            if self.needs_update:
                self.change_summary = text.replace("ΑΛΛΑΓΗ:", "").strip()
                self.reply = "Κατάλαβα, προχωράω στην αλλαγή..."
            else:
                self.change_summary = ""
                self.reply = text

    print("[DEBUG] Ολοκληρώθηκε η _get_iteration_response.")
    return ParsedResponse(raw_text_response)


def _apply_architecture_patch(current_design, change_summary):
    """
    Hybrid patch: Δοκιμάζει στοχευμένη αλλαγή πρώτα.
    Αν αποτύχει, κάνει full regeneration.
    """
    from app.backend.models.schemas import FinalArchitectureDesign

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.15,
        api_key=os.environ.get("OPENAI_API_KEY"),
    ).with_structured_output(FinalArchitectureDesign)

    design_json = json.dumps(current_design, ensure_ascii=False, indent=2)

    # ── ATTEMPT 1: Στοχευμένο Patch ──
    patch_prompt = f"""
ΤΡΕΧΟΥΣΑ ΑΡΧΙΤΕΚΤΟΝΙΚΗ (JSON):
{design_json}

ΑΠΑΙΤΟΥΜΕΝΗ ΑΛΛΑΓΗ:
{change_summary}

ΟΔΗΓΙΕΣ:
1. Ενσωμάτωσε ΜΟΝΟ την παραπάνω αλλαγή.
2. Αντίγραψε ΑΥΤΟΥΣΙΑ τα πεδία που ΔΕΝ επηρεάζονται.
3. Αν η αλλαγή επηρεάζει κόστος → ενημέρωσε cost_estimate_table.
4. Αν η αλλαγή επηρεάζει απόδοση/ασφάλεια → ενημέρωσε metrics.
5. Mermaid: ΜΟΝΟ Αγγλικά ASCII, ΧΩΡΙΣ ```mermaid fences.
6. C4: ΜΗΝ χρησιμοποιείς System_Boundary / Container_Boundary / curly braces.
7. ERD: Κάθε entity σε ξεχωριστό block με {{ }}, labels σε διπλά εισαγωγικά.
8. implementation_roadmap: πάντα "N/A".
"""
    try:
        result = llm.invoke([{"role": "user", "content": patch_prompt}])
        patched = result.model_dump()
        patched["mermaid_c4_code"] = _sanitize_mermaid(patched.get("mermaid_c4_code", ""))
        patched["mermaid_erd_code"] = _sanitize_mermaid(patched.get("mermaid_erd_code", ""))
        if len(patched.get("mermaid_c4_code", "")) > 20 and len(patched.get("mermaid_erd_code", "")) > 20:
            return patched, "patch"
    except Exception as e:
        print(f"   [ITERATION] Patch failed: {e}")

    # ── ATTEMPT 2: Full Regeneration ──
    regen_prompt = f"""
ΤΡΕΧΟΥΣΑ ΑΡΧΙΤΕΚΤΟΝΙΚΗ (JSON):
{design_json}

ΑΛΛΑΓΗ ΠΟΥ ΠΡΕΠΕΙ ΝΑ ΕΝΣΩΜΑΤΩΘΕΙ:
{change_summary}

ΟΔΗΓΙΕΣ:
Αναδημιούργησε ΟΛΟΚΛΗΡΗ την αρχιτεκτονική ενσωματώνοντας την αλλαγή.
ΜΗΝ αντιγράψεις — σκέψου ξανά τα πάντα υπό το πρίσμα της αλλαγής.

ΚΑΝΟΝΕΣ MERMAID:
- C4: Ξεκίνα με "C4Context" μόνο του. Person(), System(), SystemDb(), System_Ext(), Rel(). ΟΧΙ boundaries. ΟΧΙ curly braces. Τουλάχιστον 10 Rel().
- ERD: Ξεκίνα με "erDiagram" μόνο του. Τουλάχιστον 6 entities με {{ }} blocks. Labels σε "". Τουλάχιστον 7 relationships.
- ΜΟΝΟ Αγγλικά ASCII. ΧΩΡΙΣ ```mermaid fences.
- implementation_roadmap: "N/A".
"""
    try:
        result = llm.invoke([{"role": "user", "content": regen_prompt}])
        regenerated = result.model_dump()
        regenerated["mermaid_c4_code"] = _sanitize_mermaid(regenerated.get("mermaid_c4_code", ""))
        regenerated["mermaid_erd_code"] = _sanitize_mermaid(regenerated.get("mermaid_erd_code", ""))
        return regenerated, "full_regeneration"
    except Exception as e:
        print(f"   [ITERATION] Full regeneration also failed: {e}")
        return None, "failed"


# ==========================================
# 3. FULL-SCREEN GLASS OVERLAY (ROBUST NATIVE STYLING)
# ==========================================
def render_full_screen_selection():
    st.markdown("""
        <div style="position: fixed; top: -5%; left: -5%; width: 40vw; height: 40vw; background: #1d4ed8; border-radius: 50%; filter: blur(120px); opacity: 0.4; z-index: 0; pointer-events: none;"></div>
        <div style="position: fixed; bottom: -10%; right: -5%; width: 50vw; height: 50vw; background: #6d28d9; border-radius: 50%; filter: blur(150px); opacity: 0.3; z-index: 0; pointer-events: none;"></div>
        <div style="position: fixed; top: 40%; left: 40%; width: 30vw; height: 30vw; background: #047857; border-radius: 50%; filter: blur(100px); opacity: 0.2; z-index: 0; pointer-events: none;"></div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        [data-testid="stSidebar"], [data-testid="stHeader"] { display: none !important; }
        .stApp { background-color: #0f172a !important; color-scheme: dark !important; }
        .main .block-container {
            background: rgba(30, 41, 59, 0.4) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
            padding: 4rem 3rem !important;
            max-width: 1100px !important;
            margin: 8vh auto 0 auto !important;
            z-index: 10 !important;
            position: relative;
        }
        .main .block-container div.stButton > button {
            height: 280px !important; width: 100% !important;
            border-radius: 20px !important; color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            display: flex !important; flex-direction: column !important;
            align-items: center !important; justify-content: center !important;
            cursor: pointer !important;
        }
        .main .block-container div.stButton > button p,
        .main .block-container div.stButton > button span,
        .main .block-container div.stButton > button div {
            font-size: 1.3rem !important; font-weight: 600 !important;
            margin: 0 !important; color: #ffffff !important;
            white-space: pre-wrap !important; text-align: center !important;
            line-height: 1.6 !important;
        }
        .main div[data-testid="column"]:nth-of-type(1) div.stButton > button {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.8), rgba(29, 78, 216, 0.9)) !important;
            box-shadow: 0 10px 20px rgba(29, 78, 216, 0.3) !important;
        }
        .main div[data-testid="column"]:nth-of-type(2) div.stButton > button {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.8), rgba(4, 120, 87, 0.9)) !important;
            box-shadow: 0 10px 20px rgba(4, 120, 87, 0.3) !important;
        }
        .main div[data-testid="column"]:nth-of-type(3) div.stButton > button {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.8), rgba(109, 40, 217, 0.9)) !important;
            box-shadow: 0 10px 20px rgba(109, 40, 217, 0.3) !important;
        }
        .main .block-container div.stButton > button:hover {
            transform: translateY(-12px) !important;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            filter: brightness(1.2) !important;
        }
        
        /* ΝΕΑ ΑΣΦΑΛΗ ΚΛΑΣΗ ΑΝΤΙ ΓΙΑ ΣΚΕΤΟ h1 ΚΑΙ h3 */
        .glass-title { text-align: center; color: #f8fafc !important; font-size: 3.2rem !important; font-weight: 800 !important; margin-bottom: 0.2rem !important; margin-top: 0 !important; }
        .glass-subtitle { text-align: center; color: #94a3b8 !important; font-weight: 400 !important; font-size: 1.3rem !important; margin-bottom: 3.5rem !important; margin-top: 0 !important; }
        </style>
    """, unsafe_allow_html=True)
    
    # ΕΝΟΠΟΙΗΜΕΝΟ BLOCK: Τίτλος και υπότιτλος μαζί για να μην μπαίνουν κενά του Streamlit
    st.markdown("""
        <h1 class='glass-title'>Επιλέξτε Τύπο Έργου</h1>
        <h3 class='glass-subtitle'>Ξεκινήστε τον σχεδιασμό του επόμενου σας project</h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Web Application\n\nSaaS • Portals\nE-shops", use_container_width=True):
            st.session_state['project_type'] = "Web Application"
            st.session_state['step'] = "static_input"
            st.rerun()
    with col2:
        if st.button("Microservices\n\nBackend APIs\nDistributed", use_container_width=True):
            st.session_state['project_type'] = "Microservice Architecture"
            st.session_state['step'] = "static_input"
            st.rerun()
    with col3:
        if st.button("Data Pipeline\n\nBig Data • ML\nAnalytics", use_container_width=True):
            st.session_state['project_type'] = "Data/ML Pipeline"
            st.session_state['step'] = "static_input"
            st.rerun()
# 4. MAIN UI LAYOUT
# ==========================================
st.set_page_config(page_title="Netcompany AI Architect", layout="wide")

st.markdown("""
<style>
/* ═══════════════════════════════════════════════════
   THEME — Dark only
   ═══════════════════════════════════════════════════ */
:root {
    --nc-text-primary: #e2e8f0;
    --nc-text-secondary: #94a3b8;
    --nc-text-muted: #64748b;
    --nc-bg-card: rgba(255,255,255,0.04);
    --nc-bg-card-hover: rgba(255,255,255,0.08);
    --nc-border-card: rgba(255,255,255,0.08);
    --nc-border-subtle: rgba(255,255,255,0.06);
    --nc-section-label: rgba(255,255,255,0.40);
    --nc-panel-text: rgba(255,255,255,0.88);
    --nc-panel-header: #f1f5f9;
    --nc-panel-tradeoff-bg: rgba(234,179,8,0.08);
    --nc-panel-tradeoff-border: rgba(234,179,8,0.25);
    --nc-panel-tradeoff-accent: #fde68a;
    --nc-panel-scaling-bg: rgba(34,197,94,0.07);
    --nc-panel-scaling-border: rgba(34,197,94,0.22);
    --nc-panel-scaling-accent: #86efac;
    --nc-stack-accent: #3b82f6;
    --nc-stack-bold: #93c5fd;
    --nc-sidebar-btn-bg: rgba(255,255,255,0.03);
    --nc-sidebar-btn-border: rgba(255,255,255,0.08);
    --nc-sidebar-btn-text: #c8d1dc;
    --nc-sidebar-btn-hover-bg: rgba(59,130,246,0.12);
    --nc-sidebar-btn-hover-border: rgba(59,130,246,0.28);
    --nc-sidebar-btn-hover-text: #e2e8f0;
    --nc-del-overlay-bg: rgba(15,23,42,0.85);
    --nc-del-text: #94a3b8;
    --nc-del-hover-bg: rgba(239,68,68,0.2);
    --nc-del-hover-border: rgba(239,68,68,0.4);
    --nc-del-hover-text: #f87171;
    color-scheme: dark;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR HISTORY — HOVER-REVEAL DELETE (Streamlit DOM-aware CSS)
# ==========================================
# Στοχεύουμε τα ΠΡΑΓΜΑΤΙΚΑ Streamlit data-testid elements:
#   - stHorizontalBlock = η γραμμή που δημιουργεί το st.columns()
#   - stColumn = κάθε στήλη μέσα σε αυτή
# Το logo χρησιμοποιεί 3 columns [1,4,1], το history 2 columns.
# Στοχεύουμε ΜΟΝΟ 2-column blocks (2ο child = last-child) στο sidebar.
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════
   SIDEBAR — Global centering & polish
   ═══════════════════════════════════════════════════ */

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    text-align: center !important;
}
[data-testid="stSidebar"] p {
    text-align: center !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    text-align: center !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p {
    text-align: center !important;
}
[data-testid="stSidebar"] hr {
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ═══════════════════════════════════════════════════
   "ΝΕΟ ΕΡΓΟ" BUTTON — full-width, no extra margins
   ═══════════════════════════════════════════════════ */
[data-testid="stSidebar"] button[kind="primary"] {
    width: 100% !important;
    border-radius: 10px !important;
    margin: 0 !important;
}

/* ═══════════════════════════════════════════════════
   HISTORY ENTRIES — 2-col row, delete overlays title
   ═══════════════════════════════════════════════════ */

/* Row container: position relative so delete can overlay */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) {
    position: relative !important;
    gap: 0 !important;
    align-items: stretch !important;
    margin-bottom: 4px !important;
}

/* TITLE COLUMN: takes full width */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:first-child {
    flex: 1 1 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}

/* DELETE COLUMN: absolute overlay, top-right inside the row */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:last-child {
    position: absolute !important;
    right: 4px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    flex: none !important;
    z-index: 2 !important;
    opacity: 0 !important;
    transition: opacity 0.18s ease !important;
    pointer-events: none !important;
}

/* HOVER row → reveal delete */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
):hover > div[data-testid="stColumn"]:last-child {
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* --- TITLE BUTTON: full-width, centered text --- */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:first-child div.stButton > button {
    height: auto !important;
    width: 100% !important;
    padding: 9px 14px !important;
    border-radius: 10px !important;
    background: var(--nc-sidebar-btn-bg) !important;
    border: 1px solid var(--nc-sidebar-btn-border) !important;
    color: var(--nc-sidebar-btn-text) !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transform: none !important;
    box-shadow: none !important;
    filter: none !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:first-child div.stButton > button:hover {
    background: var(--nc-sidebar-btn-hover-bg) !important;
    border-color: var(--nc-sidebar-btn-hover-border) !important;
    color: var(--nc-sidebar-btn-hover-text) !important;
    transform: none !important;
    box-shadow: none !important;
    filter: none !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:first-child div.stButton > button p {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: inherit !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    text-align: center !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}

/* --- DELETE BUTTON: small square, grid-centered emoji --- */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:last-child div.stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:last-child div.stButton > button {
    height: 28px !important;
    width: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 6px !important;
    background: var(--nc-del-overlay-bg) !important;
    backdrop-filter: blur(4px) !important;
    border: 1px solid var(--nc-border-card) !important;
    color: var(--nc-del-text) !important;
    display: grid !important;
    place-items: center !important;
    place-content: center !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    transform: none !important;
    box-shadow: none !important;
    filter: none !important;
    line-height: 1 !important;
    font-size: 0 !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:last-child div.stButton > button:hover {
    background: var(--nc-del-hover-bg) !important;
    border-color: var(--nc-del-hover-border) !important;
    color: var(--nc-del-hover-text) !important;
    transform: none !important;
    box-shadow: none !important;
    filter: none !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(
    > div[data-testid="stColumn"]:nth-child(2):last-child
) > div[data-testid="stColumn"]:last-child div.stButton > button p {
    font-size: 0.8rem !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    color: inherit !important;
    text-align: center !important;
    width: auto !important;
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:

    logo_filename = "rollingcode-logo-1.png"
    logo_path = os.path.join(current_dir, logo_filename)
    if os.path.exists(logo_path):
        left_co, cent_co, last_co = st.columns([1, 4, 1])
        with cent_co:
            st.image(logo_path, use_container_width=True)
    else:
        # Debugging message αν το αρχείο λείπει από τον φάκελο
        st.warning(f"Logo not found in: {current_dir}")
    
    st.divider()

    if st.button("Νέο Έργο", type="primary", use_container_width=True):
        # Αντί για βίαιη διαγραφή (.clear()), μηδενίζουμε στοχευμένα τις μεταβλητές
        st.session_state['step'] = 'select_type'
        st.session_state['project_type'] = None
        st.session_state['clarification_wizard'] = None
        st.session_state['working_prompt'] = ""
        st.session_state['final_design'] = None
        
        st.session_state['context_step'] = 'chat'
        st.session_state['context_conclusions'] = []
        st.session_state['editing_index'] = None
        st.session_state['context_chat_history'] = [
            {"role": "assistant", "content": "Γεια σας. Είμαι εδώ για να καταγράψω τις εμπειρίες σας. Περιγράψτε μου ένα προηγούμενο έργο σας (π.χ. τι αφορούσε, τι τεχνολογίες χρησιμοποιήσατε και ποιες ήταν οι μεγαλύτερες προκλήσεις)."}
        ]
        
        st.session_state['iteration_chat_history'] = []
        st.session_state['iteration_pending_response'] = False
        st.session_state['design_just_updated'] = False
        st.session_state['current_project_index'] = -1
        
        st.rerun()

    st.divider()
    st.markdown("<h3 style='text-align:center;'>Ιστορικό Έργων</h3>", unsafe_allow_html=True)
    history_list = st.session_state.get('project_history', [])
    
    # --- Εκτέλεση pending delete (αν υπάρχει) ---
    pending_del = st.session_state.get('pending_delete_index')
    if pending_del is not None:
        delete_history_entry(pending_del)
        st.session_state['pending_delete_index'] = None
        st.rerun()
    
    if not history_list:
        st.info("Το ιστορικό σας είναι άδειο.")
    else:
        def _render_history_item(item, actual_idx, key_suffix=""):
            """Renders a single history entry row: [title_button | delete_button]."""
            col_title, col_del = st.columns([7, 1], gap="small")
            with col_title:
                if st.button(
                    f"{item['title']}",
                    key=f"hist_btn{key_suffix}_{actual_idx}",
                    use_container_width=True
                ):
                    st.session_state['project_type'] = item.get('type', 'Unknown')
                    st.session_state['final_design'] = item.get('design', {})
                    st.session_state['working_prompt'] = item.get('prompt', '')
                    st.session_state['iteration_chat_history'] = item.get('chat_history', [])
                    st.session_state['current_project_index'] = actual_idx
                    st.session_state['step'] = 'results'
                    st.session_state['design_just_updated'] = False
                    st.rerun()
            with col_del:
                if st.button(
                    "×",
                    key=f"del_btn{key_suffix}_{actual_idx}",
                    help="Διαγραφή έργου",
                ):
                    st.session_state['pending_delete_index'] = actual_idx
                    st.rerun()

        reversed_history = list(reversed(history_list))
        for i in range(min(3, len(reversed_history))):
            item = reversed_history[i]
            actual_idx = len(history_list) - 1 - i
            _render_history_item(item, actual_idx)
                
        if len(reversed_history) > 3:
            with st.expander("Περισσότερα έργα"):
                for i in range(3, len(reversed_history)):
                    item = reversed_history[i]
                    actual_idx = len(history_list) - 1 - i
                    _render_history_item(item, actual_idx, key_suffix="_exp")

        # Compare button — inside history section, only if 2+ projects
        if len(history_list) >= 2:
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            if st.button("Compare Architectures", use_container_width=True, key="compare_btn"):
                st.session_state['step'] = 'compare'
                st.rerun()
                        
    st.divider()
    st.markdown("<h3 style='text-align:center;'>Knowledge Base</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: gray; margin-top: -10px; text-align:center;'>Εμπλουτίστε τη μνήμη του AI.</p>", unsafe_allow_html=True)
    if st.button("Προσθήκη Προσωπικής Εμπειρίας", use_container_width=True):
        personal_context_modal()

# ==========================================
# ΚΥΡΙΩΣ ΠΕΡΙΟΧΗ (ΕΚΤΟΣ SIDEBAR)
# ==========================================

# Εμφανίζουμε τον κεντρικό τίτλο ΜΟΝΟ αν ΔΕΝ είμαστε στην αρχική οθόνη επιλογής
if st.session_state['step'] != 'select_type':
    st.title("Enterprise AI Architect")

# --- STATE 0: FULL-SCREEN GLASS OVERLAY ---
if st.session_state['step'] == 'select_type':
    render_full_screen_selection()

# --- STATE: COMPARE ARCHITECTURES ---
elif st.session_state['step'] == 'compare':
    history_list = st.session_state.get('project_history', [])
    project_names = [item.get('title', f'Project {i+1}') for i, item in enumerate(history_list)]

    st.subheader("Architecture Comparison")

    col_sel_a, col_sel_b = st.columns(2)
    with col_sel_a:
        idx_a = st.selectbox("Architecture A", range(len(project_names)),
                             format_func=lambda i: project_names[i], key="cmp_a")
    with col_sel_b:
        default_b = min(1, len(project_names) - 1)
        idx_b = st.selectbox("Architecture B", range(len(project_names)),
                             format_func=lambda i: project_names[i], index=default_b, key="cmp_b")

    if idx_a == idx_b:
        st.warning("Επιλέξτε δύο διαφορετικά έργα για σύγκριση.")
    else:
        design_a = history_list[idx_a].get('design', {})
        design_b = history_list[idx_b].get('design', {})
        m_a = design_a.get('metrics', {})
        m_b = design_b.get('metrics', {})

        st.divider()

        # ── Overlaid Radar Chart ──
        cats = ['Cost Efficiency', 'Security', 'Speed', 'Scalability']
        vals_a = [m_a.get('cost_efficiency', 5), m_a.get('security_level', 5),
                  m_a.get('performance_speed', 5), m_a.get('scalability', 5)]
        vals_b = [m_b.get('cost_efficiency', 5), m_b.get('security_level', 5),
                  m_b.get('performance_speed', 5), m_b.get('scalability', 5)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_a + [vals_a[0]], theta=cats + [cats[0]],
            fill='toself', name=project_names[idx_a],
            line=dict(color='#3b82f6'), fillcolor='rgba(59,130,246,0.15)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals_b + [vals_b[0]], theta=cats + [cats[0]],
            fill='toself', name=project_names[idx_b],
            line=dict(color='#f59e0b'), fillcolor='rgba(245,158,11,0.15)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=30, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Side-by-side Metrics Table ──
        _cmp_html = f"""<!DOCTYPE html>
<html><head><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:transparent;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;color:#e2e8f0;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th{{padding:10px 14px;text-align:left;font-weight:600;border-bottom:2px solid rgba(255,255,255,0.1);color:#94a3b8;font-size:11px;letter-spacing:0.05em;text-transform:uppercase;}}
td{{padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.05);}}
.metric-name{{color:#94a3b8;font-weight:500;}}
.val{{font-weight:700;font-size:15px;}}
.win{{color:#34d399;}}
.lose{{color:#94a3b8;}}
.tie{{color:#fbbf24;}}
.label-a{{color:#3b82f6;}}
.label-b{{color:#f59e0b;}}
</style></head><body>
<table>
<tr><th>Metric</th><th class="label-a">{project_names[idx_a][:30]}</th><th class="label-b">{project_names[idx_b][:30]}</th></tr>"""

        metrics_map = [
            ('Cost Efficiency', 'cost_efficiency'),
            ('Security Level', 'security_level'),
            ('Performance', 'performance_speed'),
            ('Scalability', 'scalability'),
        ]
        for label, key in metrics_map:
            va = m_a.get(key, 5)
            vb = m_b.get(key, 5)
            cls_a = 'win' if va > vb else ('tie' if va == vb else 'lose')
            cls_b = 'win' if vb > va else ('tie' if va == vb else 'lose')
            _cmp_html += f'<tr><td class="metric-name">{label}</td><td class="val {cls_a}">{va}/10</td><td class="val {cls_b}">{vb}/10</td></tr>'

        _cmp_html += "</table></body></html>"
        components.html(_cmp_html, height=240, scrolling=False)

        st.divider()

        # ── Side-by-side Tech Stack ──
        st.markdown("#### Tech Stack Comparison")
        col_stack_a, col_stack_b = st.columns(2)
        with col_stack_a:
            st.markdown(f"**{project_names[idx_a][:30]}**")
            st.markdown(design_a.get('tech_stack_summary', 'N/A'))
        with col_stack_b:
            st.markdown(f"**{project_names[idx_b][:30]}**")
            st.markdown(design_b.get('tech_stack_summary', 'N/A'))

        st.divider()

        # ── Side-by-side Trade-offs ──
        st.markdown("#### Trade-off Comparison")
        col_tf_a, col_tf_b = st.columns(2)
        with col_tf_a:
            st.markdown(f"**{project_names[idx_a][:30]}**")
            st.markdown(design_a.get('trade_off_analysis', 'N/A'))
        with col_tf_b:
            st.markdown(f"**{project_names[idx_b][:30]}**")
            st.markdown(design_b.get('trade_off_analysis', 'N/A'))

        st.divider()

        # ── Cost Comparison (stacked — tables too wide for side-by-side) ──
        st.markdown("#### Cost Comparison")
        st.markdown(f"**{project_names[idx_a][:30]}**")
        st.markdown(design_a.get('cost_estimate_table', 'N/A'))
        st.markdown("")
        st.markdown(f"**{project_names[idx_b][:30]}**")
        st.markdown(design_b.get('cost_estimate_table', 'N/A'))

# --- STATE 1: STATIC QUESTIONS & FREE TEXT ---
elif st.session_state['step'] == 'static_input':
    ptype = st.session_state['project_type']
    st.header(f"Τύπος: {ptype}")
    with st.form("static_form"):
        st.subheader("1. Γενική Περιγραφή")
        free_text = st.text_area("Περιγράψτε με δικά σας λόγια το όραμα και τους στόχους της εφαρμογής:", height=100)
        st.divider()
        st.subheader("2. Βασικές Προδιαγραφές")
        budget = st.selectbox("Περιορισμοί Προϋπολογισμού (Budget Constraints)", ["Χαμηλό (Startup/MVP)", "Μεσαίο (Standard)", "Υψηλό (Enterprise)"])
        load = st.selectbox("Αναμενόμενη Κίνηση (Expected Traffic)", ["Μικρή (< 10.000 χρήστες/μήνα)", "Μεσαία (10.000 - 100.000 χρήστες/μήνα)", "Μεγάλη (> 100.000 χρήστες/μήνα)"])
        security = st.selectbox("Απαιτήσεις Ασφαλείας (Security Needs)", ["Βασικές (Standard)", "Αυξημένες (GDPR / Προσωπικά Δεδομένα)", "Υψηλές (Τραπεζικές / Κυβερνητικές)"])
        static_answers = {}
        for q in STATIC_QUESTIONS.get(ptype, []):
            if q["type"] == "select":
                static_answers[q["label"]] = st.selectbox(q["label"], q["options"])
            elif q["type"] == "text":
                static_answers[q["label"]] = st.text_input(q["label"])
            elif q["type"] == "checkbox":
                static_answers[q["label"]] = "Ναι" if st.checkbox(q["label"]) else "Όχι"
        submit_static = st.form_submit_button("Εκκίνηση Σχεδιασμού", type="primary")
        if submit_static:
            if not free_text.strip():
                st.error("Παρακαλώ συμπληρώστε τη γενική περιγραφή!")
            else:
                combined_prompt = f"Τύπος Συστήματος: {ptype}\n\n"
                combined_prompt += f"--- ΓΕΝΙΚΗ ΠΕΡΙΓΡΑΦΗ ---\n{free_text}\n\n"
                combined_prompt += "--- ΣΤΑΤΙΚΕΣ ΠΡΟΔΙΑΓΡΑΦΕΣ ---\n"
                for k, v in static_answers.items():
                    combined_prompt += f"- {k} {v}\n"
                st.session_state['working_prompt'] = combined_prompt
                run_architecture_graph(combined_prompt, budget, load, security)

# --- STATE 2: DYNAMIC WIZARD ---
elif st.session_state['step'] == 'dynamic_wizard':
    st.subheader("Βήμα 2: Εξειδίκευση Απαιτήσεων")
    st.info("Βασιζόμενοι στην αρχική σας περιγραφή, ετοιμάσαμε μερικές στοχευμένες ερωτήσεις για να σχεδιάσουμε την ιδανική αρχιτεκτονική:")
    wizard = st.session_state['clarification_wizard']
    with st.form("dynamic_form"):
        dyn_answers = {}
        categories = [("essential", "Απαραίτητα"), ("recommended", "Σημαντικά"), ("optional", "Προαιρετικά")]
        for cat_key, cat_title in categories:
            fields = wizard.get(cat_key, [])
            if fields:
                if cat_key == "optional":
                    container = st.expander(cat_title)
                else:
                    st.markdown(f"#### {cat_title}")
                    container = st.container()
                with container:
                    for field in fields:
                        f_id, label = f"dyn_{field['id']}", field['label']
                        if field['field_type'] == "text":
                            dyn_answers[label] = st.text_input(label, key=f_id)
                        elif field['field_type'] == "select":
                            options = field.get('options') or ["-"]
                            dyn_answers[label] = st.selectbox(label, options, key=f_id)
                        elif field['field_type'] == "multi_select":
                            options = field.get('options') or ["-"]
                            selected = st.multiselect(label, options, key=f_id)
                            dyn_answers[label] = ", ".join(selected) if selected else ""
                        elif field['field_type'] == "checkbox":
                            dyn_answers[label] = "Ναι" if st.checkbox(label, key=f_id) else "Όχι"
        if st.form_submit_button("Συνέχεια Σχεδιασμού", type="primary"):
            missing_essential = any(
                field['field_type'] == "text" and not dyn_answers.get(field['label'], "").strip()
                for field in wizard.get('essential', [])
            )
            if missing_essential:
                st.error("Παρακαλώ απαντήστε στα απαραίτητα πεδία.")
            else:
                dyn_summary = "\n".join([f"- {k}: {v}" for k, v in dyn_answers.items() if v and v != "Όχι"])
                st.session_state['working_prompt'] += f"\n\n--- ΔΥΝΑΜΙΚΕΣ ΔΙΕΥΚΡΙΝΙΣΕΙΣ (AI) ---\n{dyn_summary}"
                run_architecture_graph(st.session_state['working_prompt'], "Medium", "Scale", "Standard")

# ══════════════════════════════════════════════════════════════════
# STATE 3: RESULTS + Floating Chat Bubble (bottom-right)
# ══════════════════════════════════════════════════════════════════
elif st.session_state['step'] == 'results':
    design = st.session_state['final_design']

    # ── A. Handle pending AI response FIRST (before rendering) ──
    if st.session_state.get('iteration_pending_response', False):
        st.session_state['iteration_pending_response'] = False
        
        with st.spinner("Ο Αρχιτέκτονας απαντά..."):
            try:
                ai_response = _get_iteration_response(
                    design,
                    st.session_state['iteration_chat_history'],
                )
                
                # Αν ο χρήστης ζήτησε αλλαγή
                if ai_response.needs_update:
                    with st.spinner("Εφαρμογή αλλαγών..."):
                        patched_design, method = _apply_architecture_patch(
                            design,
                            ai_response.change_summary
                        )
                        
                    if patched_design:
                        st.session_state['final_design'] = patched_design
                        st.session_state['design_just_updated'] = True
                        
                        # Ενημερώνουμε το Design του τρέχοντος project
                        idx = st.session_state.get('current_project_index', -1)
                        if 0 <= idx < len(st.session_state['project_history']):
                            st.session_state['project_history'][idx]['design'] = patched_design
                            pid = st.session_state['project_history'][idx].get('id')
                            if pid:
                                update_project(pid, {"design": patched_design})
                            
                        final_text = f"{ai_response.reply}\n\nΗ αρχιτεκτονική ενημερώθηκε."
                    else:
                        final_text = f"{ai_response.reply}\n\nΑποτυχία ενημέρωσης κώδικα."
                        
                    st.session_state['iteration_chat_history'].append({"role": "assistant", "content": final_text})
                
                else:
                    # Απλή ερώτηση (Ταχύτατη απάντηση)
                    st.session_state['iteration_chat_history'].append({"role": "assistant", "content": ai_response.reply})
                
                # Αποθηκεύουμε το ιστορικό του Chat στη βάση
                idx = st.session_state.get('current_project_index', -1)
                if 0 <= idx < len(st.session_state['project_history']):
                    st.session_state['project_history'][idx]['chat_history'] = st.session_state['iteration_chat_history']
                    pid = st.session_state['project_history'][idx].get('id')
                    if pid:
                        update_project(pid, {"chat_history": st.session_state['iteration_chat_history']})
                    
            except Exception as e:
                err_msg = str(e)
                if "api_key" in err_msg.lower() or "authentication" in err_msg.lower() or "401" in err_msg:
                    user_err = "API key is invalid or missing. Check your .env file."
                elif "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                    user_err = "Request timed out. Try again."
                elif "rate limit" in err_msg.lower() or "429" in err_msg:
                    user_err = "API rate limit reached. Wait a moment."
                else:
                    user_err = f"Connection error: {type(e).__name__}"
                st.session_state['iteration_chat_history'].append(
                    {"role": "assistant", "content": user_err}
                )
        
        # Rerun ONLY if the architecture was changed (diagrams need refresh).
        # For simple Q&A, the chat iframe picks up the updated history
        # naturally in this render cycle — no rerun needed, tabs stay in place.
        if st.session_state.get('design_just_updated', False):
            st.rerun()
 # ── B. CSS: reposition chat iframe + hide bridge components ──
    st.markdown("""<style>
    /* 1. Εξαφάνιση της Φόρμας Γέφυρας (μαζί με το κουμπί και το input) */
    /* Στοχεύουμε το data-testid της φόρμας που περιέχει το bridge input */
    div[data-testid="stForm"]:has(input[placeholder="__NC_BRIDGE__"]) {
        position: fixed !important;
        left: -100vw !important;
        top: -100vh !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* 2. Σταθεροποίηση του Chat Iframe Container */
    div[data-testid="stHtml"]:has(iframe[height="601"]),
    div[data-testid="element-container"]:has(iframe[height="601"]),
    div[data-testid="stElementContainer"]:has(iframe[height="601"]) {
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        width: 420px !important;
        height: 601px !important;
        z-index: 999999 !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 3. Το Iframe του Chat */
    iframe[height="601"] {
        width: 100% !important;
        height: 100% !important;
        border: none !important;
        background: transparent !important;
        pointer-events: none !important;
    }
    </style>""", unsafe_allow_html=True)    
    # ── C. Bridge form (Fixed & Silent) ──
    # Χρησιμοποιούμε clear_on_submit=True για να καθαρίζει το πεδίο αυτόματα χωρίς σφάλματα
    with st.form("nc_bridge_form", clear_on_submit=True, border=False):
        bridge_msg = st.text_input("b", placeholder="__NC_BRIDGE__", key="nc_bridge_input", label_visibility="collapsed")
        bridge_sub = st.form_submit_button("NC_SEND_BRIDGE")

    if bridge_sub and bridge_msg.strip():
        # Αποθηκεύουμε το μήνυμα στο ιστορικό
        st.session_state['iteration_chat_history'].append({"role": "user", "content": bridge_msg.strip()})
        st.session_state['iteration_pending_response'] = True
        # Κάνουμε rerun - Το πεδίο θα είναι ήδη άδειο λόγω του clear_on_submit
        st.rerun()
# ── D. Chat widget: full UI inside iframe, CSS repositions iframe ──
    _hist_json = json.dumps(st.session_state.get('iteration_chat_history', []), ensure_ascii=False)
    _pending_js = "true" if st.session_state.get('iteration_pending_response') else "false"

    _chat_html = f"""<!DOCTYPE html>
<html><head><style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{background:transparent!important;overflow:hidden;height:100%;pointer-events:none}}
#nc-fab,#nc-win{{pointer-events:auto}}
#nc-fab{{position:fixed;bottom:16px;right:16px;display:flex;align-items:center;gap:8px;background:linear-gradient(135deg,#1e3a5f,#1e2d4a);border:1px solid #2d4a7a;color:#93c5fd;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;font-weight:600;padding:10px 18px 10px 14px;border-radius:24px;cursor:pointer;box-shadow:0 4px 20px rgba(0,0,0,.5);transition:background .2s,transform .15s;user-select:none}}
#nc-fab:hover{{background:linear-gradient(135deg,#1e4080,#1e3560);transform:translateY(-1px)}}
.nc-bdg{{background:#ef4444;color:#fff;border-radius:99px;font-size:10px;font-weight:800;padding:1px 5px;min-width:16px;text-align:center;display:none}}
#nc-win{{position:fixed;bottom:70px;right:16px;width:370px;height:480px;background:#0f1e35;border:1px solid #1e3a5f;border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.65);display:none;flex-direction:column;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
#nc-win.open{{display:flex}}
.wh{{padding:13px 16px;border-bottom:1px solid #1a3050;display:flex;align-items:center;justify-content:space-between;background:#0d1a2e;flex-shrink:0}}
.wh-l{{display:flex;align-items:center;gap:9px}}
.wh-av{{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#1e40af,#4f46e5);border:1px solid #2d4a7a;display:flex;align-items:center;justify-content:center;font-size:14px}}
.wh-n{{color:#e2e8f0;font-size:13.5px;font-weight:700}}
.wh-s{{color:#3b5a8a;font-size:10.5px;margin-top:1px}}
.wh-x{{width:26px;height:26px;border-radius:6px;background:0;border:0;color:#3b5a8a;font-size:15px;cursor:pointer;display:flex;align-items:center;justify-content:center}}.wh-x:hover{{background:#1a3050;color:#94a3b8}}
.ms{{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth}}
.ms::-webkit-scrollbar{{width:3px}}.ms::-webkit-scrollbar-thumb{{background:#1a3050;border-radius:2px}}
.mr{{display:flex;gap:7px;align-items:flex-end;animation:mF .18s ease}}
@keyframes mF{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:translateY(0)}}}}
.mr.u{{flex-direction:row-reverse}}
.av{{width:24px;height:24px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:11px}}
.av.aa{{background:linear-gradient(135deg,#1e40af,#4f46e5);border:1px solid #2d4a7a}}
.av.au{{background:#1a3050;border:1px solid #2d4a7a}}
.bb{{max-width:82%;padding:8px 12px;border-radius:14px;font-size:13px;line-height:1.5;word-break:break-word}}
.bb.ba{{background:#132338;border:1px solid #1e3a5f;color:#cbd5e1;border-bottom-left-radius:3px}}
.bb.bu{{background:#1e3a5f;border:1px solid #2d5080;color:#e2e8f0;border-bottom-right-radius:3px}}
.tp{{display:flex;gap:4px;padding:9px 12px;background:#132338;border:1px solid #1e3a5f;border-radius:14px;border-bottom-left-radius:3px;width:fit-content}}
.dt{{width:5px;height:5px;border-radius:50%;background:#3b82f6;animation:dB 1.2s infinite}}
.dt:nth-child(2){{animation-delay:.2s}}.dt:nth-child(3){{animation-delay:.4s}}
@keyframes dB{{0%,80%,100%{{transform:translateY(0);opacity:.4}}40%{{transform:translateY(-4px);opacity:1}}}}
.wel{{background:#0d1a2e;border:1px solid #1e3a5f;border-radius:12px;padding:18px;text-align:center;margin:4px 0}}
.wel .wi{{font-size:28px;margin-bottom:8px}}.wel .wt{{color:#60a5fa;font-size:13px;font-weight:700;margin-bottom:5px}}.wel .ws{{color:#3b5a8a;font-size:12px;line-height:1.55}}
.ia{{padding:10px 12px 12px;border-top:1px solid #1a3050;background:#0d1a2e;display:flex;gap:7px;align-items:center;flex-shrink:0}}
.ti{{flex:1;background:#132338;border:1px solid #1e3a5f;border-radius:9px;color:#e2e8f0;font-size:13px;font-family:inherit;padding:8px 12px;outline:0}}
.ti::placeholder{{color:#2d4a6a}}.ti:focus{{border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.2)}}
.sb{{background:linear-gradient(135deg,#1e40af,#2563eb);border:0;border-radius:9px;color:#fff;font-size:14px;width:36px;height:36px;cursor:pointer;flex-shrink:0;display:flex;align-items:center;justify-content:center}}.sb:hover{{opacity:.85}}.sb:disabled{{opacity:.35;cursor:not-allowed}}
</style></head>
<body>

<div id="nc-fab" onclick="ncToggle()">
    
    <span>Architect</span>
    <span class="nc-bdg" id="nc-bdg"></span>
</div>

<div id="nc-win">
    <div class="wh">
        <div class="wh-l"><div class="wh-av">A</div><div><div class="wh-n">Principal Architect</div><div class="wh-s">Ρωτήστε · Αμφισβητήστε · Αλλάξτε</div></div></div>
        <button class="wh-x" onclick="ncClose()">✕</button>
    </div>
    <div class="ms" id="nc-ms"></div>
    <div class="ia">
        <input class="ti" id="nc-ti" type="text" placeholder="π.χ. Γιατί PostgreSQL και όχι MongoDB;" autocomplete="off" onkeydown="if(event.key==='Enter')ncSend()" />
        <button class="sb" id="nc-sb" onclick="ncSend()">&rsaquo;</button>
    </div>
</div>

<script>
/* --- ΑΠΟΛΥΤΗ ΤΟΠΟΘΕΤΗΣΗ ΚΑΤΩ ΔΕΞΙΑ ΜΕΣΩ JS --- */
try {{
    var f = window.frameElement;
    if (f) {{
        f.style.position = 'fixed'; f.style.bottom = '0'; f.style.right = '0'; f.style.width = '420px'; f.style.zIndex = '99999';
        if (f.parentElement) {{ 
            f.parentElement.style.position = 'fixed'; f.parentElement.style.bottom = '0'; f.parentElement.style.right = '0'; f.parentElement.style.width = '420px'; f.parentElement.style.zIndex = '99999'; 
        }}
    }}
}} catch(e) {{ console.log(e); }}

var HIST = {_hist_json};
var PENDING = {_pending_js};

function esc(t) {{ return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>'); }}

function ncToggle() {{
    var w = document.getElementById('nc-win');
    if (w.classList.contains('open')) {{ w.classList.remove('open'); }}
    else {{ w.classList.add('open'); document.getElementById('nc-bdg').style.display='none'; setTimeout(function(){{ var i=document.getElementById('nc-ti'); if(i) i.focus(); }}, 200); }}
}}
function ncClose() {{ document.getElementById('nc-win').classList.remove('open'); }}

function ncSend() {{
    var inp = document.getElementById('nc-ti');
    var text = inp ? inp.value.trim() : '';
    if (!text) return;
    var mc = document.getElementById('nc-ms');
    var row = document.createElement('div'); row.className = 'mr u';
    row.innerHTML = '<div class="av au">U</div><div class="bb bu">'+esc(text)+'</div>';
    mc.appendChild(row);
    var tr = document.createElement('div'); tr.className = 'mr';
    tr.innerHTML = '<div class="av aa">A</div><div class="tp"><div class="dt"></div><div class="dt"></div><div class="dt"></div></div>';
    mc.appendChild(tr); mc.scrollTop = mc.scrollHeight;
    inp.value = ''; document.getElementById('nc-sb').disabled = true;

    try {{
        var P = window.parent.document;
        var bridgeInp = P.querySelector('input[placeholder="__NC_BRIDGE__"]');
        if (bridgeInp) {{
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(bridgeInp, text);
            bridgeInp.dispatchEvent(new Event('input', {{bubbles: true}}));
            bridgeInp.dispatchEvent(new Event('change', {{bubbles: true}}));

            setTimeout(function() {{
                var buttons = P.querySelectorAll('button');
                for (var i = 0; i < buttons.length; i++) {{
                    if (buttons[i].innerText.includes("NC_SEND_BRIDGE")) {{
                        buttons[i].click();
                        break;
                    }}
                }}
            }}, 400);
        }}
    }} catch(e) {{ 
        document.getElementById('nc-ms').innerHTML += '<div style="color:#ef4444;font-size:12px;text-align:center;">Error: ' + e.message + '</div>';
    }}
}}

var mc = document.getElementById('nc-ms');
if (!HIST.length) {{
    mc.innerHTML = '<div class="wel"><div class="wt">Principal Architect AI</div><div class="ws">Είμαι ο αρχιτέκτονας αυτού του συστήματος.<br>Ρωτήστε <strong style="color:#60a5fa">γιατί</strong> επέλεξα κάτι,<br>προτείνετε εναλλακτικές ή ζητήστε αλλαγές.</div></div>';
}} else {{
    HIST.forEach(function(msg) {{
        var isU = msg.role === 'user';
        var row = document.createElement('div'); row.className = 'mr ' + (isU?'u':'');
        row.innerHTML = '<div class="av '+(isU?'au':'aa')+'">'+(isU?'U':'A')+'</div><div class="bb '+(isU?'bu':'ba')+'">'+esc(msg.content)+'</div>';
        mc.appendChild(row);
    }});
}}

// AUTO-OPEN LOGIC: Αν το τελευταίο μήνυμα είναι από τον Αρχιτέκτονα, άνοιξε το παράθυρο
if (HIST.length > 0) {{
    var lastMsg = HIST[HIST.length - 1];
    if (lastMsg.role === 'assistant') {{
        document.getElementById('nc-win').classList.add('open');
    }}
}}

if (PENDING) {{
    var tr = document.createElement('div'); tr.className = 'mr';
    tr.innerHTML = '<div class="av aa">A</div><div class="tp"><div class="dt"></div><div class="dt"></div><div class="dt"></div></div>';
    mc.appendChild(tr);
    document.getElementById('nc-win').classList.add('open');
}}
mc.scrollTop = mc.scrollHeight;

if (HIST.length && HIST[HIST.length-1].role === 'assistant' && !document.getElementById('nc-win').classList.contains('open')) {{
    var b = document.getElementById('nc-bdg'); b.style.display='inline-block'; b.textContent='1';
}}
</script>
</body></html>"""
    components.html(_chat_html, height=601, scrolling=False)

    # ── E. Architecture update banner ──
    if st.session_state.get('design_just_updated', False):
        st.info("Η αρχιτεκτονική ενημερώθηκε μέσω της συζήτησης με τον Αρχιτέκτονα. Τα παρακάτω αντικατοπτρίζουν τις αλλαγές.", )
        st.session_state['design_just_updated'] = False
    elif st.session_state['step'] == 'results':
        design = st.session_state['final_design']

    # ── E2. Export Dossier buttons ──
    _exp_col1, _exp_col2, _exp_spacer = st.columns([1, 1, 4])
    _ptype = st.session_state.get('project_type', 'Architecture')
    _prompt = st.session_state.get('working_prompt', '')
    with _exp_col1:
        _md_content = generate_markdown_dossier(design, _ptype, _prompt)
        st.download_button(
            label="Export .md",
            data=_md_content,
            file_name=f"architecture_dossier_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with _exp_col2:
        _html_content = generate_html_dossier(design, _ptype, _prompt)
        st.download_button(
            label="Export .html",
            data=_html_content,
            file_name=f"architecture_dossier_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True,
            help="Open in browser, then Print > Save as PDF",
        )

    # ── E3. Consistency badge from Design Critic ──
    _validation = st.session_state.get('final_validation', {})
    if _validation and _validation.get('consistency_score', 0) > 0:
        _score = _validation.get('consistency_score', 0)
        _issues = _validation.get('issues_found', [])
        _n_critical = sum(1 for i in _issues if i.get('severity') == 'critical')
        _n_minor = sum(1 for i in _issues if i.get('severity') == 'minor')
        _n_total = len(_issues)

        if _score >= 8:
            _ring_color = "#10b981"
            _ring_bg = "rgba(16,185,129,0.12)"
            _status_text = "Consistent"
            _status_color = "#10b981"
        elif _score >= 5:
            _ring_color = "#f59e0b"
            _ring_bg = "rgba(245,158,11,0.12)"
            _status_text = "Patched"
            _status_color = "#f59e0b"
        else:
            _ring_color = "#ef4444"
            _ring_bg = "rgba(239,68,68,0.12)"
            _status_text = "Inconsistent"
            _status_color = "#ef4444"

        _circum = 251.2
        _offset = _circum - (_circum * _score / 10)

        _lessons_stored = _validation.get('lessons_stored', 0)
        _total_lessons = _validation.get('total_lessons', 0)

        _card_full_html = f"""<!DOCTYPE html>
<html><head><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:transparent;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;color:#e2e8f0;}}
.card{{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px 26px;}}
.row{{display:flex;align-items:center;gap:26px;flex-wrap:wrap;}}
.ring-wrap{{position:relative;width:86px;height:86px;flex-shrink:0;}}
.ring-score{{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
.ring-score b{{font-size:22px;font-weight:700;color:#f1f5f9;line-height:1;}}
.ring-score small{{font-size:10px;color:#64748b;margin-top:2px;}}
.info{{flex:1;min-width:180px;}}
.title{{font-size:0.92rem;font-weight:600;color:#e2e8f0;margin-bottom:3px;}}
.pill{{display:inline-block;font-size:10.5px;font-weight:700;padding:3px 10px;border-radius:6px;margin-left:10px;letter-spacing:0.03em;}}
.sub{{color:#94a3b8;font-size:0.8rem;line-height:1.6;margin-top:3px;}}
.stats{{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;}}
.stat{{border-radius:8px;padding:5px 13px;text-align:center;}}
.stat b{{display:block;font-size:17px;font-weight:700;line-height:1.3;}}
.stat small{{font-size:9.5px;color:#94a3b8;letter-spacing:0.05em;}}
</style></head><body>
<div class="card"><div class="row">
<div class="ring-wrap">
<svg width="86" height="86" viewBox="0 0 90 90" style="transform:rotate(-90deg);">
<circle cx="45" cy="45" r="40" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="6"/>
<circle cx="45" cy="45" r="40" fill="none" stroke="{_ring_color}" stroke-width="6"
stroke-linecap="round" stroke-dasharray="{_circum}" stroke-dashoffset="{_offset}"/>
</svg>
<div class="ring-score"><b>{_score}</b><small>/10</small></div>
</div>
<div class="info">
<div class="title">Design Consistency
<span class="pill" style="background:{_ring_bg};color:{_status_color};">{_status_text}</span></div>
<div class="sub">Self-critique completed &mdash; {_n_total} issue{"s" if _n_total != 1 else ""} detected and auto-corrected{f' &middot; {_lessons_stored} new lesson{"s" if _lessons_stored != 1 else ""} learned' if _lessons_stored > 0 else ''}</div>
<div class="stats">
<div class="stat" style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);"><b style="color:#f87171;">{_n_critical}</b><small>CRITICAL</small></div>
<div class="stat" style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.18);"><b style="color:#fbbf24;">{_n_minor}</b><small>MINOR</small></div>
<div class="stat" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.18);"><b style="color:#34d399;">{_n_total}</b><small>PATCHED</small></div>
<div class="stat" style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.18);"><b style="color:#a78bfa;">{_total_lessons}</b><small>MEMORY</small></div>
</div></div></div></div>
</body></html>"""
        components.html(_card_full_html, height=140, scrolling=False)

        if _issues:
            with st.expander(f"Self-critique audit log ({_n_total} issues)"):
                for issue in _issues:
                    _sev = issue.get('severity', 'minor')
                    if _sev == 'critical':
                        _accent = "#ef4444"
                        _label = "CRITICAL"
                    else:
                        _accent = "#f59e0b"
                        _label = "MINOR"
                    _comp = issue.get('component', '?')
                    _desc = issue.get('description', '')
                    _fix = issue.get('fix_applied', 'N/A')
                    st.markdown(
                        f"<div style='border-left:3px solid {_accent};padding:8px 14px;margin-bottom:10px;"
                        f"background:rgba(255,255,255,0.02);border-radius:0 8px 8px 0;'>"
                        f"<span style='font-size:10px;font-weight:700;color:{_accent};letter-spacing:0.05em;'>{_label}</span>"
                        f" <span style='color:#e2e8f0;font-size:0.85rem;font-weight:600;'>{_comp}</span><br>"
                        f"<span style='color:#94a3b8;font-size:0.8rem;'>{_desc}</span><br>"
                        f"<span style='color:#64748b;font-size:0.76rem;font-style:italic;'>Fix: {_fix}</span></div>",
                        unsafe_allow_html=True
                    )

    # ── F. Full-width tabs (IDENTICAL content to original) ──
    t1, t2, t3 = st.tabs(["Diagrams", "Architecture Strategy", "Metrics & FinOps"])

    with t1:
        st.subheader("C4 Context Diagram")
        render_mermaid(design.get("mermaid_c4_code", ""))
        st.divider()
        st.subheader("Entity-Relationship Diagram (ERD)")
        render_mermaid(design.get("mermaid_erd_code", ""))

    with t2:
        st.markdown("""
        <style>
        .nc-stack-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 14px;
            margin: 4px 0 28px 0;
        }
        .nc-stack-card {
            background: var(--nc-bg-card);
            border: 1px solid var(--nc-border-card);
            border-left: 3px solid var(--nc-stack-accent);
            border-radius: 12px;
            padding: 14px 18px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            transition: background .2s;
        }
        .nc-stack-card:hover { background: var(--nc-bg-card-hover); }
        .nc-stack-card .nc-sc-bullet {
            font-size: 1.3rem;
            line-height: 1;
            flex-shrink: 0;
            margin-top: 2px;
        }
        .nc-stack-card .nc-sc-text {
            font-size: 0.88rem;
            line-height: 1.55;
            color: var(--nc-text-primary);
        }
        .nc-stack-card .nc-sc-text b,
        .nc-stack-card .nc-sc-text strong {
            color: var(--nc-stack-bold);
            font-weight: 700;
        }
        .nc-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
            color: var(--nc-section-label);
            margin: 28px 0 12px 0;
        }
        .nc-panel {
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 16px;
            border: 1px solid;
            line-height: 1.65;
            font-size: 0.89rem;
            color: var(--nc-panel-text);
        }
        .nc-panel-tradeoff {
            background: var(--nc-panel-tradeoff-bg);
            border-color: var(--nc-panel-tradeoff-border);
        }
        .nc-panel-scaling {
            background: var(--nc-panel-scaling-bg);
            border-color: var(--nc-panel-scaling-border);
        }
        .nc-panel-header {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--nc-panel-header);
            margin-bottom: 12px;
        }
        .nc-panel ul { margin: 6px 0 0 0; padding-left: 20px; }
        .nc-panel li { margin-bottom: 6px; }
        .nc-panel b, .nc-panel strong { color: var(--nc-panel-tradeoff-accent); }
        .nc-panel-scaling b, .nc-panel-scaling strong { color: var(--nc-panel-scaling-accent); }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nc-section-label">Τεχνολογικό Stack</div>', unsafe_allow_html=True)

        raw_stack = design.get('tech_stack_summary', '') or ''
        stack_lines = []
        for line in raw_stack.splitlines():
            s = line.strip()
            if not s:
                continue
            s = re.sub(r'^[-*•]\s+', '', s)
            s = re.sub(r'^\d+\.\s+', '', s)
            if s:
                stack_lines.append(s)

        if stack_lines:
            ACCENTS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']
            # Icons removed
            cards_html = '<div class="nc-stack-grid">'
            for i, line in enumerate(stack_lines):
                accent = ACCENTS[i % len(ACCENTS)]
                line_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                cards_html += (
                    f'<div class="nc-stack-card" style="border-left-color:{accent}">'
                    f'  <div class="nc-sc-text">{line_html}</div>'
                    f'</div>'
                )
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)
        else:
            st.info("Δεν υπάρχουν διαθέσιμα δεδομένα για το Tech Stack.")

        st.markdown('<div class="nc-section-label">Αρχιτεκτονικά Trade-offs</div>', unsafe_allow_html=True)

        raw_tradeoffs = design.get('trade_off_analysis', '') or ''
        tradeoff_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', raw_tradeoffs)
        tradeoff_html = re.sub(r'\n[-*•]\s+', '\n• ', tradeoff_html)
        tradeoff_html = tradeoff_html.replace('\n', '<br>')

        st.markdown(f"""
        <div class="nc-panel nc-panel-tradeoff">
          <div class="nc-panel-header">Διλήμματα &amp; Αποφάσεις Σχεδίασης</div>
          {tradeoff_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nc-section-label">Scaling Roadmap</div>', unsafe_allow_html=True)

        raw_scaling = design.get('future_scaling_path', '') or ''
        scaling_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', raw_scaling)
        scaling_html = re.sub(r'\n[-*•]\s+', '\n• ', scaling_html)
        scaling_html = scaling_html.replace('\n', '<br>')

        st.markdown(f"""
        <div class="nc-panel nc-panel-scaling">
          <div class="nc-panel-header">Μελλοντική Κλιμάκωση</div>
          {scaling_html}
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("### Αξιολόγηση & Οικονομική Ανάλυση")
        col_chart, col_highlights = st.columns([1.2, 1])

        with col_chart:
            m = design.get("metrics", {})
            values = [m.get("cost_efficiency", 5), m.get("security_level", 5), m.get("performance_speed", 5), m.get("scalability", 5)]
            cats = ['Cost Efficiency', 'Security', 'Speed', 'Scale']
            fig = go.Figure(data=go.Scatterpolar(r=values+[values[0]], theta=cats+[cats[0]], fill='toself', line=dict(color='#2563eb')))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_highlights:
            st.markdown("#### Architectural Highlights")
            st.success(f"**Security Level:** {m.get('security_level', 0)}/10")
            st.info(f"**Scalability Potential:** {m.get('scalability', 0)}/10")
            st.warning("Εκτιμήσεις βασισμένες σε Cloud FinOps μοντέλα (AWS/Azure Tiers).")

        st.divider()

        st.subheader("Αναλυτικό Infrastructure Breakdown (Monthly)")
        st.markdown(design.get("cost_estimate_table", "N/A"))

        with st.expander("Σημειώσεις Χρέωσης & FinOps"):
            st.write("""
            Η παραπάνω εκτίμηση περιλαμβάνει:
            - **Compute & DB Instances:** Βάσει των επιλεγμένων Tiers και Specs.
            - **Networking:** Εκτίμηση Data Transfer Out (Egress).
            - **Operational:** Backups, Snapshots και Monitoring logs.
            
            *Δεν περιλαμβάνονται κόστη ανάπτυξης (Dev hours) ή άδειες χρήσης 3rd-party λογισμικού.*
            """)