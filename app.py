"""
FormFill AI v2 — Smart form filling with Claude Vision.

Pipeline:
  Phase 0 → Claude classifies each uploaded PDF (blank form vs source doc)
  Phase 1 → Claude analyzes the blank form → identifies fields + % coordinates
  Phase 2 → Claude reads every source doc → translates (if non-English) → extracts data
  Phase 3 → Claude maps data → fill instructions JSON
  Phase 4 → reportlab overlays text onto original PDF
"""

import streamlit as st
import anthropic
import base64
import json
import io
import os
import time
import tomllib
import tomli_w
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

try:
    from pdf2image import convert_from_bytes
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

# ──────────────────────────────────────────────────────────────
# SECRETS HELPERS
# ──────────────────────────────────────────────────────────────

SECRETS_FILE = Path(".streamlit/secrets.toml")


def secrets_get(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def secrets_save(key: str, value: str) -> tuple[bool, str]:
    """Persist key=value into .streamlit/secrets.toml."""
    try:
        SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if SECRETS_FILE.exists():
            with open(SECRETS_FILE, "rb") as f:
                existing = tomllib.load(f)
        existing[key] = value
        with open(SECRETS_FILE, "wb") as f:
            tomli_w.dump(existing, f)
        return True, ""
    except Exception as e:
        return False, str(e)


def secrets_delete(key: str) -> bool:
    try:
        if not SECRETS_FILE.exists():
            return True
        with open(SECRETS_FILE, "rb") as f:
            existing = tomllib.load(f)
        existing.pop(key, None)
        with open(SECRETS_FILE, "wb") as f:
            tomli_w.dump(existing, f)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FormFill AI",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  section[data-testid="stSidebar"] { background:#0f0f13; border-right:1px solid #2a2a35; }
  section[data-testid="stSidebar"] * { color:#e8e8f0 !important; }
  section[data-testid="stSidebar"] input {
    background:#1a1a22 !important; border:1px solid #3a3a50 !important;
    color:#e8e8f0 !important; font-family:'Space Mono',monospace !important; font-size:12px !important;
  }

  .main .block-container { padding:2rem 2rem 4rem; max-width:1200px; }

  .phase-card { background:#fafafa; border:1.5px solid #e8e8ec; border-radius:12px;
                padding:16px 20px; margin-bottom:12px; }
  .phase-card.active  { border-color:#5b5bd6; background:#f5f5ff; }
  .phase-card.done    { border-color:#2da44e; background:#f0fff4; }
  .phase-card.waiting { opacity:0.45; }
  .phase-card.error   { border-color:#e53e3e; background:#fff5f5; }

  .phase-num { display:inline-block; width:26px; height:26px; border-radius:50%;
               background:#5b5bd6; color:white; font-family:'Space Mono',monospace;
               font-size:11px; font-weight:700; text-align:center; line-height:26px; margin-right:10px; }
  .phase-num.done  { background:#2da44e; }
  .phase-num.error { background:#e53e3e; }
  .phase-title { font-weight:600; font-size:14px; color:#1a1a2e; }
  .phase-desc  { font-size:12px; color:#6b6b80; margin-top:3px; margin-left:36px; }

  .doc-badge { display:inline-block; border-radius:20px; padding:3px 10px;
               font-size:11px; margin:3px; font-family:'Space Mono',monospace; }
  .doc-form   { background:#5b5bd6; color:white; }
  .doc-source { background:#2da44e; color:white; }
  .doc-unknown{ background:#888; color:white; }

  .result-box { background:linear-gradient(135deg,#f0fff4 0%,#e8f5ff 100%);
                border:1.5px solid #2da44e; border-radius:12px;
                padding:20px 24px; text-align:center; }
  .json-display { background:#1a1a22; color:#7ec8a4; font-family:'Space Mono',monospace;
                  font-size:11px; border-radius:8px; padding:12px; overflow-x:auto;
                  white-space:pre; max-height:240px; overflow-y:auto; }
  .big-title { font-family:'Space Mono',monospace; font-size:26px; font-weight:700;
               color:#1a1a2e; letter-spacing:-0.5px; }
  .subtitle  { font-size:14px; color:#6b6b80; margin-top:4px; }
  .warn-box  { background:#fffbea; border:1px solid #f6ad55; border-radius:8px;
               padding:10px 14px; font-size:13px; color:#744210; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FONT + PDF HELPERS
# ──────────────────────────────────────────────────────────────

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def get_font() -> str:
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            return fp
    return ""


def register_font(path: str) -> str:
    name = "FormFont"
    try:
        pdfmetrics.registerFont(TTFont(name, path))
        return name
    except Exception:
        return "Helvetica"


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 5, dpi: int = 150) -> list[Image.Image]:
    if HAS_PDF2IMAGE:
        try:
            return convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
        except Exception:
            pass
    return [Image.new("RGB", (595, 842), (255, 255, 255))]


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def images_to_content(images: list[Image.Image], label: str = "") -> list[dict]:
    out = []
    for i, img in enumerate(images):
        if label:
            out.append({"type": "text", "text": f"[{label} — page {i+1}]"})
        out.append({"type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": img_to_b64(img)}})
    return out


# ──────────────────────────────────────────────────────────────
# CLAUDE PIPELINE
# ──────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-6"   # overridden from session state


def call_claude(client, content: list[dict], system: str, max_tokens: int = 4000) -> str:
    resp = client.messages.create(
        model=st.session_state.get("model", MODEL),
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    raw = resp.content[0].text.strip()
    return raw.replace("```json", "").replace("```", "").strip()


# ── Phase 0: Classify each PDF ───────────────────────────────

def phase0_classify(client, uploads: list[dict]) -> list[dict]:
    """
    For each upload, ask Claude: is this a blank form to fill, or a source document?
    Returns same list with added keys: role ('form'|'source'), description, language.
    """
    results = []
    for up in uploads:
        imgs = pdf_to_images(up["bytes"], max_pages=2)
        content = images_to_content(imgs, up["name"])
        content.append({"type": "text", "text": """
Look at this document carefully.

Classify it as ONE of:
- "form"   → a blank or partially blank form/application/claim form that needs to be FILLED IN
             (has empty fields, lines, boxes, checkboxes waiting for input)
- "source" → a source document containing data/information to USE when filling a form
             (invoice, receipt, ID, insurance card, medical report, referral letter, policy document, etc.)

Also detect:
- language: main language of the document (e.g. "English", "Greek", "French")
- description: one sentence describing what this document is

Return ONLY JSON (no markdown):
{"role": "form" or "source", "language": "...", "description": "..."}
"""})
        raw = call_claude(client, content,
                          "You are a document classifier. Output ONLY valid JSON.", 300)
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {"role": "source", "language": "unknown", "description": up["name"]}
        results.append({**up, **meta})
    return results


# ── Phase 1: Analyze form fields ─────────────────────────────

def phase1_analyze_form(client, form_doc: dict) -> list[dict]:
    """Detect every fillable field in the form with % coordinates."""
    imgs = pdf_to_images(form_doc["bytes"], max_pages=6)
    content = images_to_content(imgs, "BLANK FORM")
    content.append({"type": "text", "text": """
Analyze this blank form carefully. Find EVERY fillable area:
text fields, checkboxes, radio buttons, date fields, tables, signature lines.

For each field return:
- id: short snake_case key (e.g. "policy_number", "patient_surname", "claim_new_yes")
- label: exact label text as printed on the form
- page: page number (1-based)
- x: left edge of ENTRY AREA as fraction of page width (0.0=left, 1.0=right)
- y: top edge of ENTRY AREA as fraction of page height (0.0=top, 1.0=bottom)
- w: width of entry area as fraction of page width
- h: height of entry area (fraction)
- type: "text" | "checkbox" | "radio" | "date" | "table_cell"
- section: which section of the form (e.g. "Section 1 - Claim details", "Section 2 - Policyholder")
- description: brief English description of what data goes here

Be exhaustive. Include ALL fields across ALL pages.

Return ONLY JSON (no markdown):
{"fields": [...]}
"""})
    raw = call_claude(client, content,
                      "You are a precise form analyst. Output ONLY valid JSON.", 6000)
    data = json.loads(raw)
    return data.get("fields", [])


# ── Phase 2: Extract & translate source documents ─────────────

def phase2_extract_sources(client, source_docs: list[dict]) -> dict:
    """
    For each source doc: read it, translate if non-English, extract all data.
    Returns merged dict of extracted data + per-doc summaries.
    """
    all_data: dict = {}
    summaries: list[str] = []

    for doc in source_docs:
        imgs = pdf_to_images(doc["bytes"], max_pages=4)
        content = images_to_content(imgs, doc["name"])
        lang = doc.get("language", "English")
        translate_note = (
            f"This document is in {lang}. Translate all content to English before extracting."
            if lang.lower() not in ("english", "unknown") else ""
        )
        content.append({"type": "text", "text": f"""
{translate_note}

Read this document thoroughly. Extract EVERY piece of information that could be used
to fill a medical/insurance/administrative claim form.

Return ONLY JSON (no markdown):
{{
  "document_type": "what kind of document this is",
  "language_detected": "...",
  "summary": "one sentence description",
  "data": {{
    "key": "value",
    ...
  }}
}}

Use descriptive English keys such as:
policyholder_title, policyholder_forename, policyholder_surname,
policy_number, correspondence_address, postcode, city, country,
phone, mobile, email,
patient_title, patient_forename, patient_surname, patient_dob,
illness_description, symptoms_date, treatment_type, treatment_date,
invoice_number, invoice_date, invoice_amount, invoice_currency,
provider_name, provider_address, provider_phone,
doctor_name, doctor_qualifications, doctor_license,
hospital_name, hospital_address, hospital_phone,
referral_date, diagnosis, icd_code,
is_new_claim (true/false), is_accident (true/false),
other_insurance (true/false), state_care (true/false)

Include ALL values you can read. For booleans use true/false.
For dates use DD/MM/YYYY format.
Translate any non-English text values to English.
"""})
        raw = call_claude(client, content,
                          "You are a multilingual data extraction expert. Output ONLY valid JSON.",
                          3000)
        try:
            extracted = json.loads(raw)
            doc_data = extracted.get("data", {})
            all_data.update(doc_data)
            summaries.append(f"• [{extracted.get('document_type','doc')}] {extracted.get('summary','')}")
        except Exception as e:
            summaries.append(f"• [{doc['name']}] parse error: {e}")

    all_data["_summaries"] = summaries
    return all_data


# ── Phase 3: Map data → fill instructions ────────────────────

def phase3_map_and_fill(client, fields: list[dict],
                         data: dict, form_doc: dict) -> list[dict]:
    """Generate precise fill instructions for every matched field."""
    imgs = pdf_to_images(form_doc["bytes"], max_pages=6)
    content = images_to_content(imgs, "BLANK FORM")

    # Remove internal key before sending
    clean_data = {k: v for k, v in data.items() if not k.startswith("_")}

    content.append({"type": "text", "text": f"""
You have two inputs:

A) FORM FIELDS (detected in the blank form):
{json.dumps(fields, ensure_ascii=False, indent=2)}

B) EXTRACTED DATA (from source documents, already translated to English):
{json.dumps(clean_data, ensure_ascii=False, indent=2)}

Task: For each form field that has matching data, generate a fill instruction.

Rules:
- x, y are TOP-LEFT corner of where to write (use field's x, y values exactly)
- font_size (6–10): choose so text fits within the field width w × page_width
- For radio/checkbox "Yes": use "●" at the Yes button position
- For radio/checkbox "No":  use "●" at the No button position  
- For "Is this a new claim?" → Yes, use "●" at the Yes radio position
- For "Is this related to an accident?" → No, use "●" at the No radio position
- For invoice table rows: fill date_of_treatment, description, currency+amount, payee
- For the date field in section 5 (patient signature date): use today's date
- Skip fields that have no matching data (doctor signature, stamps, etc.)
- NEVER invent data that isn't in the extracted data

Return ONLY JSON (no markdown):
{{
  "fills": [
    {{
      "field_id": "...",
      "label": "...",
      "page": 1,
      "x": 0.28,
      "y": 0.34,
      "text": "CHRISTOS IATROPOULOS",
      "font_size": 8
    }},
    ...
  ],
  "skipped": ["field_ids with no matching data"],
  "notes": "any important observations"
}}
"""})
    raw = call_claude(client, content,
                      "You are a precise form-filling expert. Output ONLY valid JSON.", 5000)
    data_out = json.loads(raw)
    return data_out.get("fills", []), data_out.get("notes", ""), data_out.get("skipped", [])


# ── Phase 4: Render filled PDF ────────────────────────────────

def phase4_render(form_bytes: bytes, fills: list[dict],
                  font_name: str, color=(0, 0, 0.6)) -> bytes:
    reader = PdfReader(io.BytesIO(form_bytes))
    writer = PdfWriter()

    by_page: dict[int, list] = {}
    for f in fills:
        by_page.setdefault(int(f.get("page", 1)), []).append(f)

    for i, page in enumerate(reader.pages):
        pn = i + 1
        pdf_w = float(page.mediabox.width)
        pdf_h = float(page.mediabox.height)
        page_fills = by_page.get(pn, [])

        if page_fills:
            pkt = io.BytesIO()
            c = canvas.Canvas(pkt, pagesize=(pdf_w, pdf_h))
            c.setFillColorRGB(*color)

            for f in page_fills:
                text = str(f.get("text", "")).strip()
                if not text:
                    continue
                fs   = float(f.get("font_size", 8))
                x_pt = float(f.get("x", 0)) * pdf_w
                y_pt = pdf_h - float(f.get("y", 0)) * pdf_h - fs
                try:
                    c.setFont(font_name, fs)
                except Exception:
                    c.setFont("Helvetica", fs)
                c.drawString(x_pt, y_pt, text)

            c.save()
            pkt.seek(0)
            overlay = PdfReader(pkt).pages[0]
            page.merge_page(overlay)

        writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, tuple]:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        # ── API Key (Streamlit Secrets) ───────────────────────
        saved_key = secrets_get("ANTHROPIC_API_KEY")
        key_is_saved = bool(saved_key)

        if key_is_saved:
            masked = saved_key[:10] + "..." + saved_key[-4:]
            st.markdown(
                "🔑 **API Key** "
                '<span style="background:#2da44e;color:white;border-radius:20px;'
                'padding:2px 8px;font-size:11px">✓ saved in secrets</span>',
                unsafe_allow_html=True)
            st.markdown(f'<code style="font-size:11px;color:#aaa">{masked}</code>',
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✏️ Change", use_container_width=True):
                    st.session_state["show_key_input"] = True
            with col2:
                if st.button("🗑 Remove", use_container_width=True):
                    secrets_delete("ANTHROPIC_API_KEY")
                    st.session_state.pop("show_key_input", None)
                    st.rerun()

            if st.session_state.get("show_key_input"):
                new_key = st.text_input("New API Key", type="password",
                                        placeholder="sk-ant-...", key="change_key_inp")
                if st.button("💾 Save", use_container_width=True, key="save_change"):
                    if new_key.startswith("sk-"):
                        ok, err = secrets_save("ANTHROPIC_API_KEY", new_key)
                        if ok:
                            st.session_state.pop("show_key_input", None)
                            st.success("Saved!")
                            st.rerun()
                        else:
                            st.error(f"Could not save: {err}")
                    else:
                        st.error("Key must start with sk-")
            api_key = saved_key

        else:
            st.markdown("🔑 **API Key**")
            api_key = st.text_input("Anthropic API Key", type="password",
                                    placeholder="sk-ant-...",
                                    key="api_key_inp",
                                    label_visibility="collapsed")
            if api_key:
                if not api_key.startswith("sk-"):
                    st.error("Key must start with sk-")
                    api_key = ""
                else:
                    if st.button("💾 Save to Secrets", use_container_width=True):
                        ok, err = secrets_save("ANTHROPIC_API_KEY", api_key)
                        if ok:
                            st.success("✓ Saved! Auto-loads next time.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Save failed: {err}")
            with st.expander("ℹ️ About Streamlit Secrets"):
                st.markdown(f"""
Saves to **`{SECRETS_FILE}`** on this machine.  
Loaded automatically via `st.secrets` on restart.

Manual setup:
```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
                """)

        st.markdown("---")

        # ── Model ─────────────────────────────────────────────
        model = st.selectbox(
            "🤖 Model",
            ["claude-sonnet-4-6", "claude-opus-4-6"],
            index=0,
            help="Sonnet: fast & cheap. Opus: most accurate.")
        st.session_state["model"] = model

        st.markdown("---")

        # ── Fill Color ────────────────────────────────────────
        st.markdown("### 🎨 Fill Color")
        color_opt = st.radio("Text color", [
            "Black (blend in)",
            "Dark Blue (visible)",
            "Red (review mode)",
        ], index=0)
        color_map = {
            "Black (blend in)":    (0.0, 0.0, 0.0),
            "Dark Blue (visible)": (0.0, 0.0, 0.6),
            "Red (review mode)":   (0.8, 0.0, 0.0),
        }
        fill_color = color_map[color_opt]

        st.markdown("---")
        st.markdown("### ℹ️ How it works")
        st.markdown("""
**Phase 0** — Classifies each PDF (blank form vs source doc)  
**Phase 1** — Detects every field in the form  
**Phase 2** — Reads source docs, translates if needed  
**Phase 3** — Maps data → fill instructions  
**Phase 4** — Renders the filled PDF  

Supports Greek, English, French, German and more.
        """)

    return api_key, model, fill_color


# ──────────────────────────────────────────────────────────────
# PHASE CARD UI
# ──────────────────────────────────────────────────────────────

def phase_card(ph: dict, num: int, title: str, desc: str):
    state = ph.get(num, "waiting")
    cls = f"phase-card {state}"
    nc = "phase-num " + ("done" if state == "done" else "error" if state == "error" else "")
    icon = "✓" if state == "done" else "✗" if state == "error" else str(num)
    st.markdown(
        f'<div class="{cls}"><span class="{nc}">{icon}</span>'
        f'<span class="phase-title">{title}</span>'
        f'<div class="phase-desc">{desc}</div></div>',
        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    api_key, model, fill_color = render_sidebar()

    st.markdown('<div class="big-title">📋 FormFill AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload any PDFs → AI classifies them → fills the form automatically</div>',
        unsafe_allow_html=True)
    st.markdown("---")

    # ── Upload — ALL files together ───────────────────────────
    st.markdown("#### 📂 Upload all your PDFs")
    st.markdown(
        "Upload **everything** — the blank form AND the source documents. "
        "The AI will automatically figure out which is which.")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s) uploaded:**")
        for f in uploaded:
            st.markdown(f"&nbsp;&nbsp;• `{f.name}` ({f.size // 1024} KB)")

    st.markdown("---")

    ready = bool(api_key and uploaded)
    if not ready:
        missing = []
        if not api_key:   missing.append("API key (sidebar)")
        if not uploaded:  missing.append("PDF files")
        st.info(f"⏳ Waiting for: {', '.join(missing)}")
        return

    run = st.button("🚀 Classify & Fill Automatically",
                    disabled=not ready, use_container_width=True, type="primary")
    if not run:
        return

    # ── Run pipeline ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔄 Processing")

    ph_state: dict[int, str] = {0: "waiting", 1: "waiting",
                                 2: "waiting", 3: "waiting", 4: "waiting"}
    slots = {i: st.empty() for i in range(5)}

    def render_phases():
        defs = {
            0: ("🔍 Classifying Documents",         "AI reads each PDF and decides: blank form vs source doc"),
            1: ("📐 Analyzing Form Structure",       "AI maps every field in the blank form with coordinates"),
            2: ("📖 Extracting & Translating Data",  "AI reads source docs, translates non-English content"),
            3: ("🧠 Mapping Data → Fields",          "AI generates precise fill instructions for each field"),
            4: ("✍️ Rendering Filled PDF",            "Overlaying extracted data onto the original form"),
        }
        for n, (title, desc) in defs.items():
            with slots[n]:
                phase_card(ph_state, n, title, desc)

    render_phases()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        t0 = time.time()

        # Prepare upload data
        uploads = [{"name": f.name, "bytes": f.read()} for f in uploaded]

        # ── Phase 0: Classify ─────────────────────────────────
        ph_state[0] = "active"; render_phases()
        classified = phase0_classify(client, uploads)

        forms   = [d for d in classified if d.get("role") == "form"]
        sources = [d for d in classified if d.get("role") == "source"]

        ph_state[0] = "done"; render_phases()

        with st.expander(f"✓ Phase 0 — Classified {len(classified)} documents", expanded=True):
            for doc in classified:
                role = doc.get("role", "unknown")
                badge = f'<span class="doc-badge doc-{role}">{role.upper()}</span>'
                lang  = doc.get("language", "")
                desc  = doc.get("description", "")
                st.markdown(
                    f'{badge} **{doc["name"]}**'
                    + (f' · {lang}' if lang else '')
                    + f'<br><span style="font-size:12px;color:#555;margin-left:4px">{desc}</span>',
                    unsafe_allow_html=True)

        if not forms:
            st.error("❌ No blank form detected among the uploaded files. "
                     "Please upload the form you want to fill.")
            ph_state[1] = ph_state[2] = ph_state[3] = ph_state[4] = "error"
            render_phases()
            return

        if len(forms) > 1:
            st.warning(f"⚠️ {len(forms)} forms detected — using: **{forms[0]['name']}**")

        form_doc = forms[0]
        st.success(f"✓ Form to fill: **{form_doc['name']}**  |  "
                   f"Source docs: {', '.join(d['name'] for d in sources) or 'none'}")

        # ── Phase 1: Form structure ───────────────────────────
        ph_state[1] = "active"; render_phases()
        fields = phase1_analyze_form(client, form_doc)
        ph_state[1] = "done"; render_phases()

        with st.expander(f"✓ Phase 1 — {len(fields)} fields detected", expanded=False):
            # Group by section
            sections: dict[str, list] = {}
            for f in fields:
                s = f.get("section", "Other")
                sections.setdefault(s, []).append(f)
            for sec, sec_fields in sections.items():
                st.markdown(f"**{sec}** ({len(sec_fields)} fields)")
                pills = " ".join(
                    f'<span style="background:#e8e8f5;border-radius:4px;'
                    f'padding:2px 6px;font-size:11px;font-family:monospace">'
                    f'{f["label"]}</span>'
                    for f in sec_fields)
                st.markdown(pills, unsafe_allow_html=True)

        # ── Phase 2: Extract data ─────────────────────────────
        ph_state[2] = "active"; render_phases()

        if not sources:
            st.warning("⚠️ No source documents detected. Only the form structure was analyzed.")
            data = {}
        else:
            data = phase2_extract_sources(client, sources)

        ph_state[2] = "done"; render_phases()

        with st.expander(f"✓ Phase 2 — {len(data) - 1} data points extracted", expanded=False):
            for s in data.get("_summaries", []):
                st.markdown(s)
            clean = {k: v for k, v in data.items() if not k.startswith("_")}
            st.markdown(
                '<div class="json-display">' +
                json.dumps(clean, ensure_ascii=False, indent=2) +
                '</div>', unsafe_allow_html=True)

        # ── Phase 3: Map & fill ───────────────────────────────
        ph_state[3] = "active"; render_phases()
        fills, notes, skipped = phase3_map_and_fill(client, fields, data, form_doc)
        ph_state[3] = "done"; render_phases()

        with st.expander(f"✓ Phase 3 — {len(fills)} fields filled, {len(skipped)} skipped",
                         expanded=False):
            if notes:
                st.info(notes)
            if skipped:
                st.markdown(f"**Skipped** (no data): {', '.join(skipped[:20])}")
            st.markdown(
                '<div class="json-display">' +
                json.dumps(fills[:20], ensure_ascii=False, indent=2) +
                ('...' if len(fills) > 20 else '') +
                '</div>', unsafe_allow_html=True)

        # ── Phase 4: Render ───────────────────────────────────
        ph_state[4] = "active"; render_phases()
        font_path = get_font()
        font_name = register_font(font_path) if font_path else "Helvetica"
        filled_bytes = phase4_render(form_doc["bytes"], fills, font_name, fill_color)
        ph_state[4] = "done"; render_phases()

        # ── Result ────────────────────────────────────────────
        elapsed = time.time() - t0
        st.markdown("---")
        st.markdown(f"""
<div class="result-box">
  <h3 style="color:#2da44e;margin:0">✅ Form Filled Successfully!</h3>
  <p style="color:#555;margin:8px 0 0">
    {len(fills)} fields filled &nbsp;·&nbsp; completed in {elapsed:.1f}s
  </p>
</div>
""", unsafe_allow_html=True)
        st.markdown("")

        out_name = form_doc["name"].replace(".pdf", "_filled.pdf")
        st.download_button(
            "⬇️ Download Filled PDF",
            data=filled_bytes,
            file_name=out_name,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

        # Preview
        st.markdown("#### Preview")
        preview_imgs = pdf_to_images(filled_bytes, max_pages=3)
        cols = st.columns(min(len(preview_imgs), 3))
        for i, img in enumerate(preview_imgs[:3]):
            with cols[i]:
                st.image(img, caption=f"Page {i+1}", use_container_width=True)

    except anthropic.AuthenticationError:
        for n in ph_state:
            ph_state[n] = "error"
        render_phases()
        st.error("❌ Invalid API key. Check your key in the sidebar.")

    except json.JSONDecodeError as e:
        st.error(f"❌ Claude returned invalid JSON: {e}. Try again.")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
