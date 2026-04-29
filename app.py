"""
FormFill AI — Streamlit app that uses Claude Vision to automatically fill PDF forms.

Pipeline:
  Phase 1 → Claude analyzes the blank form → identifies fields + positions
  Phase 2 → Claude reads source documents → extracts structured data
  Phase 3 → Claude maps data to fields → produces fill instructions (JSON)
  Phase 4 → reportlab overlays the text onto the original PDF
"""

import streamlit as st
import anthropic
import base64
import json
import io
import os
import time
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
# PAGE CONFIG & STYLE
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

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f0f13;
    border-right: 1px solid #2a2a35;
  }
  section[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
  section[data-testid="stSidebar"] .stTextInput input {
    background: #1a1a22 !important;
    border: 1px solid #3a3a50 !important;
    color: #e8e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
  }

  /* Main */
  .main .block-container { padding: 2rem 2rem 4rem; max-width: 1200px; }

  /* Phase cards */
  .phase-card {
    background: #fafafa;
    border: 1.5px solid #e8e8ec;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    position: relative;
  }
  .phase-card.active  { border-color: #5b5bd6; background: #f5f5ff; }
  .phase-card.done    { border-color: #2da44e; background: #f0fff4; }
  .phase-card.waiting { opacity: 0.5; }

  .phase-num {
    display: inline-block;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #5b5bd6;
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
    line-height: 28px;
    margin-right: 10px;
  }
  .phase-num.done { background: #2da44e; }

  .phase-title { font-weight: 600; font-size: 15px; color: #1a1a2e; }
  .phase-desc  { font-size: 13px; color: #6b6b80; margin-top: 4px; margin-left: 38px; }

  /* Upload area */
  [data-testid="stFileUploader"] {
    border: 2px dashed #d0d0e0 !important;
    border-radius: 10px !important;
    padding: 12px !important;
  }

  /* Result box */
  .result-box {
    background: linear-gradient(135deg, #f0fff4 0%, #e8f5ff 100%);
    border: 1.5px solid #2da44e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }

  /* Monospace JSON display */
  .json-display {
    background: #1a1a22;
    color: #7ec8a4;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
    white-space: pre;
    max-height: 260px;
    overflow-y: auto;
  }

  /* Field pill */
  .field-pill {
    display: inline-block;
    background: #5b5bd6;
    color: white;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    margin: 3px;
    font-family: 'Space Mono', monospace;
  }

  .big-title {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
  }
  .subtitle { font-size: 15px; color: #6b6b80; margin-top: 4px; }
  .divider  { border: none; border-top: 1px solid #e8e8ec; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
]

def get_greek_font() -> str | None:
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            return fp
    return None


def register_font(font_path: str) -> str:
    """Register a TTF with reportlab and return the registered name."""
    name = "FormFont"
    try:
        pdfmetrics.registerFont(TTFont(name, font_path))
    except Exception:
        name = "Helvetica"
    return name


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 6, dpi: int = 150) -> list[Image.Image]:
    """Convert PDF bytes → list of PIL Images (one per page, up to max_pages)."""
    if HAS_PDF2IMAGE:
        try:
            pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
            return pages
        except Exception:
            pass
    # Fallback: blank placeholder if conversion unavailable
    img = Image.new("RGB", (595, 842), (255, 255, 255))
    return [img]


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode()


def pdf_bytes_to_b64(pdf_bytes: bytes) -> str:
    return base64.standard_b64encode(pdf_bytes).decode()


def build_image_content(images_b64: list[str]) -> list[dict]:
    """Build Claude content blocks for multiple images."""
    content = []
    for i, b64 in enumerate(images_b64):
        content.append({"type": "text", "text": f"--- Page {i+1} ---"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
    return content


# ──────────────────────────────────────────────────────────────
# CLAUDE PIPELINE — 3 PHASES
# ──────────────────────────────────────────────────────────────

SYSTEM_FORM_ANALYST = """You are an expert document analyst specializing in PDF form recognition.
You analyze form images and output ONLY valid JSON — no markdown, no explanation.
"""

SYSTEM_DATA_EXTRACTOR = """You are an expert data extraction specialist.
You read documents (IDs, certificates, policies, invoices, etc.) and extract all relevant information.
Output ONLY valid JSON — no markdown, no explanation.
"""

SYSTEM_FILL_PLANNER = """You are an expert at mapping extracted data to form fields with precise positioning.
You output ONLY valid JSON fill instructions — no markdown, no explanation.
Coordinates are expressed as percentages (0.0–1.0) of page width/height, measured from top-left.
"""


def phase1_analyze_form(client: anthropic.Anthropic, model: str,
                         form_images_b64: list[str]) -> dict:
    """
    Phase 1: Identify all fillable fields in the form.
    Returns dict with 'fields' list, each having: id, label, page, x, y, width, height, type, description.
    Coordinates are 0..1 percentages from top-left of the page.
    """
    content = build_image_content(form_images_b64)
    content.append({"type": "text", "text": """
Analyze these form pages carefully. Identify EVERY fillable area: text fields, checkboxes, radio buttons, date fields, etc.

For each field output:
- "id": a short snake_case identifier (e.g. "company_name", "tax_id", "birth_date")
- "label": the human-readable label text from the form (in the original language)
- "page": page number (1-based)
- "x": left edge of the ENTRY area as fraction of page width (0.0=left, 1.0=right)
- "y": top edge of the ENTRY area as fraction of page height (0.0=top, 1.0=bottom)
- "w": width of entry area as fraction of page width
- "h": height of entry area as fraction of page height
- "type": one of "text", "checkbox", "radio", "date", "number"
- "description": brief English description of what goes here

Return ONLY this JSON structure (no markdown):
{
  "fields": [
    {"id": "...", "label": "...", "page": 1, "x": 0.0, "y": 0.0, "w": 0.1, "h": 0.01, "type": "text", "description": "..."},
    ...
  ]
}

Be thorough — include all fields. Skip pre-printed labels and instructions.
"""})

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_FORM_ANALYST,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def phase2_extract_data(client: anthropic.Anthropic, model: str,
                         source_images_b64: list[str]) -> dict:
    """
    Phase 2: Extract all data from source documents.
    Returns dict with 'data' mapping field-type → value.
    """
    content = build_image_content(source_images_b64)
    content.append({"type": "text", "text": """
Read all these source documents carefully. Extract every piece of information that could be used to fill a form.

Return ONLY this JSON (no markdown):
{
  "data": {
    "category_name": "value",
    ...
  },
  "summary": "brief description of what documents were read"
}

Use descriptive keys like: "company_name", "tax_id", "address_street", "address_number",
"postal_code", "city", "phone_mobile", "email", "vehicle_plate", "vehicle_make",
"vehicle_model", "vehicle_vin", "vehicle_year", "vehicle_cc", "vehicle_hp",
"vehicle_seats", "vehicle_use", "vehicle_value", "insurance_program",
"policy_number", "young_driver_dob", "young_driver_under23", etc.

Include ALL values found in the documents. For dates use DD/MM/YYYY format.
For booleans use true/false.
"""})

    response = client.messages.create(
        model=model,
        max_tokens=3000,
        system=SYSTEM_DATA_EXTRACTOR,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def phase3_generate_fills(client: anthropic.Anthropic, model: str,
                           form_fields: list[dict], extracted_data: dict,
                           form_images_b64: list[str]) -> dict:
    """
    Phase 3: Map extracted data to form fields and generate fill instructions.
    Returns dict with 'fills' list of {page, x, y, w, h, text, font_size, field_id}.
    """
    fields_json = json.dumps(form_fields, ensure_ascii=False, indent=2)
    data_json   = json.dumps(extracted_data, ensure_ascii=False, indent=2)

    content = build_image_content(form_images_b64)
    content.append({"type": "text", "text": f"""
You have:
A) FORM FIELDS identified in the form (with their positions):
{fields_json}

B) DATA extracted from source documents:
{data_json}

Task: Match each piece of data to the appropriate form field. Generate precise fill instructions.

Rules:
- Only fill fields where you have matching data
- For checkboxes/radio buttons: use "●" for selected, leave empty for unselected
- For text fields: use the exact value from the data
- Adjust font_size (5–10) so the text fits within w×h (page fractions)
- x, y are the TOP-LEFT of where to write the text (use the field's x, y from form fields)
- For date fields formatted as DD|MM|YYYY boxes, format the date with spaces: "12  03  2004"

Return ONLY this JSON (no markdown):
{{
  "fills": [
    {{
      "field_id": "...",
      "page": 1,
      "x": 0.28,
      "y": 0.255,
      "text": "value to write",
      "font_size": 8
    }},
    ...
  ],
  "notes": "any important notes about the filling"
}}
"""})

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_FILL_PLANNER,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ──────────────────────────────────────────────────────────────
# PDF FILLING
# ──────────────────────────────────────────────────────────────

def fill_pdf(input_pdf_bytes: bytes, fills: list[dict],
             font_name: str, color: tuple = (0, 0, 0.6)) -> bytes:
    """
    Overlay fill instructions onto the PDF using reportlab.
    Returns filled PDF as bytes.
    """
    reader = PdfReader(io.BytesIO(input_pdf_bytes))
    writer = PdfWriter()

    # Group fills by page
    fills_by_page: dict[int, list] = {}
    for f in fills:
        p = int(f.get("page", 1))
        fills_by_page.setdefault(p, []).append(f)

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        mediabox = page.mediabox
        pdf_w = float(mediabox.width)
        pdf_h = float(mediabox.height)

        page_fills = fills_by_page.get(page_num, [])
        if page_fills:
            packet = io.BytesIO()
            c = canvas.Canvas(packet, pagesize=(pdf_w, pdf_h))
            c.setFillColorRGB(*color)

            for fill in page_fills:
                text      = str(fill.get("text", "")).strip()
                font_size = float(fill.get("font_size", 8))
                x_pct     = float(fill.get("x", 0))
                y_pct     = float(fill.get("y", 0))

                # Convert from % (top-left origin) to reportlab (bottom-left origin)
                x_pt  = x_pct * pdf_w
                y_top = y_pct * pdf_h
                y_rl  = pdf_h - y_top - font_size  # baseline

                if not text:
                    continue

                try:
                    c.setFont(font_name, font_size)
                except Exception:
                    c.setFont("Helvetica", font_size)

                c.drawString(x_pt, y_rl, text)

            c.save()
            packet.seek(0)

            overlay_reader = PdfReader(packet)
            overlay_page   = overlay_reader.pages[0]
            page.merge_page(overlay_page)

        writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ──────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────

SECRETS_FILE = Path(".streamlit/secrets.toml")


def load_saved_key() -> str:
    """Read ANTHROPIC_API_KEY from .streamlit/secrets.toml if it exists."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""


def save_key_to_secrets(key: str) -> bool:
    """Write the key to .streamlit/secrets.toml."""
    try:
        SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Read existing content so we don't overwrite other secrets
        existing: dict = {}
        if SECRETS_FILE.exists():
            import tomllib
            with open(SECRETS_FILE, "rb") as f:
                existing = tomllib.load(f)
        existing["ANTHROPIC_API_KEY"] = key
        # Write back (manual TOML serialisation — only one level deep needed)
        lines = []
        for k, v in existing.items():
            lines.append(f'{k} = "{v}"')
        SECRETS_FILE.write_text("\n".join(lines) + "\n")
        return True
    except Exception as e:
        st.sidebar.error(f"Could not save: {e}")
        return False


def delete_saved_key() -> bool:
    """Remove the key from secrets.toml."""
    try:
        if not SECRETS_FILE.exists():
            return True
        import tomllib
        with open(SECRETS_FILE, "rb") as f:
            existing = tomllib.load(f)
        existing.pop("ANTHROPIC_API_KEY", None)
        lines = [f'{k} = "{v}"' for k, v in existing.items()]
        SECRETS_FILE.write_text("\n".join(lines) + "\n")
        return True
    except Exception as e:
        st.sidebar.error(f"Could not delete: {e}")
        return False


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        # ── API Key ──────────────────────────────────────────
        saved_key = load_saved_key()
        key_is_saved = bool(saved_key)

        if key_is_saved:
            st.markdown(
                "🔑 **API Key** "
                '<span style="background:#2da44e;color:white;border-radius:20px;'
                'padding:2px 8px;font-size:11px">saved</span>',
                unsafe_allow_html=True,
            )
            masked = saved_key[:8] + "..." + saved_key[-4:] if len(saved_key) > 12 else "****"
            st.markdown(
                f'<code style="font-size:11px;color:#888">{masked}</code>',
                unsafe_allow_html=True,
            )
            col_change, col_delete = st.columns(2)
            with col_change:
                change = st.button("✏️ Change", use_container_width=True)
            with col_delete:
                if st.button("🗑 Remove", use_container_width=True):
                    delete_saved_key()
                    st.session_state.pop("show_key_input", None)
                    st.rerun()

            # Show input only if user clicked Change
            if change:
                st.session_state["show_key_input"] = True

            if st.session_state.get("show_key_input"):
                new_key = st.text_input(
                    "New API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    key="new_api_key_input",
                )
                if st.button("💾 Save new key", use_container_width=True):
                    if new_key.startswith("sk-"):
                        if save_key_to_secrets(new_key):
                            st.session_state.pop("show_key_input", None)
                            st.success("Saved! Restarting…")
                            st.rerun()
                    else:
                        st.error("Key must start with sk-")

            api_key = saved_key

        else:
            st.markdown("🔑 **API Key**")
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                key="api_key_input",
                label_visibility="collapsed",
            )
            if api_key:
                if not api_key.startswith("sk-"):
                    st.error("Key must start with sk-")
                    api_key = ""
                else:
                    if st.button("💾 Save to Secrets", use_container_width=True,
                                 help="Saves to .streamlit/secrets.toml on this machine"):
                        if save_key_to_secrets(api_key):
                            st.success("✓ Key saved! Will auto-load next time.")
                            st.rerun()
            else:
                st.markdown(
                    '<div style="font-size:12px;color:#888;margin-top:4px">'
                    'Enter your key above. Click <b>Save to Secrets</b> to persist it across restarts.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("ℹ️ Where is it stored?"):
                    st.markdown(f"""
Saved to **`{SECRETS_FILE}`** on this machine.  
Streamlit loads it automatically on startup via `st.secrets`.

To set it manually, create the file:
```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
                    """)

        st.markdown("---")

        # ── Model ────────────────────────────────────────────
        model = st.selectbox(
            "Model",
            ["claude-sonnet-4-6", "claude-opus-4-6"],
            index=0,
            help="Sonnet is faster & cheaper; Opus is more thorough",
        )

        st.markdown("---")
        st.markdown("### 🎨 Fill Color")
        fill_color_opt = st.radio(
            "Text color",
            ["Dark Blue (visible)", "Black (blend in)", "Red (review mode)"],
            index=0,
        )
        color_map = {
            "Dark Blue (visible)": (0.0, 0.0, 0.6),
            "Black (blend in)":    (0.0, 0.0, 0.0),
            "Red (review mode)":   (0.8, 0.0, 0.0),
        }
        fill_color = color_map[fill_color_opt]

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
**FormFill AI** uses Claude Vision to:
1. Detect all form fields
2. Extract data from your documents
3. Map & fill automatically

Supports Greek, English, and most European languages.
        """)

    return api_key, model, fill_color


def phase_card(num: int, title: str, desc: str, state: str = "waiting"):
    """Render a phase status card."""
    cls = f"phase-card {state}"
    num_cls = "phase-num done" if state == "done" else "phase-num"
    icon = "✓" if state == "done" else str(num)
    st.markdown(f"""
    <div class="{cls}">
      <span class="{num_cls}">{icon}</span>
      <span class="phase-title">{title}</span>
      <div class="phase-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    api_key, model, fill_color = render_sidebar()

    # Header
    st.markdown('<div class="big-title">📋 FormFill AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a blank form + source documents → get a filled PDF in seconds</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Upload section
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 📄 Blank Form")
        st.markdown("The PDF form you want to fill (application, request, etc.)")
        blank_form = st.file_uploader(
            "Upload blank form",
            type=["pdf"],
            key="form",
            label_visibility="collapsed",
        )
        if blank_form:
            st.success(f"✓ {blank_form.name} ({blank_form.size // 1024} KB)")

    with col_right:
        st.markdown("#### 📂 Source Documents")
        st.markdown("PDFs with the data to fill in (IDs, certificates, policies, etc.)")
        source_docs = st.file_uploader(
            "Upload source documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="sources",
            label_visibility="collapsed",
        )
        if source_docs:
            for doc in source_docs:
                st.success(f"✓ {doc.name} ({doc.size // 1024} KB)")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Process button
    ready = bool(api_key and blank_form and source_docs)
    if not ready:
        missing = []
        if not api_key:      missing.append("API key")
        if not blank_form:   missing.append("blank form")
        if not source_docs:  missing.append("source documents")
        st.info(f"⏳ Waiting for: {', '.join(missing)}")

    run_btn = st.button(
        "🚀 Fill Form Automatically",
        disabled=not ready,
        use_container_width=True,
        type="primary",
    )

    if not run_btn:
        # Show pipeline preview
        st.markdown("#### How it works")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Phase 1**\n\n🔍 Claude scans the blank form and identifies every field, checkbox, and input area.")
        with c2:
            st.markdown("**Phase 2**\n\n📖 Claude reads your source documents and extracts all relevant data.")
        with c3:
            st.markdown("**Phase 3**\n\n🧠 Claude maps the extracted data to the correct form fields.")
        with c4:
            st.markdown("**Phase 4**\n\n✍️ The form is filled and a new PDF is generated for download.")
        return

    # ── RUN PIPELINE ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔄 Processing")

    phase_placeholders = {i: st.empty() for i in range(1, 5)}

    def set_phase(n, state):
        labels = {
            1: ("🔍 Analyzing Form Structure",    "Claude scans the blank form and identifies all fillable fields"),
            2: ("📖 Extracting Data from Sources", "Claude reads your documents and extracts all relevant information"),
            3: ("🧠 Mapping Data to Fields",       "Claude determines what goes where and at what size"),
            4: ("✍️ Generating Filled PDF",         "Overlaying the extracted data onto the original form"),
        }
        title, desc = labels[n]
        with phase_placeholders[n]:
            phase_card(n, title, desc, state)

    for n in range(1, 5):
        set_phase(n, "waiting")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Read form bytes
        form_bytes = blank_form.read()

        # ── Phase 1: Analyze Form ─────────────────────────────
        set_phase(1, "active")
        t0 = time.time()

        form_images   = pdf_to_images(form_bytes, max_pages=4)
        form_imgs_b64 = [image_to_b64(img) for img in form_images]

        form_analysis = phase1_analyze_form(client, model, form_imgs_b64)
        fields        = form_analysis.get("fields", [])

        set_phase(1, "done")

        # Show detected fields
        with st.expander(f"✓ Phase 1 — {len(fields)} fields detected", expanded=False):
            pills = " ".join(f'<span class="field-pill">{f["label"]}</span>' for f in fields[:30])
            st.markdown(pills, unsafe_allow_html=True)
            st.markdown('<div class="json-display">' +
                        json.dumps(fields[:10], ensure_ascii=False, indent=2) +
                        ("..." if len(fields) > 10 else "") + "</div>",
                        unsafe_allow_html=True)

        # ── Phase 2: Extract Data ─────────────────────────────
        set_phase(2, "active")

        all_source_imgs_b64: list[str] = []
        for doc in source_docs:
            doc_bytes = doc.read()
            doc_imgs  = pdf_to_images(doc_bytes, max_pages=3)
            all_source_imgs_b64.extend(image_to_b64(img) for img in doc_imgs)

        extracted = phase2_extract_data(client, model, all_source_imgs_b64)
        data_dict = extracted.get("data", {})

        set_phase(2, "done")

        with st.expander(f"✓ Phase 2 — {len(data_dict)} data points extracted", expanded=False):
            st.markdown(f"*{extracted.get('summary', '')}*")
            st.markdown('<div class="json-display">' +
                        json.dumps(data_dict, ensure_ascii=False, indent=2) +
                        "</div>", unsafe_allow_html=True)

        # ── Phase 3: Generate Fill Instructions ───────────────
        set_phase(3, "active")

        fill_plan = phase3_generate_fills(client, model, fields, extracted, form_imgs_b64)
        fills     = fill_plan.get("fills", [])

        set_phase(3, "done")

        with st.expander(f"✓ Phase 3 — {len(fills)} fill instructions generated", expanded=False):
            if fill_plan.get("notes"):
                st.info(fill_plan["notes"])
            st.markdown('<div class="json-display">' +
                        json.dumps(fills[:15], ensure_ascii=False, indent=2) +
                        ("..." if len(fills) > 15 else "") + "</div>",
                        unsafe_allow_html=True)

        # ── Phase 4: Fill PDF ─────────────────────────────────
        set_phase(4, "active")

        font_path = get_greek_font()
        if font_path:
            font_name = register_font(font_path)
        else:
            font_name = "Helvetica"
            st.warning("Greek font not found — falling back to Helvetica (Greek chars may not render correctly).")

        filled_bytes = fill_pdf(form_bytes, fills, font_name, color=fill_color)

        set_phase(4, "done")

        elapsed = time.time() - t0
        st.markdown("---")

        # ── Result ────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
          <h3 style="color:#2da44e;margin:0">✅ Form Filled Successfully!</h3>
          <p style="color:#555;margin:8px 0 0">
            {len(fills)} fields filled · {len(fields)} fields detected · completed in {elapsed:.1f}s
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        out_name = blank_form.name.replace(".pdf", "_filled.pdf")
        st.download_button(
            label="⬇️ Download Filled PDF",
            data=filled_bytes,
            file_name=out_name,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

        # Preview
        st.markdown("#### Preview")
        filled_images = pdf_to_images(filled_bytes, max_pages=4)
        cols = st.columns(min(len(filled_images), 3))
        for i, img in enumerate(filled_images[:3]):
            with cols[i]:
                st.image(img, caption=f"Page {i+1}", use_container_width=True)

    except anthropic.AuthenticationError:
        for n in range(1, 5):
            set_phase(n, "waiting")
        st.error("❌ Invalid API key. Please check your Anthropic API key in the sidebar.")

    except json.JSONDecodeError as e:
        st.error(f"❌ Claude returned invalid JSON: {e}\n\nTry again — this can happen occasionally.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
