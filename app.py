import os
import json
import yaml
from io import BytesIO
from datetime import datetime

import streamlit as st

from markdown import markdown  # For HTML export
from PyPDF2 import PdfReader

# ----- LLM clients -----
from openai import OpenAI
import anthropic
from openai import OpenAI as GrokClient
from google import genai  # Adjust to your chosen Gemini client

# ---------------------------------------------------
# 1. Configuration & constants
# ---------------------------------------------------

PAINTER_STYLES = [
    "Monet (Soft Pastel Clinical)",
    "Van Gogh (Expressive Teal Swirls)",
    "Picasso (Cubist Panels)",
    "Kandinsky (Abstract Geometry)",
    "Hokusai (Wave Motif)",
    "Rothko (Minimalist Blocks)",
    "Matisse (Papercut Shapes)",
    "Frida Kahlo (Vivid Focused)",
    "Dali (Surreal Gradients)",
    "Pollock (Energy Speckles)",
    "Rembrandt (Chiaroscuro)",
    "Vermeer (Soft Light)",
    "Seurat (Pointillist Dots)",
    "Cézanne (Structured Forms)",
    "Degas (Soft Motion)",
    "Magritte (Conceptual)",
    "Mondrian (Grid & Primary)",
    "Chagall (Dreamlike)",
    "O’Keeffe (Organic Forms)",
    "Bauhaus (Clinical Modern)"
]

MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

DEFAULT_MAX_TOKENS = 12000

STRINGS = {
    "en": {
        "title": "WOW FDA 510(k) Analyzer",
        "subtitle": "Upload or paste a 510(k) Summary and get a structured, interactive regulatory dashboard.",
        "upload_label": "Upload 510(k) file (PDF, TXT, MD, JSON)",
        "paste_label": "Or paste 510(k) text / markdown",
        "analyze_button": "Analyze 510(k)",
        "dashboard_tab": "Dashboard & Chat",
        "landing_tab": "Upload / Paste 510(k)",
        "pipeline_tab": "Agent Pipeline (Advanced)",
        "status_input_ready": "Input Ready",
        "status_parsing": "Parsing 510(k)",
        "status_dashboard": "Dashboard Generated",
        "status_chat": "Chat Ready",
        "select_model": "Select Model",
        "max_tokens": "Max tokens",
        "agent_prompt": "Agent system prompt",
        "agent_output": "Agent output (editable for next step)",
        "run_agent": "Run agent",
        "run_all": "Run full pipeline",
        "chat_placeholder": "Ask a question about this 510(k) summary...",
        "api_keys_section": "API Keys",
        "missing_keys_note": "If environment keys are not set, please enter them here. They will not be displayed.",
        "download_md": "Download Summary (Markdown)",
        "download_html": "Download Themed Report (HTML)",
        "appearance_section": "Output Appearance",
        "font_scale": "Font size scale",
        "density": "Layout density",
        "density_comfortable": "Comfortable",
        "density_compact": "Compact",
        "history_section": "Document History",
        "save_snapshot": "Save current document snapshot",
        "select_snapshot": "Load snapshot",
        "no_snapshots": "No snapshots saved yet.",
        "checklist_title": "Regulatory Completeness Checklist",
        "checklist_present": "Present",
        "checklist_missing": "Missing",
    },
    "zh_tw": {
        "title": "WOW FDA 510(k) 分析器",
        "subtitle": "上傳或貼上 510(k) 摘要，獲得結構化、互動式法規儀表板。",
        "upload_label": "上傳 510(k) 檔案（PDF、TXT、MD、JSON）",
        "paste_label": "或貼上 510(k) 文字／Markdown",
        "analyze_button": "分析 510(k)",
        "dashboard_tab": "儀表板與對話",
        "landing_tab": "上傳／貼上 510(k)",
        "pipeline_tab": "多代理流程（進階）",
        "status_input_ready": "輸入就緒",
        "status_parsing": "解析 510(k)",
        "status_dashboard": "已產出儀表板",
        "status_chat": "對話就緒",
        "select_model": "選擇模型",
        "max_tokens": "最大 tokens",
        "agent_prompt": "代理系統提示詞",
        "agent_output": "代理輸出（可編輯作為下一步輸入）",
        "run_agent": "執行代理",
        "run_all": "執行完整流程",
        "chat_placeholder": "針對此 510(k) 摘要提出問題……",
        "api_keys_section": "API 金鑰",
        "missing_keys_note": "若環境變數未設定，請在此輸入。系統不會顯示金鑰內容。",
        "download_md": "下載摘要（Markdown）",
        "download_html": "下載主題化報告（HTML）",
        "appearance_section": "輸出外觀",
        "font_scale": "字體大小比例",
        "density": "版面密度",
        "density_comfortable": "寬鬆",
        "density_compact": "緊湊",
        "history_section": "文件歷史",
        "save_snapshot": "儲存目前文件快照",
        "select_snapshot": "載入快照",
        "no_snapshots": "目前尚無快照。",
        "checklist_title": "法規完整性檢查表",
        "checklist_present": "已填入",
        "checklist_missing": "缺少",
    },
}

# Painter color palettes for WOW theming
PAINTER_PALETTES = [
    # 0-4
    {"accent": "#14b8a6", "accent_soft": "#ccfbf1", "card_bg_light": "#f9fafb", "card_bg_dark": "#020617"},
    {"accent": "#f97316", "accent_soft": "#ffedd5", "card_bg_light": "#fefce8", "card_bg_dark": "#030712"},
    {"accent": "#6366f1", "accent_soft": "#e0e7ff", "card_bg_light": "#eff6ff", "card_bg_dark": "#020617"},
    {"accent": "#ec4899", "accent_soft": "#fce7f3", "card_bg_light": "#fdf2ff", "card_bg_dark": "#020617"},
    {"accent": "#0ea5e9", "accent_soft": "#e0f2fe", "card_bg_light": "#f0f9ff", "card_bg_dark": "#020617"},
    # 5-9
    {"accent": "#a855f7", "accent_soft": "#f3e8ff", "card_bg_light": "#faf5ff", "card_bg_dark": "#020617"},
    {"accent": "#22c55e", "accent_soft": "#dcfce7", "card_bg_light": "#f0fdf4", "card_bg_dark": "#020617"},
    {"accent": "#e11d48", "accent_soft": "#ffe4e6", "card_bg_light": "#fff1f2", "card_bg_dark": "#020617"},
    {"accent": "#06b6d4", "accent_soft": "#cffafe", "card_bg_light": "#ecfeff", "card_bg_dark": "#020617"},
    {"accent": "#facc15", "accent_soft": "#fef9c3", "card_bg_light": "#fefce8", "card_bg_dark": "#020617"},
    # 10-14
    {"accent": "#4b5563", "accent_soft": "#e5e7eb", "card_bg_light": "#f9fafb", "card_bg_dark": "#020617"},
    {"accent": "#0f766e", "accent_soft": "#ccfbf1", "card_bg_light": "#ecfeff", "card_bg_dark": "#020617"},
    {"accent": "#7c3aed", "accent_soft": "#ede9fe", "card_bg_light": "#f5f3ff", "card_bg_dark": "#020617"},
    {"accent": "#3b82f6", "accent_soft": "#dbeafe", "card_bg_light": "#eff6ff", "card_bg_dark": "#020617"},
    {"accent": "#ea580c", "accent_soft": "#ffedd5", "card_bg_light": "#fff7ed", "card_bg_dark": "#020617"},
    # 15-19
    {"accent": "#0891b2", "accent_soft": "#e0f2fe", "card_bg_light": "#f0f9ff", "card_bg_dark": "#020617"},
    {"accent": "#f97316", "accent_soft": "#ffedd5", "card_bg_light": "#fff7ed", "card_bg_dark": "#020617"},
    {"accent": "#16a34a", "accent_soft": "#bbf7d0", "card_bg_light": "#f0fdf4", "card_bg_dark": "#020617"},
    {"accent": "#ec4899", "accent_soft": "#fce7f3", "card_bg_light": "#fff1f2", "card_bg_dark": "#020617"},
    {"accent": "#0ea5e9", "accent_soft": "#e0f2fe", "card_bg_light": "#f0f9ff", "card_bg_dark": "#020617"},
]


# ---------------------------------------------------
# 2. Utility: state initialization
# ---------------------------------------------------

def init_session_state():
    ss = st.session_state
    ss.setdefault("language", "en")
    ss.setdefault("theme_mode", "light")
    ss.setdefault("painter_style", 0)
    ss.setdefault("font_scale", 1.0)
    ss.setdefault("density", "comfortable")
    ss.setdefault("api_keys", {})
    ss.setdefault("agents_config", None)
    ss.setdefault("raw_input_text", "")
    ss.setdefault("structured_json", None)
    ss.setdefault("summary_markdown", None)
    ss.setdefault("infographics_layout", None)
    ss.setdefault("chat_history", [])
    ss.setdefault("pipeline_outputs", {})
    ss.setdefault("pipeline_status", {
        "input_ready": True,
        "parsing": False,
        "dashboard": False,
        "chat": False,
    })
    # NEW: multi-document snapshot history
    ss.setdefault("documents", [])
    ss.setdefault("current_doc_index", None)


# ---------------------------------------------------
# 3. Theming: CSS injection & HTML export
# ---------------------------------------------------

def inject_theme_css(theme_mode: str, painter_index: int, font_scale: float, density: str):
    palette = PAINTER_PALETTES[painter_index % len(PAINTER_PALETTES)]
    accent = palette["accent"]
    accent_soft = palette["accent_soft"]
    if theme_mode == "light":
        bg = "#f3f4f6"
        text = "#0f172a"
        card_bg = palette["card_bg_light"]
    else:
        bg = "#020617"
        text = "#e5e7eb"
        card_bg = palette["card_bg_dark"]

    padding = "1.1rem" if density == "comfortable" else "0.6rem"
    margin = "0.6rem" if density == "comfortable" else "0.3rem"
    border_radius = "14px"

    css = f"""
    <style>
    body {{
        background: {bg};
        color: {text};
        font-size: {font_scale}rem;
    }}
    .wow-card {{
        background: {card_bg};
        border-radius: {border_radius};
        border: 1px solid rgba(148, 163, 184, 0.5);
        padding: {padding};
        margin-bottom: {margin};
        box-shadow: 0 10px 25px rgba(15,23,42,0.05);
    }}
    .wow-highlight {{
        border-left: 4px solid {accent};
        background: {accent_soft};
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.75rem 0;
    }}
    .wow-kpi-title {{
        font-size: {0.85 * font_scale}rem;
        color: {text};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .wow-kpi-value {{
        font-size: {1.1 * font_scale}rem;
        font-weight: 600;
        color: {accent};
    }}
    .wow-section-title {{
        border-bottom: 2px solid {accent_soft};
        padding-bottom: 0.2rem;
        margin-bottom: 0.75rem;
    }}
    .wow-badge-present {{
        background: rgba(34,197,94,0.12);
        color: #16a34a;
        border-radius: 999px;
        padding: 0.1rem 0.7rem;
        font-size: 0.8rem;
    }}
    .wow-badge-missing {{
        background: rgba(239,68,68,0.12);
        color: #dc2626;
        border-radius: 999px;
        padding: 0.1rem 0.7rem;
        font-size: 0.8rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def generate_html_report(summary_md: str, lang: str, theme_mode: str,
                         painter_index: int, font_scale: float, density: str) -> str:
    """Generate a themed HTML report using current WOW theme settings."""
    palette = PAINTER_PALETTES[painter_index % len(PAINTER_PALETTES)]
    accent = palette["accent"]
    accent_soft = palette["accent_soft"]
    if theme_mode == "light":
        bg = "#f3f4f6"
        text = "#0f172a"
        card_bg = palette["card_bg_light"]
    else:
        bg = "#020617"
        text = "#e5e7eb"
        card_bg = palette["card_bg_dark"]

    padding = "1.2rem" if density == "comfortable" else "0.7rem"
    border_radius = "16px"

    html_body = markdown(summary_md, extensions=["tables", "fenced_code"])
    title = "WOW FDA 510(k) Report" if lang == "en" else "WOW FDA 510(k) 報告"

    html = f"""
    <!DOCTYPE html>
    <html lang="{lang}">
    <head>
        <meta charset="utf-8" />
        <title>{title}</title>
        <style>
        body {{
            margin: 0;
            padding: 1.5rem;
            background: {bg};
            color: {text};
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: {font_scale}rem;
        }}
        .wow-container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .wow-card {{
            background: {card_bg};
            border-radius: {border_radius};
            border: 1px solid rgba(148, 163, 184, 0.6);
            padding: {padding};
            box-shadow: 0 18px 45px rgba(15,23,42,0.12);
        }}
        h1, h2, h3, h4 {{
            color: {text};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0.5rem 0 1.2rem 0;
        }}
        th, td {{
            border: 1px solid rgba(148,163,184,0.5);
            padding: 0.4rem 0.6rem;
        }}
        th {{
            background: {accent_soft};
        }}
        a {{
            color: {accent};
        }}
        </style>
    </head>
    <body>
        <div class="wow-container">
            <div class="wow-card">
                {html_body}
            </div>
        </div>
    </body>
    </html>
    """
    return html


# ---------------------------------------------------
# 4. API key handling
# ---------------------------------------------------

def get_api_key(name: str, session_key: str):
    env_val = os.getenv(name)
    if env_val:
        return env_val
    return st.session_state["api_keys"].get(session_key)


def render_api_key_section(strings):
    st.subheader(strings["api_keys_section"])
    st.caption(strings["missing_keys_note"])

    if not os.getenv("GEMINI_API_KEY"):
        key = st.text_input("Gemini API Key", type="password")
        if key:
            st.session_state["api_keys"]["gemini"] = key
    else:
        st.text("Gemini API: Using environment configuration")

    if not os.getenv("OPENAI_API_KEY"):
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            st.session_state["api_keys"]["openai"] = key
    else:
        st.text("OpenAI API: Using environment configuration")

    if not os.getenv("ANTHROPIC_API_KEY"):
        key = st.text_input("Anthropic API Key", type="password")
        if key:
            st.session_state["api_keys"]["anthropic"] = key
    else:
        st.text("Anthropic API: Using environment configuration")

    if not os.getenv("GROK_API_KEY"):
        key = st.text_input("Grok (xAI) API Key", type="password")
        if key:
            st.session_state["api_keys"]["grok"] = key
    else:
        st.text("Grok API: Using environment configuration")


# ---------------------------------------------------
# 5. LLM call wrappers
# ---------------------------------------------------

def get_gemini_client():
    key = get_api_key("GEMINI_API_KEY", "gemini")
    if not key:
        raise RuntimeError("Gemini API key not configured.")
    client = genai.Client(api_key=key)
    return client


def get_openai_client():
    key = get_api_key("OPENAI_API_KEY", "openai")
    if not key:
        raise RuntimeError("OpenAI API key not configured.")
    return OpenAI(api_key=key)


def get_anthropic_client():
    key = get_api_key("ANTHROPIC_API_KEY", "anthropic")
    if not key:
        raise RuntimeError("Anthropic API key not configured.")
    return anthropic.Anthropic(api_key=key)


def get_grok_client():
    key = get_api_key("GROK_API_KEY", "grok")
    if not key:
        raise RuntimeError("Grok API key not configured.")
    return GrokClient(api_key=key, base_url="https://api.x.ai/v1")


def call_model(provider, model, system_prompt, user_content, max_tokens, schema=None):
    if provider == "gemini":
        client = get_gemini_client()
        config = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
        }
        if schema:
            config["response_schema"] = json.loads(schema)
            config["response_mime_type"] = "application/json"

        result = client.models.generate_content(
            model=model,
            contents=user_content,
            config=config
        )
        return result.text

    elif provider == "openai":
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        client = get_anthropic_client()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text

    elif provider == "grok":
        client = get_grok_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------
# 6. File & text handling
# ---------------------------------------------------

def extract_text_from_file(uploaded_file):
    suffix = uploaded_file.name.lower()
    data = uploaded_file.read()
    bio = BytesIO(data)

    if suffix.endswith(".pdf"):
        reader = PdfReader(bio)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text

    if suffix.endswith(".txt") or suffix.endswith(".md") or suffix.endswith(".markdown"):
        return data.decode("utf-8", errors="ignore")

    if suffix.endswith(".json"):
        raw = data.decode("utf-8", errors="ignore")
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for key in ["summary", "content", "text", "body"]:
                    if key in obj and isinstance(obj[key], str):
                        return obj[key]
            return raw
        except Exception:
            return raw

    return data.decode("utf-8", errors="ignore")


# ---------------------------------------------------
# 7. WOW status indicators
# ---------------------------------------------------

def render_status_indicators(strings):
    st.markdown("### Status")
    status = st.session_state["pipeline_status"]

    def status_chip(label, active):
        color = "#22c55e" if active else "#9ca3af"
        dot = f"<span style='color:{color};font-size:18px;'>&#9679;</span>"
        st.markdown(f"{dot} **{label}**", unsafe_allow_html=True)

    status_chip(strings["status_input_ready"], status["input_ready"])
    status_chip(strings["status_parsing"], status["parsing"])
    status_chip(strings["status_dashboard"], status["dashboard"])
    status_chip(strings["status_chat"], status["chat"])


# ---------------------------------------------------
# 8. Agents pipeline handling
# ---------------------------------------------------

def load_agents_config():
    if st.session_state["agents_config"] is None:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        st.session_state["agents_config"] = cfg["agents"]


def run_single_agent(agent_def, input_text, agent_idx):
    agent_id = agent_def["id"]
    custom_model = st.session_state.get(f"agent_{agent_id}_model", agent_def["default_model"])
    custom_max_tokens = st.session_state.get(
        f"agent_{agent_id}_max_tokens", agent_def.get("max_tokens", DEFAULT_MAX_TOKENS)
    )
    custom_prompt = st.session_state.get(
        f"agent_{agent_id}_prompt", agent_def["system_prompt"]
    )

    output = call_model(
        provider=agent_def["provider"],
        model=custom_model,
        system_prompt=custom_prompt,
        user_content=input_text,
        max_tokens=int(custom_max_tokens),
        schema=agent_def.get("schema"),
    )
    st.session_state["pipeline_outputs"][agent_id] = output
    return output


def run_full_pipeline():
    load_agents_config()
    agents = st.session_state["agents_config"]
    text_input = st.session_state["raw_input_text"]

    if not text_input.strip():
        st.warning("No 510(k) text found. Please upload or paste content.")
        return

    status = st.session_state["pipeline_status"]
    status["parsing"] = True
    status["dashboard"] = False
    status["chat"] = False

    last_output = text_input
    progress = st.progress(0)

    for idx, agent in enumerate(agents):
        with st.spinner(f"Running {agent['label']}..."):
            last_output = run_single_agent(agent, last_output, idx)
        progress.progress((idx + 1) / len(agents))

    structured_id = "extract_510k_structured"
    summary_id = "generate_dashboard_summary"
    infographics_id = "design_infographics"

    if structured_id in st.session_state["pipeline_outputs"]:
        try:
            st.session_state["structured_json"] = json.loads(
                st.session_state["pipeline_outputs"][structured_id]
            )
        except Exception:
            st.session_state["structured_json"] = None

    if summary_id in st.session_state["pipeline_outputs"]:
        st.session_state["summary_markdown"] = st.session_state["pipeline_outputs"][summary_id]

    if infographics_id in st.session_state["pipeline_outputs"]:
        try:
            st.session_state["infographics_layout"] = json.loads(
                st.session_state["pipeline_outputs"][infographics_id]
            )
        except Exception:
            st.session_state["infographics_layout"] = None

    status["parsing"] = False
    status["dashboard"] = True
    status["chat"] = True

    # NEW: auto-save snapshot after successful run
    save_current_document_snapshot(auto=True)


# ---------------------------------------------------
# 9. Document snapshot history (NEW FEATURE)
# ---------------------------------------------------

def build_snapshot_label():
    structured = st.session_state.get("structured_json") or {}
    device_name = structured.get("device_name")
    if device_name:
        return device_name
    text = st.session_state.get("raw_input_text", "") or ""
    snippet = text.strip().replace("\n", " ")
    return (snippet[:50] + "...") if len(snippet) > 50 else (snippet or "Untitled 510(k)")


def save_current_document_snapshot(auto: bool = False):
    if not st.session_state.get("raw_input_text", "").strip():
        return
    label = build_snapshot_label()
    snapshot = {
        "label": label,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "raw_input_text": st.session_state.get("raw_input_text"),
        "structured_json": st.session_state.get("structured_json"),
        "summary_markdown": st.session_state.get("summary_markdown"),
        "infographics_layout": st.session_state.get("infographics_layout"),
        "chat_history": st.session_state.get("chat_history", []),
    }
    st.session_state["documents"].append(snapshot)
    st.session_state["current_doc_index"] = len(st.session_state["documents"]) - 1
    if not auto:
        st.success("Snapshot saved for this 510(k) document.")


def load_document_snapshot(index: int):
    docs = st.session_state.get("documents", [])
    if index is None or index < 0 or index >= len(docs):
        return
    doc = docs[index]
    st.session_state["raw_input_text"] = doc["raw_input_text"]
    st.session_state["structured_json"] = doc["structured_json"]
    st.session_state["summary_markdown"] = doc["summary_markdown"]
    st.session_state["infographics_layout"] = doc["infographics_layout"]
    st.session_state["chat_history"] = doc.get("chat_history", [])
    st.session_state["current_doc_index"] = index


def render_history_section(strings):
    st.subheader(strings["history_section"])
    if st.button(strings["save_snapshot"]):
        save_current_document_snapshot(auto=False)

    docs = st.session_state.get("documents", [])
    if not docs:
        st.caption(strings["no_snapshots"])
        return

    labels = [
        f"{i + 1}. {doc['label']} ({doc['timestamp']})"
        for i, doc in enumerate(docs)
    ]
    current = st.session_state.get("current_doc_index") or 0
    selected = st.selectbox(strings["select_snapshot"], options=list(range(len(labels))),
                            format_func=lambda idx: labels[idx], index=current)
    if selected != current:
        load_document_snapshot(selected)
        st.experimental_rerun()


# ---------------------------------------------------
# 10. Dashboard rendering (with WOW cards & checklist)
# ---------------------------------------------------

def render_completeness_checklist(strings, structured):
    st.markdown(f"### {strings['checklist_title']}")
    required_fields = {
        "submitter_information": "Submitter information",
        "device_name": "Device name",
        "classification": "Classification",
        "regulation_number": "Regulation number",
        "product_code": "Product code",
        "panel": "Panel",
        "predicates": "Predicate devices",
        "indications_for_use": "Indications for use",
        "technological_characteristics": "Technological characteristics",
        "performance_data": "Performance data",
        "clinical_performance": "Clinical performance",
        "substantial_equivalence_discussion": "Substantial equivalence discussion",
    }

    cols = st.columns(2)
    items = list(required_fields.items())
    mid = len(items) // 2

    def render_list(sub_items):
        for key, label in sub_items:
            present = bool(structured.get(key))
            badge_class = "wow-badge-present" if present else "wow-badge-missing"
            status_label = strings["checklist_present"] if present else strings["checklist_missing"]
            st.markdown(
                f"<div style='margin-bottom:0.35rem;'>"
                f"<span>{label}</span> "
                f"<span class='{badge_class}'>{status_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with cols[0]:
        render_list(items[:mid])
    with cols[1]:
        render_list(items[mid:])


def render_dashboard(strings):
    st.subheader("510(k) Structured Summary")

    structured = st.session_state.get("structured_json") or {}
    summary_md = st.session_state.get("summary_markdown")

    with st.container():
        st.markdown("<div class='wow-card'>", unsafe_allow_html=True)

        if structured:
            cols = st.columns(3)
            with cols[0]:
                st.markdown("<div class='wow-kpi-title'>Device Name</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='wow-kpi-value'>{structured.get('device_name', 'N/A')}</div>",
                            unsafe_allow_html=True)
            with cols[1]:
                st.markdown("<div class='wow-kpi-title'>Product Code</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='wow-kpi-value'>{structured.get('product_code', 'N/A')}</div>",
                            unsafe_allow_html=True)
            with cols[2]:
                st.markdown("<div class='wow-kpi-title'>Panel</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='wow-kpi-value'>{structured.get('panel', 'N/A')}</div>",
                            unsafe_allow_html=True)

            st.markdown("<hr/>", unsafe_allow_html=True)

            # Completeness checklist (NEW FEATURE)
            render_completeness_checklist(strings, structured)

            # Key sections
            if "submitter_information" in structured:
                st.markdown("<h4 class='wow-section-title'>Submitter Information</h4>", unsafe_allow_html=True)
                sub = structured["submitter_information"]
                rows = "\n".join(f"| {k} | {v} |" for k, v in sub.items())
                st.markdown("| Field | Value |\n|---|---|\n" + rows)

            st.markdown("<h4 class='wow-section-title'>Indications for Use</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='wow-highlight'>{structured.get('indications_for_use', 'N/A')}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<h4 class='wow-section-title'>Technological Characteristics</h4>", unsafe_allow_html=True)
            st.write(structured.get("technological_characteristics", "N/A"))
        else:
            st.info("Structured data not available yet. Run the pipeline to generate it.")

        if summary_md:
            st.markdown("<h4 class='wow-section-title'>Narrative Summary</h4>", unsafe_allow_html=True)
            st.markdown(summary_md, unsafe_allow_html=False)
        else:
            st.info("Summary not available yet. Run the pipeline to generate it.")

        if st.session_state.get("infographics_layout"):
            st.markdown("<h4 class='wow-section-title'>Infographics & Tables</h4>", unsafe_allow_html=True)
            layout = st.session_state["infographics_layout"]
            st.json(layout)
        else:
            st.markdown("<h4 class='wow-section-title'>Infographics & Tables</h4>", unsafe_allow_html=True)
            st.caption("Infographic layout will appear here after running the ‘Design Infographics’ agent.")

        st.markdown("</div>", unsafe_allow_html=True)

    # NEW: Report export buttons
    if summary_md:
        col1, col2 = st.columns(2)
        with col1:
            md_bytes = summary_md.encode("utf-8")
            st.download_button(
                label=strings["download_md"],
                data=md_bytes,
                file_name="510k_summary.md",
                mime="text/markdown"
            )
        with col2:
            html_report = generate_html_report(
                summary_md,
                lang=st.session_state["language"],
                theme_mode=st.session_state["theme_mode"],
                painter_index=st.session_state["painter_style"],
                font_scale=st.session_state["font_scale"],
                density=st.session_state["density"],
            )
            st.download_button(
                label=strings["download_html"],
                data=html_report.encode("utf-8"),
                file_name="510k_report_themed.html",
                mime="text/html"
            )


# ---------------------------------------------------
# 11. Chat panel
# ---------------------------------------------------

def render_chat_panel(strings):
    render_status_indicators(strings)

    st.markdown("### Contextual Chat")
    for msg in st.session_state["chat_history"]:
        role = msg["role"]
        if role == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    question = st.text_input(strings["chat_placeholder"], key="chat_input")
    model = st.selectbox(strings["select_model"], MODEL_OPTIONS, key="chat_model")

    if st.button("Send", key="chat_send"):
        if not question.strip():
            return

        st.session_state["chat_history"].append({"role": "user", "content": question})

        context_parts = []
        if st.session_state.get("structured_json"):
            context_parts.append("STRUCTURED_JSON:\n" + json.dumps(st.session_state["structured_json"], indent=2))
        if st.session_state.get("summary_markdown"):
            context_parts.append("SUMMARY_MARKDOWN:\n" + st.session_state["summary_markdown"])

        context = "\n\n".join(context_parts) if context_parts else "No 510(k) data loaded yet."
        user_content = context + "\n\nUSER_QUESTION:\n" + question

        if model.startswith("gemini"):
            provider = "gemini"
        elif model.startswith("gpt-4"):
            provider = "openai"
        elif model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("grok"):
            provider = "grok"
        else:
            provider = "gemini"

        answer = call_model(
            provider=provider,
            model=model,
            system_prompt=(
                "You are an expert regulatory consultant. "
                "Answer strictly based on the provided 510(k) content. If unsure, say so."
            ),
            user_content=user_content,
            max_tokens=2000,
        )
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.experimental_rerun()


# ---------------------------------------------------
# 12. Agent pipeline UI
# ---------------------------------------------------

def render_pipeline_panel(strings):
    load_agents_config()
    agents = st.session_state["agents_config"]

    st.markdown(f"### {strings['pipeline_tab']}")

    if st.button(strings["run_all"]):
        run_full_pipeline()

    for agent in agents:
        agent_id = agent["id"]
        st.markdown("---")
        st.markdown(f"#### {agent['label']}")

        default_model = agent["default_model"]
        model_key = f"agent_{agent_id}_model"
        st.session_state.setdefault(model_key, default_model)
        st.selectbox(
            strings["select_model"],
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(default_model) if default_model in MODEL_OPTIONS else 0,
            key=model_key,
        )

        max_tokens_key = f"agent_{agent_id}_max_tokens"
        st.session_state.setdefault(max_tokens_key, agent.get("max_tokens", DEFAULT_MAX_TOKENS))
        st.number_input(
            strings["max_tokens"],
            min_value=100,
            max_value=120000,
            step=100,
            key=max_tokens_key,
        )

        prompt_key = f"agent_{agent_id}_prompt"
        st.session_state.setdefault(prompt_key, agent["system_prompt"])
        st.text_area(
            strings["agent_prompt"],
            key=prompt_key,
            height=150,
        )

        if agent_id == agents[0]["id"]:
            default_input = st.session_state["raw_input_text"]
        else:
            prev_agent = agents[agents.index(agent) - 1]
            default_input = st.session_state["pipeline_outputs"].get(prev_agent["id"], "")

        input_key = f"agent_{agent_id}_input"
        st.session_state.setdefault(input_key, default_input)
        st.text_area(
            "Input to this agent (editable)",
            key=input_key,
            height=150,
        )

        output_key = f"agent_{agent_id}_output"
        st.session_state.setdefault(output_key, st.session_state["pipeline_outputs"].get(agent_id, ""))

        if st.button(strings["run_agent"], key=f"run_{agent_id}"):
            with st.spinner(f"Running {agent['label']}..."):
                output = run_single_agent(agent, st.session_state[input_key], agents.index(agent))
                st.session_state[output_key] = output
                if agent_id == "extract_510k_structured":
                    try:
                        st.session_state["structured_json"] = json.loads(output)
                    except Exception:
                        st.session_state["structured_json"] = None
                elif agent_id == "generate_dashboard_summary":
                    st.session_state["summary_markdown"] = output
                elif agent_id == "design_infographics":
                    try:
                        st.session_state["infographics_layout"] = json.loads(output)
                    except Exception:
                        st.session_state["infographics_layout"] = None

        st.text_area(
            strings["agent_output"],
            key=output_key,
            height=200,
        )


# ---------------------------------------------------
# 13. Main app
# ---------------------------------------------------

def main():
    st.set_page_config(
        page_title="WOW FDA 510(k) Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    with st.sidebar:
        st.markdown("## WOW Controls")
        st.session_state["theme_mode"] = st.radio(
            "Theme", ["light", "dark"],
            index=0 if st.session_state["theme_mode"] == "light" else 1
        )
        st.session_state["language"] = st.radio(
            "Language", ["en", "zh_tw"],
            format_func=lambda x: "English" if x == "en" else "繁體中文"
        )
        st.session_state["painter_style"] = st.slider(
            "Magic Style Wheel", 0, len(PAINTER_STYLES) - 1, st.session_state["painter_style"]
        )
        st.caption(f"Style: {PAINTER_STYLES[st.session_state['painter_style']]}")

        strings = STRINGS[st.session_state["language"]]

        # NEW: Output appearance controls
        st.markdown(f"### {strings['appearance_section']}")
        st.session_state["font_scale"] = st.slider(
            strings["font_scale"], 0.85, 1.3, st.session_state["font_scale"], step=0.05
        )
        density_label = st.selectbox(
            strings["density"],
            ["comfortable", "compact"],
            format_func=lambda v: strings["density_comfortable"] if v == "comfortable" else strings["density_compact"],
        )
        st.session_state["density"] = density_label

        render_api_key_section(strings)

        # NEW: Document history controls
        st.markdown("---")
        render_history_section(strings)

    strings = STRINGS[st.session_state["language"]]

    # Inject WOW theme CSS for the whole page
    inject_theme_css(
        theme_mode=st.session_state["theme_mode"],
        painter_index=st.session_state["painter_style"],
        font_scale=st.session_state["font_scale"],
        density=st.session_state["density"],
    )

    st.title(strings["title"])
    st.caption(strings["subtitle"])

    tab_landing, tab_dashboard, tab_pipeline = st.tabs([
        strings["landing_tab"],
        strings["dashboard_tab"],
        strings["pipeline_tab"],
    ])

    with tab_landing:
        col1, col2 = st.columns(2)
        with col1:
            uploaded = st.file_uploader(strings["upload_label"], type=["pdf", "txt", "md", "markdown", "json"])
            if uploaded:
                text = extract_text_from_file(uploaded)
                st.session_state["raw_input_text"] = text
                st.success("File loaded into workspace.")

        with col2:
            st.text_area(
                strings["paste_label"],
                key="raw_input_text",
                height=280,
            )

        if st.button(strings["analyze_button"]):
            if not st.session_state["raw_input_text"].strip():
                st.warning("Please upload or paste 510(k) content first.")
            else:
                run_full_pipeline()

    with tab_dashboard:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            render_dashboard(strings)
        with col_right:
            render_chat_panel(strings)

    with tab_pipeline:
        render_pipeline_panel(strings)


if __name__ == "__main__":
    main()
