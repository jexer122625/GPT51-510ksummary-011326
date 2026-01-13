import os
import json
import yaml
from io import BytesIO

import streamlit as st

# ----- LLM clients -----
from openai import OpenAI
import anthropic
# For Grok (xAI) - you may need to adapt base_url depending on your setup
from openai import OpenAI as GrokClient
# Gemini Python client (adjust import to your chosen library)
from google import google-generativeai

from PyPDF2 import PdfReader

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
    },
}

# ---------------------------------------------------
# 2. Utility: state initialization
# ---------------------------------------------------

def init_session_state():
    ss = st.session_state
    ss.setdefault("language", "en")
    ss.setdefault("theme_mode", "light")
    ss.setdefault("painter_style", 0)
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

# ---------------------------------------------------
# 3. API key handling
# ---------------------------------------------------

def get_api_key(name: str, session_key: str):
    # Priority: env var -> session state input -> None
    env_val = os.getenv(name)
    if env_val:
        return env_val
    return st.session_state["api_keys"].get(session_key)

def render_api_key_section(strings):
    st.subheader(strings["api_keys_section"])
    st.caption(strings["missing_keys_note"])

    # Gemini
    if not os.getenv("GEMINI_API_KEY"):
        key = st.text_input("Gemini API Key", type="password")
        if key:
            st.session_state["api_keys"]["gemini"] = key
    else:
        st.text("Gemini API: Using environment configuration")

    # OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            st.session_state["api_keys"]["openai"] = key
    else:
        st.text("OpenAI API: Using environment configuration")

    # Anthropic
    if not os.getenv("ANTHROPIC_API_KEY"):
        key = st.text_input("Anthropic API Key", type="password")
        if key:
            st.session_state["api_keys"]["anthropic"] = key
    else:
        st.text("Anthropic API: Using environment configuration")

    # Grok
    if not os.getenv("GROK_API_KEY"):
        key = st.text_input("Grok (xAI) API Key", type="password")
        if key:
            st.session_state["api_keys"]["grok"] = key
    else:
        st.text("Grok API: Using environment configuration")

# ---------------------------------------------------
# 4. LLM call wrappers
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
    # Adjust base_url as needed for xAI's Grok API
    return GrokClient(api_key=key, base_url="https://api.x.ai/v1")

def call_model(provider, model, system_prompt, user_content, max_tokens, schema=None):
    """
    Generic LLM call.
    schema: for Gemini structured output (response_schema)
    """
    if provider == "gemini":
        client = get_gemini_client()
        # Adjust according to your installed Gemini client library
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
        # If schema enforced, result.text should be JSON
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
# 5. File & text handling
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
                # Heuristic: look for main text fields
                for key in ["summary", "content", "text", "body"]:
                    if key in obj and isinstance(obj[key], str):
                        return obj[key]
            return raw
        except Exception:
            return raw

    # Fallback
    return data.decode("utf-8", errors="ignore")

# ---------------------------------------------------
# 6. WOW status indicators
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
# 7. Agents pipeline handling
# ---------------------------------------------------

def load_agents_config():
    if st.session_state["agents_config"] is None:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        st.session_state["agents_config"] = cfg["agents"]

def run_single_agent(agent_def, input_text, agent_idx):
    # Determine overrides
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

    # Save important pieces
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

# ---------------------------------------------------
# 8. Dashboard rendering
# ---------------------------------------------------

def render_dashboard(strings):
    st.subheader("510(k) Structured Summary")

    structured = st.session_state.get("structured_json") or {}
    summary_md = st.session_state.get("summary_markdown")

    if structured:
        # KPI cards example
        cols = st.columns(3)
        with cols[0]:
            st.metric("Device Name", structured.get("device_name", "N/A"))
        with cols[1]:
            st.metric("Product Code", structured.get("product_code", "N/A"))
        with cols[2]:
            st.metric("Panel", structured.get("panel", "N/A"))

        # Submitter table
        if "submitter_information" in structured:
            st.markdown("#### Submitter Information")
            sub = structured["submitter_information"]
            rows = "\n".join(
                f"| {k} | {v} |" for k, v in sub.items()
            )
            st.markdown(
                "| Field | Value |\n|---|---|\n" + rows
            )

        # Indications & tech characteristics
        st.markdown("#### Indications for Use")
        st.info(structured.get("indications_for_use", "N/A"))

        st.markdown("#### Technological Characteristics")
        st.write(structured.get("technological_characteristics", "N/A"))

    if summary_md:
        st.markdown("#### Narrative Summary")
        st.markdown(summary_md, unsafe_allow_html=False)
    else:
        st.info("Summary not available yet. Run the pipeline to generate it.")

    # Placeholder for infographics: user layout-driven charts
    if st.session_state.get("infographics_layout"):
        st.markdown("#### Infographics & Tables")
        layout = st.session_state["infographics_layout"]
        st.json(layout)
    else:
        st.markdown("#### Infographics & Tables")
        st.caption("Infographic layout will appear here after running the ‘Design Infographics’ agent.")

# ---------------------------------------------------
# 9. Chat panel
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

        # Build context from structured JSON + summary
        context_parts = []
        if st.session_state.get("structured_json"):
            context_parts.append("STRUCTURED_JSON:\n" + json.dumps(st.session_state["structured_json"], indent=2))
        if st.session_state.get("summary_markdown"):
            context_parts.append("SUMMARY_MARKDOWN:\n" + st.session_state["summary_markdown"])

        context = "\n\n".join(context_parts) if context_parts else "No 510(k) data loaded yet."
        user_content = context + "\n\nUSER_QUESTION:\n" + question

        # Simple heuristic: choose provider by model name
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
# 10. Agent pipeline UI
# ---------------------------------------------------

def render_pipeline_panel(strings):
    load_agents_config()
    agents = st.session_state["agents_config"]

    st.markdown(f"### {strings['pipeline_tab']}")

    if st.button(strings["run_all"]):
        run_full_pipeline()

    for agent in agents:
        agent_id = agent["id"]
        st.markdown(f"---")
        st.markdown(f"#### {agent['label']}")

        # Model selector
        default_model = agent["default_model"]
        model_key = f"agent_{agent_id}_model"
        st.session_state.setdefault(model_key, default_model)
        st.selectbox(
            strings["select_model"],
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(default_model) if default_model in MODEL_OPTIONS else 0,
            key=model_key,
        )

        # Max tokens
        max_tokens_key = f"agent_{agent_id}_max_tokens"
        st.session_state.setdefault(max_tokens_key, agent.get("max_tokens", DEFAULT_MAX_TOKENS))
        st.number_input(
            strings["max_tokens"],
            min_value=100,
            max_value=120000,
            step=100,
            key=max_tokens_key,
        )

        # Prompt editor
        prompt_key = f"agent_{agent_id}_prompt"
        st.session_state.setdefault(prompt_key, agent["system_prompt"])
        st.text_area(
            strings["agent_prompt"],
            key=prompt_key,
            height=150,
        )

        # Input for this agent (previous agent’s output, editable)
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

        # Output area
        output_key = f"agent_{agent_id}_output"
        st.session_state.setdefault(output_key, st.session_state["pipeline_outputs"].get(agent_id, ""))

        if st.button(strings["run_agent"], key=f"run_{agent_id}"):
            with st.spinner(f"Running {agent['label']}..."):
                output = run_single_agent(agent, st.session_state[input_key], agents.index(agent))
                st.session_state[output_key] = output
                # If it's one of the key agents, update global structured/summary/infographics
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
# 11. Main app
# ---------------------------------------------------

def main():
    st.set_page_config(
        page_title="WOW FDA 510(k) Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    # Sidebar: Theme, language, style, API
    with st.sidebar:
        st.markdown("## WOW Controls")
        st.session_state["theme_mode"] = st.radio(
            "Theme", ["light", "dark"], index=0 if st.session_state["theme_mode"] == "light" else 1
        )
        st.session_state["language"] = st.radio(
            "Language", ["en", "zh_tw"], format_func=lambda x: "English" if x == "en" else "繁體中文"
        )
        st.session_state["painter_style"] = st.slider(
            "Magic Style Wheel", 0, len(PAINTER_STYLES) - 1, st.session_state["painter_style"]
        )
        st.caption(f"Style: {PAINTER_STYLES[st.session_state['painter_style']]}")

        strings = STRINGS[st.session_state["language"]]
        render_api_key_section(strings)

    strings = STRINGS[st.session_state["language"]]

    st.title(strings["title"])
    st.caption(strings["subtitle"])

    tab_landing, tab_dashboard, tab_pipeline = st.tabs([
        strings["landing_tab"],
        strings["dashboard_tab"],
        strings["pipeline_tab"],
    ])

    # Landing: upload/paste + analyze button
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

    # Dashboard & Chat
    with tab_dashboard:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            render_dashboard(strings)
        with col_right:
            render_chat_panel(strings)

    # Agent pipeline (advanced controls)
    with tab_pipeline:
        render_pipeline_panel(strings)


if __name__ == "__main__":
    main()
