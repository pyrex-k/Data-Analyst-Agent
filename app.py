"""
Streamlit web interface for the data-analyst agent.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.agent import TextStep, ThinkingStep, ToolCallStep, run_agent

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analyst Agent")
st.caption("Powered by Claude claude-opus-4-8 with adaptive thinking + tool use")

# ─── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    show_steps   = st.checkbox("Show agent steps", value=True)
    show_thinking = st.checkbox("Show thinking excerpts", value=False)

    st.divider()
    st.header("💡 Example questions")
    examples = [
        "Load the sales dataset from data/sales_data.csv and describe it",
        "Which region generated the most total revenue?",
        "What are the top 5 products by profit margin?",
        "Show monthly revenue trends with a line chart",
        "Is there a correlation between units sold and revenue?",
        "Which marketing channel drives the highest average order value?",
        "Compare revenue across product categories with a bar chart",
        "What percentage of revenue comes from each region? (pie chart)",
        "Filter to Electronics only and show which products are most profitable",
        "Create a full sales performance report",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.divider()
    st.markdown("**Session outputs** are saved to `outputs/`")

# ─── chat history ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# Display past exchanges
for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["query"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        if item.get("steps") and show_steps:
            with st.expander("Agent steps", expanded=False):
                for s in item["steps"]:
                    _render_step(s, show_thinking) if False else None

# ─── query input ──────────────────────────────────────────────────────────────
query = st.chat_input(
    "Ask a question about your data…",
    key="chat_input",
)
# Allow sidebar buttons to pre-fill the input
if not query and "query_input" in st.session_state:
    query = st.session_state.pop("query_input")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        step_container = st.container()
        final_placeholder = st.empty()

        collected_steps: list[TextStep | ThinkingStep | ToolCallStep] = []

        def on_step(step: TextStep | ThinkingStep | ToolCallStep) -> None:
            collected_steps.append(step)
            if not show_steps:
                return
            with step_container:
                if isinstance(step, ThinkingStep):
                    if show_thinking:
                        with st.expander("🧠 Thinking…", expanded=False):
                            st.text(step.thinking[:800] + ("…" if len(step.thinking) > 800 else ""))
                elif isinstance(step, TextStep):
                    st.info(step.text)
                elif isinstance(step, ToolCallStep):
                    with st.expander(f"🔧 `{step.tool_name}`", expanded=False):
                        st.json(step.tool_input)
                        if "error" in step.tool_result:
                            st.error(step.tool_result["error"])
                        elif "path" in step.tool_result:
                            path = step.tool_result["path"]
                            st.success(f"Saved: {path}")
                            if path.endswith(".png"):
                                try:
                                    st.image(path)
                                except Exception:
                                    pass
                        elif "chart_saved" in step.tool_result:
                            path = step.tool_result["chart_saved"]
                            st.success(f"Saved: {path}")
                            try:
                                st.image(path)
                            except Exception:
                                pass
                        else:
                            st.json(step.tool_result)

        with st.spinner("Analysing…"):
            result = run_agent(query, on_step=on_step)

        final_placeholder.markdown(result.final_answer)
        st.caption(f"Iterations: {result.iterations}  |  Stop reason: {result.stop_reason}")

        # Show any charts that were generated
        output_dir = Path("outputs")
        if output_dir.exists():
            new_pngs = sorted(output_dir.glob("*.png"))
            if new_pngs:
                st.subheader("Generated charts")
                cols = st.columns(min(len(new_pngs), 2))
                for i, png in enumerate(new_pngs[-4:]):   # show last 4
                    with cols[i % 2]:
                        st.image(str(png), caption=png.stem, use_container_width=True)

    st.session_state.history.append({
        "query": query,
        "answer": result.final_answer,
        "steps": collected_steps,
    })
