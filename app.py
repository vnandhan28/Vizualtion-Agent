# app.py

import streamlit as st
from AgentClass import Agent, create_client
import pandas as pd

# --- Page Setup ---
st.set_page_config(page_title="AI DataViz Agent", layout="wide")
st.title("📊 AI-Powered Data Visualization Agent")
st.markdown("Upload your dataset and ask natural-language questions to generate interactive plots using LLMs.")

# --- API & Agent Setup ---
api_key = st.secrets["HF_key"]  # Or use st.text_input("Enter API key:")
client = create_client(api_key)
agent = Agent(client)

# --- Session Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File Upload ---
uploaded = st.file_uploader("📎 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded:
    try:
        # Read file
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)

        st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)

        # --- User Question ---
        question = st.text_input("💬 Ask a question about the data:")
        if question:
            with st.spinner("🤖 Thinking..."):
                result = agent.answer(question, {"data": df})

            st.subheader("📈 Result")
            if result.kind == "plotly":
                st.plotly_chart(result.obj, use_container_width=True)
            elif result.kind == "matplotlib":
                st.pyplot(result.obj)
            elif result.kind == "altair":
                st.altair_chart(result.obj, use_container_width=True)

            # Save result in session history
            st.session_state.chat_history.append({
                "q": question,
                "a": result.explanation,
                "kind": result.kind,
                "chart": result.obj
            })

        # --- History Viewer ---
        if st.session_state.chat_history:
            st.markdown("## 🕘 Previous Queries")
            for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"{i}. {entry['q']}"):
                    st.markdown(f"**Explanation:** {entry['a']}")
                    if entry["kind"] == "plotly":
                        st.plotly_chart(entry["chart"], use_container_width=True)
                    elif entry["kind"] == "matplotlib":
                        st.pyplot(entry["chart"])
                    elif entry["kind"] == "altair":
                        st.altair_chart(entry["chart"], use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load or process file: {e}")

else:
    st.info("Upload a `.csv` or `.xlsx` file to begin.")
