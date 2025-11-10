# vibrant_app.py

import streamlit as st
from AgentClass import Agent, create_client
import pandas as pd
import io

# --- Page Setup ---
st.set_page_config(page_title="🔍 AI DataViz Agent", layout="wide", page_icon="📊")
st.title("📊 AI-Powered Data Visualization Agent")
st.markdown(
    """
    Upload a **CSV or Excel** dataset and ask natural-language questions like:
    
    - "Show a 3D scatter plot of Age, BMI, and Overall_Risk_Score colored by Cancer_Type."
    - "Create a stacked area chart showing how the number of Movies and TV Shows released each year has changed since 2000, and color it by type."
    - "Count number of rides by Booking Status and show as pie chart."
    """
)
# --- Sample datasets ---
@st.cache_data
def load_sample_data():
    cancer_df = pd.read_csv("cancer-risk-factors.csv")
    netflix_df = pd.read_csv("netflix_titles.csv")
    uber_df = pd.read_csv("ncr_ride_bookings.csv")
    return {
        "Cancer risk factors": cancer_df,
        "Netflix title": netflix_df,
        "Uber Bookings" : uber_df,
    }

samples = load_sample_data()
sample_names = list(samples.keys())

sample_choice = st.selectbox("Or try a sample dataset:", ["None"] + sample_names)
# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧹 Clear chat history"):
        st.session_state.chat_history = []

# --- Agent Setup ---
api_key = st.secrets["HF_key"]
client = create_client(api_key)
agent = Agent(client)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File Upload ---
st.markdown("### 📎 Upload Your Dataset")
uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded:
    try:
        file_bytes = uploaded.read()
        file_io = io.BytesIO(file_bytes)

        if uploaded.name.endswith(".csv"):
            file_io.seek(0)
            df = pd.read_csv(file_io)
        elif uploaded.name.endswith(".xlsx"):
            file_io.seek(0)
            df = pd.read_excel(file_io, engine="openpyxl")

        st.success(f"✅ Loaded `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} columns")
        with st.expander("📄 Preview Data"):
            st.dataframe(df.head(), use_container_width=True)

        # --- Ask a Question ---
        st.markdown("### 💬 Ask a Question")
        question = st.text_input("Type your question here:")

        if question:
            with st.spinner(" Thinking..."):
                result = agent.answer(question, {"data": df})

            st.markdown("### 📈 Answer")
            if result.kind == "plotly":
                st.plotly_chart(result.obj, use_container_width=True)
            elif result.kind == "matplotlib":
                st.pyplot(result.obj)
            elif result.kind == "altair":
                st.altair_chart(result.obj, use_container_width=True)

            st.success("✅ Chart generated successfully!")
            st.session_state.chat_history.append({
                "q": question,
                "a": result.explanation,
                "kind": result.kind,
                "chart": result.obj
            })

        # --- History ---
        if st.session_state.chat_history:
            st.markdown("### 🕘 Previous Queries")
            for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"{i}. {entry['q']}"):
                    st.markdown(f"**Explanation:** {entry['a']}")
                    if entry["kind"] == "plotly":
                        st.plotly_chart(entry["chart"], use_container_width=True, key=f"plotly-{i}")
                    elif entry["kind"] == "matplotlib":
                        st.pyplot(entry["chart"], key=f"matplotlib-{i}")
                    elif entry["kind"] == "altair":
                        st.altair_chart(entry["chart"], use_container_width=True, key=f"altair-{i}")

    except Exception as e:
        st.error(f"❌ Failed to load or process file: `{e}`")
else:
    st.info("👆 Upload a `.csv` or `.xlsx` file to begin.")
