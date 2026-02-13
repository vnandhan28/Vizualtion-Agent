import streamlit as st
import pandas as pd
import io
from langchain_agent import LangChainVizAgent

st.set_page_config(
    page_title="🔍 AI DataViz Agent",
    layout="wide",
    page_icon="📊"
)

st.title("📊 AI-Powered Data Visualization Agent")
st.markdown(
    """
Upload a **CSV or Excel** dataset and ask natural-language questions like:

- "Show a 3D scatter plot of Age, BMI, and Overall_Risk_Score colored by Cancer_Type."
- "Create a stacked area chart showing Movies and TV Shows released each year."
- "Count number of rides by Booking Status and show as a pie chart."

"""
)

@st.cache_data
def load_sample_data():
    cancer_df = pd.read_csv("cancer-risk-factors.csv")
    netflix_df = pd.read_csv("netflix_titles.csv")
    uber_df = pd.read_csv("ncr_ride_bookings.csv")
    return {
        "Cancer risk factors": cancer_df,
        "Netflix titles": netflix_df,
        "Uber bookings": uber_df,
    }

samples = load_sample_data()
sample_names = list(samples.keys())
sample_choice = st.selectbox(" Try a sample dataset:", ["None"] + sample_names)

#side bar
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🧹 Clear Conversation"):
        st.session_state.chat_messages = []
        if "agent" in st.session_state:
            st.session_state.agent.memory.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("🧠 Memory Panel")
    if "agent" in st.session_state:
        messages = st.session_state.agent.memory.chat_memory.messages
        for msg in messages[-4:]:
            role = "User" if msg.type == "human" else "Assistant"
            st.write(f"**{role}**: {msg.content}")

# agent connection
api_key = st.secrets["HF_key"]

if "agent" not in st.session_state:
    st.session_state.agent = LangChainVizAgent(api_key)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# files upload 
st.markdown("### 📎 Upload Your Dataset")
uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

df = None

if uploaded:
    try:
        file_bytes = uploaded.read()
        file_io = io.BytesIO(file_bytes)

        if uploaded.name.endswith(".csv"):
            file_io.seek(0)
            df = pd.read_csv(file_io)
        else:
            file_io.seek(0)
            df = pd.read_excel(file_io, engine="openpyxl")

        st.success(f" Uploaded `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} columns")

    except Exception as e:
        st.error(f" Failed to load file: {e}")

elif sample_choice != "None":
    df = samples[sample_choice]
    st.success(f" Loaded sample dataset: {sample_choice}")

#  user interface 
if df is None:
    st.info(" Upload a `.csv` or `.xlsx` file OR use a sample dataset.")
else:
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 💬 Ask a question")

    # exisitng msgs
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            block = st.chat_message("assistant")
            block.markdown(msg["content"])
            if "chart" in msg:
                st.write(msg["chart"])

    # input box
    prompt = st.chat_input("Ask something about the data...")

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Run agent
        with st.spinner("🧠 Thinking..."):
            try:
                chart_result, explanation = st.session_state.agent.ask(prompt, {"data": df})

                # Save to UI chat
                msg_to_save = {
                    "role": "assistant",
                    "content": explanation,
                }
                if chart_result:
                    msg_to_save["chart"] = chart_result.obj
                
                st.session_state.chat_messages.append(msg_to_save)

                # display result
                block = st.chat_message("assistant")
                block.markdown(explanation)
                if chart_result:
                    st.write(chart_result.obj)

            except Exception as e:
                st.error(f" Error: {e}")
