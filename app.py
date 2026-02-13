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

if sample_choice != "None" and "df" not in st.session_state:
    st.session_state.df = samples[sample_choice]

# side bar
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.chat_messages = []
        if "df" in st.session_state:
            del st.session_state.df
        if "agent" in st.session_state:
            st.session_state.agent.memory.clear()
        st.rerun()

    st.markdown("---")
    
    current_df = st.session_state.get("df")
    if current_df is not None:
        st.subheader("📊 Dataset Overview")
        st.write(f"**Rows:** {current_df.shape[0]}")
        st.write(f"**Columns:** {current_df.shape[1]}")
        
        with st.expander("📝 Column Details"):
            for col in current_df.columns:
                dtype = current_df[col].dtype
                nulls = current_df[col].isnull().sum()
                st.write(f"- **{col}** ({dtype}) | {nulls} nulls")

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

if uploaded:
    try:
        file_bytes = uploaded.read()
        file_io = io.BytesIO(file_bytes)

        if uploaded.name.endswith(".csv"):
            file_io.seek(0)
            new_df = pd.read_csv(file_io)
        else:
            file_io.seek(0)
            new_df = pd.read_excel(file_io, engine="openpyxl")

        st.session_state.df = new_df
        st.success(f" Uploaded `{uploaded.name}` — {new_df.shape[0]} rows × {new_df.shape[1]} columns")

    except Exception as e:
        st.error(f" Failed to load file: {e}")

elif sample_choice != "None":
    st.success(f" Loaded sample dataset: {sample_choice}")

df = st.session_state.get("df")

#  user interface 
if df is None:
    st.info(" Upload a `.csv` or `.xlsx` file OR use a sample dataset.")
else:
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 💬 Ask a question")

    # exisitng msgs
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "chart" in msg:
                st.write(msg["chart"])
                # Add download buttons for history
                if hasattr(msg["chart"], "savefig"): # Matplotlib
                    buf = io.BytesIO()
                    msg["chart"].savefig(buf, format="png", bbox_inches='tight')
                    st.download_button("💾 Download Plot", buf.getvalue(), f"plot_{st.session_state.chat_messages.index(msg)}.png", "image/png", key=f"dl_{st.session_state.chat_messages.index(msg)}")
                elif hasattr(msg["chart"], "to_html"): # Plotly
                    html_str = msg["chart"].to_html()
                    st.download_button("🌐 Download HTML", html_str, f"chart_{st.session_state.chat_messages.index(msg)}.html", "text/html", key=f"dl_{st.session_state.chat_messages.index(msg)}")

    # Suggested Questions
    if df is not None and len(st.session_state.chat_messages) == 0:
        st.markdown("##### 💡 Suggested Questions")
        cols = ", ".join(df.columns[:3])
        suggestions = [
            f"Show a summary of {df.columns[0]}",
            f"Compare {df.columns[0]} and {df.columns[1]}",
            f"Visualize distribution of {df.columns[0]}"
        ]
        
        cols_ui = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            if cols_ui[i].button(s, use_container_width=True):
                st.session_state.pending_prompt = s
                st.rerun()

    # input box
    prompt = st.chat_input("Ask something about the data...")
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Run agent
        with st.spinner("🧠 Thinking..."):
            try:
                chart_result, explanation = st.session_state.agent.ask(prompt, {"data": df})

                # Handle data prep updates
                if chart_result and chart_result.kind == "data_prep" and chart_result.modified_datasets:
                    if "data" in chart_result.modified_datasets:
                        st.session_state.df = chart_result.modified_datasets["data"]
                        st.success("✅ Dataset updated successfully!")
                        st.rerun()

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
                    
                    # Download buttons
                    if chart_result.kind == "matplotlib":
                        buf = io.BytesIO()
                        chart_result.obj.savefig(buf, format="png", bbox_inches='tight')
                        st.download_button("💾 Download Plot", buf.getvalue(), "plot.png", "image/png")
                    elif chart_result.kind == "plotly":
                        html_str = chart_result.obj.to_html()
                        st.download_button("🌐 Download HTML", html_str, "chart.html", "text/html")

            except Exception as e:
                st.error(f" Error: {e}")
