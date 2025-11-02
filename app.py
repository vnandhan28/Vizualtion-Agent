import streamlit as st
import pandas as pd

from agent_core import Agent  # your existing class
from agent_core import run_python_chart  # if defined separately

st.set_page_config(page_title="AI Data Viz Agent", layout="wide")
st.title("🧠📊 AI Data Visualization Agent")

st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

question = st.text_area("Ask a question about your dataset", placeholder="e.g. Compare number of completed vs cancelled rides")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("Generate Chart"):
        agent = Agent()  # or pass model name
        out = agent.answer(question, {"data": df}, prefer="plotly")
        if out.ok:
            st.success(out.result.explanation)
            if out.result.kind == "plotly":
                st.plotly_chart(out.result.obj, use_container_width=True)
            elif out.result.kind == "matplotlib":
                st.pyplot(out.result.obj)
            elif out.result.kind == "altair":
                st.altair_chart(out.result.obj, use_container_width=True)
        else:
            st.error("Agent failed to generate a chart.")
            st.exception(out.error)
