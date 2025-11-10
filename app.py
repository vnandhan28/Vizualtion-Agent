# app.py (streamlit)

import streamlit as st
from AgentClass import Agent, create_client

api_key = st.secrets["HF_key"]   # or st.text_input(...)
client = create_client(api_key)

agent = Agent(client)

uploaded = st.file_uploader("Upload CSV")
if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)

    question = st.text_input("Ask something about the data:")
    if question:
        result = agent.answer(question, {"data": df})

        if result.kind == "plotly":
            st.plotly_chart(result.obj)
        elif result.kind == "matplotlib":
            st.pyplot(result.obj)
        else:
            st.altair_chart(result.obj)
