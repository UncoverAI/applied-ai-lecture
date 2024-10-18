import streamlit as st
import ollama

MODEL_ID = 'llama3.1:8b-instruct-q4_0'

st.markdown(f"### Prompt: ``{MODEL_ID}``")

if prompt := st.chat_input("Type your message"):
    st.write("You:<br>" + prompt, unsafe_allow_html=True)
    response = ollama.generate(model=MODEL_ID, prompt=prompt)
    st.write("Model:<br>" + response["response"], unsafe_allow_html=True)
