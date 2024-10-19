import streamlit as st
import ollama

SYSTEM_PROMPT = "You are a helpful AI assistant."
SYSTEM_MESSAGE = [{"role": "system","content": SYSTEM_PROMPT}]

MODEL_ID = 'llama3.1:8b-instruct-q4_0'

st.markdown(f"### Chat with: ``{MODEL_ID}``")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("New Chat"):
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        messages = SYSTEM_MESSAGE + st.session_state.messages
        msg = ollama.chat(model=MODEL_ID, messages=messages, keep_alive="24h")['message']['content']
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.rerun()
