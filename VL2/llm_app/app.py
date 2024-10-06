import streamlit as st
import ollama
import datetime
import numpy as np

st.set_page_config(layout="wide")

SYSTEM_PROMPT = "You are a helpful AI assistant."
SYSTEM_MESSAGE = [{"role": "system","content": SYSTEM_PROMPT}]
models = ollama.list()

# Global Timer Variables
if "token_times" not in st.session_state:
    st.session_state.token_times = []
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.datetime.now()

def format_func(option: dict) -> str:
    return f"{option['name']} ({round(option['size'] / 10**9, 1)} Gb)"

def stream_response(model: str, messages: list):
    try:
        messages = SYSTEM_MESSAGE + messages
        stream = ollama.chat(model=model, messages=messages, stream=True)
        st.session_state.start_time = datetime.datetime.now()
        st.session_state.token_times = []
    except Exception as e:
        stream = f"LLM ERROR: {e}"
    return stream

def stream_wrapper(stream):
    for chunk in stream:
        # date_format = '%Y-%m-%dT%H:%M:%S.%f%Z'
        # date_obj = datetime.strptime(chunk['created_at'], date_format)
        st.session_state.token_times.append(datetime.datetime.now())
        yield chunk['message']['content']

with st.sidebar:
    st.markdown("### Select a Model:")
    model_dict = st.selectbox(label="model selector", label_visibility="collapsed", options=models["models"], index=5, format_func=format_func)
    model = model_dict["name"]

    st.markdown("### Performance:")
    if st.session_state.token_times:
        st.markdown(f"- Time to first token: {st.session_state.token_times[0] - st.session_state.start_time}")
        st.markdown(f"- Average token time: {np.mean(np.diff(st.session_state.token_times))}")

st.markdown(f"### Chat with: ``{model}``")

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

    stream = stream_wrapper(stream_response(model, st.session_state.messages))
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        msg = st.write_stream(stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.rerun()



# import ollama

# stream = ollama.chat(
#     model='llama3.1',
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

# with st.chat_message(“assistant”):
# msg = st.write_stream(response)