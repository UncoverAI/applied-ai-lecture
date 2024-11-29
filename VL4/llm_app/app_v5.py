import streamlit as st
import ollama
import datetime
import numpy as np
import wikipedia
import json

def wiki_search(query: str) -> str:
    try:
        return wikipedia.summary(query)
    except Exception as e:
        return str(e)

def calculator(problem: str) -> str:
    try:
        terms = problem.split(" ")
        if len(terms) != 3:
            return "Only input 2 terms and one operator e.g. '5 + 3'"
        n1, n2 = float(terms[0]), float(terms[2])
        op = terms[1]
        if op == "+":
            return str(n1 + n2)
        elif op == "-":
            return str(n1 - n2)
        elif op == "*":
            return str(n1 * n2)
        elif op == "/":
            if n2 != 0:
                return str(n1 / n2)
            else:
                return "Can't divide by zero!"
        else:
            return "Can't recognize operator!"
    except Exception as e:
        return str(e)

def finish(result: str) -> str:
    print(result)
    return "Finished"

ACTIONS = {
    wiki_search.__name__: wiki_search,
    calculator.__name__: calculator,
    finish.__name__: finish
}

SYSTEM_PROMPT = """
You are an AI Agent that uses thought, action, action input and observation to solve a problem.
You get a list of tools available to you. You work in a loop where you generate thoughts and actions + action_inputs. You will get observations from your actions.
Think step by step. Split the problem into a sequence of actions.
Use the finish action once you are done.

Always uset this json format to generate a repsonse. NEVER WRITE ANYTHING OUTSIDE.
```json
{
    "thought": "<YOUR THOUGHT>",
    "action": "<YOUR ACTION>",
    "action_input": "<YOUR ACTION INPUT>"
}
```

These Tools are available to you:
- wiki_search:
{
    "thought": "The user wants me to search the web to find the president of America",
    "action": "wiki_search",
    "action_input": "President of a America"
}
- calculator:
{
    "thought": "The user wants me to calculate the age of the president plus twenty. I have to remember that this tool can only take 2 numbers and an operator as input.",
    "action": "calculator",
    "action_input": "1994 + 20"
}
- finish:
{
    "thought": "I am done with the task",
    "action": "finish",
    "action_input": "The answer to the questions is ..."
}
"""

DEFAULT_MODEL = {"name": "Error no models!", "size": 0}
TEMPERATURE = 0
DEFAULT_TASK = 'How old is the oldest crocodile and what is that number divided by 3.67897?'

st.set_page_config(layout="wide")

try:
    models = ollama.list()["models"]
except:
    models = [DEFAULT_MODEL]

# Global Timer Variables
if "token_times" not in st.session_state:
    st.session_state.token_times = []
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.datetime.now()

def format_func(option: dict) -> str:
    return f"{option['name']} ({round(option['size'] / 10**9, 1)} Gb)"

def stream_response(model: str, messages: list):
    try:
        stream = ollama.chat(model=model, messages=messages, stream=True, keep_alive="24h", options={"temperature": TEMPERATURE})
        st.session_state.start_time = datetime.datetime.now()
        st.session_state.token_times = []
    except Exception as e:
        stream = f"LLM ERROR: {e}"
    return stream

def stream_wrapper(stream):
    for chunk in stream:
        st.session_state.token_times.append(datetime.datetime.now())
        yield chunk['message']['content']

with st.sidebar:
    st.markdown("### Select a Model:")
    model_dict = st.selectbox(label="model selector", label_visibility="collapsed", options=models, index=len(models) - 1, format_func=format_func)
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
if prompt := st.chat_input(DEFAULT_TASK):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Initialize values for agent loop
    action = None
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n"}, {'role': 'user',  'content': prompt + "\n"}]
    while action != "finish":
        # Get model response
        stream = stream_wrapper(stream_response(model, st.session_state.messages))
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            msg = st.write_stream(stream)
        model_message = {"role": "assistant", "content": msg + "\n"}
        # Add model response to memory
        st.session_state.messages.append(model_message)
        # Parse model response
        try:
            output = json.loads(msg)
            action = output["action"]
            # Execute action and get observation
            observation = ACTIONS[action](output["action_input"])
        except Exception as e:
            # In case of error, try and let agent fix the error
            observation = "Error: Can't parse response, because: " + str(e)
            print(observation)
        # Display observation in chat message container
        st.chat_message("user").markdown(observation)
        observation_message = {"role": "user", "content": observation + "\n"}
        # Add observation to memory
        st.session_state.messages.append(observation_message)
        
    st.rerun()
