import streamlit as st
import ollama
import datetime
import numpy as np
from pymongo import MongoClient
import PyPDF2
import io

st.set_page_config(layout="wide")

SYSTEM_PROMPT = "You are a helpful AI assistant."
SYSTEM_MESSAGE = [{"role": "system","content": SYSTEM_PROMPT}]
DEFAULT_MODEL = {"name": "Error no models!", "size": 0}

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
        messages = SYSTEM_MESSAGE + messages
        stream = ollama.chat(model=model, messages=messages, stream=True, keep_alive="24h")
        st.session_state.start_time = datetime.datetime.now()
        st.session_state.token_times = []
    except Exception as e:
        stream = f"LLM ERROR: {e}"
    return stream

def stream_wrapper(stream):
    for chunk in stream:
        st.session_state.token_times.append(datetime.datetime.now())
        yield chunk['message']['content']
        
def upload_and_embed(file_bytes):
    collection = get_embeddings_collection()
    
    pdf_text = extract_text_from_pdf(file_bytes)
    
    embeddings, text_chunks = create_embedding(pdf_text)
    
    total_chunks = 0
    for embedding, text in zip(embeddings, text_chunks):
        # Document to insert
        document = {
                "vector": embedding,  # The vector field
                "magnitude": sum(x**2 for x in embedding)**0.5,  # Optional: precompute the vector's magnitude
                "text": text
        }

        # Insert the document
        collection.insert_one(document)
        total_chunks +=1
        
    return total_chunks

def get_embeddings_collection():
    username = "mongoadmin"  
    password = "super-save-password"  
    host = "localhost"          
    port = 27017                
    database_name = "vector_store"  

    # Construct the MongoDB URI with authentication
    connection_string = f"mongodb://{username}:{password}@{host}:{port}"

    # Connect to MongoDB
    client = MongoClient(connection_string, connect=True)
    database = client.get_database(database_name)
    return database.get_collection("lecture_embeddings")

def extract_text_from_pdf(file_bytes: str):
    text = ""
    reader = PyPDF2.PdfReader(file_bytes)
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text: str, chunk_size=500):
    # Split text into chunks of specified size
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def create_embedding(text):
    chunk_size = 1000
    model_name: str="nomic-embed-text:latest"
    
    # Step 2: Chunk the PDF text
    pdf_chunks = chunk_text(text, chunk_size)

    # Step 3: Embed the PDF chunks
    return ollama.embed(model=model_name, input=pdf_chunks)["embeddings"], pdf_chunks

def do_retrival(question, model_name: str = "nomic-embed-text:latest", k_entries: int = 3):
    collection = get_embeddings_collection()
    question_embedding = ollama.embed(model=model_name, input=question)["embeddings"][0]
    query_magnitude = np.sqrt(np.sum(np.square(question_embedding)))
    
    pipeline =  [
        {
            # Compute the dot product
            "$addFields": {
                "dot_product": {
                    "$sum": {
                        "$map": {
                            "input": {"$range": [0, {"$size": "$vector"}]},  # Iterate over indices
                            "as": "index",
                            "in": {
                                "$multiply": [
                                    {"$arrayElemAt": ["$vector", "$$index"]},
                                    {"$arrayElemAt": [question_embedding, "$$index"]}
                                ]
                            }
                        }
                    }
                }
            }
        },
        {
            # Compute magnitude of stored vectors dynamically if not precomputed
            "$addFields": {
                "vector_magnitude": {
                    "$sqrt": {
                        "$sum": {
                            "$map": {
                                "input": "$vector",
                                "as": "x",
                                "in": {"$multiply": ["$$x", "$$x"]}
                            }
                        }
                    }
                }
            }
        },
        {
            # Compute cosine similarity
            "$addFields": {
                "cosine_similarity": {
                    "$cond": {
                        "if": {"$and": [{"$gt": ["$vector_magnitude", 0]}, {"$gt": [query_magnitude, 0]}]},
                        "then": {
                            "$divide": [
                                "$dot_product",
                                {"$multiply": ["$vector_magnitude", query_magnitude]}
                            ]
                        },
                        "else": 0
                    }
                }
            }
        },
        {
            # Sort by cosine similarity in descending order
            "$sort": {"cosine_similarity": -1}
        },
        {
            # Limit the number of results
            "$limit": k_entries
        },
        {
            # Project the fields you want in the output
            "$project": {"_id": 1, "cosine_similarity": 1, "text": 1}
        }
    ]
    
    results = list(collection.aggregate(pipeline))

    return results


with st.sidebar:
    st.markdown("### Select a Model:")
    model_dict = st.selectbox(label="model selector", label_visibility="collapsed", options=models, index=len(models) - 1, format_func=format_func)
    model = model_dict["name"]

    st.markdown("### Performance:")
    if st.session_state.token_times:
        st.markdown(f"- Time to first token: {st.session_state.token_times[0] - st.session_state.start_time}")
        st.markdown(f"- Average token time: {np.mean(np.diff(st.session_state.token_times))}")
    
    st.markdown("### Settings")
    enable_retrival = st.checkbox("Enable Vector Search")
    no_of_entries = st.number_input("Number of matches", min_value=1, max_value=100, value=1)
    
    file = st.file_uploader("Upload a PDF file")
    if file is not None:
        # To read file as bytes:
        bytes_buffer = io.BytesIO(file.read())        

        # To convert to a string based IO:
        number_embedded_chunks = upload_and_embed(bytes_buffer)
        st.write(f"Embedded Document '{file.name}' into {number_embedded_chunks} chunks")
        st.success("File processed!")
        
st.markdown(f"### Chat with: ``{model}``")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("New Chat"):
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    prompt_context = ""

    if enable_retrival:
        prompt_context = do_retrival(prompt, k_entries=no_of_entries)
        st.session_state.messages.append({"role": "system", "content": f"Answer the following question based on the following context: {prompt_context}"})

    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = stream_wrapper(stream_response(model, st.session_state.messages))
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        msg = st.write_stream(stream)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.rerun()