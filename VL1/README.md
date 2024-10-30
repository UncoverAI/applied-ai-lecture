# Lecture 1 - LLM Applications

Code examples for the lecture

## Requirements
- Download & Install Ollama: https://ollama.com/
- Pull you first model (e.g. `ollama pull llama3.1:8b-instruct-q4_0`)
- Install Python
- Create a Virtual Environment for Python (e.g. with conda: https://docs.anaconda.com/miniconda/miniconda-install/)
- Install the requirements (e.g. `pip install -r requirements.txt`)

## Tipps:
- A good model to start with is ``llama3.1:8b-instruct-q4_0`` 
 
## Components for RAG
1. Content Splitter -> text chunks
2. Tokenizer
3. Embedding Model -> HF 
4. Vector Datenbank
5. Retriever -> Similarity Algorithm

