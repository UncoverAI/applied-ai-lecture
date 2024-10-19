# Lecture 1 - LLM Applications

In this lecture we will learn how we can use Python, Streamlit and Ollama to build a simple LLM Chat App, where the LLM used is a Llama model run locally on your device.

## Preperation
- Setup a Python Developement Environment (Python + Virtual Environment, VS Code + Python & Notebook Extensions)
- Install the Python requirements in your virtual environment `pip install -r requirements.txt`
- Download & Install Ollama: https://ollama.com/

## Lecture content
1. Setup Ollama
2. Learn RAG
3. Learn Agents
4. Build an LLM App

## Tipps:
- While ``llama3.1:8b-instruct-q4_0`` is a good model to get started with, ollama has a lot of cool models available: https://ollama.com/library
 
## Components for RAG
1. Content Splitter -> text chunks
2. Tokenizer
3. Embedding Model -> HF 
4. Vector Datenbank
5. Retriever -> Similarity Algorithm

