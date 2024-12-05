from fastapi import FastAPI, Body
from pydantic import BaseModel
import ollama


ollama.pull("nomic-embed-text:latest")
app = FastAPI(docs_url="/")

class EmbedRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return "OK"

@app.post("/embed")
def embed(request: EmbedRequest):
    return {"embedding": ollama.embed(model="nomic-embed-text:latest", input=request.text), "text": request.text}