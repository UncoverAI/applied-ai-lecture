FROM python:3.11

RUN export OLLAMA_MODELS=/
RUN curl -o nomic-embed-text https://ollama.com/api/model/download?model_name=nomic-embed-text:latest

COPY ./src .
RUN pip install -r ./requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--port", "8000", "--host", "0.0.0.0"]