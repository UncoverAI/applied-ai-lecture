FROM python:3.11

COPY ./src . 
RUN pip install -r ./requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app_v6.py"]