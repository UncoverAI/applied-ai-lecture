import streamlit as st
from pymongo import MongoClient
import requests
import json


def get_collection():
    host = "embedding-database"          
    port = 27017                
    database_name = "vector_store"  

    # Construct the MongoDB URI with authentication
    connection_string = f"mongodb://{host}:{port}"

    # Connect to MongoDB
    client = MongoClient(connection_string, connect=True)
    database = client.get_database(database_name)
    return database.get_collection("something")

if st.button("Send 'Hello World' to database"):
    # Display user message in chat message container
    collection = get_collection()
    collection.insert_one({"msg": "Hello World!"})
    
if st.button("Get everything from DB"):
    collection = get_collection()
    data = list(collection.find())
    for record in data:
        st.write(record)
        
if st.button("Clear Database"):
    collection = get_collection()
    collection.delete_many({})

text = st.text_input("Text")
if st.button("Embed"):
    resonse = requests.post("http://backend:8000/embed", json={"text":text})
    st.write(resonse.json())

if st.button("Embed to DB"):
    resonse = requests.post("http://backend:8000/embed", json={"text":text})
    embedding = resonse.json()["embedding"]["embeddings"][0]
    document = {
        "vector": embedding,  # The vector field
        "magnitude": sum(x**2 for x in embedding)**0.5,  # Optional: precompute the vector's magnitude
        "text": text
    }
    collection = get_collection()
    _id = collection.insert_one(document)
    st.success(f"Inserted with id: {_id}")