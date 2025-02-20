import streamlit as st
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone
import uuid
from sentence_transformers import SentenceTransformer
import os

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"
index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=100):
    """Splits text into chunks of a given size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def store_in_pinecone(text_chunks):
    """Stores text chunks in Pinecone."""
    vectors = [(str(uuid.uuid4()), model.encode(chunk).tolist(), {"text": chunk}) for chunk in text_chunks]
    index.upsert(vectors)

def extract_text_from_url(url):
    """Fetches and extracts clean text from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract paragraphs and headers
        extracted_text = ' '.join([elem.get_text(strip=True) for elem in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        return extracted_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the URL: {e}")
        return ""

# Streamlit UI
st.title("Pinecone Chunk Storage")
url_input = st.text_input("Enter URL:")
user_input = st.text_area("Or enter your text:")
chunk_size = st.slider("Chunk Size", 50, 500, 100)

if st.button("Store in Pinecone"):
    if url_input:
        user_input = extract_text_from_url(url_input)
    
    if user_input:
        chunks = chunk_text(user_input, chunk_size)
        store_in_pinecone(chunks)
        st.success(f"Stored {len(chunks)} chunks in Pinecone!")
    else:
        st.error("Please enter a valid URL or some text.")
