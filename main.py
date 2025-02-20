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

def chunk_text(text, chunk_size=500):
    """Splits text into meaningful paragraphs instead of fixed-size chunks."""
    paragraphs = text.split("\n\n")  # Splitting by double newlines for structured text
    return [para.strip() for para in paragraphs if len(para.strip()) > 50]  # Remove short/noise text

def store_in_pinecone(text_chunks):
    """Stores text chunks in Pinecone."""
    vectors = [(str(uuid.uuid4()), model.encode(chunk).tolist(), {"text": chunk}) for chunk in text_chunks]
    index.upsert(vectors)

def extract_text_from_url(url):
    """Fetches and extracts well-structured text from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract and structure content properly
        paragraphs = [elem.get_text(strip=True) for elem in soup.find_all(['p', 'h1', 'h2', 'h3'])]
        structured_text = "\n\n".join(paragraphs)  # Maintain paragraph breaks
        return structured_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the URL: {e}")
        return ""

# Streamlit UI
st.title("Pinecone Article Storage")
url_input = st.text_input("Enter URL:")
user_input = st.text_area("Or enter your text:")

if st.button("Store in Pinecone"):
    if url_input:
        user_input = extract_text_from_url(url_input)
    
    if user_input:
        chunks = chunk_text(user_input)
        store_in_pinecone(chunks)
        st.success(f"Stored {len(chunks)} structured paragraphs in Pinecone!")
    else:
        st.error("Please enter a valid URL or some text.")
