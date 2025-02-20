import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "helpdesk"
DIMENSION = 786

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text(url):
    """Extract and clean text from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(soup.get_text().split())  # Clean text
        return text if len(text) > 50 else None  # Ignore empty or very short text
    except Exception as e:
        st.error(f"❌ Failed to extract text from {url}: {e}")
        return None

def chunk_text(text, max_words=100):
    """Split text into meaningful chunks of max_words words each."""
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

def store_in_pinecone(urls):
    """Extracts text from URLs, chunks it, and stores all chunks together under one ID."""
    vectors = []
    for url in urls:
        text = extract_text(url)
        if text:
            text_chunks = chunk_text(text)
            combined_text = " ||| ".join(text_chunks)  # Combine chunks with a separator
            
            # Generate a single embedding for all chunks combined
            avg_embedding = model.encode(combined_text).tolist()
            
            doc_id = f"url_{hash(url)}"  # Single ID for the whole document
            
            vectors.append({
                "id": doc_id,
                "values": avg_embedding,
                "metadata": {"url": url, "chunks": text_chunks}
            })
    
    if vectors:
        index.upsert(vectors)
        st.success(f"✅ Stored {len(vectors)} URLs in Pinecone with all chunks combined!")

# Streamlit UI
st.title("🔗 URL Text Extractor & Pinecone Storage")
urls = st.text_area("Enter URLs (comma-separated):")

if st.button("Process URLs"):
    url_list = [url.strip() for url in urls.split(",") if url.strip()]
    if url_list:
        store_in_pinecone(url_list)
    else:
        st.warning("⚠️ Please enter at least one valid URL.")
