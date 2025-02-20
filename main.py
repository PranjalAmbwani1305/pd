import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"
DIMENSION = 384  

if not PINECONE_API_KEY:
    st.error("❌ Pinecone API key is missing.")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure correct index setup
if INDEX_NAME in [i.name for i in pc.list_indexes()]:
    index_info = pc.describe_index(INDEX_NAME)
    if index_info.dimension != DIMENSION:
        pc.delete_index(INDEX_NAME)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine")

index = pc.Index(INDEX_NAME)

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

def store_in_pinecone(urls):
    """Extracts text from URLs and stores text + embeddings in Pinecone."""
    vectors = []
    for url in urls:
        text = extract_text(url)
        if text:
            embedding = model.encode(text).tolist()
            doc_id = f"url_{hash(url)}"
            
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {"url": url, "text": text[:5000]}  # Store up to 5000 chars
            })
    
    if vectors:
        index.upsert(vectors)
        st.success(f"✅ Stored {len(vectors)} URLs in Pinecone with text!")

# Streamlit UI
st.title("🔗 URL Text Extractor & Pinecone Storage")
urls = st.text_area("Enter URLs (comma-separated):")

if st.button("Process URLs"):
    url_list = [url.strip() for url in urls.split(",") if url.strip()]
    if url_list:
        store_in_pinecone(url_list)
    else:
        st.warning("⚠️ Please enter at least one valid URL.")
