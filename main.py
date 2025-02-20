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
DIMENSION = 768  # Matching the model dimension

if not PINECONE_API_KEY:
    st.error("❌ Pinecone API key is missing.")
    st.stop()

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ List existing indexes and check if our index exists
existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME in existing_indexes:
    index_info = pc.describe_index(INDEX_NAME)
    if index_info.dimension != DIMENSION:
        st.warning(f"⚠️ Index dimension mismatch! Deleting and recreating {INDEX_NAME}.")
        pc.delete_index(INDEX_NAME)

# ✅ Create index if it doesn’t exist
if INDEX_NAME not in existing_indexes:
    pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine")

# ✅ Correct way to initialize Pinecone index
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

def chunk_text(text, max_words=100):
    """Split text into meaningful chunks of max_words words each."""
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

def store_articles_in_pinecone(urls):
    """Extracts text, chunks it, and stores each article in Pinecone as a structured document."""
    for url in urls:
        text = extract_text(url)
        if text:
            text_chunks = chunk_text(text)
            
            # Generate an embedding for the whole article
            avg_embedding = model.encode(" ".join(text_chunks)).tolist()
            
            doc_id = f"url_{hash(url)}"  # Single ID for the article
            
            # Store each article with structured chunks in metadata
            index.upsert([{
                "id": doc_id,
                "values": avg_embedding,
                "metadata": {
                    "url": url,
                    "article": text_chunks  # Full article split into chunks
                }
            }]])
    
    st.success(f"✅ Stored {len(urls)} articles in Pinecone!")

# Streamlit UI
st.title("📚 Article Extractor & Pinecone Storage")
urls = st.text_area("Enter article URLs (comma-separated):")

if st.button("Process Articles"):
    url_list = [url.strip() for url in urls.split(",") if url.strip()]
    if url_list:
        store_articles_in_pinecone(url_list)
    else:
        st.warning("⚠️ Please enter at least one valid URL.")
