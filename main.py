import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
from pinecone import Pinecone

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"
DIMENSION = 384  # Embedding size

if not PINECONE_API_KEY:
    st.error("❌ Pinecone API key is missing.")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Ensure correct index dimensions
if INDEX_NAME in [i.name for i in pc.list_indexes()]:
    index_info = pc.describe_index(INDEX_NAME)
    if index_info.dimension != DIMENSION:
        pc.delete_index(INDEX_NAME)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, max_words=200):
    """Split text into chunks of max_words."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def process_and_store(data_list, data_type):
    """Process multiple URLs/PDFs and store in Pinecone."""
    vectors = []
    for data in data_list:
        try:
            if data_type == "url":
                response = requests.get(data)
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text()
            else:
                text = extract_text_from_pdf(data)
            
            text_chunks = chunk_text(" ".join(text.split()))
            avg_embedding = np.mean([model.encode(chunk) for chunk in text_chunks], axis=0).tolist()
            doc_id = f"{data_type}_{hash(data)}"
            
            vectors.append({
                "id": doc_id,
                "values": avg_embedding,
                "metadata": {"source": data, "chunks": text_chunks}
            })
        except Exception as e:
            st.error(f"❌ Error processing {data}: {e}")

    if vectors:
        index.upsert(vectors)
        st.success(f"✅ Stored {len(vectors)} {data_type}(s) in Pinecone!")

# Streamlit UI
st.title("📄 Multi PDF & URL Processor")
st.write("Upload PDFs or enter URLs to store in Pinecone.")

pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
urls = st.text_area("Enter URLs (comma-separated):")

if st.button("Process"):
    url_list = [url.strip() for url in urls.split(",") if url.strip()]
    
    if pdfs:
        process_and_store(pdfs, "pdf")
    if url_list:
        process_and_store(url_list, "url")

    if not pdfs and not url_list:
        st.warning("⚠️ No valid URLs or PDFs provided.")
