import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec
import pdfkit

def scrape_website(url):
    """Scrapes the given website and extracts text content."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = '\n'.join([p.get_text() for p in paragraphs])
            return text
        else:
            return None
    except Exception as e:
        return None

def store_in_pinecone(text, index_name="web-scraper-index"):
    """Stores extracted text embeddings into Pinecone."""
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
    
    index = pc.Index(index_name)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # 1536-dim model
    lines = text.split('\n')
    embeddings = model.encode(lines)
    
    upsert_data = []
    for i, emb in enumerate(embeddings):
        metadata = {"text": lines[i]} if len(lines[i]) < 500 else {}
        upsert_data.append((f"doc_{i}", emb.tolist(), metadata))
    
    if upsert_data:
        index.upsert(vectors=upsert_data)

def save_as_pdf(text, filename="scraped_data.pdf"):
    """Saves the extracted text as a PDF."""
    pdfkit.from_string(text, filename)
    return filename

st.title("Website Scraper & Pinecone Storage")
url = st.text_input("Enter website URL:")
if st.button("Scrape & Store"):
    with st.spinner("Scraping website..."):
        text = scrape_website(url)
        if text:
            st.success("Website scraped successfully!")
            st.text_area("Extracted Text:", text, height=300)
            
            with st.spinner("Storing in Pinecone..."):
                store_in_pinecone(text)
                st.success("Data stored in Pinecone!")
                
            pdf_file = save_as_pdf(text)
            st.success("PDF saved!")
            st.download_button("Download PDF", data=open(pdf_file, "rb"), file_name="scraped_data.pdf", mime="application/pdf")
        else:
            st.error("Failed to scrape website. Check URL.")
