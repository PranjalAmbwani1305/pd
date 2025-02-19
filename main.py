import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pinecone
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

def store_in_pinecone(text, index_name="helpdesk"):
    """Stores extracted text embeddings into Pinecone."""
    pinecone.init(api_key="pcsk_77tP2W_671WX1BP2SkmMW6WimJR4jnNRigUMzMH8kZy4qdnDHMXQduiPT4EC3CgiTTE9WF", environment="us-east-1")
    index = pinecone.Index(index_name)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text.split('\n'))
    
    for i, emb in enumerate(embeddings):
        index.upsert([(f"doc_{i}", emb.tolist(), {"text": text.split('\n')[i]})])

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
