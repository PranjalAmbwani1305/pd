import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
from pinecone import Pinecone

# Load Pinecone API Key from Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "helpdesk"

if not PINECONE_API_KEY:
    st.error("❌ Pinecone API key is missing. Set it as an environment variable.")
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    # Delete the index if dimensions don't match
    if INDEX_NAME in existing_indexes:
        index_info = pc.describe_index(INDEX_NAME)
        if index_info.dimension != 384:
            st.warning(f"⚠️ Deleting old index (wrong dimension: {index_info.dimension}) and creating a new one.")
            pc.delete_index(INDEX_NAME)

    # Create index if it doesn't exist
    if INDEX_NAME not in existing_indexes:
        pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine")
    
    # Connect to the index
    index = pc.Index(INDEX_NAME)

    # Load embedding model
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(MODEL_NAME)

    # Function to split text into small chunks
    def chunk_text(text, max_words=200):
        """Splits text into smaller chunks based on word count."""
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    # Function to extract text from PDF
    def extract_text_from_pdf(pdf_file):
        """Extracts text from a PDF file."""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return " ".join(text.split())  # Clean text

    # Streamlit UI
    st.title("📄 Multi PDF & URL to Pinecone (Single ID per File/URL)")
    st.write("Upload multiple PDFs or enter multiple URLs (comma-separated) to store them in Pinecone.")

    # File uploader for multiple PDFs
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Text input for multiple URLs (comma-separated)
    urls = st.text_area("Enter URLs (comma-separated):")

    if st.button("Process"):
        processed_count = 0

        # Process multiple PDFs
        if uploaded_pdfs:
            for uploaded_pdf in uploaded_pdfs:
                try:
                    # Extract text from PDF
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    text_chunks = chunk_text(pdf_text)

                    # Generate embeddings
                    embeddings = np.array([model.encode(chunk) for chunk in text_chunks])
                    avg_embedding = np.mean(embeddings, axis=0).tolist()

                    # Store in Pinecone under a unique ID
                    doc_id = f"pdf_{hash(uploaded_pdf.name)}"
                    index.upsert(
                        vectors=[{
                            "id": doc_id,
                            "values": avg_embedding,
                            "metadata": {"filename": uploaded_pdf.name, "text_chunks": text_chunks}
                        }]
                    )

                    processed_count += 1
                except Exception as e:
                    st.error(f"❌ Error processing PDF {uploaded_pdf.name}: {e}")

        # Process multiple URLs
        if urls:
            url_list = [url.strip() for url in urls.split(",") if url.strip()]  # Split and clean URLs

            for url in url_list:
                try:
                    # Extract text from URL
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text()
                    text = " ".join(text.split())  # Clean text

                    # Split text into chunks
                    text_chunks = chunk_text(text)

                    # Generate embeddings
                    embeddings = np.array([model.encode(chunk) for chunk in text_chunks])
                    avg_embedding = np.mean(embeddings, axis=0).tolist()

                    # Store in Pinecone under a unique ID
                    doc_id = f"url_{hash(url)}"
                    index.upsert(
                        vectors=[{
                            "id": doc_id,
                            "values": avg_embedding,
                            "metadata": {"url": url, "text_chunks": text_chunks}
                        }]
                    )

                    processed_count += 1
                except Exception as e:
                    st.error(f"❌ Error processing URL {url}: {e}")

        if processed_count > 0:
            st.success(f"✅ Successfully stored {processed_count} files/URLs in Pinecone!")
        else:
            st.warning("⚠️ No valid URLs or PDFs processed.")
