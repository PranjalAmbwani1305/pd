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
    st.title("📄 PDF & URL to Pinecone (Single ID Storage)")
    st.write("Upload a PDF or enter a URL to extract text, split it into chunks, and store it in Pinecone under one ID.")

    # File uploader for PDFs
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    
    # Text input for URL
    url = st.text_input("Or enter a URL:")

    if st.button("Process"):
        if uploaded_pdf:
            try:
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(uploaded_pdf)
                text_chunks = chunk_text(pdf_text)

                # Generate embeddings
                embeddings = np.array([model.encode(chunk) for chunk in text_chunks])
                avg_embedding = np.mean(embeddings, axis=0).tolist()

                # Store in Pinecone under a single ID
                doc_id = f"pdf_{hash(uploaded_pdf.name)}"
                index.upsert(
                    vectors=[{
                        "id": doc_id,
                        "values": avg_embedding,
                        "metadata": {"filename": uploaded_pdf.name, "text_chunks": text_chunks}
                    }]
                )

                st.success(f"✅ Successfully stored {len(text_chunks)} chunks from {uploaded_pdf.name} under one ID in Pinecone!")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")

        elif url:
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

                # Store in Pinecone under a single ID
                doc_id = f"url_{hash(url)}"
                index.upsert(
                    vectors=[{
                        "id": doc_id,
                        "values": avg_embedding,
                        "metadata": {"url": url, "text_chunks": text_chunks}
                    }]
                )

                st.success(f"✅ Successfully stored {len(text_chunks)} chunks under one ID for {url} in Pinecone!")
            except Exception as e:
                st.error(f"❌ Error processing URL: {e}")

        else:
            st.warning("⚠️ Please upload a PDF or enter a URL!")
