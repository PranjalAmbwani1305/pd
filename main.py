import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
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

    # Streamlit UI
    st.title("🔗 URL to Pinecone (Chunked Text, Single ID)")
    st.write("Enter a URL to extract text, split it into chunks, and store in Pinecone under one ID.")

    url = st.text_input("Enter the URL:")

    if st.button("Process URL"):
        if url:
            try:
                # Extract text from URL
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text()
                text = " ".join(text.split())  # Clean text

                # Split text into chunks
                text_chunks = chunk_text(text)

                # Generate embeddings for each chunk
                embeddings = np.array([model.encode(chunk) for chunk in text_chunks])

                # Compute the average embedding to store under one ID
                avg_embedding = np.mean(embeddings, axis=0).tolist()

                # Store in Pinecone (Single ID, Metadata contains all chunks)
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
            st.warning("⚠️ Please enter a valid URL!")
