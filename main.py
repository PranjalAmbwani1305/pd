import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pinecone

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_77tP2W_671WX1BP2SkmMW6WimJR4jnNRigUMzMH8kZy4qdnDHMXQduiPT4EC3CgiTTE9WF"
INDEX_NAME = "helpdesk"

pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=384)  # Adjust based on model output dimension

index = pinecone.Index(INDEX_NAME)

# Load embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Streamlit UI
st.title("🔗 URL to Vector Storage")
st.write("Enter a URL, and we'll extract its content, convert it to embeddings, and store it in Pinecone.")

url = st.text_input("Enter the URL:")

if st.button("Process URL"):
    if url:
        # Extract text from URL
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            text = " ".join(text.split())  # Clean text

            # Generate embedding
            embedding = model.encode(text)

            # Store in Pinecone
            doc_id = f"url_{hash(url)}"
            index.upsert([(doc_id, embedding.tolist())])

            st.success(f"✅ Successfully stored embeddings for {url} in Pinecone!")
        except Exception as e:
            st.error(f"❌ Error processing URL: {e}")
    else:
        st.warning("⚠️ Please enter a valid URL!")

