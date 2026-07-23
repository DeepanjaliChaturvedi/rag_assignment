# -------------------- IMPORTS --------------------
def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cohere


# -------------------- FUNCTION 1 --------------------
# Load file and split into chunks

def load_and_chunk_document(file_path, chunk_size=300, overlap=50):
    text = ""

    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    chunks = split_text(text, chunk_size, overlap)
    return chunks


# -------------------- FUNCTION 2 --------------------
# Create embeddings using TF-IDF

def create_embeddings(chunks):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks)
    return embeddings, vectorizer


# -------------------- FUNCTION 3 --------------------
# Search most relevant chunks

def search_chunks(query, chunks, embeddings, vectorizer, k=3):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_k_indices = np.argsort(similarities)[-k:][::-1]
    results = [chunks[i] for i in top_k_indices]

    return results


# -------------------- FUNCTION 4 --------------------
# Generate answer using Cohere

def generate_answer(query, context, api_key):
    co = cohere.Client(api_key)

    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=200
    )

    return response.generations[0].text.strip()