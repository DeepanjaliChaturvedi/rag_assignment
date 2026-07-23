import streamlit as st
from functions import (
    load_and_chunk_document,
    create_embeddings,
    search_chunks,
    generate_answer
)

st.set_page_config(page_title="RAG App", layout="centered")

st.title("📄 RAG Q&A System")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
query = st.text_input("Ask a question:")
api_key = st.text_input("Enter your Cohere API Key", type="password")

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    chunks = load_and_chunk_document(uploaded_file.name)
    embeddings, vectorizer = create_embeddings(chunks)

    if query and api_key:
        relevant_chunks = search_chunks(query, chunks, embeddings, vectorizer)

        context = "\n".join(relevant_chunks)

        answer = generate_answer(query, context, api_key)

        st.subheader("Answer:")
        st.write(answer)