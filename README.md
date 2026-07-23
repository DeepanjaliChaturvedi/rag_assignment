# RAG Assignment

Project Description
This project is a Retrieval-Augmented Generation (RAG) application built using Python and Streamlit. It allows users to upload PDF or TXT files, splits the documents into chunks, creates embeddings using Sentence Transformers, retrieves the most relevant chunks using cosine similarity, and generates answers using the Cohere API.

Features
- Upload PDF or TXT files
- Document chunking
- Embedding generation
- Similarity search
- Answer generation using Cohere
- Streamlit web interface

Technologies Used
- Python
- Streamlit
- Sentence Transformers
- LangChain
- Cohere
- Scikit-learn

Project Structure

rag_assignment/
├── app.py
├── functions.py
├── notebook.ipynb
├── requirements.txt
├── README.md
└── screenshots/
```

 Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```
