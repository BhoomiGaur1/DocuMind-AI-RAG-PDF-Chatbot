# 📄 DocuMind AI

A RAG-based Multi-PDF Chatbot built using Streamlit, FAISS, Sentence Transformers, and Gemini API.

## Features

- Upload multiple PDF documents
- Semantic search using FAISS
- Context-aware question answering
- Gemini API integration
- Chat history support
- Follow-up question handling
- Source chunk citation
- Multi-document retrieval

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Gemini API
- NumPy

## How It Works

1. Upload one or more PDF files
2. Extract text using PyPDF
3. Split text into chunks
4. Generate embeddings using BGE Small
5. Store embeddings in FAISS
6. Retrieve relevant chunks for user query
7. Generate grounded answers using Gemini

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project Structure

```text
streamlit_app.py
requirements.txt
README.md
```

## Future Improvements

- File-level citations
- Page-level citations
- Persistent vector database
- Authentication
- Cloud deployment
