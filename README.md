# 📄 DocuMind AI

A RAG-based Multi-PDF Chatbot built using Streamlit, FAISS, Sentence Transformers, and Gemini API.

## Live Demo

🔗 https://docu-ai-bhoomi.streamlit.app

## Features

- 📚 Multi-PDF Support
- 🔍 Semantic Search using FAISS
- 🤖 Gemini-Powered Question Answering
- 💬 Conversational Chat History
- 📄 Source Chunk Citation
- 📑 Multi-Document Retrieval
- 🔄 Follow-up Question Handling
- 🧠 Context-Aware Responses
- ⚡ Sentence Transformer Embeddings
- ☁️ Streamlit Cloud Deployment

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
