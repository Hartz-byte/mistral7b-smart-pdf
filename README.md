# Mistral-7B RAG Assistant
An AI-powered document question-answering, summarization, and named entity recognition (NER) app using a local Mistral-7B-instruct GGUF model with LangChain, deployed via Streamlit for an interactive chat interface.

---

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline for question answering, summarization, and entity recognition on PDF documents or arbitrary text, powered by a local Mistral-7B-instruct-v0.2.Q4_K_M.gguf model.

It uses:
- LangChain to orchestrate document processing, embeddings, and LLM interaction
- FAISS vector store for efficient similarity search
- SentenceTransformers embeddings for document indexing
- Streamlit to create a friendly web UI for uploading documents, chatting, and viewing results

The key strength is running everything locally with your own LLM, preserving privacy and enabling full control without relying on external APIs.

---

## Features
- Upload multiple PDF documents and process them into searchable chunks
- Ask questions based on document content with contextual retrieval + LLM generation
- Generate comprehensive summaries of uploaded documents
- Perform Named Entity Recognition (NER) on documents or arbitrary input text
- Direct chat interface with local Mistral-7B LLM without document context
- GPU-accelerated local inference (configurable layer offloading)
- Session state persistence for seamless multi-step conversations
- Debug panel with model, document, and pipeline status

---

## Technologies Used
- Mistral-7B-instruct GGUF local LLM loaded via llama-cpp-python
- LangChain (with langchain-community embeddings and vectorstores)
- FAISS for vector similarity search
- SentenceTransformers for embedding computations
- PyPDF2 for PDF text extraction
- Streamlit for app UI and session management
- Python 3.10+ (matching llama-cpp-python requirements)

---

## Prerequisites
- NVIDIA GPU with CUDA (recommended for reasonable performance)
- WSL 2 with Ubuntu (or Linux environment)
- Python 3.10 environment with required packages installed
- Downloaded mistral-7b-instruct-v0.2.Q4_K_M.gguf model file located at expected project-relative path: "../../../mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

---

## Installation
1. Clone the repo

2. Create and activate a Python 3.10 environment:
```
python3.10 -m venv mistral_env
source mistral_env/bin/activate
```

3. Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

4. Ensure CUDA is installed and visible to llama-cpp-python

5. Place your Mistral model file at the path specified in app.py or modify accordingly.

---

## Usage
Start the Streamlit app:
```
streamlit run app.py
```

- The app will auto-load the Mistral-7B model (takes some seconds)
- Upload one or more PDFs using the sidebar uploader
- Click Process Documents to extract, chunk, embed, and index
- Use the UI tabs for Q&A, Summarization, or NER on your documents
- Use the direct chat tab to interact directly with the model

---

## Development Notes
- All core "heavy" stateful objects persist in st.session_state to avoid redundant reloads
- PDF text extraction relies on PyPDF2, but can be swapped for better parsers if needed
- Embeddings and FAISS vector stores handle efficient retrieval of relevant chunks for context
- Mistral-7B model uses GPU offloading layers configurable via UI
- This project uses local LLM inference only — no external API calls

---

## Troubleshooting
- Model not found error: Verify correct model path in app.py
- CUDA errors: Ensure NVIDIA drivers & CUDA toolkit installed and compatible with llama-cpp-python
- Slow performance: GPU offloading layers can be increased or the model moved to higher-end GPUs
- No docs found in summarization/Q&A: Make sure documents are processed and the vector store created before querying
- Streamlit rerunning too often: Manage session state properly (see code)
- Import errors with LangChain: Use compatible langchain-community and update imports accordingly

---

