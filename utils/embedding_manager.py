import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Optional
import pickle
import os

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.embeddings = None
        self.vector_store = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'}
            )
            print(f"✅ Embedding model loaded: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents."""
        if not documents:
            st.error("No documents provided for vector store creation")
            return None
        
        try:
            with st.spinner("Creating vector embeddings..."):
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            st.success(f"✅ Vector store created with {len(documents)} chunks")
            return self.vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None
    
    def save_vector_store(self, path: str):
        """Save vector store to disk."""
        if self.vector_store:
            try:
                self.vector_store.save_local(path)
                st.success(f"Vector store saved to {path}")
            except Exception as e:
                st.error(f"Error saving vector store: {e}")
    
    def load_vector_store(self, path: str):
        """Load vector store from disk."""
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings)
            st.success(f"Vector store loaded from {path}")
            return self.vector_store
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if not self.vector_store:
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            st.error(f"Error performing similarity search: {e}")
            return []
