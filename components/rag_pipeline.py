import streamlit as st
from utils.llm_manager import LocalLLMManager
from utils.embedding_manager import EmbeddingManager
from typing import List, Dict, Any
import re

class RAGPipeline:
    def __init__(self, llm_manager: LocalLLMManager, embedding_manager: EmbeddingManager):
        """
        Initialize RAG pipeline.
        
        Args:
            llm_manager: Instance of LocalLLMManager
            embedding_manager: Instance of EmbeddingManager
        """
        self.llm_manager = llm_manager
        self.embedding_manager = embedding_manager
    
    def answer_question(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Answer a question using RAG."""
        # Retrieve relevant documents
        relevant_docs = self.embedding_manager.similarity_search(question, k=k)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in the provided documents.",
                "sources": [],
                "context": ""
            }
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt for Q&A
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
        Please provide accurate and detailed answers based only on the information given in the context. 
        If the context doesn't contain enough information to answer the question, say so clearly."""
        
        prompt = self.llm_manager.format_mistral_prompt(system_prompt, question, context)
        
        # Generate answer
        answer = self.llm_manager.generate_response(prompt, max_tokens=512)
        
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in relevant_docs],
            "context": context
        }
    
    def summarize_document(self, max_chunks: int = 10) -> str:
        """Summarize the entire document."""
        if not self.embedding_manager.vector_store:
            return "No documents available for summarization."
        
        # Get a representative sample of chunks
        sample_query = "main topics key points summary overview"
        relevant_docs = self.embedding_manager.similarity_search(sample_query, k=max_chunks)
        
        if not relevant_docs:
            return "No content available for summarization."
        
        # Combine content
        combined_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create summarization prompt
        system_prompt = """You are an expert at creating comprehensive summaries. 
        Please provide a detailed summary of the following content, highlighting the main topics, 
        key points, and important insights. Organize your summary in a clear and structured manner."""
        
        prompt = self.llm_manager.format_mistral_prompt(system_prompt, "Please summarize this content:", combined_text)
        
        # Generate summary
        summary = self.llm_manager.generate_response(prompt, max_tokens=1024)
        
        return summary
    
    def extract_entities(self, text: str = None, k: int = 5) -> Dict[str, List[str]]:
        """Extract named entities from the document or provided text."""
        if text is None:
            # Use document content
            if not self.embedding_manager.vector_store:
                return {"error": ["No documents available for entity extraction."]}
            
            # Get representative chunks
            query = "names people organizations locations dates"
            relevant_docs = self.embedding_manager.similarity_search(query, k=k)
            text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create NER prompt
        system_prompt = """You are an expert at named entity recognition. 
        Please extract and categorize the following types of entities from the text:
        - PERSON: Names of people
        - ORGANIZATION: Companies, institutions, organizations
        - LOCATION: Places, cities, countries, landmarks
        - DATE: Dates, times, periods
        - MISCELLANEOUS: Other important entities
        
        Format your response as:
        **PERSON:**
        - Entity 1
        - Entity 2
        
        **ORGANIZATION:**
        - Entity 1
        - Entity 2
        
        (Continue for all categories found)"""
        
        prompt = self.llm_manager.format_mistral_prompt(system_prompt, "Extract named entities from this text:", text)
        
        # Generate NER response
        ner_result = self.llm_manager.generate_response(prompt, max_tokens=512)
        
        # Parse the result into categories
        entities = self._parse_ner_result(ner_result)
        
        return entities
    
    def _parse_ner_result(self, ner_text: str) -> Dict[str, List[str]]:
        """Parse NER result into structured format."""
        entities = {}
        current_category = None
        
        lines = ner_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check if line is a category header
            if line.startswith('**') and line.endswith(':**'):
                current_category = line.replace('**', '').replace(':', '').strip()
                entities[current_category] = []
            elif line.startswith('-') and current_category:
                entity = line.replace('-', '').strip()
                if entity:
                    entities[current_category].append(entity)
        
        return entities
