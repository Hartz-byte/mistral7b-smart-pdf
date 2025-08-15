import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import io

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def process_text(self, text: str) -> List[Document]:
        """Split text into chunks and create Document objects."""
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        documents = [
            Document(page_content=chunk, metadata={"chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]
        return documents
    
    def process_pdf_files(self, uploaded_files) -> List[Document]:
        """Process multiple PDF files."""
        all_documents = []
        
        for i, file in enumerate(uploaded_files):
            text = self.extract_text_from_pdf(file)
            if text:
                chunks = self.text_splitter.split_text(text)
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={"source": file.name, "chunk_id": j, "file_id": i}
                    )
                    for j, chunk in enumerate(chunks)
                ]
                all_documents.extend(documents)
        
        return all_documents
