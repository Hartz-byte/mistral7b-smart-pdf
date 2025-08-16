import streamlit as st
import os
from pathlib import Path

# Import custom components
from utils.llm_manager import LocalLLMManager
from utils.pdf_processor import PDFProcessor
from utils.embedding_manager import EmbeddingManager
from components.rag_pipeline import RAGPipeline
from components.chat_interface import ChatInterface

# Page configuration
st.set_page_config(
    page_title="Mistral-7B RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    margin-bottom: 30px;
}
.feature-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

class RAGApp:
    MODEL_PATH = "../../../mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    def __init__(self):
        """Initialize the RAG application."""
        self.initialize_session_state()
        
        # Initialize lightweight objects in session state if not already done
        if "pdf_processor" not in st.session_state:
            st.session_state.pdf_processor = PDFProcessor()
        if "chat_interface" not in st.session_state:
            st.session_state.chat_interface = ChatInterface()
        
        # Load model if not already loaded
        if not st.session_state.model_loaded and os.path.exists(self.MODEL_PATH):
            self.load_model(self.MODEL_PATH, n_gpu_layers=35)
        elif not os.path.exists(self.MODEL_PATH):
            st.error(f"Model file not found at fixed path: {self.MODEL_PATH}")
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        # Status flags
        if "model_loaded" not in st.session_state:
            st.session_state.model_loaded = False
        if "documents_processed" not in st.session_state:
            st.session_state.documents_processed = False
        if "vector_store_ready" not in st.session_state:
            st.session_state.vector_store_ready = False
        
        # Object containers
        if "llm_manager" not in st.session_state:
            st.session_state.llm_manager = None
        if "embedding_manager" not in st.session_state:
            st.session_state.embedding_manager = None
        if "rag_pipeline" not in st.session_state:
            st.session_state.rag_pipeline = None
    
    def setup_sidebar(self):
        """Setup sidebar configuration."""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            st.subheader("1. Model Configuration")
            st.markdown(f"**Model path:** `{self.MODEL_PATH}`")
            
            # GPU layers slider
            n_gpu_layers = st.slider("GPU Layers", min_value=0, max_value=50, value=35,
                                    help="Number of layers to offload to GPU (0 for CPU only)")
            
            # Button to reload model with updated GPU layers
            if st.button("Reload Model with current GPU Layers"):
                # Reset states before reloading
                st.session_state.model_loaded = False
                st.session_state.documents_processed = False
                st.session_state.vector_store_ready = False
                self.load_model(self.MODEL_PATH, n_gpu_layers)
            
            st.divider()
            
            st.subheader("2. Document Processing")
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files to chat with"
            )
            
            if uploaded_files and st.button("Process Documents"):
                self.process_documents(uploaded_files)
            
            st.divider()
            
            st.subheader("3. Options")
            if st.button("Clear Chat History"):
                if "chat_interface" in st.session_state:
                    st.session_state.chat_interface.clear_chat()
                st.rerun()
            
            # Debug info
            with st.expander("🔍 Debug Info"):
                st.write("Session State Status:")
                st.write(f"- Model Loaded: {st.session_state.model_loaded}")
                st.write(f"- Documents Processed: {st.session_state.documents_processed}")
                st.write(f"- Vector Store Ready: {st.session_state.vector_store_ready}")
                st.write(f"- LLM Manager: {st.session_state.llm_manager is not None}")
                st.write(f"- Embedding Manager: {st.session_state.embedding_manager is not None}")
                st.write(f"- RAG Pipeline: {st.session_state.rag_pipeline is not None}")
                if st.session_state.embedding_manager:
                    st.write(f"- Vector Store Exists: {st.session_state.embedding_manager.vector_store is not None}")
    
    def load_model(self, model_path: str, n_gpu_layers: int):
        """Load the LLM model and store in session state."""
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return
        
        try:
            with st.spinner("Loading Mistral-7B model..."):
                # Create and store managers in session state
                st.session_state.llm_manager = LocalLLMManager(model_path, n_gpu_layers=n_gpu_layers)
                st.session_state.embedding_manager = EmbeddingManager()
                st.session_state.rag_pipeline = RAGPipeline(
                    st.session_state.llm_manager, 
                    st.session_state.embedding_manager
                )
                
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.session_state.model_loaded = False
    
    def process_documents(self, uploaded_files):
        """Process uploaded PDF documents."""
        if not st.session_state.embedding_manager:
            st.error("Please load the model first!")
            return
        
        try:
            with st.spinner("Processing PDF documents..."):
                # Process PDFs
                documents = st.session_state.pdf_processor.process_pdf_files(uploaded_files)
                
                if documents:
                    # Create vector store
                    vector_store = st.session_state.embedding_manager.create_vector_store(documents)
                    
                    if vector_store:
                        st.session_state.documents_processed = True
                        st.session_state.vector_store_ready = True
                        st.success(f"✅ Processed {len(documents)} document chunks from {len(uploaded_files)} files")
                    else:
                        st.error("Failed to create vector store")
                        st.session_state.documents_processed = False
                        st.session_state.vector_store_ready = False
                else:
                    st.error("No text could be extracted from the uploaded files.")
                    st.session_state.documents_processed = False
                    st.session_state.vector_store_ready = False
                    
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            st.session_state.documents_processed = False
            st.session_state.vector_store_ready = False
    
    def main_interface(self):
        """Display the main chat interface."""
        st.markdown('<h1 class="main-header">🤖 Mistral-7B RAG Assistant</h1>', unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ Loaded" if st.session_state.model_loaded else "❌ Not Loaded"
            st.metric("Model Status", status)
        with col2:
            status = "✅ Processed" if st.session_state.documents_processed else "❌ No Documents"
            st.metric("Documents", status)
        with col3:
            status = "✅ Ready" if st.session_state.vector_store_ready else "❌ Not Ready"
            st.metric("Vector Store", status)
        
        # Feature tabs
        if st.session_state.model_loaded:
            tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A Chat", "📝 Summarization", "🏷️ Named Entity Recognition", "📄 Direct Text Input"])
            with tab1:
                self.qa_interface()
            with tab2:
                self.summarization_interface()
            with tab3:
                self.ner_interface()
            with tab4:
                self.direct_text_interface()
        else:
            st.info("👈 Model failed to load. Please check the model file path.")

    def qa_interface(self):
        st.subheader("💬 Ask Questions About Your Documents")
        if not st.session_state.vector_store_ready:
            st.warning("Please upload and process documents first!")
            return
        if "chat_interface" in st.session_state:
            st.session_state.chat_interface.display_chat_history()
        if prompt := st.chat_input("Ask a question about your documents..."):
            if "chat_interface" in st.session_state:
                st.session_state.chat_interface.add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.rag_pipeline:
                        result = st.session_state.rag_pipeline.answer_question(prompt)
                        st.markdown(result["answer"])
                        if "chat_interface" in st.session_state:
                            st.session_state.chat_interface.display_sources(result["sources"])
                            st.session_state.chat_interface.display_context(result["context"])
                        if "chat_interface" in st.session_state:
                            st.session_state.chat_interface.add_message("assistant", result["answer"])
                    else:
                        st.error("RAG pipeline not available")

    def summarization_interface(self):
        st.subheader("📝 Document Summarization")
        if not st.session_state.vector_store_ready:
            st.warning("Please upload and process documents first!")
            return
        col1, col2 = st.columns([3, 1])
        with col1:
            max_chunks = st.slider("Number of chunks to summarize", 5, 20, 10)
        with col2:
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    if st.session_state.rag_pipeline:
                        summary = st.session_state.rag_pipeline.summarize_document(max_chunks=max_chunks)
                        st.subheader("📋 Summary")
                        st.markdown(summary)
                    else:
                        st.error("RAG pipeline not available")

    def ner_interface(self):
        st.subheader("🏷️ Named Entity Recognition")
        option = st.radio("Choose input source:", ["From processed documents", "From text input"])
        if option == "From processed documents":
            if not st.session_state.vector_store_ready:
                st.warning("Please upload and process documents first!")
                return
            if st.button("Extract Entities from Documents", type="primary"):
                with st.spinner("Extracting entities..."):
                    if st.session_state.rag_pipeline:
                        entities = st.session_state.rag_pipeline.extract_entities()
                        st.subheader("🏷️ Extracted Entities")
                        for category, entity_list in entities.items():
                            if entity_list:
                                st.write(f"**{category}:**")
                                for entity in entity_list:
                                    st.write(f"- {entity}")
                    else:
                        st.error("RAG pipeline not available")
        else:
            text_input = st.text_area("Enter text for entity extraction:", height=200)
            if st.button("Extract Entities from Text", type="primary") and text_input:
                with st.spinner("Extracting entities..."):
                    if st.session_state.rag_pipeline:
                        entities = st.session_state.rag_pipeline.extract_entities(text=text_input)
                        st.subheader("🏷️ Extracted Entities")
                        for category, entity_list in entities.items():
                            if entity_list:
                                st.write(f"**{category}:**")
                                for entity in entity_list:
                                    st.write(f"- {entity}")
                    else:
                        st.error("RAG pipeline not available")

    def direct_text_interface(self):
        st.subheader("📄 Direct Text Chat")
        st.info("Ask any question directly to Mistral-7B (without document context)")
        if not st.session_state.model_loaded:
            st.warning("Please load the model first!")
            return
        direct_query = st.text_area("Enter your question or text:", height=100)
        if st.button("Get Response", type="primary") and direct_query:
            with st.spinner("Generating response..."):
                if st.session_state.llm_manager:
                    system_prompt = "You are a helpful AI assistant. Please provide a clear and informative response to the user's question."
                    prompt = st.session_state.llm_manager.format_mistral_prompt(system_prompt, direct_query)
                    response = st.session_state.llm_manager.generate_response(prompt, max_tokens=512)
                    st.subheader("🤖 Response")
                    st.markdown(response)
                else:
                    st.error("LLM manager not available")

    def run(self):
        self.setup_sidebar()
        self.main_interface()

if __name__ == "__main__":
    app = RAGApp()
    app.run()
