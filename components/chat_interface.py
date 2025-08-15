import streamlit as st
from typing import List, Dict, Any

class ChatInterface:
    def __init__(self):
        """Initialize chat interface."""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
    def display_chat_history(self):
        """Display chat history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def add_message(self, role: str, content: str):
        """Add message to chat history."""
        st.session_state.messages.append({"role": role, "content": content})
    
    def clear_chat(self):
        """Clear chat history."""
        st.session_state.messages = []
        st.session_state.chat_history = []
    
    def display_sources(self, sources: List[Dict[str, Any]]):
        """Display document sources."""
        if sources:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(sources):
                    st.write(f"**Source {i+1}:**")
                    for key, value in source.items():
                        st.write(f"- {key}: {value}")
                    st.write("---")
    
    def display_context(self, context: str):
        """Display retrieved context."""
        if context:
            with st.expander("🔍 Retrieved Context", expanded=False):
                st.text_area("Context used for answering:", value=context, height=200, disabled=True)
