from llama_cpp import Llama
import os
from typing import Optional

class LocalLLMManager:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 35):
        """
        Initialize the local LLM manager.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model."""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=6,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1
            )
            print(f"âœ… Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"âŒ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response from the LLM."""
        if not self.llm:
            raise ValueError("Model not loaded")
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["</s>", "[INST]", "[/INST]"],
                echo=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error generating response: {e}"
    
    def format_mistral_prompt(self, system_prompt: str, user_query: str, context: str = "") -> str:
        """Format prompt for Mistral model."""
        if context:
            prompt = f"""[INST] {system_prompt}

Context: {context}

Question: {user_query} [/INST]"""
        else:
            prompt = f"[INST] {system_prompt}\n\nQuestion: {user_query} [/INST]"
        
        return prompt
