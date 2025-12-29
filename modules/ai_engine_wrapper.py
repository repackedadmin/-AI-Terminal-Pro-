# ================================================
# FILE: modules/ai_engine_wrapper.py
# AIEngine Wrapper for TTS Pro Mode
# Integrates AI Terminal Pro's AIEngine with MiraiAssist's MemoryManager
# ================================================

import logging
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class AIEngineWrapperError(Exception):
    """Raised for AIEngineWrapper initialization or runtime errors."""
    pass


class AIEngineWrapper:
    """
    Wrapper around AI Terminal Pro's AIEngine that integrates with
    MiraiAssist's MemoryManager for RAG-enhanced context.
    
    Supports both HuggingFace and Ollama backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AIEngine with configuration from main config.
        
        Args:
            config: Configuration dictionary from AI Terminal Pro
        """
        logger.info("Initializing AIEngineWrapper...")
        self.config = config
        self.backend = config.get("backend", "huggingface")
        self.model_name = config.get("model_name", "gpt2")
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._load()
        logger.info("AIEngineWrapper initialized successfully.")
    
    def _load(self):
        """Load the model based on backend configuration."""
        logger.info(f"Loading Backend: {self.backend} (Model: {self.model_name})...")
        
        if self.backend == "huggingface":
            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Determine device
                if torch.cuda.is_available(): 
                    self.device = "cuda"
                elif torch.backends.mps.is_available(): 
                    self.device = "mps"
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info(f"Model loaded on {self.device}.")
                
            except Exception as e:
                logger.critical(f"Model Load Failed: {e}", exc_info=True)
                raise AIEngineWrapperError(f"Failed to load HuggingFace model: {e}")
                
        elif self.backend == "ollama":
            try:
                # Test Ollama connection
                requests.get("http://localhost:11434", timeout=2)
                logger.info("Ollama connection established.")
            except Exception as e:
                logger.warning(f"Ollama is not running on localhost:11434: {e}")
                raise AIEngineWrapperError("Ollama backend not available")
        else:
            raise AIEngineWrapperError(f"Unknown backend: {self.backend}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt.
        
        Args:
            prompt: The full prompt including system message and context
            
        Returns:
            Generated response text
        """
        if self.backend == "huggingface":
            return self._generate_huggingface(prompt)
        elif self.backend == "ollama":
            return self._generate_ollama(prompt)
        else:
            return f"Error: Unknown backend {self.backend}"
    
    def _generate_huggingface(self, prompt: str) -> str:
        """Generate response using HuggingFace model."""
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.config.get("max_response_tokens", 2000),
                    do_sample=True,
                    temperature=self.config.get("temperature", 0.7),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Remove prompt from output
            response = full[len(prompt):].strip()
            
            # Stop sequence trimming
            if "You:" in response: 
                response = response.split("You:")[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}", exc_info=True)
            return f"Error generating response: {e}"
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate response using Ollama."""
        try:
            res = requests.post("http://localhost:11434/api/generate", json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.get("temperature", 0.7),
                    "num_predict": self.config.get("max_response_tokens", 2000)
                }
            }, timeout=120)
            
            if res.status_code == 200:
                return res.json()['response'].strip()
            return f"Ollama Error: {res.text}"
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}", exc_info=True)
            return f"Connection Error: {e}"
    
    def generate_with_memory(self, user_text: str, memory_manager) -> str:
        """
        Generate response using MemoryManager for RAG-enhanced context.
        
        Args:
            user_text: The user's current input
            memory_manager: MemoryManager instance for context retrieval
            
        Returns:
            Generated response text
        """
        try:
            # Get context from MemoryManager (includes RAG + recent messages)
            context_messages = memory_manager.construct_prompt_context(user_text)
            
            # Build prompt
            system_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")
            prompt = f"{system_prompt}\n\n"
            
            # Add context messages
            for msg in context_messages:
                role = "You" if msg["role"] == "user" else "AI"
                content = msg["content"]
                prompt += f"{role}: {content}\n"
            
            # Add current user input
            prompt += f"You: {user_text}\nAI:"
            
            logger.debug(f"Generating response with context ({len(context_messages)} messages)...")
            
            # Generate response
            response = self.generate(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_with_memory: {e}", exc_info=True)
            return f"Error: {e}"

