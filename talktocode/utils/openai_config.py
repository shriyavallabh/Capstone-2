"""
Centralized OpenAI Configuration Module

This module provides a unified configuration system for all OpenAI-related settings.
It's designed to be easily replaceable with a company-specific implementation.

To use the company's API management system:
1. Replace this file with your implementation
2. Ensure it provides the same interface (methods and properties)
3. The rest of the codebase will work without further modifications
"""

import os
from typing import Dict, Any, Optional, List, Union, Callable
import warnings
from dotenv import load_dotenv
import requests

# Import the OpenAI client - this would be replaced with your company's client
from openai import OpenAI

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class EmbeddingModelConfig:
    """Configuration for embedding models"""
    
    def __init__(self, 
                 model_name: str,
                 dimensions: int,
                 max_batch_size: int,
                 max_batch_items: int,
                 cost_per_1k_tokens: float,
                 context_length: int,
                 description: str):
        self.model_name = model_name
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self.max_batch_items = max_batch_items
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.context_length = context_length
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "dimensions": self.dimensions,
            "max_batch_size": self.max_batch_size,
            "max_batch_items": self.max_batch_items,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "context_length": self.context_length,
            "description": self.description
        }


class LLMConfig:
    """Configuration for Large Language Models"""
    
    def __init__(self,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.5,
                 max_tokens: int = 4096,
                 request_timeout: int = 120):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout
        }


class FaissConfig:
    """Configuration for FAISS indexing"""
    
    def __init__(self,
                 enabled: bool = True,
                 index_directory: str = ".talktocode_cache/faiss_indices",
                 index_factory_string: str = "Flat",
                 normalize_vectors: bool = True):
        self.enabled = enabled
        self.index_directory = index_directory
        self.index_factory_string = index_factory_string
        self.normalize_vectors = normalize_vectors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "enabled": self.enabled,
            "index_directory": self.index_directory,
            "index_factory_string": self.index_factory_string,
            "normalize_vectors": self.normalize_vectors
        }


class CacheConfig:
    """Configuration for embedding cache"""
    
    def __init__(self,
                 enabled: bool = True,
                 directory: str = ".talktocode_cache/embeddings",
                 expiration_days: int = 30,
                 max_size_mb: int = 500):
        self.enabled = enabled
        self.directory = directory
        self.expiration_days = expiration_days
        self.max_size_mb = max_size_mb
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "enabled": self.enabled,
            "directory": self.directory,
            "expiration_days": self.expiration_days,
            "max_size_mb": self.max_size_mb
        }


class EmbeddingConfig:
    """Configuration for embeddings"""
    
    def __init__(self,
                 model: str = "text-embedding-3-small",
                 cache: Optional[CacheConfig] = None,
                 faiss: Optional[FaissConfig] = None,
                 batch_enabled: bool = True,
                 batch_size: int = 50,
                 batch_timeout_ms: int = 5000):
        self.model = model
        self.cache = cache or CacheConfig()
        self.faiss = faiss or FaissConfig()
        self.batch_enabled = batch_enabled
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Will be set when embedding model is chosen
        self.dimensions = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "cache": self.cache.to_dict(),
            "faiss": self.faiss.to_dict(),
            "batch": {
                "enabled": self.batch_enabled,
                "size": self.batch_size,
                "timeout_ms": self.batch_timeout_ms
            }
        }


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

class OpenAIConfiguration:
    """
    Centralized configuration for OpenAI
    
    This class manages all OpenAI-related configurations, including:
    - API key management
    - Client initialization
    - Model selection
    - Embedding configuration
    
    It's designed to be replaced with a company-specific implementation
    that integrates with your company's API management system.
    """
    
    def __init__(self):
        # Initialize the API key from environment
        self._api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Define available embedding models
        self._embedding_models = {
            # OpenAI's text-embedding-ada-002 (legacy but widely used)
            "text-embedding-ada-002": EmbeddingModelConfig(
                model_name="text-embedding-ada-002",
                dimensions=1536,
                max_batch_size=1000,
                max_batch_items=100,
                cost_per_1k_tokens=0.0001,
                context_length=8191,
                description="OpenAI's text-embedding-ada-002 model (legacy)"
            ),
            # OpenAI's text-embedding-3-small (newer, better performance/cost ratio)
            "text-embedding-3-small": EmbeddingModelConfig(
                model_name="text-embedding-3-small",
                dimensions=1536,
                max_batch_size=8192,
                max_batch_items=2048,
                cost_per_1k_tokens=0.00002,
                context_length=8191,
                description="OpenAI's smaller text-embedding-3 model (recommended)"
            ),
            # OpenAI's text-embedding-3-large (highest quality)
            "text-embedding-3-large": EmbeddingModelConfig(
                model_name="text-embedding-3-large",
                dimensions=3072,
                max_batch_size=8192,
                max_batch_items=2048,
                cost_per_1k_tokens=0.00013,
                context_length=8191,
                description="OpenAI's larger text-embedding-3 model (highest quality)"
            )
        }
        
        # Define default embedding model
        self._default_embedding_model = "text-embedding-3-small"
        
        # Create configuration objects
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig(model=self._default_embedding_model)
        
        # Set dimensions based on the default model
        self.embedding.dimensions = self._embedding_models[self._default_embedding_model].dimensions
        
        # Create a client instance (lazy loaded when needed)
        self._client = None
    
    @property
    def api_key(self) -> str:
        """Get the current API key"""
        return self._api_key
    
    @api_key.setter
    def api_key(self, value: str) -> None:
        """Set the API key and recreate the client"""
        self._api_key = value
        os.environ["OPENAI_API_KEY"] = value
        # Reset client so it's recreated with the new key
        self._client = None
    
    def is_api_key_valid(self) -> bool:
        """
        Check if the OpenAI API key is valid by making a test request.
        
        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        if not self._api_key:
            return False
            
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Using models endpoint as a lightweight way to validate the API key
            response = requests.get("https://api.openai.com/v1/models", headers=headers)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_client(self) -> OpenAI:
        """
        Get an OpenAI client instance.
        
        This is a lazy-loading factory that ensures the client is
        created with the latest API key.
        
        Returns:
            OpenAI: An OpenAI client instance
        """
        if self._client is None:
            # In a company implementation, this would use your company's API wrapper
            self._client = OpenAI(api_key=self._api_key)
        return self._client
    
    def set_embedding_model(self, model_name: str) -> bool:
        """
        Set the embedding model to use.
        
        Args:
            model_name: Name of the model to use
        
        Returns:
            bool: True if successful, False if model not found
        """
        if model_name in self._embedding_models:
            self.embedding.model = model_name
            self.embedding.dimensions = self._embedding_models[model_name].dimensions
            
            # Update batch size based on model's max batch items
            self.embedding.batch_size = min(
                50,  # Default reasonable batch size
                self._embedding_models[model_name].max_batch_items
            )
            return True
        else:
            warnings.warn(f"Unknown embedding model: {model_name}. Using {self.embedding.model} instead.")
            return False
    
    def get_embedding_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific embedding model or the current default.
        
        Args:
            model_name: Optional name of the model to get info for
                       If None, returns info for the current default model
        
        Returns:
            dict: Model information dictionary
        """
        if model_name is None:
            model_name = self.embedding.model
            
        if model_name in self._embedding_models:
            return self._embedding_models[model_name].to_dict()
        else:
            raise ValueError(f"Unknown embedding model: {model_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        This is mainly used for compatibility with the existing codebase.
        
        Returns:
            Dict[str, Any]: Configuration as a dictionary
        """
        embedding_models_dict = {name: model.to_dict() for name, model in self._embedding_models.items()}
        
        return {
            "models": {
                "embedding": self.embedding.model,
                "chat": self.llm.model_name,
                "code_analysis": self.llm.model_name,
                "entity_extraction": self.llm.model_name,
            },
            "embedding": self.embedding.to_dict(),
            "llm": self.llm.to_dict(),
            "embedding_models": embedding_models_dict,
            "graph": {
                "similarity_threshold": 0.75,
                "max_connections": 5,
            },
            "community_detection": {
                "resolution": 1.0,
                "randomize": True,
                "seed": 42,
                "min_community_size": 3,
            },
            "retrieval": {
                "top_k": 5,
                "max_distance": 2,
                "weight_decay": 0.8,
            }
        }
    
    def validate(self) -> tuple:
        """
        Validate the configuration settings.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not self._api_key:
            return False, "OpenAI API key is missing. Please set it in your .env file or environment variables."
        
        if not self.is_api_key_valid():
            return False, "Invalid OpenAI API key. Please check your API key."
        
        # Check if embedding model exists
        current_embedding_model = self.embedding.model
        if current_embedding_model not in self._embedding_models:
            return False, f"Invalid embedding model: {current_embedding_model}. Available models: {', '.join(self._embedding_models.keys())}."
        
        return True, "Configuration is valid."


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Create a global instance for use throughout the application
openai_config = OpenAIConfiguration()

# For backward compatibility with existing code
MODEL_CONFIG = openai_config.to_dict()
OPENAI_API_KEY = openai_config.api_key

# Backward compatibility functions
def get_openai_client() -> OpenAI:
    """Get an OpenAI client instance"""
    return openai_config.get_client()

def set_embedding_model(model_name: str) -> bool:
    """Set the embedding model to use"""
    return openai_config.set_embedding_model(model_name)

def is_api_key_valid() -> bool:
    """Check if the API key is valid"""
    return openai_config.is_api_key_valid()

def get_embedding_model_info(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get information about an embedding model"""
    return openai_config.get_embedding_model_info(model_name)

def validate_config() -> tuple:
    """Validate the configuration"""
    return openai_config.validate() 