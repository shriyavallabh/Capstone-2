import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# UBS Color Constants
UBS_RED = "#FF0000"
LIGHT_GREY = "#F5F5F5"
DARK_GREY = "#333333"
BORDER_GREY = "#CCCCCC"

# Embedding Model Configuration
EMBEDDING_MODELS = {
    # OpenAI's text-embedding-ada-002 (legacy but widely used)
    "text-embedding-ada-002": {
        "dimensions": 1536,
        "max_batch_size": 1000,  # Maximum tokens per batch
        "max_batch_items": 100,  # Maximum items per batch
        "cost_per_1k_tokens": 0.0001,  # Cost in USD per 1K tokens
        "context_length": 8191,  # Maximum input length in tokens
        "description": "OpenAI's text-embedding-ada-002 model (legacy)"
    },
    # OpenAI's text-embedding-3-small (newer, better performance/cost ratio)
    "text-embedding-3-small": {
        "dimensions": 1536,
        "max_batch_size": 8192,  # Maximum tokens per batch
        "max_batch_items": 2048,  # Maximum items per batch
        "cost_per_1k_tokens": 0.00002,  # Cost in USD per 1K tokens
        "context_length": 8191,  # Maximum input length in tokens
        "description": "OpenAI's smaller text-embedding-3 model (recommended)"
    },
    # OpenAI's text-embedding-3-large (highest quality)
    "text-embedding-3-large": {
        "dimensions": 3072,
        "max_batch_size": 8192,  # Maximum tokens per batch
        "max_batch_items": 2048,  # Maximum items per batch
        "cost_per_1k_tokens": 0.00013,  # Cost in USD per 1K tokens
        "context_length": 8191,  # Maximum input length in tokens
        "description": "OpenAI's larger text-embedding-3 model (highest quality)"
    }
}

# Default embedding model to use
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Model Configuration
MODEL_CONFIG = {
    # Model names for different functions
    "models": {
        "embedding": DEFAULT_EMBEDDING_MODEL,  # Default embedding model
        "chat": "gpt-3.5-turbo",
        "code_analysis": "gpt-3.5-turbo-16k",
        "entity_extraction": "gpt-3.5-turbo-16k",
    },
    
    # Embedding configuration
    "embedding": {
        "model": DEFAULT_EMBEDDING_MODEL,
        "dimensions": EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]["dimensions"],
        "models_info": EMBEDDING_MODELS,
        "cache": {
            "enabled": True,
            "directory": ".talktocode_cache/embeddings",
            "expiration_days": 30,  # Cache entries expire after 30 days
            "max_size_mb": 500,  # Maximum cache size in MB
        },
        "batch": {
            "enabled": True,
            "size": min(50, EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]["max_batch_items"]),  # Default batch size
            "timeout_ms": 5000,  # Maximum time to wait for batch completion
        }
    },
    
    # Graph parameters
    "graph": {
        "similarity_threshold": 0.75,
        "max_connections": 5,
    },
    
    # Community detection parameters
    "community_detection": {
        "resolution": 1.0,
        "randomize": True,
        "seed": 42,
        "min_community_size": 3,
    },
    
    # Retrieval parameters
    "retrieval": {
        "top_k": 5,
        "max_distance": 2,
        "weight_decay": 0.8,
    }
}

def is_api_key_valid():
    """
    Check if the OpenAI API key is valid by making a test request.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    if not OPENAI_API_KEY:
        return False
        
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Using models endpoint as a lightweight way to validate the API key
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_embedding_model_info(model_name=None):
    """
    Get information about a specific embedding model or the current default.
    
    Args:
        model_name: Optional name of the model to get info for
                   If None, returns info for the current default model
    
    Returns:
        dict: Model information dictionary
    """
    if model_name is None:
        model_name = MODEL_CONFIG["models"]["embedding"]
        
    if model_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_name]
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")

def set_embedding_model(model_name):
    """
    Set the embedding model to use.
    
    Args:
        model_name: Name of the model to use
    
    Returns:
        bool: True if successful, False if model not found
    """
    if model_name in EMBEDDING_MODELS:
        MODEL_CONFIG["models"]["embedding"] = model_name
        MODEL_CONFIG["embedding"]["model"] = model_name
        MODEL_CONFIG["embedding"]["dimensions"] = EMBEDDING_MODELS[model_name]["dimensions"]
        
        # Update batch size based on model's max batch items
        MODEL_CONFIG["embedding"]["batch"]["size"] = min(
            50,  # Default reasonable batch size
            EMBEDDING_MODELS[model_name]["max_batch_items"]
        )
        return True
    else:
        return False

# Validate configurations
def validate_config():
    """
    Validate the configuration settings.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not OPENAI_API_KEY:
        return False, "OpenAI API key is missing. Please set it in your .env file or environment variables."
    
    if not is_api_key_valid():
        return False, "Invalid OpenAI API key. Please check your API key."
    
    # Check if embedding model exists
    current_embedding_model = MODEL_CONFIG["models"]["embedding"]
    if current_embedding_model not in EMBEDDING_MODELS:
        return False, f"Invalid embedding model: {current_embedding_model}. Available models: {', '.join(EMBEDDING_MODELS.keys())}."
    
    return True, "Configuration is valid." 