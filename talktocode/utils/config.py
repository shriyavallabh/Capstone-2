"""
Configuration module for the Talk to Code application.

This module now imports OpenAI configurations from the centralized 
openai_config.py module, which can be replaced with a company-specific
implementation without requiring changes to the rest of the codebase.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import OpenAI configuration from the centralized module
from talktocode.utils.openai_config import (
    # Main configuration object
    openai_config,
    
    # Backward compatibility variables
    OPENAI_API_KEY,
    MODEL_CONFIG,
    
    # Backward compatibility functions
    get_openai_client,
    set_embedding_model,
    is_api_key_valid,
    get_embedding_model_info,
    validate_config
)

# COMPANY Color Constants (Renamed from UBS)
# Moved from ui_components.py to break circular import
COMPANY_COLORS = {
    "red": "#EC0016", # Keeping original color value for now
    "light_red": "#FF6D6A",
    "dark_red": "#B30012",
    "blue": "#0205A8",
    "light_blue": "#9A9CFF",
    "dark_blue": "#000066",
    "black": "#000000",
    "dark_gray": "#333333",
    "medium_gray": "#666666",
    "light_gray": "#CCCCCC",
    "white": "#FFFFFF",
}

# Migration note for future reference
"""
NOTE: All OpenAI-related configurations have been moved to openai_config.py,
which serves as a centralized configuration hub. This makes it easier to replace
with a company-specific implementation.

To update the API key programmatically:
    from talktocode.utils.openai_config import openai_config
    openai_config.api_key = "your_api_key"

To change the embedding model:
    from talktocode.utils.openai_config import openai_config
    openai_config.set_embedding_model("text-embedding-3-large")

To customize LLM parameters:
    from talktocode.utils.openai_config import openai_config
    openai_config.llm.temperature = 0.7
    openai_config.llm.max_tokens = 2048
""" 