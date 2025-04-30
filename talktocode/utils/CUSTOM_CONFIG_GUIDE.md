# Customizing OpenAI Configuration for Your Company Environment

This guide explains how to adapt the Talk to Code application to work with your company's API management system for OpenAI.

## Overview

The application has been designed with a centralized configuration system for all OpenAI-related settings in `talktocode/utils/openai_config.py`. This makes it easy to replace with a company-specific implementation without modifying the rest of the codebase.

## How to Customize

### Option 1: Replace the entire `openai_config.py` file

1. Create a new `openai_config.py` file that implements the same interface
2. Replace the existing file with your implementation
3. Ensure it provides the same variables and functions for backward compatibility

Required exports:
- `openai_config`: The main configuration object
- `OPENAI_API_KEY`: The API key string
- `MODEL_CONFIG`: Dictionary with model settings
- `get_openai_client()`: Function to get an OpenAI client
- `set_embedding_model(model_name)`: Function to change the embedding model
- `is_api_key_valid()`: Function to check if the API key is valid
- `get_embedding_model_info(model_name)`: Function to get model details
- `validate_config()`: Function to validate the configuration

### Option 2: Extend the existing configuration

1. Create a subclass of `OpenAIConfiguration` in your own module
2. Override the necessary methods to use your company's API management
3. Replace the global instance with your custom one

Example:

```python
from talktocode.utils.openai_config import OpenAIConfiguration

class CompanyOpenAIConfiguration(OpenAIConfiguration):
    def __init__(self):
        super().__init__()
        self.company_client = None
        
    def get_client(self):
        """Override to use company's client"""
        if self.company_client is None:
            from company.api import ApiClient
            self.company_client = ApiClient()
        return self.company_client
        
    # Override other methods as needed

# Replace the global instance
import talktocode.utils.openai_config
talktocode.utils.openai_config.openai_config = CompanyOpenAIConfiguration()
```

## Key Methods to Override

The most important methods to customize for your company environment are:

1. `get_client()`: Replace with your company's API client
2. `api_key` property: Handle API key access through your company's system
3. `set_embedding_model()`: Adapt model selection to your company's available models

## Testing Your Implementation

After customizing, verify that all OpenAI-related functionality works by:

1. Running the application
2. Checking that the API key validation works
3. Testing embedding generation
4. Testing report generation

## Company-Specific Configurations

If your company has specific configurations for:

- API endpoints
- Model availability
- Rate limiting
- Authentication
- Proxy settings

These can all be handled in your custom implementation without affecting the rest of the codebase. 