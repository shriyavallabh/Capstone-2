"""
Helper utilities for the sample application.
"""

import random
import string
import json
from typing import Any, Dict


def generate_random_id(length: int = 10) -> str:
    """
    Generate a random ID with the specified length.
    
    Args:
        length: Length of the ID to generate
        
    Returns:
        Random ID string
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def format_output(obj: Any) -> str:
    """
    Format an object for display.
    
    Args:
        obj: Object to format
        
    Returns:
        Formatted string representation
    """
    if hasattr(obj, 'to_dict'):
        obj_dict = obj.to_dict()
        return json.dumps(obj_dict, indent=2)
    
    return str(obj)


def validate_email(email: str) -> bool:
    """
    Validate an email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if the email is valid, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def create_error_response(message: str, code: int = 400) -> Dict[str, Any]:
    """
    Create an error response.
    
    Args:
        message: Error message
        code: Error code
        
    Returns:
        Error response dictionary
    """
    return {
        "error": True,
        "code": code,
        "message": message
    }


def create_success_response(data: Any) -> Dict[str, Any]:
    """
    Create a success response.
    
    Args:
        data: Response data
        
    Returns:
        Success response dictionary
    """
    return {
        "error": False,
        "data": data
    } 