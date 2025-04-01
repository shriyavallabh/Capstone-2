"""
User model for the sample application.
"""

from typing import Dict, Any
from datetime import datetime


class User:
    """Represents a user in the system."""
    
    def __init__(self, id: str, name: str, email: str):
        """
        Initialize a user.
        
        Args:
            id: Unique identifier for the user
            name: User's full name
            email: User's email address
        """
        self.id = id
        self.name = name
        self.email = email
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def update_email(self, new_email: str) -> None:
        """
        Update the user's email address.
        
        Args:
            new_email: New email address
        """
        self.email = new_email
        self.updated_at = datetime.now().isoformat()
    
    def update_name(self, new_name: str) -> None:
        """
        Update the user's name.
        
        Args:
            new_name: New name
        """
        self.name = new_name
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user to a dictionary.
        
        Returns:
            Dictionary representation of the user
        """
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        Create a user from a dictionary.
        
        Args:
            data: Dictionary containing user data
            
        Returns:
            User object
        """
        user = cls(
            id=data["id"],
            name=data["name"],
            email=data["email"]
        )
        user.created_at = data.get("created_at", user.created_at)
        user.updated_at = data.get("updated_at", user.updated_at)
        return user
    
    def __str__(self) -> str:
        """String representation of the user."""
        return f"User({self.id}, {self.name}, {self.email})" 