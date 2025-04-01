#!/usr/bin/env python3
"""
Sample Project Main Module

This is a simple sample project to test the Talk to Code application.
It provides basic functionality to demonstrate code analysis capabilities.
"""

import os
import json
from typing import Dict, List, Any, Optional
from utils.helper import format_output, generate_random_id
from models.user import User
from models.product import Product


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    if not os.path.exists(config_path):
        return {"default": True, "debug": False}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {"default": True, "error": str(e)}


class DataManager:
    """Manages data operations for the sample application."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        self.users: List[User] = []
        self.products: List[Product] = []
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def add_user(self, name: str, email: str) -> User:
        """
        Add a new user.
        
        Args:
            name: User's name
            email: User's email address
            
        Returns:
            Newly created User object
        """
        user_id = generate_random_id()
        user = User(user_id, name, email)
        self.users.append(user)
        self._save_users()
        return user
    
    def add_product(self, name: str, price: float, description: str) -> Product:
        """
        Add a new product.
        
        Args:
            name: Product name
            price: Product price
            description: Product description
            
        Returns:
            Newly created Product object
        """
        product_id = generate_random_id()
        product = Product(product_id, name, price, description)
        self.products.append(product)
        self._save_products()
        return product
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Find a user by ID.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            User object if found, None otherwise
        """
        for user in self.users:
            if user.id == user_id:
                return user
        return None
    
    def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """
        Find a product by ID.
        
        Args:
            product_id: Product's unique identifier
            
        Returns:
            Product object if found, None otherwise
        """
        for product in self.products:
            if product.id == product_id:
                return product
        return None
    
    def _save_users(self) -> None:
        """Save users to a JSON file."""
        user_data = [user.to_dict() for user in self.users]
        with open(os.path.join(self.data_dir, "users.json"), 'w') as f:
            json.dump(user_data, f, indent=2)
    
    def _save_products(self) -> None:
        """Save products to a JSON file."""
        product_data = [product.to_dict() for product in self.products]
        with open(os.path.join(self.data_dir, "products.json"), 'w') as f:
            json.dump(product_data, f, indent=2)
    
    def load_data(self) -> None:
        """Load all data from files."""
        self._load_users()
        self._load_products()
    
    def _load_users(self) -> None:
        """Load users from a JSON file."""
        file_path = os.path.join(self.data_dir, "users.json")
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r') as f:
                user_data = json.load(f)
                self.users = [User.from_dict(data) for data in user_data]
        except Exception as e:
            print(f"Error loading users: {str(e)}")
    
    def _load_products(self) -> None:
        """Load products from a JSON file."""
        file_path = os.path.join(self.data_dir, "products.json")
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r') as f:
                product_data = json.load(f)
                self.products = [Product.from_dict(data) for data in product_data]
        except Exception as e:
            print(f"Error loading products: {str(e)}")


def main():
    """Main function to run the sample application."""
    config = load_config()
    debug_mode = config.get("debug", False)
    
    if debug_mode:
        print("Running in debug mode")
    
    data_manager = DataManager()
    data_manager.load_data()
    
    # Add some example data if empty
    if not data_manager.users:
        data_manager.add_user("John Doe", "john.doe@example.com")
        data_manager.add_user("Jane Smith", "jane.smith@example.com")
    
    if not data_manager.products:
        data_manager.add_product("Laptop", 999.99, "High-performance laptop")
        data_manager.add_product("Smartphone", 499.99, "Latest smartphone model")
    
    # Display data
    print("\nUsers:")
    for user in data_manager.users:
        print(format_output(user))
    
    print("\nProducts:")
    for product in data_manager.products:
        print(format_output(product))
    
    print("\nSample application completed successfully.")


if __name__ == "__main__":
    main() 