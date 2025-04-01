"""
Product model for the sample application.
"""

from typing import Dict, Any, Optional
from datetime import datetime


class Product:
    """Represents a product in the system."""
    
    def __init__(self, id: str, name: str, price: float, description: str, 
                in_stock: bool = True, category: Optional[str] = None):
        """
        Initialize a product.
        
        Args:
            id: Unique identifier for the product
            name: Product name
            price: Product price
            description: Product description
            in_stock: Whether the product is in stock
            category: Optional product category
        """
        self.id = id
        self.name = name
        self.price = price
        self.description = description
        self.in_stock = in_stock
        self.category = category
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def update_price(self, new_price: float) -> None:
        """
        Update the product price.
        
        Args:
            new_price: New price
        """
        self.price = new_price
        self.updated_at = datetime.now().isoformat()
    
    def update_description(self, new_description: str) -> None:
        """
        Update the product description.
        
        Args:
            new_description: New description
        """
        self.description = new_description
        self.updated_at = datetime.now().isoformat()
    
    def set_category(self, category: str) -> None:
        """
        Set the product category.
        
        Args:
            category: Product category
        """
        self.category = category
        self.updated_at = datetime.now().isoformat()
    
    def set_stock_status(self, in_stock: bool) -> None:
        """
        Set whether the product is in stock.
        
        Args:
            in_stock: Whether the product is in stock
        """
        self.in_stock = in_stock
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert product to a dictionary.
        
        Returns:
            Dictionary representation of the product
        """
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "in_stock": self.in_stock,
            "category": self.category,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """
        Create a product from a dictionary.
        
        Args:
            data: Dictionary containing product data
            
        Returns:
            Product object
        """
        product = cls(
            id=data["id"],
            name=data["name"],
            price=data["price"],
            description=data["description"],
            in_stock=data.get("in_stock", True),
            category=data.get("category")
        )
        product.created_at = data.get("created_at", product.created_at)
        product.updated_at = data.get("updated_at", product.updated_at)
        return product
    
    def __str__(self) -> str:
        """String representation of the product."""
        status = "In Stock" if self.in_stock else "Out of Stock"
        category_info = f", {self.category}" if self.category else ""
        return f"Product({self.id}, {self.name}, ${self.price}, {status}{category_info})" 