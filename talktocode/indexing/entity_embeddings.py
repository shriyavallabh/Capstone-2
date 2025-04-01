"""
Code entity embedding generator for the TalkToCode system.

This module creates and manages embeddings for code entities using
OpenAI's embedding models, focusing on text-embedding-ada-002 model.
It creates separate embeddings for different aspects of code entities:
1. Full code snippets
2. Function/class names and signatures
3. Docstrings and comments

These embeddings enable efficient similar entity lookups for relationship extraction.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from pathlib import Path
import hashlib
from tqdm import tqdm

# Import OpenAI library
from openai import OpenAI

# Import local modules
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import OPENAI_API_KEY, MODEL_CONFIG, set_embedding_model
from talktocode.indexing.entity_extractor import (
    CodeEntity, FunctionEntity, ClassEntity, 
    VariableEntity, ImportEntity, extract_code_with_context
)

# Configure the embedding model
# Set to text-embedding-ada-002 as required
set_embedding_model("text-embedding-ada-002")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class EntityEmbeddings:
    """Stores and manages embeddings for a code entity."""
    
    def __init__(self, entity_id: str):
        """
        Initialize embeddings for a code entity.
        
        Args:
            entity_id: Unique identifier for the entity (typically name:file:line)
        """
        self.entity_id = entity_id
        
        # Initialize embeddings for different aspects
        self.full_code_embedding: Optional[np.ndarray] = None
        self.name_signature_embedding: Optional[np.ndarray] = None
        self.docstring_embedding: Optional[np.ndarray] = None
        
        # Track when embeddings were last updated
        self.last_updated = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert embeddings to dictionary representation for storage."""
        return {
            "entity_id": self.entity_id,
            "full_code_embedding": self.full_code_embedding.tolist() if self.full_code_embedding is not None else None,
            "name_signature_embedding": self.name_signature_embedding.tolist() if self.name_signature_embedding is not None else None,
            "docstring_embedding": self.docstring_embedding.tolist() if self.docstring_embedding is not None else None,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityEmbeddings':
        """Create an EntityEmbeddings object from a dictionary."""
        embeddings = cls(data["entity_id"])
        
        if data.get("full_code_embedding") is not None:
            embeddings.full_code_embedding = np.array(data["full_code_embedding"])
            
        if data.get("name_signature_embedding") is not None:
            embeddings.name_signature_embedding = np.array(data["name_signature_embedding"])
            
        if data.get("docstring_embedding") is not None:
            embeddings.docstring_embedding = np.array(data["docstring_embedding"])
            
        embeddings.last_updated = data.get("last_updated")
        
        return embeddings


class EntityEmbeddingGenerator:
    """Generates and manages embeddings for code entities."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            cache_dir: Directory to cache embeddings (uses default from config if None)
        """
        # Use the cache directory from config or the provided directory
        if cache_dir is None:
            cache_dir = MODEL_CONFIG["embedding"]["cache"]["directory"]
        
        self.cache_dir = os.path.join(cache_dir, "entity_embeddings")
        self.embeddings_cache: Dict[str, EntityEmbeddings] = {}
        self.loaded_cache = False
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if we need to load the cache
        if MODEL_CONFIG["embedding"]["cache"]["enabled"]:
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load embeddings from cache files."""
        if not os.path.exists(self.cache_dir):
            return
        
        try:
            # Load the cache index if it exists
            index_file = os.path.join(self.cache_dir, "index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                # Load each embedding file referenced in the index
                for entity_id, file_path in index.items():
                    full_path = os.path.join(self.cache_dir, file_path)
                    if os.path.exists(full_path):
                        with open(full_path, 'r') as f:
                            data = json.load(f)
                            self.embeddings_cache[entity_id] = EntityEmbeddings.from_dict(data)
            
            self.loaded_cache = True
        except Exception as e:
            print(f"Error loading embedding cache: {str(e)}")
    
    def _save_to_cache(self, entity_embeddings: EntityEmbeddings) -> None:
        """
        Save entity embeddings to cache.
        
        Args:
            entity_embeddings: EntityEmbeddings object to save
        """
        if not MODEL_CONFIG["embedding"]["cache"]["enabled"]:
            return
        
        try:
            # Create a filename from the entity ID (using a hash to avoid invalid filenames)
            entity_hash = hashlib.md5(entity_embeddings.entity_id.encode()).hexdigest()
            file_name = f"{entity_hash}.json"
            file_path = os.path.join(self.cache_dir, file_name)
            
            # Save the embeddings to file
            with open(file_path, 'w') as f:
                json.dump(entity_embeddings.to_dict(), f)
            
            # Update the index file
            index_file = os.path.join(self.cache_dir, "index.json")
            index = {}
            
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    index = json.load(f)
            
            index[entity_embeddings.entity_id] = file_name
            
            with open(index_file, 'w') as f:
                json.dump(index, f)
                
        except Exception as e:
            print(f"Error saving embedding to cache: {str(e)}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding for a text string using OpenAI's API.
        
        Args:
            text: Text to embed
            
        Returns:
            NumPy array containing the embedding vector
        """
        try:
            # Get the embedding model from config
            embedding_model = MODEL_CONFIG["models"]["embedding"]
            
            # Call OpenAI API to generate embedding
            response = client.embeddings.create(
                model=embedding_model,
                input=text
            )
            
            # Extract embedding
            embedding = response.data[0].embedding
            return np.array(embedding)
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return a zero vector with the correct dimensions
            return np.zeros(MODEL_CONFIG["embedding"]["dimensions"])
    
    def get_entity_id(self, entity: CodeEntity) -> str:
        """
        Generate a unique identifier for a code entity.
        
        Args:
            entity: Code entity
            
        Returns:
            String containing the entity ID
        """
        return f"{entity.name}:{entity.source_file}:{entity.lineno}"
    
    def extract_name_and_signature(self, entity: CodeEntity) -> str:
        """
        Extract the name and signature for a code entity.
        
        Args:
            entity: Code entity
            
        Returns:
            String containing the name and signature
        """
        if isinstance(entity, FunctionEntity):
            # For functions, include name and parameters
            params_str = ", ".join([
                f"{p.get('name', '')}: {p.get('type', '')}" 
                if p.get('type') else p.get('name', '')
                for p in entity.parameters
            ])
            
            return_str = f" -> {entity.returns}" if entity.returns else ""
            
            if entity.is_method:
                return f"method {entity.name}({params_str}){return_str} of class {entity.parent_class}"
            else:
                return f"function {entity.name}({params_str}){return_str}"
                
        elif isinstance(entity, ClassEntity):
            # For classes, include name and base classes
            bases_str = ", ".join(entity.base_classes) if entity.base_classes else ""
            if bases_str:
                return f"class {entity.name}({bases_str})"
            else:
                return f"class {entity.name}"
                
        elif isinstance(entity, VariableEntity):
            # For variables, include name and type
            type_str = f": {entity.var_type}" if entity.var_type else ""
            value_str = f" = {entity.value}" if entity.value else ""
            
            if entity.is_class_property:
                return f"property {entity.name}{type_str}{value_str} of class {entity.parent_class}"
            else:
                return f"variable {entity.name}{type_str}{value_str}"
                
        elif isinstance(entity, ImportEntity):
            # For imports, include import statement
            if entity.import_from:
                return f"from {entity.import_from} import {entity.name}"
            else:
                alias_str = f" as {entity.alias}" if entity.alias else ""
                return f"import {entity.name}{alias_str}"
        
        # Default case
        return entity.name
    
    def extract_docstring_and_comments(self, entity: CodeEntity, context_lines: int = 5) -> str:
        """
        Extract docstring and nearby comments for a code entity.
        
        Args:
            entity: Code entity
            context_lines: Number of context lines to consider
            
        Returns:
            String containing docstring and comments
        """
        result = []
        
        # Add docstring if available
        if hasattr(entity, 'docstring') and entity.docstring:
            result.append(entity.docstring)
        
        # Extract code context
        context = extract_code_with_context(entity, context_lines)
        
        # Extract comments from context
        lines = context.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                result.append(line[1:].strip())
            elif '#' in line:
                # Get in-line comment
                comment = line.split('#', 1)[1].strip()
                if comment:
                    result.append(comment)
        
        return '\n'.join(result)
    
    def generate_embeddings_for_entity(self, 
                                      entity: CodeEntity, 
                                      context_lines: int = 5,
                                      use_cache: bool = True) -> EntityEmbeddings:
        """
        Generate all embeddings for a code entity.
        
        Args:
            entity: Code entity
            context_lines: Number of context lines to consider
            use_cache: Whether to use cached embeddings
            
        Returns:
            EntityEmbeddings object with all embeddings
        """
        # Generate entity ID
        entity_id = self.get_entity_id(entity)
        
        # Check if we have cached embeddings
        if use_cache and entity_id in self.embeddings_cache:
            return self.embeddings_cache[entity_id]
        
        # Create new embeddings
        embeddings = EntityEmbeddings(entity_id)
        
        # 1. Generate full code embedding
        full_code = extract_code_with_context(entity, context_lines)
        embeddings.full_code_embedding = self.get_embedding(full_code)
        
        # 2. Generate name and signature embedding
        name_signature = self.extract_name_and_signature(entity)
        embeddings.name_signature_embedding = self.get_embedding(name_signature)
        
        # 3. Generate docstring and comments embedding
        docstring_comments = self.extract_docstring_and_comments(entity, context_lines)
        if docstring_comments:
            embeddings.docstring_embedding = self.get_embedding(docstring_comments)
        
        # Set last updated timestamp
        embeddings.last_updated = time.time()
        
        # Cache the embeddings
        self.embeddings_cache[entity_id] = embeddings
        self._save_to_cache(embeddings)
        
        return embeddings
    
    def generate_embeddings_for_entities(self, 
                                        entities: Dict[str, List[CodeEntity]],
                                        context_lines: int = 5,
                                        use_cache: bool = True) -> Dict[str, EntityEmbeddings]:
        """
        Generate embeddings for all entities.
        
        Args:
            entities: Dictionary of entities by type
            context_lines: Number of context lines to consider
            use_cache: Whether to use cached embeddings
            
        Returns:
            Dictionary mapping entity IDs to EntityEmbeddings objects
        """
        results = {}
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        
        # Process entities with a progress bar
        with tqdm(total=total_entities, desc="Generating embeddings") as pbar:
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    # Generate entity ID
                    entity_id = self.get_entity_id(entity)
                    
                    # Generate embeddings
                    embeddings = self.generate_embeddings_for_entity(
                        entity, context_lines, use_cache
                    )
                    
                    # Store results
                    results[entity_id] = embeddings
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Add a small delay to avoid API rate limits
                    time.sleep(0.1)
        
        return results
    
    def find_similar_entities(self, 
                             query_entity: CodeEntity,
                             entities: Dict[str, List[CodeEntity]],
                             embedding_type: str = "full_code",
                             top_k: int = 5,
                             threshold: float = 0.7) -> List[Tuple[CodeEntity, float]]:
        """
        Find entities similar to the query entity.
        
        Args:
            query_entity: Entity to find similar entities for
            entities: Dictionary of entities to search in
            embedding_type: Type of embedding to use for comparison
                           (options: full_code, name_signature, docstring)
            top_k: Number of similar entities to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples containing (entity, similarity score)
        """
        # Generate embeddings for the query entity
        query_embeddings = self.generate_embeddings_for_entity(query_entity)
        
        # Get the appropriate embedding vector based on the type
        if embedding_type == "name_signature":
            query_vector = query_embeddings.name_signature_embedding
        elif embedding_type == "docstring":
            query_vector = query_embeddings.docstring_embedding
        else:  # Default to full_code
            query_vector = query_embeddings.full_code_embedding
        
        # If no embedding is available for the requested type, return empty list
        if query_vector is None:
            return []
        
        # Prepare a flat list of all entities
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        # Calculate similarity scores
        similarities = []
        for entity in all_entities:
            # Skip self-comparison
            if entity.name == query_entity.name and entity.lineno == query_entity.lineno and entity.source_file == query_entity.source_file:
                continue
            
            # Generate embeddings for this entity
            entity_embeddings = self.generate_embeddings_for_entity(entity)
            
            # Get the appropriate embedding vector based on the type
            if embedding_type == "name_signature":
                entity_vector = entity_embeddings.name_signature_embedding
            elif embedding_type == "docstring":
                entity_vector = entity_embeddings.docstring_embedding
            else:  # Default to full_code
                entity_vector = entity_embeddings.full_code_embedding
            
            # Skip if no embedding is available for this type
            if entity_vector is None:
                continue
            
            # Calculate cosine similarity
            similarity = self.cosine_similarity(query_vector, entity_vector)
            
            # Add to results if above threshold
            if similarity >= threshold:
                similarities.append((entity, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        dot_product = np.dot(v1, v2)
        return dot_product / (norm1 * norm2)
    
    def save_embeddings(self, output_dir: Optional[str] = None) -> None:
        """
        Save all embeddings to files.
        
        Args:
            output_dir: Directory to save embeddings (uses cache_dir if None)
        """
        save_dir = output_dir if output_dir else self.cache_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create an index of all embeddings
        index = {}
        
        # Save each embedding
        for entity_id, embeddings in self.embeddings_cache.items():
            # Create a filename from the entity ID
            entity_hash = hashlib.md5(entity_id.encode()).hexdigest()
            file_name = f"{entity_hash}.json"
            file_path = os.path.join(save_dir, file_name)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(embeddings.to_dict(), f)
            
            # Add to index
            index[entity_id] = file_name
        
        # Save the index
        index_path = os.path.join(save_dir, "index.json")
        with open(index_path, 'w') as f:
            json.dump(index, f)
    
    def load_embeddings(self, input_dir: str) -> None:
        """
        Load embeddings from files.
        
        Args:
            input_dir: Directory containing embedding files
        """
        if not os.path.exists(input_dir):
            print(f"Embeddings directory does not exist: {input_dir}")
            return
        
        # Load the index file
        index_path = os.path.join(input_dir, "index.json")
        if not os.path.exists(index_path):
            print(f"Embeddings index file not found: {index_path}")
            return
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        # Load each embedding file
        for entity_id, file_name in index.items():
            file_path = os.path.join(input_dir, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.embeddings_cache[entity_id] = EntityEmbeddings.from_dict(data)
        
        self.loaded_cache = True


# Utility function to use in other modules
def generate_entity_embeddings(entities: Dict[str, List[CodeEntity]], 
                              cache_dir: Optional[str] = None,
                              context_lines: int = 5) -> Dict[str, EntityEmbeddings]:
    """
    Generate embeddings for code entities.
    
    Args:
        entities: Dictionary of entities by type
        cache_dir: Directory to cache embeddings
        context_lines: Number of context lines to consider
        
    Returns:
        Dictionary mapping entity IDs to EntityEmbeddings objects
    """
    # Set the embedding model to text-embedding-ada-002
    set_embedding_model("text-embedding-ada-002")
    
    # Create the embedding generator
    generator = EntityEmbeddingGenerator(cache_dir)
    
    # Generate embeddings for all entities
    embeddings = generator.generate_embeddings_for_entities(
        entities, context_lines=context_lines
    )
    
    # Save embeddings to cache
    generator.save_embeddings()
    
    return embeddings 