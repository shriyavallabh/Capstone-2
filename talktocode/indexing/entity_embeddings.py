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
from talktocode.indexing.faiss_manager import FaissIndexManager

# Configure the embedding model
# Set to text-embedding-ada-002 as required
set_embedding_model("text-embedding-ada-002")

# Initialize OpenAI client - but do it lazily to avoid initialization issues
# We'll initialize it when needed in the functions instead
client = None

def get_client():
    """Get or initialize the OpenAI client."""
    global client
    if client is None:
        try:
            # Import the shared client function from the main app
            from app import get_openai_client
            client = get_openai_client()
        except ImportError:
            # Fallback if the import fails
            # Initialize without extra parameters that might cause issues
            client = OpenAI(api_key=OPENAI_API_KEY)
    return client


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
        self.last_updated = time.time()
        
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
    
    def __init__(self, faiss_manager: Optional[FaissIndexManager] = None):
        """
        Initialize the embedding generator.
        
        Args:
            faiss_manager: Optional FaissIndexManager instance
        """
        # Initialize FAISS manager
        self.faiss_enabled = MODEL_CONFIG["embedding"].get("faiss", {}).get("enabled", False)
        self.faiss_manager = faiss_manager
        if self.faiss_enabled and self.faiss_manager is None:
            # Attempt to initialize if not provided (e.g., for standalone use)
            print("Warning: FAISS enabled but no manager provided to EntityEmbeddingGenerator. Initializing default.")
            faiss_config = MODEL_CONFIG["embedding"]["faiss"]
            embed_config = MODEL_CONFIG["embedding"]
            self.faiss_manager = FaissIndexManager(
                index_directory=faiss_config["index_directory"],
                dimensions=embed_config["dimensions"],
                index_factory_string=faiss_config["index_factory_string"],
                normalize_vectors=faiss_config.get("normalize_vectors", False)
            )
        elif not self.faiss_enabled:
            self.faiss_manager = None # Ensure it's None if not enabled
        
        self._client = None # Lazy init OpenAI client
        self.embedding_model = MODEL_CONFIG["embedding"]["model"]
        self.dimensions = MODEL_CONFIG["embedding"]["dimensions"]
        
    def get_client(self):
        """Lazy loads OpenAI client."""
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a text string using OpenAI's API.
        
        Args:
            text: Text to embed
            
        Returns:
            NumPy array containing the embedding vector
        """
        if not text or not isinstance(text, str):
            return None
        try:
            response = self.get_client().embeddings.create(
                model=self.embedding_model,
                input=[text] # API expects a list
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embedding for text snippet: {e}")
            # Return zero vector on error? Or handle upstream?
            return np.zeros(self.dimensions, dtype=np.float32)
    
    def get_entity_id(self, entity: CodeEntity) -> str:
        """
        Generate a unique identifier for a code entity.
        
        Args:
            entity: Code entity
            
        Returns:
            String containing the entity ID
        """
        return f"{entity.source_file}:{entity.__class__.__name__}:{entity.name}:{entity.lineno}"
    
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
    
    def extract_docstring_and_comments(self, entity: CodeEntity, context_lines: int = 0) -> str:
        """
        Extract the docstring and comments for a code entity.
        
        Args:
            entity: Code entity
            context_lines: Number of context lines to include
            
        Returns:
            String containing the docstring and comments
        """
        docstring = entity.docstring if hasattr(entity, 'docstring') else ""
        
        # Get comments from the code
        code_with_context = extract_code_with_context(entity, context_lines)
        
        # Simple comment extraction (can be improved)
        comments = []
        for line in code_with_context.split('\n'):
            line = line.strip()
            if line.startswith('#') or line.startswith('//'):
                comments.append(line)
                
        # Combine docstring and comments
        result = []
        if docstring:
            result.append(f"Docstring: {docstring}")
        if comments:
            result.append("Comments: " + " ".join(comments))
            
        return "\n".join(result)
    
    def generate_embeddings_for_entity(self, entity: CodeEntity, context_lines: int) -> Optional[EntityEmbeddings]:
        """
        Generate all embeddings for a code entity.
        
        Args:
            entity: Code entity
            context_lines: Number of context lines to consider
            
        Returns:
            EntityEmbeddings object with all embeddings
        """
        entity_id = self.get_entity_id(entity)
        entity_embeddings = EntityEmbeddings(entity_id)
        added_to_faiss = False
        
        # Generate Full Code Embedding
        code_text = extract_code_with_context(entity, context_lines)
        embedding = self.get_embedding(code_text)
        if embedding is not None:
            entity_embeddings.full_code_embedding = embedding
            if self.faiss_enabled and self.faiss_manager:
                self.faiss_manager.add_embeddings(
                    index_name="full_code",
                    ids=[entity_id],
                    embeddings=embedding.reshape(1, -1)
                )
                added_to_faiss = True
        
        # Generate Name/Signature Embedding
        signature_text = self.extract_name_and_signature(entity)
        embedding = self.get_embedding(signature_text)
        if embedding is not None:
            entity_embeddings.name_signature_embedding = embedding
            if self.faiss_enabled and self.faiss_manager:
                self.faiss_manager.add_embeddings(
                    index_name="name_signature",
                    ids=[entity_id],
                    embeddings=embedding.reshape(1, -1)
                )
                added_to_faiss = True
        
        # Generate Docstring/Comment Embedding
        doc_text = self.extract_docstring_and_comments(entity, context_lines=0)
        if doc_text:
            embedding = self.get_embedding(doc_text)
            if embedding is not None:
                entity_embeddings.docstring_embedding = embedding
                if self.faiss_enabled and self.faiss_manager:
                    self.faiss_manager.add_embeddings(
                        index_name="docstring",
                        ids=[entity_id],
                        embeddings=embedding.reshape(1, -1)
                    )
                    added_to_faiss = True
        
        return entity_embeddings if added_to_faiss else None
    
    def generate_embeddings_for_entities(self, entities_input, context_lines: int = 5):
        """
        Generate embeddings for multiple entities.
        
        Args:
            entities_input: Either a dictionary of entity lists or a flat list of entities
            context_lines: Number of context lines to include
            
        Returns:
            Tuple containing (list of entity_ids, list of embeddings)
        """
        # Determine the input type and process accordingly
        if isinstance(entities_input, dict):
            # Dictionary of entity lists (by type)
            all_entities = []
            for entity_list in entities_input.values():
                all_entities.extend(entity_list)
        elif isinstance(entities_input, list):
            # Already a flat list of entities
            all_entities = entities_input
        else:
            raise ValueError(f"Invalid input type: {type(entities_input)}. Expected dict or list.")
        
        entity_ids = []
        embeddings = []
        
        # Process all entities with progress bar
        for i, entity in enumerate(tqdm(all_entities, desc="Generating Embeddings")):
            # Get the entity ID
            entity_id = self.get_entity_id(entity)
            
            # Generate embeddings
            entity_embeddings = self.generate_embeddings_for_entity(entity, context_lines)
            
            if entity_embeddings is None or entity_embeddings.full_code_embedding is None:
                # Skip if embeddings couldn't be generated
                continue
                
            # Store entity ID and embedding
            entity_ids.append(entity_id)
            embeddings.append(entity_embeddings.full_code_embedding)
            
            # Add to FAISS index if enabled
            if self.faiss_enabled and self.faiss_manager is not None:
                try:
                    # Add to the full_code index
                    self.faiss_manager.add_embeddings(
                        index_name="full_code",
                        ids=[entity_id],
                        embeddings=entity_embeddings.full_code_embedding.reshape(1, -1)
                    )
                    print(f"Added 1 vectors to index 'full_code'. Total vectors: {self.faiss_manager.get_index_size('full_code')}")
                    
                    # Add to the name_signature index if available
                    if entity_embeddings.name_signature_embedding is not None:
                        self.faiss_manager.add_embeddings(
                            index_name="name_signature",
                            ids=[entity_id],
                            embeddings=entity_embeddings.name_signature_embedding.reshape(1, -1)
                        )
                        print(f"Added 1 vectors to index 'name_signature'. Total vectors: {self.faiss_manager.get_index_size('name_signature')}")
                    
                    # Add to the docstring index if available
                    if entity_embeddings.docstring_embedding is not None:
                        self.faiss_manager.add_embeddings(
                            index_name="docstring",
                            ids=[entity_id],
                            embeddings=entity_embeddings.docstring_embedding.reshape(1, -1)
                        )
                        print(f"Added 1 vectors to index 'docstring'. Total vectors: {self.faiss_manager.get_index_size('docstring')}")
                except Exception as e:
                    print(f"Error adding entity {entity_id} to FAISS index: {e}")
        
        # Save FAISS indices if enabled
        if self.faiss_enabled and self.faiss_manager is not None:
            try:
                self.faiss_manager.save_all_indices()
            except Exception as e:
                print(f"Error saving FAISS indices: {e}")
        
        return entity_ids, np.array(embeddings)
    
    def find_similar_entities(self,
                              query_text: str, 
                              embedding_type: str = 'full_code', 
                              top_k: int = 10, 
                              min_similarity: Optional[float] = None # Similarity check done by caller now
                             ) -> List[Tuple[str, float]]:
        """
        Find entities similar to the query entity.
        
        Args:
            query_text: Entity to find similar entities for
            embedding_type: Type of embedding to use for comparison
            top_k: Number of similar entities to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of tuples containing (entity_id, similarity_score)
        """
        if not self.faiss_enabled or not self.faiss_manager:
            print("FAISS is not enabled or manager not available for similarity search.")
            return []

        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            print("Could not generate embedding for query text.")
            return []

        # Perform search using FaissManager
        # The FaissManager.search method now returns sorted (id, similarity) tuples
        similar_ids_scores = self.faiss_manager.search(
            index_name=embedding_type, # Use the specific index
            query_vector=query_embedding,
            top_k=top_k
        )

        # Optional: Filter further by min_similarity if needed, 
        # though thresholding might be better handled by the caller (SearchEngine)
        if min_similarity is not None:
             similar_ids_scores = [(id, score) for id, score in similar_ids_scores if score >= min_similarity]
             
        return similar_ids_scores


# Utility function to use in other modules
def generate_entity_embeddings(entities: Dict[str, List[CodeEntity]], 
                              faiss_manager: Optional[FaissIndexManager] = None,
                              context_lines: int = 5) -> None:
    """
    Generate embeddings for code entities.
    
    Args:
        entities: Dictionary of entities by type
        faiss_manager: Optional FaissIndexManager instance
        context_lines: Number of context lines to consider
    """
    # Set the embedding model to text-embedding-ada-002
    set_embedding_model("text-embedding-ada-002")
    
    # Create the embedding generator
    generator = EntityEmbeddingGenerator(faiss_manager)
    
    # Generate embeddings for all entities
    generator.generate_embeddings_for_entities(entities, context_lines)
    
    # Save embeddings to FAISS
    generator.faiss_manager.save_all_indices()
    
    print("Finished generating and indexing embeddings.")

# Cosine similarity function is removed as FAISS handles similarity search
# def cosine_similarity(v1, v2):
#    ...

# Cosine similarity function is removed as FAISS handles similarity search
# def cosine_similarity(v1, v2):
#    ... 