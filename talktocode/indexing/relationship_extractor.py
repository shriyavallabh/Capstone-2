import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Set
import openai
from collections import defaultdict
import numpy as np

# Import local modules
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import OPENAI_API_KEY, MODEL_CONFIG
from talktocode.indexing.entity_extractor import CodeEntity, FunctionEntity, ClassEntity, VariableEntity, ImportEntity, extract_code_with_context
from talktocode.indexing.entity_embeddings import generate_entity_embeddings, EntityEmbeddingGenerator

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Relationship types
RELATIONSHIP_TYPES = [
    "CALLS",           # Function calls another function
    "IMPORTS",         # Module imports another module
    "INHERITS_FROM",   # Class inherits from another class
    "CONTAINS",        # Class contains a method or property
    "USES",            # Function uses a variable
    "DEFINES",         # Module/class defines a function
    "OVERRIDES",       # Method overrides parent method
    "REFERENCES",      # Any entity references another
    "DEPENDS_ON",      # Entity depends on another entity
    "SEMANTICALLY_SIMILAR" # New: Entity is semantically similar to another entity
]

class Relationship:
    """Represents a relationship between two code entities."""
    
    def __init__(self, source: CodeEntity, target: CodeEntity, 
                 relationship_type: str, strength: int, description: str):
        self.source = source
        self.target = target
        self.relationship_type = relationship_type
        self.strength = strength  # 1-10
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "source": {
                "name": self.source.name,
                "type": self.source.__class__.__name__,
                "source_file": self.source.source_file,
                "lineno": self.source.lineno
            },
            "target": {
                "name": self.target.name,
                "type": self.target.__class__.__name__,
                "source_file": self.target.source_file,
                "lineno": self.target.lineno
            },
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "description": self.description
        }
    
    def __str__(self) -> str:
        return f"{self.source.name} --[{self.relationship_type}({self.strength})]-> {self.target.name}"


def create_prompt_for_entity_pair(entity1: CodeEntity, entity2: CodeEntity, 
                                 context1: str, context2: str) -> str:
    """
    Create a prompt for the LLM to analyze the relationship between two code entities.
    
    Args:
        entity1: First code entity
        entity2: Second code entity
        context1: Code context for the first entity
        context2: Code context for the second entity
    
    Returns:
        String containing the prompt for the LLM
    """
    entity1_type = entity1.__class__.__name__.replace("Entity", "")
    entity2_type = entity2.__class__.__name__.replace("Entity", "")
    
    prompt = f"""As a code analysis expert, examine the relationship between these two code entities:

ENTITY 1 ({entity1_type}): {entity1.name}
Located in: {entity1.source_file}, Line {entity1.lineno}
{"-" * 50}
{context1}
{"-" * 50}

ENTITY 2 ({entity2_type}): {entity2.name}
Located in: {entity2.source_file}, Line {entity2.lineno}
{"-" * 50}
{context2}
{"-" * 50}

Analyze the potential relationship between these two entities based on the code context.
Possible relationship types include: {", ".join(RELATIONSHIP_TYPES)}

Your task:
1. Determine if there's a relationship between ENTITY 1 and ENTITY 2
2. If a relationship exists, identify:
   - The relationship type (from the list above)
   - The strength of the relationship (score 1-10, where 10 is strongest)
   - A brief description of how they're related (1-2 sentences)
3. If no relationship exists, reply with "NO_RELATIONSHIP"

Output format (JSON):
{{
  "relationship_type": "<TYPE>", 
  "strength": <1-10>,
  "description": "<brief description of the relationship>"
}}

Or if no relationship:
{{
  "relationship_type": "NO_RELATIONSHIP",
  "strength": 0,
  "description": "These entities are not related."
}}
"""
    return prompt


def analyze_entity_pair(entity1: CodeEntity, entity2: CodeEntity, 
                      context_lines: int = 5) -> Optional[Relationship]:
    """
    Use the LLM to analyze the relationship between two code entities.
    
    Args:
        entity1: First code entity
        entity2: Second code entity
        context_lines: Number of context lines to include
    
    Returns:
        Relationship object or None if no relationship exists
    """
    # Get code context for both entities
    context1 = extract_code_with_context(entity1, context_lines)
    context2 = extract_code_with_context(entity2, context_lines)
    
    # Create prompt
    prompt = create_prompt_for_entity_pair(entity1, entity2, context1, context2)
    
    # Call OpenAI API
    try:
        chat_model = MODEL_CONFIG["models"]["chat"]
        response = openai.ChatCompletion.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a code analysis assistant that identifies relationships between code entities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        # Extract and parse response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                result = json.loads(content)
        except json.JSONDecodeError:
            print(f"Error parsing response as JSON: {content}")
            return None
        
        # Create relationship if one was found
        if result["relationship_type"] != "NO_RELATIONSHIP":
            return Relationship(
                source=entity1,
                target=entity2,
                relationship_type=result["relationship_type"],
                strength=result["strength"],
                description=result["description"]
            )
        return None
        
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        time.sleep(5)  # Back off on rate limit errors
        return None


def create_entity_batches(entities: Dict[str, List[CodeEntity]], 
                         batch_size: int = 10) -> List[List[Tuple[CodeEntity, CodeEntity]]]:
    """
    Create batches of entity pairs for efficient processing.
    
    Args:
        entities: Dictionary of entities by type
        batch_size: Number of entity pairs in each batch
    
    Returns:
        List of batches, each containing entity pairs
    """
    all_entities = []
    for entity_list in entities.values():
        all_entities.extend(entity_list)
    
    # Create all potential pairs while avoiding self-pairs
    pairs = [(e1, e2) for i, e1 in enumerate(all_entities) 
             for e2 in all_entities[i+1:]]
    
    # Create batches
    batches = []
    for i in range(0, len(pairs), batch_size):
        batches.append(pairs[i:i + batch_size])
    
    return batches


def process_entity_batch(batch: List[Tuple[CodeEntity, CodeEntity]], 
                        context_lines: int = 5) -> List[Relationship]:
    """
    Process a batch of entity pairs to find relationships.
    
    Args:
        batch: List of entity pairs to process
        context_lines: Number of context lines to include
    
    Returns:
        List of relationships found in the batch
    """
    relationships = []
    
    for entity1, entity2 in batch:
        relationship = analyze_entity_pair(entity1, entity2, context_lines)
        if relationship:
            relationships.append(relationship)
    
    return relationships


def extract_relationships_from_entities(entities: Dict[str, List[CodeEntity]], 
                                      max_pairs: Optional[int] = None,
                                      batch_size: int = 10,
                                      context_lines: int = 5) -> List[Relationship]:
    """
    Extract relationships between code entities using the LLM.
    
    Args:
        entities: Dictionary of entities by type
        max_pairs: Maximum number of entity pairs to process (for limiting API calls)
        batch_size: Number of entity pairs in each batch
        context_lines: Number of context lines to include
    
    Returns:
        List of relationships between entities
    """
    # Create batches
    batches = create_entity_batches(entities, batch_size)
    
    # Limit the number of pairs if specified
    if max_pairs:
        total_pairs = sum(len(batch) for batch in batches)
        if total_pairs > max_pairs:
            # Trim batches to not exceed max_pairs
            pairs_so_far = 0
            for i, batch in enumerate(batches):
                if pairs_so_far + len(batch) > max_pairs:
                    # Trim this batch and remove the rest
                    batches[i] = batch[:max_pairs - pairs_so_far]
                    batches = batches[:i+1]
                    break
                pairs_so_far += len(batch)
    
    # Process batches
    all_relationships = []
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} entity pairs)...")
        relationships = process_entity_batch(batch, context_lines)
        all_relationships.extend(relationships)
        time.sleep(1)  # Avoid rate limiting
    
    return all_relationships


def extract_implicit_relationships(entities: Dict[str, List[CodeEntity]]) -> List[Relationship]:
    """
    Extract relationships that can be determined without using the LLM.
    
    Args:
        entities: Dictionary of entities by type
    
    Returns:
        List of implicit relationships
    """
    relationships = []
    
    # Map entities by name and file for faster lookup
    entity_map = {}
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            key = (entity.name, entity.source_file)
            entity_map[key] = entity
    
    # Process functions to find CALLS relationships
    for function in entities.get("functions", []):
        for called_func_name in function.called_functions:
            # Try to find the called function
            for target_func in entities.get("functions", []):
                if target_func.name == called_func_name:
                    relationship = Relationship(
                        source=function,
                        target=target_func,
                        relationship_type="CALLS",
                        strength=8,  # Direct call is strong
                        description=f"{function.name} calls {target_func.name}"
                    )
                    relationships.append(relationship)
    
    # Process classes to find INHERITS_FROM relationships
    for class_entity in entities.get("classes", []):
        for base_class_name in class_entity.base_classes:
            # Try to find the base class
            for target_class in entities.get("classes", []):
                if target_class.name == base_class_name:
                    relationship = Relationship(
                        source=class_entity,
                        target=target_class,
                        relationship_type="INHERITS_FROM",
                        strength=10,  # Inheritance is very strong
                        description=f"{class_entity.name} inherits from {target_class.name}"
                    )
                    relationships.append(relationship)
    
    # Process imports to find IMPORTS relationships
    for import_entity in entities.get("imports", []):
        module_name = import_entity.name
        # Look for other entities that might be imported
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity.name == module_name or (
                    hasattr(entity, 'import_from') and 
                    entity.import_from == import_entity.import_from
                ):
                    relationship = Relationship(
                        source=import_entity,
                        target=entity,
                        relationship_type="IMPORTS",
                        strength=7,  # Import is medium-strong
                        description=f"Import of {entity.name}"
                    )
                    relationships.append(relationship)
    
    return relationships


def extract_semantic_relationships(entities: Dict[str, List[CodeEntity]], 
                                cache_dir: Optional[str] = None,
                                similarity_threshold: float = 0.75,
                                max_relationships_per_entity: int = 5) -> List[Relationship]:
    """
    Extract semantic relationships between code entities using embeddings.
    
    Args:
        entities: Dictionary of entities by type
        cache_dir: Directory to cache embeddings
        similarity_threshold: Minimum similarity threshold
        max_relationships_per_entity: Maximum number of semantic relationships per entity
    
    Returns:
        List of semantic relationships
    """
    print("Extracting semantic relationships using embeddings...")
    
    # Initialize embedding generator
    embedding_generator = EntityEmbeddingGenerator(cache_dir)
    
    # Create a flat list of all entities
    all_entities = []
    for entity_list in entities.values():
        all_entities.extend(entity_list)
    
    # Store all relationships
    relationships = []
    
    # Track processed pairs to avoid duplicates
    processed_pairs = set()
    
    # Generate embeddings for all entities first to utilize caching
    _ = embedding_generator.generate_embeddings_for_entities(entities)
    
    # Configuration for similarity types
    similarity_types = [
        # (embedding_type, description_prefix, threshold_modifier)
        ("full_code", "Similar implementation context", 0.0),
        ("name_signature", "Similar function/class signature", -0.05),  # Lower threshold slightly
        ("docstring", "Similar documentation", -0.1)  # Lower threshold more for docstrings
    ]
    
    # For each entity, find related entities
    print(f"Finding semantic relationships for {len(all_entities)} entities...")
    for i, entity in enumerate(all_entities):
        if i % 10 == 0:
            print(f"Processed {i}/{len(all_entities)} entities...")
        
        # Try each similarity type
        for embedding_type, description_prefix, threshold_modifier in similarity_types:
            # Adjust threshold based on similarity type
            adjusted_threshold = similarity_threshold + threshold_modifier
            
            # Find similar entities
            similar_entities = embedding_generator.find_similar_entities(
                query_entity=entity,
                entities=entities,
                embedding_type=embedding_type,
                top_k=max_relationships_per_entity,
                threshold=adjusted_threshold
            )
            
            # Create relationships
            for similar_entity, similarity_score in similar_entities:
                # Skip if we've already processed this pair
                pair_id = (
                    f"{entity.name}:{entity.source_file}:{entity.lineno}",
                    f"{similar_entity.name}:{similar_entity.source_file}:{similar_entity.lineno}"
                )
                
                if pair_id in processed_pairs:
                    continue
                
                # Add to processed pairs (in both directions)
                processed_pairs.add(pair_id)
                processed_pairs.add((pair_id[1], pair_id[0]))
                
                # Calculate strength (1-10 scale)
                strength = int(round(similarity_score * 10))
                
                # Description based on embedding type
                description = f"{description_prefix}: {similarity_score:.2f} similarity score"
                
                # Create relationship
                relationship = Relationship(
                    source=entity,
                    target=similar_entity,
                    relationship_type="SEMANTICALLY_SIMILAR",
                    strength=strength,
                    description=description
                )
                
                relationships.append(relationship)
    
    print(f"Found {len(relationships)} semantic relationships")
    return relationships


def extract_all_relationships(entities: Dict[str, List[CodeEntity]], 
                            use_llm: bool = True,
                            use_embeddings: bool = True,
                            max_llm_pairs: Optional[int] = 100,
                            cache_dir: Optional[str] = None) -> List[Relationship]:
    """
    Extract all relationships between code entities.
    
    Args:
        entities: Dictionary of entities by type
        use_llm: Whether to use LLM to analyze relationships
        use_embeddings: Whether to use embeddings for semantic relationships
        max_llm_pairs: Maximum number of entity pairs to analyze with LLM
        cache_dir: Directory to cache embeddings
    
    Returns:
        List of all relationships
    """
    relationships = []
    
    # Extract implicit relationships from code structure
    print("Extracting implicit relationships...")
    implicit_relationships = extract_implicit_relationships(entities)
    relationships.extend(implicit_relationships)
    print(f"Found {len(implicit_relationships)} implicit relationships.")
    
    # Extract semantic relationships using embeddings
    if use_embeddings:
        print("Extracting semantic relationships using embeddings...")
        semantic_relationships = extract_semantic_relationships(
            entities, cache_dir=cache_dir
        )
        relationships.extend(semantic_relationships)
        print(f"Found {len(semantic_relationships)} semantic relationships.")
    
    # Use LLM to extract more complex relationships
    if use_llm:
        print("Extracting relationships using LLM...")
        llm_relationships = extract_relationships_from_entities(
            entities, max_pairs=max_llm_pairs
        )
        relationships.extend(llm_relationships)
        print(f"Found {len(llm_relationships)} LLM-identified relationships.")
    
    # Remove duplicate relationships
    unique_relationships = []
    relationship_pairs = set()
    
    for rel in relationships:
        # Create a tuple to identify the relationship
        rel_key = (
            rel.source.name, rel.source.source_file, rel.source.lineno,
            rel.target.name, rel.target.source_file, rel.target.lineno,
            rel.relationship_type
        )
        
        if rel_key not in relationship_pairs:
            relationship_pairs.add(rel_key)
            unique_relationships.append(rel)
    
    print(f"Total unique relationships: {len(unique_relationships)}")
    return unique_relationships


def export_relationships_csv(relationships: List[Relationship], output_file: str) -> None:
    """
    Export relationships to a CSV file.
    
    Args:
        relationships: List of relationships to export
        output_file: Path to the output CSV file
    """
    with open(output_file, "w") as f:
        # Write header
        f.write("Entity1,Entity2,RelationshipType,Strength,Description\n")
        
        # Write relationships
        for rel in relationships:
            f.write(f"{rel.source.name},{rel.target.name},{rel.relationship_type},{rel.strength},\"{rel.description}\"\n")


def extract_entity_purpose(entity: CodeEntity, context_lines: int = 5) -> str:
    """
    Use the LLM to extract the purpose of a code entity.
    
    Args:
        entity: Code entity to analyze
        context_lines: Number of context lines to include
    
    Returns:
        String describing the entity's purpose
    """
    # Get code context for the entity
    context = extract_code_with_context(entity, context_lines)
    entity_type = entity.__class__.__name__.replace("Entity", "")
    
    # Create prompt
    prompt = f"""As a code analysis expert, examine this {entity_type} and describe its purpose:

{entity_type}: {entity.name}
Located in: {entity.source_file}, Line {entity.lineno}
{"-" * 50}
{context}
{"-" * 50}

Provide a concise description (1-3 sentences) of this {entity_type}'s purpose and functionality.
"""
    
    # Call OpenAI API
    try:
        chat_model = MODEL_CONFIG["models"]["chat"]
        response = openai.ChatCompletion.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a code analysis assistant that identifies the purpose of code entities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Extract response
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        time.sleep(5)  # Back off on rate limit errors
        return f"Purpose not available due to error: {str(e)}"


def enrich_entities_with_purpose(entities: Dict[str, List[CodeEntity]], 
                                max_entities: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    """
    Enrich entities with purpose descriptions using the LLM.
    
    Args:
        entities: Dictionary of entities by type
        max_entities: Maximum number of entities to process
    
    Returns:
        Dictionary mapping entity names to their purposes
    """
    purposes = {}
    
    # Flatten the entity structure
    all_entities = []
    for entity_list in entities.values():
        all_entities.extend(entity_list)
    
    # Limit the number of entities if specified
    if max_entities and len(all_entities) > max_entities:
        all_entities = all_entities[:max_entities]
    
    # Process entities
    for i, entity in enumerate(all_entities):
        print(f"Extracting purpose for entity {i+1}/{len(all_entities)}: {entity.name}...")
        purpose = extract_entity_purpose(entity)
        
        # Add to purposes dictionary
        entity_key = f"{entity.__class__.__name__}:{entity.name}:{entity.source_file}:{entity.lineno}"
        purposes[entity_key] = {
            "name": entity.name,
            "type": entity.__class__.__name__,
            "source_file": entity.source_file,
            "lineno": entity.lineno,
            "purpose": purpose
        }
        
        # Avoid rate limiting
        time.sleep(1)
    
    return purposes 