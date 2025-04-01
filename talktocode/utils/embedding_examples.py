"""
Examples for working with embeddings using the updated configuration.

This file demonstrates how to use the embedding configuration
options added to the config.py file.
"""

import os
import numpy as np
from typing import List, Dict, Any
import time

from talktocode.utils.config import (
    MODEL_CONFIG, 
    EMBEDDING_MODELS,
    get_embedding_model_info,
    set_embedding_model
)
from openai import OpenAI

def get_current_embedding_config():
    """Get the current embedding configuration."""
    
    # Access the current embedding model
    current_model = MODEL_CONFIG["models"]["embedding"]
    print(f"Current embedding model: {current_model}")
    
    # Get model-specific information
    model_info = get_embedding_model_info()
    print(f"Dimensions: {model_info['dimensions']}")
    print(f"Max batch size: {model_info['max_batch_size']} tokens")
    print(f"Max batch items: {model_info['max_batch_items']} items")
    print(f"Cost per 1K tokens: ${model_info['cost_per_1k_tokens']}")
    
    # Show caching configuration
    cache_config = MODEL_CONFIG["embedding"]["cache"]
    print(f"\nCaching enabled: {cache_config['enabled']}")
    print(f"Cache directory: {cache_config['directory']}")
    print(f"Cache expiration: {cache_config['expiration_days']} days")
    
    # Show batching configuration
    batch_config = MODEL_CONFIG["embedding"]["batch"]
    print(f"\nBatching enabled: {batch_config['enabled']}")
    print(f"Batch size: {batch_config['size']} items")
    print(f"Batch timeout: {batch_config['timeout_ms']} ms")

def switch_embedding_model(model_name: str):
    """
    Switch to a different embedding model.
    
    Args:
        model_name: Name of the model to switch to
    """
    # Get info on all available models
    print("Available embedding models:")
    for name, info in EMBEDDING_MODELS.items():
        print(f"- {name}: {info['description']} ({info['dimensions']} dims)")
    
    # Try to switch to the specified model
    success = set_embedding_model(model_name)
    
    if success:
        print(f"\nSuccessfully switched to {model_name}")
        # Show the updated configuration
        get_current_embedding_config()
    else:
        print(f"\nFailed to switch to {model_name}. Model not found.")
        print(f"Available models: {', '.join(EMBEDDING_MODELS.keys())}")

def batch_embedding_example(texts: List[str]):
    """
    Demonstrate batched embedding processing.
    
    Args:
        texts: List of texts to embed
    """
    client = OpenAI()
    
    # Get current configuration
    model_name = MODEL_CONFIG["models"]["embedding"]
    batch_size = MODEL_CONFIG["embedding"]["batch"]["size"]
    
    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size}")
    
    # Process in batches
    embeddings = []
    total_tokens = 0
    
    # Process multiple batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_size_actual = len(batch)
        
        print(f"Processing batch {i//batch_size + 1}: {batch_size_actual} items")
        
        start_time = time.time()
        response = client.embeddings.create(
            model=model_name,
            input=batch
        )
        end_time = time.time()
        
        # Extract embeddings and append to results
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        
        # Update token count
        total_tokens += response.usage.total_tokens
        
        print(f"Batch processed in {end_time - start_time:.2f} seconds")
        print(f"Tokens in this batch: {response.usage.total_tokens}")
        
        # Sleep to avoid rate limits
        if i + batch_size < len(texts):
            time.sleep(0.5)
    
    # Calculate costs
    cost_per_1k = EMBEDDING_MODELS[model_name]["cost_per_1k_tokens"]
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    
    print(f"\nProcessed {len(texts)} texts in {len(embeddings)} embeddings")
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated cost: ${estimated_cost:.6f}")
    print(f"Average tokens per text: {total_tokens / len(texts):.1f}")
    
    # Show dimensions of first embedding
    if embeddings:
        dims = len(embeddings[0])
        print(f"Embedding dimensions: {dims}")
        
    return embeddings

def cost_comparison_example():
    """Compare costs between different embedding models."""
    
    # Sample text for comparison
    sample_text = """
    This is a sample of code documentation that would be processed by our 
    embedding models. It describes a function that calculates the Fibonacci 
    sequence using dynamic programming for efficiency.
    
    The function takes an integer n as input and returns the nth Fibonacci number.
    It uses a bottom-up approach to avoid the exponential time complexity of a 
    naive recursive implementation.
    
    Time complexity: O(n)
    Space complexity: O(n)
    """
    
    # Create 10 variations for a batch
    sample_texts = [f"{sample_text} Example {i+1}." for i in range(10)]
    
    # Compare models
    print("Cost comparison for embedding models:\n")
    print("Model                   | Dimensions | Cost per 1K tokens | Relative Cost")
    print("------------------------|------------|--------------------|-------------")
    
    for model_name, info in EMBEDDING_MODELS.items():
        dims = info["dimensions"]
        cost = info["cost_per_1k_tokens"]
        
        # Calculate relative cost (using ada-002 as base)
        base_cost = EMBEDDING_MODELS["text-embedding-ada-002"]["cost_per_1k_tokens"]
        relative = cost / base_cost
        
        print(f"{model_name.ljust(24)} | {dims:<10} | ${cost:<18.5f} | {relative:<13.2f}x")
    
    # Calculate estimated costs for a large codebase
    print("\nEstimated costs for processing a codebase:")
    
    # Assumptions
    num_files = 1000
    avg_tokens_per_file = 500
    total_tokens = num_files * avg_tokens_per_file
    
    print(f"Assumptions: {num_files} files, {avg_tokens_per_file} tokens per file")
    print(f"Total tokens: {total_tokens}")
    
    print("\nModel                   | Total Cost")
    print("-----------------------|------------")
    
    for model_name, info in EMBEDDING_MODELS.items():
        cost = info["cost_per_1k_tokens"]
        total_cost = (total_tokens / 1000) * cost
        
        print(f"{model_name.ljust(23)} | ${total_cost:.2f}")

if __name__ == "__main__":
    print("Current Embedding Configuration:")
    print("--------------------------------")
    get_current_embedding_config()
    
    print("\nSwitching Models Example:")
    print("-------------------------")
    # Try switching to ada-002
    switch_embedding_model("text-embedding-ada-002")
    
    print("\nCost Comparison:")
    print("---------------")
    cost_comparison_example()
    
    # Sample texts for embedding
    sample_texts = [
        "The function calculates the Fibonacci sequence efficiently.",
        "Graph-based code analysis helps understand complex codebases.",
        "Embedding models transform text into high-dimensional vectors.",
        "Dynamic programming optimizes recursive algorithms.",
        "Python decorators add functionality to existing functions."
    ]
    
    print("\nBatch Embedding Example:")
    print("-----------------------")
    batch_embedding_example(sample_texts) 