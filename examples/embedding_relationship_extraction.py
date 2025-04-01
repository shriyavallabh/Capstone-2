#!/usr/bin/env python3
"""
Example script demonstrating how to use entity embeddings for relationship extraction.

This script shows how to:
1. Use OpenAI's text-embedding-ada-002 model to embed code entities
2. Create separate embeddings for different aspects of code entities
3. Use these embeddings to find semantically similar entities
4. Visualize the relationships between entities
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TalkToCode modules
from talktocode.utils.config import set_embedding_model
from talktocode.indexing.entity_extractor import extract_entities_from_directory
from talktocode.indexing.relationship_extractor import extract_all_relationships
from talktocode.indexing.graph_builder import CodeKnowledgeGraph
from talktocode.indexing.entity_embeddings import EntityEmbeddingGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract relationships from code entities using embeddings")
    parser.add_argument("--code_dir", type=str, required=True, help="Directory containing Python code to analyze")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store output files")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for relationship extraction")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--similarity_threshold", type=float, default=0.75, help="Minimum similarity threshold (0-1)")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002", 
                        help="Embedding model to use (default: text-embedding-ada-002)")
    return parser.parse_args()


def process_directory(code_dir: str, args):
    """Process a directory of Python files and extract relationships."""
    print(f"Processing directory: {code_dir}")
    
    # Set the embedding model
    print(f"Using embedding model: {args.embedding_model}")
    set_embedding_model(args.embedding_model)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Extract entities from the code
    print("Extracting code entities...")
    all_entities = extract_entities_from_directory(code_dir)
    
    # Print entity counts
    entity_counts = {entity_type: len(entities) for entity_type, entities in all_entities.items()}
    print(f"Extracted entities: {entity_counts}")
    
    # Extract relationships using semantic embeddings
    print("Extracting relationships...")
    relationships = extract_all_relationships(
        all_entities,
        use_llm=args.use_llm,
        use_embeddings=True,
        cache_dir=str(cache_dir)
    )
    
    # Build knowledge graph
    print("Building knowledge graph...")
    graph = CodeKnowledgeGraph()
    graph.build_graph_from_entities_and_relationships(all_entities, relationships)
    
    # Save graph to output directory
    graph_file = output_dir / "code_graph.json"
    graph.save_graph(str(graph_file))
    print(f"Saved graph to {graph_file}")
    
    # Return the graph for visualization
    return graph, all_entities, relationships


def count_relationship_types(relationships):
    """Count the number of each type of relationship."""
    counts = {}
    for rel in relationships:
        rel_type = rel.relationship_type
        counts[rel_type] = counts.get(rel_type, 0) + 1
    
    return counts


def visualize_graph(graph: CodeKnowledgeGraph, output_file: str, highlight_semantic: bool = True):
    """Visualize the code knowledge graph."""
    # Get the NetworkX graph
    nx_graph = graph.graph
    
    # Create a figure
    plt.figure(figsize=(20, 14))
    
    # Create a dictionary to map entity types to colors
    entity_types = set(nx_graph.nodes[node].get('type', 'Unknown') for node in nx_graph.nodes)
    cmap = get_cmap('viridis', len(entity_types))
    color_map = {t: cmap(i) for i, t in enumerate(entity_types)}
    
    # Create node colors based on entity type
    node_colors = [color_map.get(nx_graph.nodes[node].get('type', 'Unknown'), 'gray') for node in nx_graph.nodes]
    
    # Create edge colors based on relationship type (red for semantic relationships)
    edge_colors = []
    edge_widths = []
    
    for u, v, data in nx_graph.edges(data=True):
        if data.get('type') == 'SEMANTICALLY_SIMILAR' and highlight_semantic:
            edge_colors.append('red')
            edge_widths.append(2.0)
        else:
            edge_colors.append('black')
            edge_widths.append(0.5)
    
    # Create a layout for the graph
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # Draw the graph
    nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, alpha=0.8, node_size=100)
    nx.draw_networkx_edges(nx_graph, pos, edge_color=edge_colors, width=edge_widths, alpha=0.5)
    
    # Add node labels for larger graphs (limited to maintain readability)
    if len(nx_graph) < 100:
        node_labels = {n: nx_graph.nodes[n].get('name', n) for n in nx_graph.nodes}
        nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=8)
    
    # Add a title
    plt.title(f"Code Knowledge Graph ({len(nx_graph.nodes)} entities, {len(nx_graph.edges)} relationships)")
    
    # Add a legend for entity types
    legend_elements = []
    for entity_type, color in color_map.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=entity_type))
    
    # Add semantic relationship to legend if highlighted
    if highlight_semantic:
        legend_elements.append(plt.Line2D([0], [0], color='red', lw=2, label='Semantic Similarity'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Remove axis
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved graph visualization to {output_file}")
    plt.close()


def explore_semantic_relationships(relationships, all_entities, output_dir):
    """Explore and analyze semantic relationships in the extracted data."""
    # Filter semantic relationships
    semantic_rels = [r for r in relationships if r.relationship_type == "SEMANTICALLY_SIMILAR"]
    
    # Create a dict to track which embedding types led to the relationships
    embedding_type_counts = {"full_code": 0, "name_signature": 0, "docstring": 0}
    
    for rel in semantic_rels:
        desc = rel.description.lower()
        if "implementation" in desc:
            embedding_type_counts["full_code"] += 1
        elif "signature" in desc:
            embedding_type_counts["name_signature"] += 1
        elif "documentation" in desc:
            embedding_type_counts["docstring"] += 1
    
    # Print statistics
    print("\nSemantic Relationship Analysis:")
    print(f"Total semantic relationships: {len(semantic_rels)}")
    print("Breakdown by embedding type:")
    for emb_type, count in embedding_type_counts.items():
        percentage = (count / len(semantic_rels) * 100) if semantic_rels else 0
        print(f"  - {emb_type}: {count} ({percentage:.1f}%)")
    
    # Create a visualization of embedding type distribution
    plt.figure(figsize=(10, 6))
    plt.bar(embedding_type_counts.keys(), embedding_type_counts.values())
    plt.title("Semantic Relationships by Embedding Type")
    plt.xlabel("Embedding Type")
    plt.ylabel("Number of Relationships")
    
    # Save the figure
    chart_file = os.path.join(output_dir, "semantic_relationship_types.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Saved semantic relationship analysis to {chart_file}")
    plt.close()
    
    # Analyze relationship strengths
    strengths = [r.strength for r in semantic_rels]
    
    plt.figure(figsize=(10, 6))
    plt.hist(strengths, bins=10, range=(1, 11))
    plt.title("Semantic Relationship Strength Distribution")
    plt.xlabel("Relationship Strength (1-10)")
    plt.ylabel("Number of Relationships")
    
    # Save the figure
    strength_file = os.path.join(output_dir, "semantic_relationship_strengths.png")
    plt.savefig(strength_file, dpi=300, bbox_inches='tight')
    print(f"Saved strength distribution to {strength_file}")
    plt.close()
    
    # Export a sample of semantic relationships to a JSON file
    sample_size = min(20, len(semantic_rels))
    sample_rels = semantic_rels[:sample_size]
    
    sample_data = []
    for rel in sample_rels:
        sample_data.append({
            "source": f"{rel.source.name} ({rel.source.__class__.__name__})",
            "target": f"{rel.target.name} ({rel.target.__class__.__name__})",
            "strength": rel.strength,
            "description": rel.description
        })
    
    sample_file = os.path.join(output_dir, "semantic_relationship_samples.json")
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Saved sample relationships to {sample_file}")


def main():
    """Main function."""
    args = parse_args()
    
    # Process the directory
    start_time = time.time()
    graph, all_entities, relationships = process_directory(args.code_dir, args)
    end_time = time.time()
    
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    
    # Print relationship statistics
    rel_counts = count_relationship_types(relationships)
    print("\nRelationship Types:")
    for rel_type, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {rel_type}: {count}")
    
    # Visualize the graph
    vis_file = os.path.join(args.output_dir, "knowledge_graph.png")
    visualize_graph(graph, vis_file)
    
    # Create visualization highlighting only semantic relationships
    semantic_vis_file = os.path.join(args.output_dir, "semantic_relationships.png")
    visualize_graph(graph, semantic_vis_file, highlight_semantic=True)
    
    # Explore semantic relationships
    explore_semantic_relationships(relationships, all_entities, args.output_dir)
    
    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main() 