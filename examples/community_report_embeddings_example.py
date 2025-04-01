#!/usr/bin/env python3
"""
Example script demonstrating the use of hierarchical community report embeddings.

This script shows how to:
1. Generate and store community reports with embeddings at different hierarchical levels
2. Use report embeddings for semantic search across multiple levels
3. Compare the effectiveness of different embedding aspects (title, summary, full report)
4. Visualize relationships between communities using embedding similarity
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TalkToCode modules
from talktocode.utils.config import set_embedding_model
from talktocode.indexing.entity_extractor import extract_entities_from_directory
from talktocode.indexing.relationship_extractor import extract_all_relationships
from talktocode.indexing.graph_builder import CodeKnowledgeGraph
from talktocode.indexing.community_detector import detect_hierarchical_communities
from talktocode.indexing.report_generator import (
    generate_community_reports,
    search_community_reports,
    CommunityReport
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate and explore community reports with embeddings")
    parser.add_argument("--code_dir", type=str, required=True, help="Directory containing Python code to analyze")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store output files")
    parser.add_argument("--num_levels", type=int, default=3, help="Number of hierarchical levels to generate")
    parser.add_argument("--query", type=str, default=None, help="Optional query to search for in reports")
    parser.add_argument("--skip_processing", action="store_true", help="Skip processing and use existing reports")
    parser.add_argument("--visualize_only", action="store_true", help="Only generate visualizations from existing reports")
    return parser.parse_args()


def process_directory(code_dir: str, output_dir: str, num_levels: int) -> Dict[Tuple[int, int], CommunityReport]:
    """Process codebase and generate hierarchical community reports with embeddings."""
    print(f"Processing directory: {code_dir}")
    start_time = time.time()
    
    # Set the embedding model to text-embedding-ada-002
    print("Setting embedding model to text-embedding-ada-002")
    set_embedding_model("text-embedding-ada-002")
    
    # Create output directories
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Extract entities from the code
    print("Extracting code entities...")
    all_entities = extract_entities_from_directory(code_dir)
    entity_counts = {entity_type: len(entities) for entity_type, entities in all_entities.items()}
    print(f"Extracted entities: {entity_counts}")
    
    # Extract relationships
    print("Extracting relationships...")
    relationships = extract_all_relationships(
        all_entities, 
        use_llm=False,  # Faster without LLM for this example
        use_embeddings=True
    )
    print(f"Extracted {len(relationships)} relationships")
    
    # Build knowledge graph
    print("Building knowledge graph...")
    graph = CodeKnowledgeGraph()
    graph.build_graph_from_entities_and_relationships(all_entities, relationships)
    print(f"Graph built with {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges")
    
    # Detect communities at multiple hierarchical levels
    print(f"Detecting communities at {num_levels} hierarchical levels...")
    community_detector = detect_hierarchical_communities(
        graph.graph, 
        num_levels=num_levels
    )
    
    # Print community statistics
    for level_idx in range(num_levels):
        communities = community_detector.get_communities_at_level(level_idx)
        print(f"Level {level_idx+1}: {len(communities)} communities")
    
    # Generate community reports with embeddings
    print("Generating community reports with embeddings...")
    reports = generate_community_reports(
        graph, 
        community_detector,
        output_dir=reports_dir
    )
    print(f"Generated {len(reports)} community reports")
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    return reports


def load_existing_reports(output_dir: str) -> Dict[Tuple[int, int], CommunityReport]:
    """Load existing reports from the output directory."""
    from talktocode.indexing.report_generator import CommunityReportGenerator
    
    reports_dir = os.path.join(output_dir, "reports")
    if not os.path.exists(reports_dir):
        print(f"Reports directory not found: {reports_dir}")
        return {}
    
    # Create a temporary generator to load reports
    generator = CommunityReportGenerator(None, None)
    
    # Load reports
    reports = generator.load_reports(reports_dir)
    print(f"Loaded {len(reports)} existing reports")
    
    return reports


def search_reports(reports: Dict[Tuple[int, int], CommunityReport], query: str):
    """Search for reports matching a query across all aspects and levels."""
    aspects = ["title", "summary", "full"]
    
    print(f"\nSearching for: '{query}'")
    print("-" * 50)
    
    # Search across all aspects
    for aspect in aspects:
        print(f"\nResults using {aspect.upper()} embeddings:")
        results = search_community_reports(reports, query, aspect=aspect)
        
        if not results:
            print("  No matching reports found")
            continue
        
        # Print top results
        for i, (report, score) in enumerate(results[:5]):
            level_name = "Level 1 (Fine)" if report.level_idx == 0 else (
                "Level 2 (Module)" if report.level_idx == 1 else "Level 3 (Arch)"
            )
            print(f"  {i+1}. [{level_name}] {report.title} (Score: {score:.2f})")
            print(f"     {report.summary[:100]}...")
    
    # Compare results across hierarchical levels
    print("\nResults by hierarchical level:")
    for level_idx in range(3):
        level_name = "Level 1 (Fine-grained)" if level_idx == 0 else (
            "Level 2 (Module-level)" if level_idx == 1 else "Level 3 (Architectural)"
        )
        print(f"\n{level_name} results:")
        
        results = search_community_reports(reports, query, level_idx=level_idx)
        
        if not results:
            print("  No matching reports found at this level")
            continue
        
        # Print top results for this level
        for i, (report, score) in enumerate(results[:3]):
            print(f"  {i+1}. {report.title} (Score: {score:.2f})")
            print(f"     {report.summary[:100]}...")


def visualize_community_embeddings(reports: Dict[Tuple[int, int], CommunityReport], output_dir: str):
    """Visualize community embeddings using t-SNE dimensionality reduction."""
    print("\nVisualizing community embeddings...")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Prepare embeddings by level
    embeddings_by_level = {0: [], 1: [], 2: []}
    report_info_by_level = {0: [], 1: [], 2: []}
    
    for (level_idx, community_id), report in reports.items():
        if level_idx > 2:  # Focus on first three levels
            continue
            
        # Use full report embedding
        embedding = report.get_embedding_for_aspect("full")
        if embedding is not None:
            embeddings_by_level[level_idx].append(embedding)
            report_info_by_level[level_idx].append({
                "community_id": community_id,
                "title": report.title,
                "num_key_entities": len(report.key_entities)
            })
    
    # Process each level
    for level_idx in range(3):
        if not embeddings_by_level[level_idx]:
            print(f"No embeddings found for Level {level_idx+1}")
            continue
            
        level_embeddings = np.array(embeddings_by_level[level_idx])
        level_info = report_info_by_level[level_idx]
        
        # Apply t-SNE to reduce dimensionality
        print(f"Applying t-SNE to Level {level_idx+1} embeddings...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(level_embeddings)-1) if len(level_embeddings) > 1 else 1)
        embeddings_2d = tsne.fit_transform(level_embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot points
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
        
        # Add labels
        for i, info in enumerate(level_info):
            plt.annotate(
                f"C{info['community_id']}: {info['title'][:20]}...",
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.8
            )
        
        # Set title and labels
        level_name = "Fine-grained" if level_idx == 0 else ("Module-level" if level_idx == 1 else "Architectural")
        plt.title(f"Level {level_idx+1} ({level_name}) Community Embeddings", fontsize=14)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"community_embeddings_level{level_idx+1}.png"), dpi=300)
        plt.close()
        
    # Create a combined visualization of all levels
    all_embeddings = []
    all_level_markers = []
    all_info = []
    
    # Combine embeddings from all levels
    for level_idx in range(3):
        if embeddings_by_level[level_idx]:
            all_embeddings.extend(embeddings_by_level[level_idx])
            all_level_markers.extend([level_idx] * len(embeddings_by_level[level_idx]))
            all_info.extend(report_info_by_level[level_idx])
    
    # Apply t-SNE to combined embeddings
    print("Creating combined visualization of all levels...")
    if len(all_embeddings) > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(all_embeddings)-1))
        all_embeddings_2d = tsne.fit_transform(np.array(all_embeddings))
        
        # Create visualization
        plt.figure(figsize=(14, 12))
        
        # Define colors and markers for each level
        colors = ['#FF9999', '#66B2FF', '#99CC99']
        markers = ['o', 's', '^']  # circle, square, triangle
        labels = ['Level 1 (Fine-grained)', 'Level 2 (Module-level)', 'Level 3 (Architectural)']
        
        # Plot points by level
        for level in range(3):
            level_indices = [i for i, marker in enumerate(all_level_markers) if marker == level]
            if level_indices:
                plt.scatter(
                    all_embeddings_2d[level_indices, 0], 
                    all_embeddings_2d[level_indices, 1],
                    color=colors[level],
                    marker=markers[level],
                    s=100,
                    alpha=0.7,
                    label=labels[level]
                )
        
        # Add labels for a subset of points to avoid clutter
        for i, info in enumerate(all_info):
            # Only label some points to avoid overcrowding
            if i % 3 == 0:
                plt.annotate(
                    f"L{all_level_markers[i]+1}-C{info['community_id']}",
                    (all_embeddings_2d[i, 0], all_embeddings_2d[i, 1]),
                    fontsize=7,
                    alpha=0.7
                )
        
        # Add legend and title
        plt.legend(fontsize=12)
        plt.title("Hierarchical Community Embeddings", fontsize=16)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "hierarchical_community_embeddings.png"), dpi=300)
        plt.close()


def visualize_cross_level_similarity(reports: Dict[Tuple[int, int], CommunityReport], output_dir: str):
    """Visualize similarity between communities across different hierarchical levels."""
    print("\nAnalyzing cross-level community relationships...")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Group reports by level
    reports_by_level = {0: {}, 1: {}, 2: {}}
    
    for (level_idx, community_id), report in reports.items():
        if level_idx <= 2:  # Focus on first three levels
            reports_by_level[level_idx][community_id] = report
    
    # For level pairs to analyze (level 1->2, level 2->3)
    level_pairs = [(0, 1), (1, 2)]
    
    for lower_level, higher_level in level_pairs:
        print(f"Analyzing similarity between Level {lower_level+1} and Level {higher_level+1}...")
        
        lower_reports = reports_by_level[lower_level]
        higher_reports = reports_by_level[higher_level]
        
        if not lower_reports or not higher_reports:
            print(f"  Not enough data for levels {lower_level+1} and {higher_level+1}")
            continue
        
        # Create matrices for embedding similarity
        lower_ids = sorted(lower_reports.keys())
        higher_ids = sorted(higher_reports.keys())
        
        similarity_matrix = np.zeros((len(lower_ids), len(higher_ids)))
        
        # Compute similarity between lower and higher level communities
        for i, lower_id in enumerate(lower_ids):
            lower_embedding = lower_reports[lower_id].get_embedding_for_aspect("full")
            
            for j, higher_id in enumerate(higher_ids):
                higher_embedding = higher_reports[higher_id].get_embedding_for_aspect("full")
                
                if lower_embedding is not None and higher_embedding is not None:
                    # Compute cosine similarity
                    similarity = cosine_similarity([lower_embedding], [higher_embedding])[0][0]
                    similarity_matrix[i, j] = similarity
        
        # Create heatmap visualization
        plt.figure(figsize=(max(8, len(higher_ids) * 0.8), max(6, len(lower_ids) * 0.6)))
        
        # Adjust font sizes based on matrix dimensions
        font_size = max(8, min(12, 200 / max(len(lower_ids), len(higher_ids))))
        
        # Create heatmap
        ax = sns.heatmap(
            similarity_matrix,
            cmap="YlGnBu",
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={"shrink": 0.8, "label": "Similarity Score"},
            vmin=0.3,  # Minimum similarity threshold
            vmax=1.0,  # Maximum similarity (1.0)
            annot=True,
            fmt=".2f",
            annot_kws={"size": font_size}
        )
        
        # Add labels
        level_lower_name = "Fine-grained" if lower_level == 0 else ("Module-level" if lower_level == 1 else "Architectural")
        level_higher_name = "Fine-grained" if higher_level == 0 else ("Module-level" if higher_level == 1 else "Architectural")
        
        plt.title(f"Community Similarity: Level {lower_level+1} ({level_lower_name}) → Level {higher_level+1} ({level_higher_name})")
        plt.xlabel(f"Level {higher_level+1} Communities")
        plt.ylabel(f"Level {lower_level+1} Communities")
        
        # Use community IDs and shortened titles as tick labels
        lower_labels = [f"C{c_id}: {lower_reports[c_id].title[:15]}..." for c_id in lower_ids]
        higher_labels = [f"C{c_id}: {higher_reports[c_id].title[:15]}..." for c_id in higher_ids]
        
        ax.set_xticklabels(higher_labels, rotation=45, ha="right", fontsize=font_size)
        ax.set_yticklabels(lower_labels, rotation=0, fontsize=font_size)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"cross_level_similarity_{lower_level+1}_to_{higher_level+1}.png"), dpi=300)
        plt.close()
        
        # Identify most similar communities between levels
        print(f"\nMost similar communities between Level {lower_level+1} and Level {higher_level+1}:")
        
        # Find top 3 most similar pairs
        top_pairs = []
        for i, lower_id in enumerate(lower_ids):
            for j, higher_id in enumerate(higher_ids):
                similarity = similarity_matrix[i, j]
                if similarity > 0.5:  # Only consider reasonably similar communities
                    top_pairs.append((lower_id, higher_id, similarity))
        
        # Sort by similarity (descending)
        top_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Print top pairs
        for lower_id, higher_id, similarity in top_pairs[:5]:
            print(f"  Level {lower_level+1} Community {lower_id} and Level {higher_level+1} Community {higher_id}: {similarity:.2f}")
            print(f"    - L{lower_level+1} C{lower_id}: {lower_reports[lower_id].title}")
            print(f"    - L{higher_level+1} C{higher_id}: {higher_reports[higher_id].title}")


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process codebase or load existing reports
    if args.visualize_only or args.skip_processing:
        reports = load_existing_reports(args.output_dir)
        if not reports:
            print("No existing reports found. Please run without --skip_processing first.")
            return
    else:
        reports = process_directory(args.code_dir, args.output_dir, args.num_levels)
    
    # Search for reports if query is provided
    if args.query:
        search_reports(reports, args.query)
    
    # Generate visualizations
    visualize_community_embeddings(reports, args.output_dir)
    visualize_cross_level_similarity(reports, args.output_dir)
    
    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main() 