#!/usr/bin/env python3
"""
Example script demonstrating the use of embedding-based search strategies.

This script shows how to:
1. Use entity embeddings for Local Search to find similar code entities
2. Use community report embeddings for Global Search to find relevant communities
3. Compare different search strategies and configurations
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TalkToCode modules
from talktocode.utils.config import set_embedding_model
from talktocode.indexing.entity_extractor import extract_entities_from_directory
from talktocode.indexing.relationship_extractor import extract_all_relationships
from talktocode.indexing.graph_builder import CodeKnowledgeGraph
from talktocode.indexing.community_detector import detect_hierarchical_communities
from talktocode.indexing.report_generator import generate_community_reports
from talktocode.retrieval.search import search_codebase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate embedding-based code search strategies")
    parser.add_argument("--code_dir", type=str, required=True, help="Directory containing code to analyze")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store output files")
    parser.add_argument("--query", type=str, default="How does the error handling work?", 
                        help="Query to search for in the codebase")
    parser.add_argument("--skip_processing", action="store_true", help="Skip processing and use existing reports")
    parser.add_argument("--strategy", type=str, default="both", 
                        choices=["local", "global", "drift", "both", "all"], 
                        help="Search strategy to use")
    return parser.parse_args()


def process_codebase(code_dir: str, output_dir: str) -> Tuple[CodeKnowledgeGraph, Any, Dict]:
    """
    Process a codebase to build a knowledge graph, detect communities, and generate reports.
    
    Args:
        code_dir: Path to the code directory
        output_dir: Path to the output directory
        
    Returns:
        Tuple containing:
        - CodeKnowledgeGraph instance
        - HierarchicalCommunityDetector instance
        - Dictionary of community reports
    """
    print(f"Processing codebase in {code_dir}...")
    start_time = time.time()
    
    # Set embedding model to text-embedding-ada-002
    set_embedding_model("text-embedding-ada-002")
    
    # Create output directories
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Extract entities from code
    print("Extracting code entities...")
    all_entities = extract_entities_from_directory(code_dir)
    entity_counts = {entity_type: len(entities) for entity_type, entities in all_entities.items()}
    print(f"Extracted entities: {entity_counts}")
    
    # Extract relationships
    print("Extracting relationships...")
    relationships = extract_all_relationships(
        all_entities, 
        use_llm=False,  # Faster without LLM
        use_embeddings=True  # Use embeddings for semantic relationships
    )
    print(f"Extracted {len(relationships)} relationships")
    
    # Build knowledge graph
    print("Building knowledge graph...")
    graph = CodeKnowledgeGraph()
    graph.build_graph_from_entities_and_relationships(all_entities, relationships)
    print(f"Graph built with {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges")
    
    # Detect communities
    print("Detecting hierarchical communities...")
    community_detector = detect_hierarchical_communities(
        graph.graph, 
        num_levels=3  # 3 levels: fine-grained, module, architecture
    )
    
    # Print community statistics
    for level_idx in range(3):
        communities = community_detector.get_communities_at_level(level_idx)
        print(f"Level {level_idx+1}: {len(communities)} communities")
    
    # Generate community reports
    print("Generating community reports with embeddings...")
    reports = generate_community_reports(
        graph, 
        community_detector,
        output_dir=reports_dir
    )
    print(f"Generated {len(reports)} community reports")
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    return graph, community_detector, reports


def load_existing_data(output_dir: str) -> Tuple[CodeKnowledgeGraph, Any, Dict]:
    """
    Load existing graph, community detector, and reports from files.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Tuple containing:
        - CodeKnowledgeGraph instance
        - HierarchicalCommunityDetector instance
        - Dictionary of community reports
    """
    print("Loading existing data...")
    
    # Load graph
    graph_path = os.path.join(output_dir, "graph.json")
    graph = CodeKnowledgeGraph()
    graph.load_graph(graph_path)
    print(f"Loaded graph with {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges")
    
    # Load community detector
    communities_path = os.path.join(output_dir, "communities.json")
    community_detector = detect_hierarchical_communities(graph.graph, load_from=communities_path)
    
    # Load reports
    reports_dir = os.path.join(output_dir, "reports")
    from talktocode.indexing.report_generator import CommunityReportGenerator
    temp_generator = CommunityReportGenerator(graph, community_detector)
    reports = temp_generator.load_reports(reports_dir)
    print(f"Loaded {len(reports)} community reports")
    
    return graph, community_detector, reports


def run_search(query: str, graph: CodeKnowledgeGraph, community_detector: Any, 
               reports: Dict, strategy: str, output_dir: str):
    """
    Run a search using the specified strategy.
    
    Args:
        query: The search query
        graph: The code knowledge graph
        community_detector: HierarchicalCommunityDetector instance
        reports: Dictionary of community reports
        strategy: Search strategy ('local', 'global', 'drift')
        output_dir: Path to the output directory
    """
    print(f"\n{'='*80}")
    print(f"Running {strategy.upper()} search for query: '{query}'")
    print(f"{'='*80}")
    
    # Configure search parameters
    params = None
    if strategy == "local":
        params = {
            "max_hops": 2,
            "top_k_entities": 15,
            "min_similarity": 0.6,  # Lower threshold to get more results
            "include_code": True
        }
    elif strategy == "global":
        params = {
            "top_k_communities": 5,
            "min_similarity": 0.5,  # Lower threshold to get more results
            "community_levels": [0, 1, 2],  # All levels
            "report_aspect": "full"  # Use full report content
        }
    elif strategy == "drift":
        params = {
            "num_hypotheses": 2,
            "max_steps": 2,
            "branching_factor": 2
        }
    
    # Run search
    start_time = time.time()
    results = search_codebase(
        query=query,
        graph=graph,
        community_detector=community_detector,
        community_reports=reports,
        strategy=strategy,
        params=params
    )
    end_time = time.time()
    
    # Print search time
    print(f"\nSearch completed in {end_time - start_time:.2f} seconds")
    
    # Print results overview
    if strategy == "local":
        print(f"\nFound {len(results['entities'])} relevant entities")
        
        # Print top entities
        print("\nTop entities found:")
        for i, entity in enumerate(results['entities'][:5]):
            print(f"  {i+1}. {entity['name']} ({entity['type']}) - Score: {entity['score']:.3f}")
            print(f"     File: {entity['source_file']}")
            print(f"     Description: {entity['description'][:100]}...")
            
    elif strategy == "global":
        print(f"\nFound {len(results['communities'])} relevant communities")
        
        # Print top communities
        print("\nTop communities found:")
        for i, community in enumerate(results['communities'][:3]):
            print(f"  {i+1}. Level {community['level']} Community {community['id']}: {community['title']}")
            print(f"     Similarity: {community['similarity']:.3f}")
            print(f"     Summary: {community['summary'][:150]}...")
            
    elif strategy == "drift":
        print(f"\nGenerated {len(results['reasoning'])} reasoning paths")
        
        # Print reasoning summary
        print("\nReasoning summary:")
        print(results['summary'])
    
    # Save results to file
    results_dir = os.path.join(output_dir, "search_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, f"{strategy}_search_results.json")
    with open(results_path, 'w') as f:
        # Remove large arrays for smaller output file
        simplified_results = results.copy()
        if 'entities' in simplified_results and len(simplified_results['entities']) > 5:
            simplified_results['entities'] = simplified_results['entities'][:5]
            simplified_results['entities'].append({"note": f"{len(results['entities']) - 5} more entities omitted"})
        
        json.dump(simplified_results, f, indent=2)
    
    print(f"\nDetailed results saved to {results_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process or load codebase
    if args.skip_processing:
        try:
            graph, community_detector, reports = load_existing_data(args.output_dir)
        except Exception as e:
            print(f"Error loading existing data: {e}")
            print("Processing codebase from scratch...")
            graph, community_detector, reports = process_codebase(args.code_dir, args.output_dir)
    else:
        graph, community_detector, reports = process_codebase(args.code_dir, args.output_dir)
    
    # Run searches
    if args.strategy == "both":
        run_search(args.query, graph, community_detector, reports, "local", args.output_dir)
        run_search(args.query, graph, community_detector, reports, "global", args.output_dir)
    elif args.strategy == "all":
        run_search(args.query, graph, community_detector, reports, "local", args.output_dir)
        run_search(args.query, graph, community_detector, reports, "global", args.output_dir)
        run_search(args.query, graph, community_detector, reports, "drift", args.output_dir)
    else:
        run_search(args.query, graph, community_detector, reports, args.strategy, args.output_dir)
    
    print("\nSearch demonstration completed successfully!")


if __name__ == "__main__":
    main() 