import os
import time
import json
import hashlib
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
import shutil
from datetime import datetime
import traceback

# Import local modules
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import MODEL_CONFIG
from talktocode.indexing.entity_extractor import (
    extract_entities_from_file, 
    extract_entities_from_directory,
    CodeEntity
)
from talktocode.indexing.relationship_extractor import (
    extract_all_relationships,
    extract_entity_purpose,
    enrich_entities_with_purpose,
    Relationship
)
from talktocode.indexing.graph_builder import (
    CodeKnowledgeGraph,
    build_graph_from_entities_and_relationships
)
from talktocode.indexing.community_detector import (
    detect_hierarchical_communities,
    map_nodes_to_hierarchical_communities,
    HierarchicalCommunityDetector
)
from talktocode.indexing.report_generator import (
    generate_community_reports,
    CommunityReport,
    CommunityReportGenerator
)


class IndexingProgress:
    """Tracks progress of the indexing process."""
    
    def __init__(self, total_files: int = 0):
        self.total_files = total_files
        self.processed_files = 0
        self.skipped_files = 0
        self.failed_files = []
        self.current_stage = "Initializing"
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.update_interval = 1.0  # Update progress at most once per second
    
    def update(self, stage: str = None, increment_processed: bool = False, 
              increment_skipped: bool = False, failed_file: str = None):
        """Update progress information."""
        current_time = time.time()
        
        if stage:
            self.current_stage = stage
        
        if increment_processed:
            self.processed_files += 1
        
        if increment_skipped:
            self.skipped_files += 1
        
        if failed_file:
            self.failed_files.append(failed_file)
        
        # Only print updates at the specified interval to avoid console spam
        if current_time - self.last_update_time >= self.update_interval:
            self.print_progress()
            self.last_update_time = current_time
    
    def print_progress(self):
        """Print current progress to console."""
        if self.total_files > 0:
            percentage = (self.processed_files + self.skipped_files) / self.total_files * 100
            elapsed_time = time.time() - self.start_time
            
            print(f"[{self.current_stage}] Progress: {percentage:.1f}% "
                  f"({self.processed_files} processed, {self.skipped_files} skipped) "
                  f"in {elapsed_time:.1f}s")
        else:
            print(f"[{self.current_stage}] Processed {self.processed_files} files, "
                  f"skipped {self.skipped_files}")
        
        if self.failed_files:
            print(f"  Failed files: {len(self.failed_files)}")
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the indexing process."""
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "failed_files": self.failed_files,
            "elapsed_time_seconds": elapsed_time,
            "completion_time": datetime.now().isoformat()
        }


class CodeIndexer:
    """
    Integrates all components to index code files, build a knowledge graph,
    detect communities, and generate reports.
    
    This class provides a simple interface to the entire code analysis pipeline.
    """
    
    def __init__(self, cache_dir: str = ".talktocode_cache", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the code indexer.
        
        Args:
            cache_dir: Directory to store cache files
            config: Optional configuration parameters
        """
        self.cache_dir = Path(cache_dir)
        self.config = config or {}
        
        # Initialize cache
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.file_cache_path = self.cache_dir / "file_cache.json"
        self.file_cache = self._load_file_cache()
        
        # Initialize components
        self.entities = {}
        self.entity_purposes = {}
        self.relationships = []
        self.graph = None
        self.community_detector = None
        self.reports = {}
        
        # Initialize progress tracker
        self.progress = IndexingProgress()
    
    def index_files(self, file_paths: List[str], force_reindex: bool = False) -> IndexingProgress:
        """
        Process a list of Python files to build the knowledge graph.
        
        Args:
            file_paths: List of file paths to process
            force_reindex: Whether to force reindexing of all files
            
        Returns:
            IndexingProgress object with indexing statistics
        """
        # Initialize progress tracker
        self.progress = IndexingProgress(total_files=len(file_paths))
        
        # Extract entities from files
        self.progress.update(stage="Extracting entities")
        self._extract_entities(file_paths, force_reindex)
        
        if not self.entities or not any(len(entities) > 0 for entities in self.entities.values()):
            print("No entities found in the provided files.")
            return self.progress
        
        # Extract relationships between entities
        self.progress.update(stage="Extracting entity purposes")
        self._extract_entity_purposes()
        
        # Extract relationships between entities
        self.progress.update(stage="Extracting relationships")
        self._extract_relationships()
        
        # Build knowledge graph
        self.progress.update(stage="Building knowledge graph")
        self._build_graph()
        
        # Detect communities
        self.progress.update(stage="Detecting communities")
        self._detect_communities()
        
        # Generate reports
        self.progress.update(stage="Generating reports")
        self._generate_reports()
        
        # Save cache
        self._save_file_cache()
        
        # Print final summary
        self.progress.print_progress()
        return self.progress
    
    def get_graph(self) -> Optional[CodeKnowledgeGraph]:
        """
        Get the built knowledge graph.
        
        Returns:
            CodeKnowledgeGraph instance or None if not built yet
        """
        return self.graph
    
    def get_communities(self) -> Optional[HierarchicalCommunityDetector]:
        """
        Get the community structure.
        
        Returns:
            HierarchicalCommunityDetector instance or None if not built yet
        """
        return self.community_detector
    
    def get_community_reports(self) -> Dict[Tuple[int, int], CommunityReport]:
        """
        Get the generated community reports.
        
        Returns:
            Dictionary mapping (level_idx, community_id) to CommunityReport objects
        """
        return self.reports
    
    def save_results(self, output_dir: str) -> None:
        """
        Save all results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save entities
        entities_path = output_path / "entities.json"
        with open(entities_path, 'w') as f:
            # Convert entities to serializable format
            serializable_entities = {}
            for entity_type, entity_list in self.entities.items():
                serializable_entities[entity_type] = [entity.to_dict() for entity in entity_list]
            json.dump(serializable_entities, f, indent=2)
        
        # Save graph as JSON
        if self.graph:
            graph_path = output_path / "graph.json"
            graph_json = self.graph.to_json()
            with open(graph_path, 'w') as f:
                f.write(graph_json)
            
            # Save graph visualization as HTML
            graph_viz_path = output_path / "graph_visualization.html"
            self.graph.save_plotly_html(str(graph_viz_path))
        
        # Save community reports
        if self.reports:
            reports_dir = output_path / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Create report generator to use its save method
            if self.graph and self.community_detector:
                generator = CommunityReportGenerator(self.graph, self.community_detector)
                generator.reports = self.reports
                generator.save_reports(str(reports_dir))
        
        # Save community visualization if available
        if self.community_detector:
            community_viz_path = output_path / "community_hierarchy.png"
            self.community_detector.visualize_hierarchy(str(community_viz_path))
        
        # Save progress summary
        summary_path = output_path / "indexing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.progress.summary(), f, indent=2)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of a file for caching.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash of the file contents
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"Error computing hash for {file_path}: {str(e)}")
            return ""
    
    def _load_file_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Load file cache from disk.
        
        Returns:
            Dictionary mapping file paths to their metadata
        """
        if self.file_cache_path.exists():
            try:
                with open(self.file_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading file cache: {str(e)}")
        
        return {}
    
    def _save_file_cache(self) -> None:
        """Save file cache to disk."""
        try:
            with open(self.file_cache_path, 'w') as f:
                json.dump(self.file_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving file cache: {str(e)}")
    
    def _extract_entities(self, file_paths: List[str], force_reindex: bool) -> None:
        """
        Extract entities from files.
        
        Args:
            file_paths: List of file paths to process
            force_reindex: Whether to force reindexing of all files
        """
        all_entities = {
            "functions": [],
            "classes": [],
            "variables": [],
            "imports": []
        }
        
        # Process each file
        for file_path in file_paths:
            self.progress.update(stage=f"Extracting entities from {Path(file_path).name}")
            
            try:
                # Check if file has changed
                file_hash = self._compute_file_hash(file_path)
                
                if not force_reindex and file_path in self.file_cache and self.file_cache[file_path]["hash"] == file_hash:
                    # File hasn't changed, skip processing
                    self.progress.update(increment_skipped=True)
                    continue
                
                # Extract entities from the file
                file_entities = extract_entities_from_file(file_path)
                
                # Add entities to the combined list
                for entity_type, entity_list in file_entities.items():
                    all_entities[entity_type].extend(entity_list)
                
                # Update cache
                self.file_cache[file_path] = {
                    "hash": file_hash,
                    "last_processed": datetime.now().isoformat()
                }
                
                self.progress.update(increment_processed=True)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                traceback.print_exc()
                self.progress.update(failed_file=file_path)
        
        self.entities = all_entities
    
    def _extract_entity_purposes(self) -> None:
        """Extract purposes for each entity using GPT-3.5."""
        if not self.entities:
            return
        
        # Get the maximum number of entities to process from config
        max_entities = self.config.get("max_entities_for_purposes", 100)
        
        # Extract purposes using GPT-3.5
        self.entity_purposes = enrich_entities_with_purpose(self.entities, max_entities=max_entities)
        
        # Update progress
        self.progress.update(stage=f"Extracted purposes for {len(self.entity_purposes)} entities")
    
    def _extract_relationships(self) -> None:
        """Extract relationships between entities."""
        if not self.entities:
            return
        
        # Get the maximum number of LLM pairs to process from config
        max_llm_pairs = self.config.get("max_llm_pairs", 100)
        
        # Extract relationships using both static analysis and LLM
        self.relationships = extract_all_relationships(
            self.entities, 
            use_llm=True, 
            max_llm_pairs=max_llm_pairs
        )
        
        # Update progress
        self.progress.update(stage=f"Extracted {len(self.relationships)} relationships")
    
    def _build_graph(self) -> None:
        """Build knowledge graph from entities and relationships."""
        if not self.entities or not self.relationships:
            return
        
        # Build the graph
        self.graph = build_graph_from_entities_and_relationships(
            self.entities,
            self.relationships,
            self.entity_purposes
        )
        
        # Update progress
        num_nodes = len(self.graph.graph.nodes)
        num_edges = len(self.graph.graph.edges)
        self.progress.update(stage=f"Built graph with {num_nodes} nodes and {num_edges} edges")
    
    def _detect_communities(self) -> None:
        """Detect communities in the knowledge graph."""
        if not self.graph:
            return
        
        # Get the number of community levels from config
        num_levels = self.config.get("community_levels", 3)
        
        # Detect communities
        self.community_detector = detect_hierarchical_communities(
            self.graph.graph,
            num_levels=num_levels,
            auto_detect_resolutions=True
        )
        
        # Add community information to the graph
        map_nodes_to_hierarchical_communities(self.graph.graph, num_levels=num_levels)
        
        # Update progress
        community_counts = []
        for level_idx in range(num_levels):
            communities = self.community_detector.get_communities_at_level(level_idx)
            community_counts.append(len(communities))
        
        self.progress.update(stage=f"Detected communities: {community_counts}")
    
    def _generate_reports(self) -> None:
        """Generate reports for each community."""
        if not self.graph or not self.community_detector:
            return
        
        # Generate reports
        self.reports = generate_community_reports(
            self.graph,
            self.community_detector
        )
        
        # Update progress
        self.progress.update(stage=f"Generated {len(self.reports)} community reports")


def index_codebase(directory_path: str, output_dir: str = "results", 
                 file_extensions: List[str] = ['.py'], 
                 exclude_dirs: List[str] = ['__pycache__', '.git', 'venv', 'env'],
                 force_reindex: bool = False,
                 config: Optional[Dict[str, Any]] = None) -> CodeIndexer:
    """
    Index a codebase and save the results.
    
    Args:
        directory_path: Path to the directory containing the codebase
        output_dir: Directory to save the results
        file_extensions: List of file extensions to process
        exclude_dirs: List of directories to exclude
        force_reindex: Whether to force reindexing of all files
        config: Optional configuration parameters
        
    Returns:
        CodeIndexer instance with the indexed codebase
    """
    # Find all files to process
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    # Create indexer
    indexer = CodeIndexer(config=config)
    
    # Index files
    indexer.index_files(file_paths, force_reindex=force_reindex)
    
    # Save results
    indexer.save_results(output_dir)
    
    return indexer 