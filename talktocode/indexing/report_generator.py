import os
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Set
import networkx as nx
import numpy as np
import openai
from pathlib import Path
from tqdm import tqdm

# Import local modules
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import OPENAI_API_KEY, MODEL_CONFIG, set_embedding_model
from talktocode.indexing.entity_extractor import CodeEntity, extract_code_with_context
from talktocode.indexing.community_detector import HierarchicalCommunityDetector
from talktocode.indexing.graph_builder import CodeKnowledgeGraph
from talktocode.indexing.faiss_manager import FaissIndexManager
from talktocode.indexing.entity_embeddings import EntityEmbeddingGenerator

# Configure OpenAI
openai.api_key = OPENAI_API_KEY
from openai import OpenAI

# Use lazy initialization to avoid recursion issues
def get_client():
    """Get or initialize the OpenAI client."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

# Configure the embedding model to use text-embedding-ada-002
set_embedding_model("text-embedding-ada-002")


class CommunityReport:
    """Represents a generated report for a community of code entities."""
    
    def __init__(self, community_id: int, level_idx: int, resolution: float):
        self.community_id = community_id
        self.level_idx = level_idx
        self.resolution = resolution
        self.title = ""
        self.summary = ""
        self.key_entities = []
        self.related_communities = []
        self.architectural_patterns = []
        self.full_report = ""
        
        # Embeddings for different aspects of the report
        self.title_embedding: Optional[np.ndarray] = None
        self.summary_embedding: Optional[np.ndarray] = None
        self.full_report_embedding: Optional[np.ndarray] = None
        
        # Legacy embedding field for backward compatibility
        self.embedding: Optional[np.ndarray] = None
        
        # When the report was last updated
        self.last_updated = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "community_id": self.community_id,
            "level_idx": self.level_idx,
            "resolution": self.resolution,
            "title": self.title,
            "summary": self.summary,
            "key_entities": self.key_entities,
            "related_communities": self.related_communities,
            "architectural_patterns": self.architectural_patterns,
            "full_report": self.full_report,
            # Store all embeddings in optimized format (list of floats)
            "title_embedding": self.title_embedding.tolist() if self.title_embedding is not None else None,
            "summary_embedding": self.summary_embedding.tolist() if self.summary_embedding is not None else None,
            "full_report_embedding": self.full_report_embedding.tolist() if self.full_report_embedding is not None else None,
            # Legacy embedding field
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunityReport':
        """Create a report from dictionary representation."""
        report = cls(
            community_id=data["community_id"],
            level_idx=data["level_idx"],
            resolution=data["resolution"]
        )
        report.title = data["title"]
        report.summary = data["summary"]
        report.key_entities = data["key_entities"]
        report.related_communities = data["related_communities"]
        report.architectural_patterns = data["architectural_patterns"]
        report.full_report = data["full_report"]
        
        # Load embeddings if available
        if data.get("title_embedding") is not None:
            report.title_embedding = np.array(data["title_embedding"])
        
        if data.get("summary_embedding") is not None:
            report.summary_embedding = np.array(data["summary_embedding"])
        
        if data.get("full_report_embedding") is not None:
            report.full_report_embedding = np.array(data["full_report_embedding"])
        
        # Legacy embedding field
        if data.get("embedding") is not None:
            report.embedding = np.array(data["embedding"])
        
        report.last_updated = data.get("last_updated")
        
        return report
    
    def get_embedding_for_aspect(self, aspect: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a specific aspect of the report.
        
        Args:
            aspect: Which aspect of the report to get the embedding for
                   (options: "title", "summary", "full")
                   
        Returns:
            NumPy array containing the embedding vector or None if not available
        """
        if aspect == "title":
            return self.title_embedding
        elif aspect == "summary":
            return self.summary_embedding
        elif aspect == "full":
            return self.full_report_embedding
        else:
            # Default to full report embedding or legacy embedding
            return self.full_report_embedding or self.embedding


class CommunityReportGenerator:
    """Generates comprehensive reports for communities using LLMs and indexes them using FAISS."""
    
    def __init__(self, 
                 graph: CodeKnowledgeGraph, 
                 community_detector: HierarchicalCommunityDetector,
                 faiss_manager: Optional[FaissIndexManager] = None, # Accept FaissManager
                 embedding_generator: Optional[EntityEmbeddingGenerator] = None): # Accept Embeddings Generator
        """Initializes the report generator with graph, detector, and optional managers."""
        self.graph = graph
        self.community_detector = community_detector
        self.reports: Dict[Tuple[int, int], CommunityReport] = {} # (level_idx, community_id) -> Report object
        self._client = None # Lazy init OpenAI client
        
        # --- FAISS Integration --- 
        self.faiss_config = MODEL_CONFIG["embedding"].get("faiss", {})
        self.faiss_enabled = self.faiss_config.get("enabled", False)
        self.faiss_manager = faiss_manager
        # Disable FAISS specifically for reports if manager not provided, even if globally enabled
        if self.faiss_enabled and self.faiss_manager is None:
             print("Warning: FaissManager not provided to CommunityReportGenerator. FAISS indexing will be disabled for reports.")
             self.faiss_enabled = False 

        # --- Embedding Generator --- 
        self.embedding_generator = embedding_generator
        if self.embedding_generator is None:
             print("Warning: No embedding generator provided to CommunityReportGenerator. Initializing default.")
             # If initializing default embedding generator, pass it the faiss_manager if available
             # Ensure EntityEmbeddingGenerator accepts faiss_manager in its init
             self.embedding_generator = EntityEmbeddingGenerator(faiss_manager=self.faiss_manager) 
             
    def get_openai_client(self):
         """Lazy loads OpenAI client."""
         if self._client is None:
              # Consider using a shared client instance (e.g., from app.py or a util func)
              self._client = OpenAI() # Assumes API key is set in environment
         return self._client

    def _get_community_context_string(self, level_idx: int, community_id: int, max_entities: int = 10) -> str:
        """Helper to get formatted context string for a community."""
        members = self.community_detector.get_nodes_in_community(level_idx, community_id)
        
        # Check if members list is empty
        if not members:
            return "This community is empty."
        
        # Build context string
        context_str = f"Context for Community {community_id} (Level {level_idx}):\n"
        context_str += f"Number of members: {len(members)}\n"
        context_str += "Sample Members:\n"
        count = 0
        for node_id in members:
            if node_id in self.graph.graph.nodes:
                node_data = self.graph.graph.nodes[node_id]
                context_str += f"- {node_data.get('name','?')} ({node_data.get('type','?')}) in {os.path.basename(node_data.get('source_file','?'))}\n"
                # Add description if available
                desc = node_data.get('description', '')
                if desc and desc != "No description available":
                    context_str += f"  Description: {desc[:100]}{'...' if len(desc) > 100 else ''}\n"
                count += 1
                if count >= max_entities:
                    break 
        return context_str

    def _generate_report_with_gpt(self, context_str: str) -> str:
        """Generates the report text using an LLM call."""
        # Define the system prompt instructing the LLM
        system_prompt = ( 
            "You are an expert code analyst. Based on the provided context about a code community "
            "(a group of related code entities like functions and classes), generate a concise report "
            "in JSON format. The report should have the following keys: 'title' (a short, descriptive name), "
            "'summary' (a 1-2 sentence overview of the community's purpose), 'key_entities' (a list of up to 5 "
            "most important entity names within the community mentioned in the context), and 'architectural_patterns' "
            "(a list of any potential software design patterns observed, e.g., Singleton, Factory, Observer). "
            "Focus only on the provided context."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_str}
        ]
        
        try:
            client = self.get_openai_client()
            response = client.chat.completions.create(
                model=MODEL_CONFIG["models"]["code_analysis"], # Use appropriate model
                messages=messages,
                response_format={ "type": "json_object" }, # Request JSON output
                temperature=0.3,
                max_tokens=500 # Adjust as needed
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI for report generation: {e}")
            # Return placeholder JSON on error
            return json.dumps({ 
                "title": "Error Generating Report", 
                "summary": f"Failed to generate report: {e}",
                "key_entities": [],
                "architectural_patterns": []
            })

    def _parse_report_sections(self, report_json_str: str) -> Dict[str, Any]:
        """Parses the JSON string returned by the LLM."""
        try:
            report_data = json.loads(report_json_str)
            # Basic validation
            if not isinstance(report_data, dict): return {}
            return {
                "title": report_data.get("title", "Untitled Community"),
                "summary": report_data.get("summary", "No summary provided."),
                "key_entities": report_data.get("key_entities", []), # Expecting list of strings
                "architectural_patterns": report_data.get("architectural_patterns", []) # Expecting list
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM report JSON: {e}\nContent: {report_json_str[:500]}...")
            return {}
        except Exception as e:
            print(f"Unexpected error parsing report: {e}")
            return {}

    def generate_report_for_community(self, level_idx: int, community_id: int) -> Optional[CommunityReport]:
        """Generates a single community report and indexes its embeddings."""
        print(f"Generating report for Level {level_idx}, Community {community_id}...")
        try:
            resolution = self.community_detector.community_levels[level_idx]
            report = CommunityReport(community_id, level_idx, resolution)
            
            context_str = self._get_community_context_string(level_idx, community_id)
            if context_str == "This community is empty.":
                report.title = f"Empty Community {community_id}"
                report.summary = "This community has no members."
                self.reports[(level_idx, community_id)] = report
                return report # Return empty report

            llm_report_json = self._generate_report_with_gpt(context_str)
            parsed_sections = self._parse_report_sections(llm_report_json)

            report.title = parsed_sections.get("title", f"Community {community_id} - Level {level_idx}")
            report.summary = parsed_sections.get("summary", "")
            report.key_entities = parsed_sections.get("key_entities", [])
            report.architectural_patterns = parsed_sections.get("architectural_patterns", [])
            # Store the original JSON string as the 'full_report' for reference?
            report.full_report = llm_report_json 
            report.last_updated = time.time()

            # Generate and Index Embeddings
            report_faiss_id = f"report_{level_idx}_{community_id}"
            if self.embedding_generator:
                if report.title:
                    title_embedding = self.embedding_generator.get_embedding(report.title)
                    if title_embedding is not None:
                        report.title_embedding = title_embedding
                        if self.faiss_enabled and self.faiss_manager:
                            self.faiss_manager.add_embeddings("report_title", [report_faiss_id], title_embedding.reshape(1, -1))
                
                if report.summary:
                    summary_embedding = self.embedding_generator.get_embedding(report.summary)
                    if summary_embedding is not None:
                        report.summary_embedding = summary_embedding
                        if self.faiss_enabled and self.faiss_manager:
                            self.faiss_manager.add_embeddings("report_summary", [report_faiss_id], summary_embedding.reshape(1, -1))
                        # Use summary for legacy embedding field
                        report.embedding = summary_embedding 
                
                # Avoid embedding the full JSON report string unless needed
                # if report.full_report:
                #    full_report_embedding = self.embedding_generator.get_embedding(report.full_report)
                #    if full_report_embedding is not None:
                #        report.full_report_embedding = full_report_embedding
                #        if self.faiss_enabled and self.faiss_manager:
                #            self.faiss_manager.add_embeddings("report_full", [report_faiss_id], full_report_embedding.reshape(1, -1))
            
            self.reports[(level_idx, community_id)] = report
            return report
        except Exception as e:
            print(f"ERROR generating report for L{level_idx} C{community_id}: {e}")
            # traceback.print_exc() # Uncomment for debugging
            return None
    
    def generate_all_reports(self) -> Dict[Tuple[int, int], CommunityReport]:
        """Generates reports for all communities across all levels and ensures FAISS indices are saved."""
        print("Generating community reports for all levels...")
        start_time = time.time()
        self.reports = {}
        num_levels = len(self.community_detector.community_levels)
        
        for level_idx in range(num_levels):
            level_communities = self.community_detector.get_communities_at_level(level_idx)
            print(f"Generating reports for {len(level_communities)} communities at Level {level_idx}...")
            
            with tqdm(total=len(level_communities), desc=f"Level {level_idx} Reports") as pbar:
                for comm_id in level_communities.keys():
                    self.generate_report_for_community(level_idx, comm_id)
                    pbar.update(1)
                    # Optional: time.sleep(0.05) # Small delay for API rate limits
        
        # Save FAISS indices after processing all reports
        if self.faiss_enabled and self.faiss_manager:
             print("Saving FAISS indices for reports...")
             self.faiss_manager.save_all_indices()
             
        end_time = time.time()
        print(f"Generated all reports in {end_time - start_time:.2f} seconds.")
        return self.reports
    
    def find_similar_reports(self,
                           query_text: str,
                           level_idx: Optional[int] = None,
                           aspect: str = "summary", 
                           top_k: int = 5,
                           min_similarity: Optional[float] = None) -> List[Tuple[CommunityReport, float]]:
        """Finds similar reports using FAISS based on query text."""
        if not self.faiss_enabled or not self.faiss_manager:
             print("FAISS not enabled/available for report search.")
             return []
        if not self.embedding_generator:
             print("Embedding generator not available for report search.")
             return []

        query_embedding = self.embedding_generator.get_embedding(query_text)
        if query_embedding is None:
            print("Could not generate query embedding for report search.")
            return []

        # Determine the FAISS index name based on the aspect
        index_name = f"report_{aspect}" # e.g., report_summary, report_title
        if self.faiss_manager.get_index_size(index_name) == 0:
             print(f"Warning: FAISS index '{index_name}' is empty or not found.")
             return []

        # Search FAISS
        similar_report_ids_scores = self.faiss_manager.search(
            index_name=index_name,
            query_vector=query_embedding,
            top_k=top_k
        )
        
        # Retrieve report objects and filter by level/similarity
        results = []
        for report_faiss_id, score in similar_report_ids_scores:
            if min_similarity is not None and score < min_similarity:
                continue
            try:
                parts = report_faiss_id.split('_')
                if len(parts) == 3 and parts[0] == 'report':
                    r_level = int(parts[1])
                    r_comm_id = int(parts[2])
                    # Filter by level if specified
                    if level_idx is not None and r_level != level_idx:
                         continue
                    # Find the actual report object from the in-memory store
                    report_obj = self.reports.get((r_level, r_comm_id))
                    if report_obj:
                        results.append((report_obj, score))
                else: print(f"Warning: Could not parse FAISS ID for report: {report_faiss_id}")
            except (ValueError, IndexError): print(f"Warning: Could not parse level/comm_id from FAISS ID: {report_faiss_id}")
                 
        # Results from FAISS search should already be sorted by similarity/distance
        # Re-sorting might not be necessary unless converting distance to similarity inverted order
        # results.sort(key=lambda x: x[1], reverse=True) # Ensure highest similarity first
        return results[:top_k]

    # --- save_reports/load_reports need review --- 
    # These currently handle the CommunityReport objects which might still hold embeddings
    # If embeddings are ONLY in FAISS, these might just save/load metadata.
    def save_reports(self, output_dir: str) -> None:
        """Saves generated reports (metadata only?) to disk."""
        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        index_data = {}
        print(f"Saving {len(self.reports)} reports metadata to {output_dir}...")
        for (level, comm_id), report in self.reports.items():
            filename = f"report_l{level}_c{comm_id}.json"
            filepath = report_dir / filename
            try:
                # Save report metadata (potentially excluding embedding arrays now)
                report_dict = report.to_dict()
                # Optionally remove embedding keys before saving if they only live in FAISS
                # report_dict.pop('title_embedding', None)
                # report_dict.pop('summary_embedding', None)
                # report_dict.pop('full_report_embedding', None)
                # report_dict.pop('embedding', None)
                with open(filepath, 'w') as f:
                    json.dump(report_dict, f, indent=4)
                index_data[f"{level}_{comm_id}"] = filename
            except Exception as e:
                print(f"Error saving report {level}-{comm_id}: {e}")
        # Save index file
        index_filepath = report_dir / "reports_index.json"
        try:
             with open(index_filepath, 'w') as f:
                  json.dump(index_data, f, indent=4)
        except Exception as e:
             print(f"Error saving reports index: {e}")

    def load_reports(self, input_dir: str):
        """Loads report metadata from disk. Assumes embeddings are loaded into FAISS separately."""
        report_dir = Path(input_dir)
        index_filepath = report_dir / "reports_index.json"
        if not index_filepath.exists():
            print(f"Reports index file not found at {index_filepath}, cannot load reports.")
            return
        print(f"Loading reports metadata from {input_dir}...")
        self.reports = {}
        try:
            with open(index_filepath, 'r') as f:
                index_data = json.load(f)
            for key, filename in index_data.items():
                filepath = report_dir / filename
                if filepath.exists():
                    try:
                        with open(filepath, 'r') as f:
                            report_dict = json.load(f)
                        # Recreate report object (embeddings will be None unless saved)
                        report_obj = CommunityReport.from_dict(report_dict)
                        level, comm_id = map(int, key.split('_'))
                        self.reports[(level, comm_id)] = report_obj
                    except Exception as e:
                        print(f"Error loading report file {filename}: {e}")
            print(f"Loaded {len(self.reports)} reports metadata.")
        except Exception as e:
            print(f"Error loading reports index: {e}")


def generate_community_reports(graph: CodeKnowledgeGraph, 
                              community_detector: HierarchicalCommunityDetector,
                              faiss_manager: Optional[FaissIndexManager] = None, # Add param
                              embedding_generator: Optional[EntityEmbeddingGenerator] = None, # Add param
                              output_dir: Optional[str] = None) -> Dict[Tuple[int, int], CommunityReport]:
    """Wrapper function to generate community reports and handle FAISS indexing."""
    print("Initializing CommunityReportGenerator...")
    # Ensure embedding generator gets the faiss manager if provided
    if embedding_generator is None:
         embedding_generator = EntityEmbeddingGenerator(faiss_manager=faiss_manager)
    elif faiss_manager is not None:
         # If both are provided, ensure generator uses the correct manager
         embedding_generator.faiss_manager = faiss_manager
         embedding_generator.faiss_enabled = MODEL_CONFIG["embedding"].get("faiss", {}).get("enabled", False)
         
    generator = CommunityReportGenerator(
        graph=graph, 
        community_detector=community_detector, 
        faiss_manager=faiss_manager, # Pass manager
        embedding_generator=embedding_generator # Pass generator
    )
    reports = generator.generate_all_reports()
    # Save report metadata (embeddings are in FAISS)
    if output_dir:
        print(f"Saving reports metadata to {output_dir}...")
        generator.save_reports(output_dir)
    return reports