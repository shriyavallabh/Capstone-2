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

# Configure OpenAI
openai.api_key = OPENAI_API_KEY
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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
    """Generates comprehensive reports for communities using GPT-3.5."""
    
    def __init__(self, graph: CodeKnowledgeGraph, community_detector: HierarchicalCommunityDetector):
        self.graph = graph
        self.community_detector = community_detector
        self.reports: Dict[Tuple[int, int], CommunityReport] = {}  # (level_idx, community_id) -> Report
        
        # Set up embedding caching
        self.cache_dir = MODEL_CONFIG["embedding"]["cache"]["directory"]
        self.reports_cache_dir = os.path.join(self.cache_dir, "community_report_embeddings")
        os.makedirs(self.reports_cache_dir, exist_ok=True)
    
    def generate_report_for_community(self, level_idx: int, community_id: int) -> CommunityReport:
        """
        Generate a comprehensive report for a specific community.
        
        Args:
            level_idx: Index of the hierarchy level
            community_id: ID of the community
            
        Returns:
            CommunityReport object with the generated report
        """
        # Get resolution for this level
        resolution = self.community_detector.community_levels[level_idx]
        
        # Create report object
        report = CommunityReport(community_id, level_idx, resolution)
        
        # Get community members
        communities = self.community_detector.get_communities_at_level(level_idx)
        members = communities.get(community_id, [])
        
        if not members:
            report.title = f"Empty Community {community_id}"
            report.summary = "This community has no members."
            return report
        
        # Get node data for members
        member_data = []
        for node_id in members:
            if node_id in self.graph.graph.nodes:
                member_data.append(self.graph.graph.nodes[node_id])
        
        # Get relationships between members
        internal_relationships = []
        for member1 in members:
            for member2 in members:
                if member1 != member2 and self.graph.graph.has_edge(member1, member2):
                    edge_data = self.graph.graph.edges[member1, member2]
                    internal_relationships.append({
                        "source": member1,
                        "target": member2,
                        "type": edge_data.get("type", "Unknown"),
                        "strength": edge_data.get("strength", 0),
                        "description": edge_data.get("description", "")
                    })
        
        # Get relationships to other communities
        external_relationships = []
        other_communities = set()
        
        for member in members:
            # Get all neighbors
            for neighbor in self.graph.graph.neighbors(member):
                if neighbor not in members:
                    # Find which community this neighbor belongs to
                    neighbor_comm = self.community_detector.get_community_at_level(
                        neighbor, level_idx
                    )
                    
                    if neighbor_comm != community_id:
                        # Add to the list of other communities
                        other_communities.add(neighbor_comm)
                        
                        # Get edge data
                        edge_data = self.graph.graph.edges[member, neighbor]
                        
                        external_relationships.append({
                            "source_member": member,
                            "target_community": neighbor_comm,
                            "type": edge_data.get("type", "Unknown"),
                            "strength": edge_data.get("strength", 0),
                            "description": edge_data.get("description", "")
                        })
        
        # Generate entity type statistics
        entity_types = {}
        for node_data in member_data:
            entity_type = node_data.get("type", "Unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Prepare data for GPT-3.5
        community_data = {
            "level_idx": level_idx,
            "community_id": community_id,
            "num_members": len(members),
            "entity_types": entity_types,
            "members": [
                {
                    "id": node_id,
                    "name": self.graph.graph.nodes[node_id].get("name", "Unknown"),
                    "type": self.graph.graph.nodes[node_id].get("type", "Unknown"),
                    "file": self.graph.graph.nodes[node_id].get("source_file", "Unknown"),
                    "description": self.graph.graph.nodes[node_id].get("description", "No description")
                }
                for node_id in members[:10]  # Limit to 10 members to avoid large prompts
            ],
            "internal_relationships": internal_relationships[:10],  # Limit to 10 relationships
            "related_communities": list(other_communities),
            "external_relationships": external_relationships[:10]  # Limit to 10 relationships
        }
        
        # Generate report using GPT-3.5
        full_report = self._generate_report_with_gpt(community_data)
        
        # Parse the report sections
        parsed_report = self._parse_report_sections(full_report)
        
        # Update report object
        report.title = parsed_report.get("title", f"Community {community_id}")
        report.summary = parsed_report.get("summary", "")
        report.key_entities = parsed_report.get("key_entities", [])
        report.related_communities = parsed_report.get("related_communities", [])
        report.architectural_patterns = parsed_report.get("architectural_patterns", [])
        report.full_report = full_report
        
        # Generate embeddings for different aspects of the report
        report.title_embedding = self._generate_embedding(report.title)
        report.summary_embedding = self._generate_embedding(report.summary)
        report.full_report_embedding = self._generate_embedding(full_report)
        
        # Maintain backward compatibility
        report.embedding = report.full_report_embedding
        
        # Set last updated timestamp
        report.last_updated = time.time()
        
        # Store the report
        self.reports[(level_idx, community_id)] = report
        
        return report
    
    def _generate_report_with_gpt(self, community_data: Dict[str, Any]) -> str:
        """
        Generate a report for a community using GPT-3.5.
        
        Args:
            community_data: Dictionary containing community information
            
        Returns:
            String containing the generated report
        """
        # Create prompt for GPT-3.5
        prompt = self._create_report_prompt(community_data)
        
        try:
            # Call GPT-3.5 API
            chat_model = MODEL_CONFIG["models"]["chat"]
            
            response = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": "You are a code analysis assistant that generates comprehensive reports about communities of code entities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract response
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            time.sleep(5)  # Back off on rate limit errors
            return f"Error generating report: {str(e)}"
    
    def _create_report_prompt(self, community_data: Dict[str, Any]) -> str:
        """
        Create a prompt for GPT-3.5 to generate a community report.
        
        Args:
            community_data: Dictionary containing community information
            
        Returns:
            String containing the prompt
        """
        level_names = ["Fine-grained (Level 1)", "Module-level (Level 2)", "Architectural (Level 3)"]
        level_idx = community_data["level_idx"]
        level_name = level_names[level_idx] if level_idx < len(level_names) else f"Level {level_idx+1}"
        
        prompt = f"""Generate a comprehensive report for a {level_name} community of code entities.

COMMUNITY INFORMATION:
- Community ID: {community_data['community_id']}
- Hierarchy Level: {level_name}
- Number of Members: {community_data['num_members']}
- Entity Types: {', '.join([f"{k}: {v}" for k, v in community_data['entity_types'].items()])}

KEY MEMBERS (up to 10):
"""
        
        # Add members
        for member in community_data["members"]:
            prompt += f"- {member['name']} ({member['type']}): {member['description'][:100]}...\n"
        
        # Add internal relationships
        if community_data["internal_relationships"]:
            prompt += "\nINTERNAL RELATIONSHIPS (between members):\n"
            for rel in community_data["internal_relationships"]:
                prompt += f"- {rel['source']} --[{rel['type']}]--> {rel['target']} (Strength: {rel['strength']}): {rel['description'][:50]}...\n"
        
        # Add external relationships
        if community_data["external_relationships"]:
            prompt += "\nEXTERNAL RELATIONSHIPS (to other communities):\n"
            for rel in community_data["external_relationships"]:
                prompt += f"- {rel['source_member']} --[{rel['type']}]--> Community {rel['target_community']} (Strength: {rel['strength']}): {rel['description'][:50]}...\n"
            
            prompt += f"\nRelated Communities: {', '.join([f'Community {c}' for c in community_data['related_communities']])}"
        
        prompt += """

Based on this information, please generate a comprehensive report with the following sections:

1. TITLE: A descriptive name for this community based on its purpose (one line).

2. SUMMARY: A concise summary of what this community of code entities does in the codebase (2-3 sentences).

3. KEY ENTITIES: List the 3-5 most important entities in this community and their roles.

4. RELATED COMMUNITIES: Describe how this community relates to other communities (if any).

5. ARCHITECTURAL PATTERNS: Identify any software architectural patterns or design patterns that might be present in this community.

Format your response with clear section headers (TITLE, SUMMARY, etc.) and be concise but informative.
"""
        
        return prompt
    
    def _parse_report_sections(self, report: str) -> Dict[str, Any]:
        """
        Parse the sections of a generated report.
        
        Args:
            report: String containing the generated report
            
        Returns:
            Dictionary with parsed sections
        """
        sections = {
            "title": "",
            "summary": "",
            "key_entities": [],
            "related_communities": [],
            "architectural_patterns": []
        }
        
        # Split by lines
        lines = report.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if "title:" in lower_line:
                current_section = "title"
                section_content = [line.split(":", 1)[1].strip() if ":" in line else line]
            elif "summary:" in lower_line:
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                
                current_section = "summary"
                section_content = []
            elif "key entities:" in lower_line:
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                
                current_section = "key_entities"
                section_content = []
            elif "related communities:" in lower_line:
                # Save previous section
                if current_section and section_content:
                    if current_section == "key_entities":
                        # Parse key entities as a list
                        sections[current_section] = self._parse_list_items('\n'.join(section_content))
                    else:
                        sections[current_section] = '\n'.join(section_content).strip()
                
                current_section = "related_communities"
                section_content = []
            elif "architectural patterns:" in lower_line:
                # Save previous section
                if current_section and section_content:
                    if current_section == "related_communities":
                        # Parse related communities as a list
                        sections[current_section] = self._parse_list_items('\n'.join(section_content))
                    else:
                        sections[current_section] = '\n'.join(section_content).strip()
                
                current_section = "architectural_patterns"
                section_content = []
            elif current_section:
                # Add line to current section content
                section_content.append(line)
        
        # Save the last section
        if current_section and section_content:
            if current_section in ["key_entities", "related_communities", "architectural_patterns"]:
                # Parse as a list
                sections[current_section] = self._parse_list_items('\n'.join(section_content))
            else:
                sections[current_section] = '\n'.join(section_content).strip()
        
        return sections
    
    def _parse_list_items(self, text: str) -> List[str]:
        """
        Parse list items from text.
        
        Args:
            text: Text potentially containing list items
            
        Returns:
            List of parsed items
        """
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                items.append(line[1:].strip())
            elif line and not any(line.startswith(s) for s in ["1.", "2.", "3.", "4.", "5."]):
                items.append(line)
                
        return items
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the report text using text-embedding-ada-002.
        
        Args:
            text: Text to embed
            
        Returns:
            NumPy array containing the embedding vector
        """
        try:
            # Make sure we're using text-embedding-ada-002
            embedding_model = "text-embedding-ada-002"
            
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
    
    def generate_reports_for_level(self, level_idx: int) -> Dict[int, CommunityReport]:
        """
        Generate reports for all communities at a specific level.
        
        Args:
            level_idx: Index of the hierarchy level
            
        Returns:
            Dictionary mapping community IDs to reports
        """
        # Get communities at this level
        communities = self.community_detector.get_communities_at_level(level_idx)
        
        # Generate report for each community
        level_reports = {}
        
        # Process communities with a progress bar
        print(f"Generating reports for Level {level_idx}...")
        for community_id in tqdm(communities.keys(), desc=f"Level {level_idx} Reports"):
            report = self.generate_report_for_community(level_idx, community_id)
            level_reports[community_id] = report
            
            # Sleep to avoid API rate limits
            time.sleep(1)
        
        return level_reports
    
    def generate_all_reports(self) -> Dict[Tuple[int, int], CommunityReport]:
        """
        Generate reports for all communities at all levels.
        
        Returns:
            Dictionary mapping (level_idx, community_id) to reports
        """
        # Generate reports for each level
        for level_idx in range(len(self.community_detector.community_levels)):
            self.generate_reports_for_level(level_idx)
        
        return self.reports
    
    def find_similar_reports(self,
                           query_text: str,
                           level_idx: Optional[int] = None,
                           aspect: str = "full",
                           top_k: int = 5,
                           threshold: float = 0.7) -> List[Tuple[CommunityReport, float]]:
        """
        Find community reports similar to the query text.
        
        Args:
            query_text: Text to find similar reports for
            level_idx: Optional specific level to search in (None for all levels)
            aspect: Which aspect of the report to compare (title, summary, full)
            top_k: Number of similar reports to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples containing (report, similarity score)
        """
        # Generate embedding for the query text
        query_embedding = self._generate_embedding(query_text)
        
        # Prepare the list of reports to search
        reports_to_search = []
        if level_idx is not None:
            # Only include reports from the specified level
            reports_to_search = [
                report for (idx, _), report in self.reports.items() 
                if idx == level_idx
            ]
        else:
            # Include all reports
            reports_to_search = list(self.reports.values())
        
        # Calculate similarity scores
        similarities = []
        for report in reports_to_search:
            # Get the appropriate embedding for the specified aspect
            report_embedding = report.get_embedding_for_aspect(aspect)
            
            # Skip if no embedding is available
            if report_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, report_embedding)
            
            # Add to results if above threshold
            if similarity >= threshold:
                similarities.append((report, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
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
    
    def save_reports(self, output_dir: str) -> None:
        """
        Save all generated reports to files in an optimized format.
        
        Args:
            output_dir: Directory to save the reports
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create an embeddings directory
        embeddings_dir = os.path.join(output_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Save reports metadata
        metadata = {
            "num_levels": len(self.community_detector.community_levels),
            "resolutions": self.community_detector.community_levels,
            "reports": {},
            "level_info": {}
        }
        
        # Add level-specific information
        for level_idx in range(len(self.community_detector.community_levels)):
            level_communities = self.community_detector.get_communities_at_level(level_idx)
            
            # Add level info to metadata
            if level_idx == 0:
                level_name = "Level 1 (Fine-grained)"
                level_description = "Detailed code components and specific implementations"
            elif level_idx == 1:
                level_name = "Level 2 (Module-level)"
                level_description = "Mid-level code organization and module relationships"
            elif level_idx == 2:
                level_name = "Level 3 (Architectural)"
                level_description = "High-level architecture and system design"
            else:
                level_name = f"Level {level_idx+1}"
                level_description = f"Community level at resolution {self.community_detector.community_levels[level_idx]}"
            
            metadata["level_info"][level_idx] = {
                "name": level_name,
                "description": level_description,
                "resolution": self.community_detector.community_levels[level_idx],
                "community_count": len(level_communities)
            }
        
        # Create index for embeddings
        embeddings_index = {}
        
        # Save individual reports
        for (level_idx, community_id), report in self.reports.items():
            # Save report content
            report_filename = f"report_level{level_idx}_community{community_id}.json"
            report_path = os.path.join(output_dir, report_filename)
            
            # Add to metadata
            metadata["reports"][(level_idx, community_id)] = {
                "filename": report_filename,
                "title": report.title,
                "community_id": community_id,
                "level_idx": level_idx
            }
            
            # Generate a unique ID for the report
            report_id = f"level{level_idx}_community{community_id}"
            
            # Save report content (without embeddings to reduce file size)
            report_content = {
                "community_id": report.community_id,
                "level_idx": report.level_idx,
                "resolution": report.resolution,
                "title": report.title,
                "summary": report.summary,
                "key_entities": report.key_entities,
                "related_communities": report.related_communities,
                "architectural_patterns": report.architectural_patterns,
                "full_report": report.full_report,
                "last_updated": report.last_updated
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_content, f, indent=2)
            
            # Save embeddings separately for efficient retrieval
            for aspect in ["title", "summary", "full_report"]:
                # Get embedding for this aspect
                embedding = None
                if aspect == "title" and report.title_embedding is not None:
                    embedding = report.title_embedding
                elif aspect == "summary" and report.summary_embedding is not None:
                    embedding = report.summary_embedding
                elif aspect == "full_report" and report.full_report_embedding is not None:
                    embedding = report.full_report_embedding
                
                if embedding is not None:
                    # Create a filename for the embedding
                    embedding_id = f"{report_id}_{aspect}"
                    embedding_hash = hashlib.md5(embedding_id.encode()).hexdigest()
                    embedding_filename = f"{embedding_hash}.npy"
                    embedding_path = os.path.join(embeddings_dir, embedding_filename)
                    
                    # Save as numpy array for faster loading
                    np.save(embedding_path, embedding)
                    
                    # Add to index
                    embeddings_index[embedding_id] = embedding_filename
        
        # Save embeddings index
        embeddings_index_path = os.path.join(embeddings_dir, "index.json")
        with open(embeddings_index_path, 'w') as f:
            json.dump(embeddings_index, f, indent=2)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "reports_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_metadata = {
                "num_levels": metadata["num_levels"],
                "resolutions": metadata["resolutions"],
                "level_info": metadata["level_info"],
                "reports": {
                    f"{level_idx}_{community_id}": info 
                    for (level_idx, community_id), info in metadata["reports"].items()
                }
            }
            json.dump(serializable_metadata, f, indent=2)
    
    def load_reports(self, input_dir: str) -> Dict[Tuple[int, int], CommunityReport]:
        """
        Load reports from files.
        
        Args:
            input_dir: Directory containing the reports
            
        Returns:
            Dictionary mapping (level_idx, community_id) to reports
        """
        # Load metadata
        metadata_path = os.path.join(input_dir, "reports_metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if we have the new format with embeddings directory
        embeddings_dir = os.path.join(input_dir, "embeddings")
        has_embeddings_dir = os.path.exists(embeddings_dir)
        
        # Load embeddings index if available
        embeddings_index = {}
        if has_embeddings_dir:
            embeddings_index_path = os.path.join(embeddings_dir, "index.json")
            if os.path.exists(embeddings_index_path):
                with open(embeddings_index_path, 'r') as f:
                    embeddings_index = json.load(f)
        
        # Load individual reports
        reports = {}
        for report_key, report_info in metadata.get("reports", {}).items():
            # Parse level_idx and community_id from the key
            level_idx, community_id = map(int, report_key.split('_'))
            
            # Load report from file
            report_path = os.path.join(input_dir, report_info["filename"])
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                    
                    # Check if we're using the new format (without embeddings in the main file)
                    if not any(key.endswith("_embedding") for key in report_data.keys()):
                        # New format - load embeddings separately
                        report = CommunityReport(
                            community_id=report_data["community_id"],
                            level_idx=report_data["level_idx"],
                            resolution=report_data["resolution"]
                        )
                        
                        # Copy report data
                        report.title = report_data["title"]
                        report.summary = report_data["summary"]
                        report.key_entities = report_data["key_entities"]
                        report.related_communities = report_data["related_communities"]
                        report.architectural_patterns = report_data["architectural_patterns"]
                        report.full_report = report_data["full_report"]
                        report.last_updated = report_data.get("last_updated")
                        
                        # Load embeddings from separate files
                        report_id = f"level{level_idx}_community{community_id}"
                        
                        # Try to load each aspect embedding
                        for aspect in ["title", "summary", "full_report"]:
                            embedding_id = f"{report_id}_{aspect}"
                            if embedding_id in embeddings_index:
                                embedding_path = os.path.join(embeddings_dir, embeddings_index[embedding_id])
                                if os.path.exists(embedding_path):
                                    embedding = np.load(embedding_path)
                                    
                                    if aspect == "title":
                                        report.title_embedding = embedding
                                    elif aspect == "summary":
                                        report.summary_embedding = embedding
                                    elif aspect == "full_report":
                                        report.full_report_embedding = embedding
                                        # Maintain backward compatibility
                                        report.embedding = embedding
                    else:
                        # Old format - embeddings are in the report file
                        report = CommunityReport.from_dict(report_data)
                    
                    reports[(level_idx, community_id)] = report
            else:
                print(f"Report file not found: {report_path}")
        
        self.reports = reports
        return reports


def generate_community_reports(graph: CodeKnowledgeGraph, 
                              community_detector: HierarchicalCommunityDetector,
                              output_dir: Optional[str] = None) -> Dict[Tuple[int, int], CommunityReport]:
    """
    Generate comprehensive reports for all communities in the code knowledge graph.
    
    Args:
        graph: CodeKnowledgeGraph instance
        community_detector: HierarchicalCommunityDetector instance
        output_dir: Optional directory to save the reports
        
    Returns:
        Dictionary mapping (level_idx, community_id) to reports
    """
    # Configure to use text-embedding-ada-002 model
    set_embedding_model("text-embedding-ada-002")
    
    # Create report generator
    generator = CommunityReportGenerator(graph, community_detector)
    
    # Generate reports for all levels
    reports = generator.generate_all_reports()
    
    # Save reports if output directory is provided
    if output_dir:
        generator.save_reports(output_dir)
    
    return reports


def search_community_reports(reports: Dict[Tuple[int, int], CommunityReport],
                           query: str,
                           level_idx: Optional[int] = None,
                           aspect: str = "full") -> List[Tuple[CommunityReport, float]]:
    """
    Search for community reports that match a query string.
    
    Args:
        reports: Dictionary of reports to search (level_idx, community_id) -> report
        query: Query string to search for
        level_idx: Optional level index to restrict search to (None for all levels)
        aspect: Which aspect of reports to search (title, summary, full)
        
    Returns:
        List of tuples containing (report, similarity score) sorted by similarity
    """
    # Create a temporary report generator to use its search functionality
    generator = CommunityReportGenerator(None, None)
    
    # Add reports to the generator
    generator.reports = reports
    
    # Execute search
    return generator.find_similar_reports(
        query_text=query,
        level_idx=level_idx,
        aspect=aspect,
        top_k=10,
        threshold=0.6
    ) 