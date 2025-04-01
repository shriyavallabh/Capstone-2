import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import random
import networkx as nx
from collections import defaultdict
import heapq
from scipy.spatial.distance import cosine

# Import from talktocode
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import MODEL_CONFIG
from talktocode.indexing.entity_extractor import CodeEntity
from talktocode.indexing.relationship_extractor import Relationship
from talktocode.indexing.graph_builder import CodeKnowledgeGraph
from talktocode.indexing.community_detector import HierarchicalCommunityDetector
from talktocode.indexing.report_generator import CommunityReport

# For OpenAI embeddings and chat
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


class GraphSearchEngine:
    """
    Implements various Graph RAG search strategies for code knowledge graphs.
    
    This class provides three primary search strategies:
    1. Local Search: Entity-level search with graph traversal
    2. Global Search: Community-level search leveraging report summaries
    3. Drift Search: Exploratory search with follow-up questions and reasoning
    """
    
    def __init__(
        self, 
        graph: CodeKnowledgeGraph,
        community_detector: Optional[HierarchicalCommunityDetector] = None,
        community_reports: Optional[Dict[Tuple[int, int], CommunityReport]] = None,
        openai_client: Optional[OpenAI] = None,
        embedding_model: str = MODEL_CONFIG["embedding"]["model"],
        chat_model: str = MODEL_CONFIG["models"]["chat"]
    ):
        """
        Initialize the search engine.
        
        Args:
            graph: The code knowledge graph to search
            community_detector: Optional hierarchical community detector
            community_reports: Optional dictionary of community reports
            openai_client: Optional OpenAI client instance
            embedding_model: OpenAI embedding model to use
            chat_model: OpenAI chat model to use
        """
        self.graph = graph
        self.community_detector = community_detector
        self.community_reports = community_reports
        
        # Create OpenAI client if not provided
        self.client = openai_client or OpenAI()
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        
        # Cache for embeddings
        self.entity_embeddings = {}
        self.report_embeddings = {}
        
        # Default search parameters
        self.default_search_params = {
            "local": {
                "max_hops": 2,
                "top_k_entities": 10,
                "min_similarity": 0.7,
                "max_context_items": 20,
                "include_code": True
            },
            "global": {
                "top_k_communities": 3,
                "min_similarity": 0.6,
                "community_levels": [1, 2],  # Module and architecture levels
                "include_reports": True,
                "report_aspect": "full"  # Which aspect of community reports to use (title, summary, full)
            },
            "drift": {
                "num_hypotheses": 3,
                "max_steps": 3,
                "branching_factor": 2,
                "exploration_width": 5
            }
        }
        
        # Print a message about using text-embedding-ada-002
        print(f"Using embedding model: {self.embedding_model}")
    
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using OpenAI's API with retry logic.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return a zero vector with the correct dimensions
            return np.zeros(MODEL_CONFIG["embedding"]["dimensions"]).tolist()
    
    def _compute_entity_embeddings(self, entities: List[str]) -> Dict[str, List[float]]:
        """
        Compute embeddings for a list of entity IDs.
        
        Args:
            entities: List of entity IDs
            
        Returns:
            Dictionary mapping entity IDs to their embeddings
        """
        embeddings = {}
        for entity_id in entities:
            if entity_id in self.entity_embeddings:
                embeddings[entity_id] = self.entity_embeddings[entity_id]
                continue
                
            # Get entity data
            entity_data = self.graph.graph.nodes[entity_id]
            
            # Create text representation for embedding
            # Include different aspects of the entity for better semantic matching
            text = f"{entity_data.get('name', '')} ({entity_data.get('type', '')})\n"
            
            if 'description' in entity_data and entity_data['description'] != "No description available":
                text += entity_data['description'] + "\n"
                
            if 'code_snippet' in entity_data:
                text += entity_data['code_snippet']
            
            # Get embedding
            embedding = self._get_embedding(text)
            
            # Cache and return
            self.entity_embeddings[entity_id] = embedding
            embeddings[entity_id] = embedding
            
        return embeddings
    
    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to numpy arrays if they're not already
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
            
        # Calculate cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        return max(0.0, min(1.0, similarity))  # Ensure range 0-1
    
    def _find_similar_entities(
        self, 
        query_embedding: List[float], 
        top_k: int = 10, 
        min_similarity: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find entities similar to the query embedding.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of top entities to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (entity_id, similarity) tuples, sorted by similarity
        """
        # Get all entity IDs
        entity_ids = list(self.graph.graph.nodes())
        
        # Compute embeddings for all entities (uses cache)
        entity_embeddings = self._compute_entity_embeddings(entity_ids)
        
        # Compute similarities
        similarities = []
        for entity_id, embedding in entity_embeddings.items():
            similarity = self._compute_similarity(query_embedding, embedding)
            if similarity >= min_similarity:
                similarities.append((entity_id, similarity))
        
        # Return top-k
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _traverse_graph(
        self, 
        start_entities: List[str], 
        max_hops: int = 2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Traverse the graph starting from seed entities up to max_hops.
        
        Args:
            start_entities: List of starting entity IDs
            max_hops: Maximum number of hops
            
        Returns:
            Dictionary mapping entity IDs to their metadata including distance
        """
        visited = {}  # entity_id -> metadata
        queue = [(entity_id, 0) for entity_id in start_entities]  # (entity_id, distance)
        
        while queue:
            entity_id, distance = queue.pop(0)
            
            # Skip if already visited with shorter distance
            if entity_id in visited and visited[entity_id]["distance"] <= distance:
                continue
                
            # Get entity data
            entity_data = self.graph.graph.nodes[entity_id]
            
            # Add to visited
            visited[entity_id] = {
                "id": entity_id,
                "name": entity_data.get("name", ""),
                "type": entity_data.get("type", ""),
                "source_file": entity_data.get("source_file", ""),
                "lineno": entity_data.get("lineno", ""),
                "description": entity_data.get("description", "No description available"),
                "code_snippet": entity_data.get("code_snippet", ""),
                "distance": distance,
                "community": entity_data.get("community", None)
            }
            
            # Stop if max distance reached
            if distance >= max_hops:
                continue
                
            # Add neighbors to queue
            for neighbor in self.graph.graph.successors(entity_id):
                edge_data = self.graph.graph.edges[entity_id, neighbor]
                queue.append((
                    neighbor, 
                    distance + 1,  # Increment distance
                ))
                
            for neighbor in self.graph.graph.predecessors(entity_id):
                edge_data = self.graph.graph.edges[neighbor, entity_id]
                queue.append((
                    neighbor, 
                    distance + 1,  # Increment distance
                ))
        
        return visited
    
    def _rank_entities(
        self, 
        entities: Dict[str, Dict[str, Any]], 
        similarities: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Rank entities based on similarity, relationship strength, and distance.
        
        Args:
            entities: Dictionary of entity data from traversal
            similarities: Dictionary mapping entity IDs to similarity scores
            
        Returns:
            List of entity data dictionaries, sorted by rank
        """
        ranked_entities = []
        
        for entity_id, entity_data in entities.items():
            # Base score is similarity if available, else distance-based fallback
            base_score = similarities.get(entity_id, 1.0 / (1 + entity_data["distance"]))
            
            # Decay score based on distance
            distance_factor = 1.0 / (1 + entity_data["distance"])
            
            # Final score
            final_score = base_score * distance_factor
            
            # Add score to entity data
            entity_data["score"] = final_score
            ranked_entities.append(entity_data)
        
        # Sort by score
        return sorted(ranked_entities, key=lambda x: x["score"], reverse=True)
    
    def _collect_context(
        self, 
        ranked_entities: List[Dict[str, Any]], 
        max_items: int = 20,
        include_code: bool = True
    ) -> Dict[str, Any]:
        """
        Collect context from ranked entities.
        
        Args:
            ranked_entities: List of ranked entity data
            max_items: Maximum number of context items to include
            include_code: Whether to include code snippets
            
        Returns:
            Dictionary with formatted context data
        """
        # Select top entities for context
        selected_entities = ranked_entities[:max_items]
        
        # Group entities by file
        files = defaultdict(list)
        for entity in selected_entities:
            files[entity["source_file"]].append(entity)
        
        # Sort entities within each file by line number
        for file_path in files:
            files[file_path].sort(key=lambda x: x["lineno"])
        
        # Format context
        context = {
            "entities": selected_entities,
            "files": dict(files),
            "summary": f"Found {len(selected_entities)} relevant code elements across {len(files)} files."
        }
        
        # Build text context for LLM
        text_context = []
        for entity in selected_entities:
            entry = f"--- {entity['name']} ({entity['type']}) ---\n"
            entry += f"File: {entity['source_file']}, Line: {entity['lineno']}\n"
            
            if entity['description'] != "No description available":
                entry += f"Description: {entity['description']}\n"
                
            if include_code and entity['code_snippet']:
                entry += f"Code:\n{entity['code_snippet']}\n"
                
            text_context.append(entry)
        
        context["text"] = "\n\n".join(text_context)
        
        return context
    
    def local_search(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a local search in the knowledge graph around semantically similar entities.
        
        This search:
        1. Finds semantically similar entities to the query
        2. Traverses the graph to collect related entities up to N hops away
        3. Ranks the results based on relationship strength and relevance
        4. Collects context including code snippets and descriptions
        
        Args:
            query: The search query
            params: Optional parameters to override defaults
                - max_hops: Maximum hops for graph traversal
                - top_k_entities: Number of initial entities to consider
                - min_similarity: Minimum similarity threshold
                - max_context_items: Maximum number of context items to include
                - include_code: Whether to include code snippets
                
        Returns:
            Dictionary with search results and context
        """
        # Merge parameters with defaults
        p = self.default_search_params["local"].copy()
        if params:
            p.update(params)
            
        # Get query embedding using text-embedding-ada-002
        print(f"Generating embedding for query: '{query}'")
        query_embedding = self._get_embedding(query)
        
        # Find similar entities using embeddings
        print(f"Finding semantically similar entities (top {p['top_k_entities']}, min similarity {p['min_similarity']})")
        similar_entities = self._find_similar_entities(
            query_embedding,
            top_k=p["top_k_entities"],
            min_similarity=p["min_similarity"]
        )
        
        # Early return if no similar entities found
        if not similar_entities:
            return {
                "query": query,
                "entities": [],
                "context": {"text": "No relevant code entities found."},
                "status": "no_results"
            }
            
        # Get similarities as dict for ranking
        similarities = {entity_id: similarity for entity_id, similarity in similar_entities}
        
        # Print top entities found and their similarities
        print(f"Found {len(similar_entities)} similar entities:")
        for entity_id, similarity in similar_entities[:3]:  # Print top 3
            entity_name = self.graph.graph.nodes[entity_id].get('name', 'Unknown')
            entity_type = self.graph.graph.nodes[entity_id].get('type', 'Unknown')
            print(f"  - {entity_name} ({entity_type}): similarity = {similarity:.3f}")
        
        # Get starting entities
        start_entities = [entity_id for entity_id, _ in similar_entities]
        
        # Traverse graph
        print(f"Traversing graph up to {p['max_hops']} hops from {len(start_entities)} seed entities")
        traversed_entities = self._traverse_graph(
            start_entities=start_entities,
            max_hops=p["max_hops"]
        )
        
        # Rank entities using embeddings-based similarity scores
        ranked_entities = self._rank_entities(
            entities=traversed_entities,
            similarities=similarities
        )
        
        # Collect context
        context = self._collect_context(
            ranked_entities=ranked_entities,
            max_items=p["max_context_items"],
            include_code=p["include_code"]
        )
        
        # Return results
        results = {
            "query": query,
            "entities": ranked_entities,
            "context": context,
            "status": "success"
        }
        
        return results
        
    def _compute_report_embeddings(self) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
        """
        Compute embeddings for all community reports.
        
        Returns:
            Dictionary mapping (level_idx, community_id) to a dictionary of aspect embeddings
        """
        # Check if community reports are available
        if not self.community_reports:
            return {}
            
        # Compute embeddings
        embeddings = {}
        for (level_idx, community_id), report in self.community_reports.items():
            if (level_idx, community_id) in self.report_embeddings:
                embeddings[(level_idx, community_id)] = self.report_embeddings[(level_idx, community_id)]
                continue
            
            # Get embeddings for different aspects of the report
            aspect_embeddings = {}
            
            # Use pre-computed embeddings from the report if available
            if report.title_embedding is not None:
                aspect_embeddings["title"] = report.title_embedding.tolist()
            else:
                aspect_embeddings["title"] = self._get_embedding(report.title)
                
            if report.summary_embedding is not None:
                aspect_embeddings["summary"] = report.summary_embedding.tolist()
            else:
                aspect_embeddings["summary"] = self._get_embedding(report.summary)
                
            if report.full_report_embedding is not None:
                aspect_embeddings["full"] = report.full_report_embedding.tolist()
            else:
                # Create text representation for full report embedding
                text = f"Community {community_id} (Level {level_idx})\n"
                text += f"Title: {report.title}\n"
                text += f"Summary: {report.summary}\n"
                text += f"Key Entities: {', '.join(report.key_entities)}\n"
                
                # Add related communities and architectural patterns if available
                if hasattr(report, 'related_communities') and report.related_communities:
                    if isinstance(report.related_communities, list):
                        text += f"Related Communities: {', '.join(report.related_communities)}\n"
                    else:
                        text += f"Related Communities: {report.related_communities}\n"
                
                if hasattr(report, 'architectural_patterns') and report.architectural_patterns:
                    if isinstance(report.architectural_patterns, list):
                        text += f"Architectural Patterns: {', '.join(report.architectural_patterns)}\n"
                    else:
                        text += f"Architectural Patterns: {report.architectural_patterns}\n"
                
                aspect_embeddings["full"] = self._get_embedding(text)
            
            # Cache and return
            self.report_embeddings[(level_idx, community_id)] = aspect_embeddings
            embeddings[(level_idx, community_id)] = aspect_embeddings
        
        return embeddings
    
    def _find_relevant_communities(
        self, 
        query_embedding: List[float],
        community_levels: List[int] = [1, 2],
        top_k: int = 3,
        min_similarity: float = 0.6,
        aspect: str = "full"
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Find communities relevant to the query.
        
        Args:
            query_embedding: Query embedding
            community_levels: List of community levels to consider
            top_k: Number of top communities to return
            min_similarity: Minimum similarity threshold
            aspect: Which aspect of community reports to use (title, summary, full)
            
        Returns:
            List of ((level_idx, community_id), similarity) tuples
        """
        # Check if community detector and reports are available
        if not self.community_detector or not self.community_reports:
            return []
            
        # Compute report embeddings
        report_embeddings = self._compute_report_embeddings()
        if not report_embeddings:
            return []
            
        # Compute similarities
        similarities = []
        for (level_idx, community_id), aspect_embeddings in report_embeddings.items():
            # Filter by community level
            if level_idx not in community_levels:
                continue
                
            # Use the appropriate aspect embedding
            if aspect in aspect_embeddings:
                embedding = aspect_embeddings[aspect]
            else:
                # Fall back to full report if the requested aspect is not available
                embedding = aspect_embeddings.get("full", None)
                
            if embedding is None:
                continue
                
            # Compute similarity
            similarity = self._compute_similarity(query_embedding, embedding)
            
            # Add if above threshold
            if similarity >= min_similarity:
                similarities.append(((level_idx, community_id), similarity))
        
        # Sort and return top-k
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _get_community_nodes(
        self,
        level_idx: int,
        community_id: int
    ) -> List[str]:
        """
        Get all nodes in a community.
        
        Args:
            level_idx: Level index
            community_id: Community ID
            
        Returns:
            List of node IDs in the community
        """
        if not self.community_detector:
            return []
            
        community_nodes = []
        
        # Get community mapping for this level
        community_mapping = self.community_detector.get_node_to_community_mapping(level_idx)
        
        # Get nodes in this community
        for node_id, node_community_id in community_mapping.items():
            if node_community_id == community_id:
                community_nodes.append(node_id)
        
        return community_nodes
    
    def _extract_community_context(
        self,
        relevant_communities: List[Tuple[Tuple[int, int], float]],
        include_reports: bool = True
    ) -> Dict[str, Any]:
        """
        Extract context from relevant communities.
        
        Args:
            relevant_communities: List of ((level_idx, community_id), similarity) tuples
            include_reports: Whether to include community reports
            
        Returns:
            Dictionary with community context
        """
        community_context = {
            "communities": [],
            "text": ""
        }
        
        text_context = []
        
        for (level_idx, community_id), similarity in relevant_communities:
            # Get community report
            report = self.community_reports.get((level_idx, community_id))
            if not report:
                continue
                
            # Get community nodes
            community_nodes = self._get_community_nodes(level_idx, community_id)
            
            # Build community info
            community_info = {
                "level": level_idx,
                "id": community_id,
                "title": report.title,
                "summary": report.summary,
                "key_entities": report.key_entities,
                "external_relationships": report.external_relationships,
                "patterns": report.patterns,
                "similarity": similarity,
                "node_count": len(community_nodes)
            }
            
            community_context["communities"].append(community_info)
            
            # Add to text context
            entry = f"--- Community {community_id} (Level {level_idx}) ---\n"
            entry += f"Title: {report.title}\n"
            entry += f"Summary: {report.summary}\n"
            entry += f"Key Entities: {', '.join(report.key_entities)}\n"
            
            if report.external_relationships:
                entry += f"External Relationships: {report.external_relationships}\n"
                
            if report.patterns:
                entry += f"Architectural Patterns: {report.patterns}\n"
                
            text_context.append(entry)
        
        # Set text context
        community_context["text"] = "\n\n".join(text_context)
        
        return community_context
    
    def global_search(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a global search using community-level information.
        
        This search:
        1. Matches the query against community reports
        2. Retrieves relevant community-level information
        3. Scores communities based on relevance to the query
        4. Summarizes the high-level code organization
        
        Args:
            query: The search query
            params: Optional parameters to override defaults
                - top_k_communities: Number of communities to return
                - min_similarity: Minimum similarity threshold
                - community_levels: List of community levels to consider
                - include_reports: Whether to include community reports
                - report_aspect: Which aspect of community reports to use (title, summary, full)
                
        Returns:
            Dictionary with search results and context
        """
        # Check if community detector and reports are available
        if not self.community_detector or not self.community_reports:
            return {
                "query": query,
                "communities": [],
                "context": {"text": "No community information available."},
                "status": "no_community_info"
            }
            
        # Merge parameters with defaults
        p = self.default_search_params["global"].copy()
        if params:
            p.update(params)
            
        # Get query embedding using text-embedding-ada-002
        print(f"Generating embedding for query: '{query}'")
        query_embedding = self._get_embedding(query)
        
        # Find relevant communities using embeddings
        print(f"Finding relevant communities (levels {p['community_levels']}, aspect '{p['report_aspect']}')")
        relevant_communities = self._find_relevant_communities(
            query_embedding=query_embedding,
            community_levels=p["community_levels"],
            top_k=p["top_k_communities"],
            min_similarity=p["min_similarity"],
            aspect=p["report_aspect"]
        )
        
        # Early return if no relevant communities found
        if not relevant_communities:
            return {
                "query": query,
                "communities": [],
                "context": {"text": "No relevant communities found."},
                "status": "no_results"
            }
            
        # Print top communities found and their similarities
        print(f"Found {len(relevant_communities)} relevant communities:")
        for (level_idx, community_id), similarity in relevant_communities[:3]:  # Print top 3
            report = self.community_reports.get((level_idx, community_id))
            if report:
                print(f"  - Level {level_idx}, Community {community_id}: '{report.title}', similarity = {similarity:.3f}")
            
        # Extract community context
        community_context = self._extract_community_context(
            relevant_communities=relevant_communities,
            include_reports=p["include_reports"]
        )
        
        # Return results
        results = {
            "query": query,
            "communities": [c for c in community_context["communities"]],
            "context": community_context,
            "status": "success"
        }
        
        return results
    
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _generate_chat_completion(
        self, 
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Generate a chat completion using OpenAI's API with retry logic.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated response text
        """
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages
        )
        return response.choices[0].message.content
    
    def _generate_hypothetical_answers(
        self, 
        query: str, 
        initial_context: Dict[str, Any],
        num_hypotheses: int = 3
    ) -> List[str]:
        """
        Generate hypothetical answers to guide exploration.
        
        Args:
            query: The search query
            initial_context: Context from initial search
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of hypothetical answers
        """
        # Create messages for the chat API
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert code analyst. Given a query about a codebase "
                    "and some initial context, generate multiple hypothetical answers "
                    "that could guide further exploration. Each hypothesis should be "
                    "specific enough to verify with targeted code searches."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Initial context from codebase:\n{initial_context['text']}\n\n"
                    f"Generate {num_hypotheses} distinct hypothetical answers to the query. "
                    f"Each hypothesis should be concise (1-2 sentences) and make a specific claim "
                    f"that could be verified by further code exploration. Format your response "
                    f"as a numbered list with each hypothesis on a new line."
                )
            }
        ]
        
        # Generate hypotheses
        response = self._generate_chat_completion(messages)
        
        # Parse response
        hypotheses = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove any numbering or bullet points
                hypothesis = line.split(".", 1)[-1].split(":", 1)[-1].strip()
                hypotheses.append(hypothesis)
                
                if len(hypotheses) >= num_hypotheses:
                    break
        
        # Ensure we have the requested number of hypotheses
        while len(hypotheses) < num_hypotheses:
            hypotheses.append(f"The code might implement {query} in a different way not covered by initial context.")
            
        return hypotheses
    
    def _generate_follow_up_questions(
        self, 
        query: str, 
        hypothesis: str,
        previous_context: Dict[str, Any],
        num_questions: int = 2
    ) -> List[str]:
        """
        Generate follow-up questions based on a hypothesis.
        
        Args:
            query: The original search query
            hypothesis: The current hypothesis
            previous_context: Context from previous searches
            num_questions: Number of follow-up questions to generate
            
        Returns:
            List of follow-up questions
        """
        # Create messages for the chat API
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert code analyst. Given a query about a codebase, "
                    "a current hypothesis, and context from previous searches, generate "
                    "specific follow-up questions that could help verify or refine the hypothesis."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Current hypothesis: {hypothesis}\n\n"
                    f"Context from previous searches:\n{previous_context['text']}\n\n"
                    f"Generate {num_questions} specific follow-up questions that would help verify "
                    f"or refine this hypothesis. These questions should be directly answerable by "
                    f"searching the codebase. Format your response as a numbered list with each "
                    f"question on a new line."
                )
            }
        ]
        
        # Generate questions
        response = self._generate_chat_completion(messages)
        
        # Parse response
        questions = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove any numbering or bullet points
                question = line.split(".", 1)[-1].split(":", 1)[-1].strip()
                if question.endswith("?"):
                    questions.append(question)
                else:
                    questions.append(question + "?")
                    
                if len(questions) >= num_questions:
                    break
        
        # Ensure we have the requested number of questions
        while len(questions) < num_questions:
            questions.append(f"What parts of the code are most related to {query}?")
            
        return questions
    
    def _search_for_question(
        self, 
        question: str
    ) -> Dict[str, Any]:
        """
        Search for a specific question using local search.
        
        Args:
            question: The question to search for
            
        Returns:
            Search results
        """
        # Use local search with slightly modified parameters
        search_params = {
            "max_hops": 1,  # Focused search
            "top_k_entities": 5,  # Fewer initial entities
            "min_similarity": 0.65,  # Slightly lower threshold
            "max_context_items": 10,  # Fewer context items
            "include_code": True
        }
        
        # Generate embedding for this specific question
        print(f"Searching follow-up question: '{question}'")
        
        return self.local_search(question, params=search_params)
    
    def _aggregate_search_paths(
        self, 
        search_paths: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Aggregate results from different search paths.
        
        Args:
            search_paths: List of search paths, each containing search results
            
        Returns:
            Aggregated results and context
        """
        # Collect all entities and their scores
        all_entities = {}
        
        for path in search_paths:
            for result in path:
                for entity in result.get("entities", []):
                    entity_id = entity["id"]
                    
                    if entity_id not in all_entities:
                        all_entities[entity_id] = entity.copy()
                        all_entities[entity_id]["path_count"] = 1
                        all_entities[entity_id]["merged_score"] = entity["score"]
                    else:
                        # Update score
                        existing = all_entities[entity_id]
                        existing["path_count"] += 1
                        existing["merged_score"] = max(existing["merged_score"], entity["score"])
        
        # Rank entities
        ranked_entities = sorted(
            all_entities.values(), 
            key=lambda x: (x["path_count"], x["merged_score"]), 
            reverse=True
        )
        
        # Collect context
        aggregated_context = self._collect_context(
            ranked_entities=ranked_entities,
            max_items=20,
            include_code=True
        )
        
        # Return results
        return {
            "entities": ranked_entities,
            "context": aggregated_context,
            "paths": len(search_paths),
            "total_steps": sum(len(path) for path in search_paths)
        }
    
    def drift_search(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a drift search that explores the codebase through multiple steps.
        
        This search:
        1. Generates hypothetical answers to guide exploration
        2. Creates follow-up questions based on initial findings
        3. Implements multi-step reasoning via graph traversal
        4. Aggregates results from different search paths
        
        Args:
            query: The search query
            params: Optional parameters to override defaults
                - num_hypotheses: Number of hypotheses to generate
                - max_steps: Maximum number of search steps per path
                - branching_factor: Number of branches at each step
                - exploration_width: Number of entities to consider at each step
                
        Returns:
            Dictionary with search results, reasoning paths, and context
        """
        # Merge parameters with defaults
        p = self.default_search_params["drift"].copy()
        if params:
            p.update(params)
        
        # Get query embedding using text-embedding-ada-002
        print(f"Generating embedding for drift search query: '{query}'")
        query_embedding = self._get_embedding(query)
            
        # Start with a local search to get initial context (uses the embedding)
        print("Starting with local search to establish initial context")
        initial_results = self.local_search(query)
        
        # Early return if initial search finds nothing
        if initial_results["status"] == "no_results":
            return {
                "query": query,
                "entities": [],
                "context": {"text": "No relevant code entities found."},
                "reasoning": [],
                "status": "no_results"
            }
        
        # Generate hypothetical answers
        print(f"Generating {p['num_hypotheses']} hypothetical answers to guide exploration")
        hypotheses = self._generate_hypothetical_answers(
            query=query,
            initial_context=initial_results["context"],
            num_hypotheses=p["num_hypotheses"]
        )
        
        # Store the reasoning paths
        reasoning_paths = []
        search_paths = []
        
        # Explore each hypothesis
        for hypothesis_idx, hypothesis in enumerate(hypotheses):
            print(f"Exploring hypothesis {hypothesis_idx+1}/{len(hypotheses)}: {hypothesis[:100]}...")
            
            # Create a reasoning path for this hypothesis
            reasoning_path = [
                {
                    "step": 0,
                    "type": "hypothesis",
                    "content": hypothesis,
                    "context": initial_results["context"]["text"]
                }
            ]
            
            # Create a search path for this hypothesis
            search_path = [initial_results]
            
            # Iteratively ask follow-up questions
            current_context = initial_results["context"]
            
            for step in range(1, p["max_steps"] + 1):
                # Generate follow-up questions
                follow_up_questions = self._generate_follow_up_questions(
                    query=query,
                    hypothesis=hypothesis,
                    previous_context=current_context,
                    num_questions=p["branching_factor"]
                )
                
                step_results = []
                
                # Search for each question
                for question_idx, question in enumerate(follow_up_questions):
                    # Search for this question using embeddings
                    question_results = self._search_for_question(question)
                    
                    # Add to reasoning path
                    reasoning_path.append({
                        "step": step,
                        "type": "question",
                        "content": question,
                        "context": question_results["context"]["text"]
                    })
                    
                    # Add to search results for this step
                    step_results.append(question_results)
                
                # Update current context (combine all step results)
                combined_text = "\n\n".join([r["context"]["text"] for r in step_results])
                current_context = {"text": combined_text}
                
                # Add step results to search path
                search_path.extend(step_results)
                
                # Add analysis to reasoning path
                reasoning_path.append({
                    "step": step,
                    "type": "analysis",
                    "content": f"Analysis of findings after step {step}",
                    "context": current_context["text"]
                })
            
            # Add completed paths
            reasoning_paths.append(reasoning_path)
            search_paths.append(search_path)
        
        # Aggregate results from all search paths
        aggregated_results = self._aggregate_search_paths(search_paths)
        
        # Generate a summary of findings
        summary_prompt = (
            f"Query: {query}\n\n"
            f"Context from multiple search paths:\n{aggregated_results['context']['text']}\n\n"
            f"Provide a concise summary of the findings, focusing on how the code implements or addresses the query."
        )
        
        summary_message = [
            {"role": "system", "content": "You are an expert code analyst. Summarize findings from code search results."},
            {"role": "user", "content": summary_prompt}
        ]
        
        summary = self._generate_chat_completion(summary_message)
        
        # Return combined results
        return {
            "query": query,
            "entities": aggregated_results["entities"],
            "context": aggregated_results["context"],
            "reasoning": reasoning_paths,
            "summary": summary,
            "paths": aggregated_results["paths"],
            "total_steps": aggregated_results["total_steps"],
            "status": "success"
        }
    

# Usage example
def search_codebase(
    query: str,
    graph: CodeKnowledgeGraph,
    community_detector: Optional[HierarchicalCommunityDetector] = None,
    community_reports: Optional[Dict[Tuple[int, int], CommunityReport]] = None,
    strategy: str = "local",
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search a codebase using one of the Graph RAG strategies.
    
    Args:
        query: Search query
        graph: Code knowledge graph
        community_detector: Optional hierarchical community detector
        community_reports: Optional dictionary of community reports
        strategy: Search strategy ('local', 'global', or 'drift')
        params: Optional parameters for the search
        
    Returns:
        Search results
    """
    # Initialize search engine
    engine = GraphSearchEngine(
        graph=graph,
        community_detector=community_detector,
        community_reports=community_reports
    )
    
    # Execute search
    if strategy == "local":
        return engine.local_search(query, params)
    elif strategy == "global":
        return engine.global_search(query, params)
    elif strategy == "drift":
        return engine.drift_search(query, params)
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")
    