import os
import json
import networkx as nx
import plotly.graph_objs as go
# Make community import optional
try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    print("Warning: 'community' module (python-louvain) not found. Community detection will be disabled.")
    COMMUNITY_AVAILABLE = False
    # Create a simple placeholder function for compatibility
    class DummyCommunity:
        @staticmethod
        def best_partition(graph, **kwargs):
            # Just assign all nodes to the same community
            return {node: 0 for node in graph.nodes()}
    community_louvain = DummyCommunity()

from typing import Dict, List, Any, Optional, Tuple
import matplotlib.colors as mcolors
import random

# Import local modules
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import MODEL_CONFIG
from talktocode.indexing.entity_extractor import CodeEntity, extract_code_with_context
from talktocode.indexing.relationship_extractor import Relationship


class CodeKnowledgeGraph:
    """Builds and manages a knowledge graph of code entities and their relationships."""
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.graph = nx.DiGraph()
        self.entity_map = {}  # Maps entity IDs to entities
        self.communities = {}  # Community detection results
        self.community_available = COMMUNITY_AVAILABLE
    
    def generate_entity_id(self, entity: CodeEntity) -> str:
        """
        Generate a unique ID for an entity.
        
        Args:
            entity: The code entity
            
        Returns:
            A unique string ID
        """
        return f"{entity.__class__.__name__}:{entity.name}:{entity.source_file}:{entity.lineno}"
    
    def add_entity(self, entity: CodeEntity, description: Optional[str] = None, 
                  context_lines: int = 5) -> str:
        """
        Add a code entity as a node to the graph.
        
        Args:
            entity: The code entity to add
            description: Optional description of the entity's purpose
            context_lines: Number of context lines for the code snippet
            
        Returns:
            The ID of the added entity node
        """
        # Generate a unique ID
        entity_id = self.generate_entity_id(entity)
        
        # Extract the code snippet
        code_snippet = extract_code_with_context(entity, context_lines)
        
        # Create node attributes
        node_attrs = {
            "name": entity.name,
            "type": entity.__class__.__name__.replace("Entity", ""),
            "source_file": entity.source_file,
            "lineno": entity.lineno,
            "end_lineno": entity.end_lineno,
            "code_snippet": code_snippet,
            "description": description or "No description available",
            # Additional attributes based on entity type
            "entity_data": entity.to_dict()
        }
        
        # Add the node to the graph
        self.graph.add_node(entity_id, **node_attrs)
        
        # Store the entity in the map
        self.entity_map[entity_id] = entity
        
        return entity_id
    
    def add_relationship(self, relationship: Relationship) -> Tuple[str, str]:
        """
        Add a relationship as an edge to the graph.
        
        Args:
            relationship: The relationship to add
            
        Returns:
            A tuple of (source_id, target_id)
        """
        source_id = self.generate_entity_id(relationship.source)
        target_id = self.generate_entity_id(relationship.target)
        
        # Ensure the nodes exist
        if source_id not in self.graph:
            self.add_entity(relationship.source)
        
        if target_id not in self.graph:
            self.add_entity(relationship.target)
        
        # Create edge attributes
        edge_attrs = {
            "type": relationship.relationship_type,
            "strength": relationship.strength,
            "description": relationship.description,
            "weight": relationship.strength / 10.0  # Normalize to 0-1 range
        }
        
        # Add the edge to the graph
        self.graph.add_edge(source_id, target_id, **edge_attrs)
        
        return source_id, target_id
    
    def build_from_entities_and_relationships(self, 
                                            entities: Dict[str, List[CodeEntity]],
                                            relationships: List[Relationship],
                                            entity_purposes: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        """
        Build the knowledge graph from a set of entities and relationships.
        
        Args:
            entities: Dictionary of code entities by type
            relationships: List of relationships between entities
            entity_purposes: Optional dictionary mapping entity IDs to their purposes
        """
        # Add all entities as nodes
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_id = self.generate_entity_id(entity)
                
                # Get the description if available
                description = None
                if entity_purposes and entity_id in entity_purposes:
                    description = entity_purposes[entity_id].get("purpose")
                
                self.add_entity(entity, description)
        
        # Add all relationships as edges
        for relationship in relationships:
            self.add_relationship(relationship)
    
    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Apply community detection to group related entities.
        
        Args:
            resolution: Resolution parameter for the Louvain algorithm
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        # Convert the directed graph to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Apply the Louvain algorithm for community detection
        if not COMMUNITY_AVAILABLE:
            print("Warning: Using simplified community detection as python-louvain is not installed")
            
        self.communities = community_louvain.best_partition(
            undirected, 
            weight='weight', 
            resolution=resolution
        )
        
        # Add community information to node attributes
        for node_id, community_id in self.communities.items():
            self.graph.nodes[node_id]['community'] = community_id
        
        return self.communities
    
    def get_subgraph_for_entity(self, entity_id: str, max_distance: int = 2) -> nx.DiGraph:
        """
        Get a subgraph centered around a specific entity.
        
        Args:
            entity_id: The ID of the central entity
            max_distance: Maximum distance from the central entity
            
        Returns:
            A NetworkX subgraph
        """
        # Get the ego network centered on the entity
        nodes = {entity_id}
        current_distance = 0
        
        while current_distance < max_distance:
            new_nodes = set()
            for node in nodes:
                # Add neighbors
                new_nodes.update(self.graph.successors(node))
                new_nodes.update(self.graph.predecessors(node))
            
            nodes.update(new_nodes)
            current_distance += 1
        
        # Create the subgraph
        return self.graph.subgraph(nodes)
    
    def get_similar_entities(self, entity_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar entities to a given entity based on graph connectivity.
        
        Args:
            entity_id: The ID of the entity
            top_k: Number of similar entities to return
            
        Returns:
            List of tuples containing (entity_id, similarity_score)
        """
        if not nx.has_path(self.graph, entity_id, entity_id):
            # If the entity is not in the graph or not connected to itself
            return []
        
        # Use personalized PageRank to find similar entities
        personalization = {node: 0.0 for node in self.graph.nodes()}
        personalization[entity_id] = 1.0
        
        # Calculate PageRank scores
        pagerank = nx.pagerank(self.graph, alpha=0.85, personalization=personalization)
        
        # Sort entities by score and return top k (excluding the entity itself)
        similar_entities = [(node, score) for node, score in pagerank.items() if node != entity_id]
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        
        return similar_entities[:top_k]
    
    def to_json(self, output_file: Optional[str] = None) -> str:
        """
        Export the graph to a JSON format.
        
        Args:
            output_file: Optional path to save the JSON file
            
        Returns:
            JSON string representation of the graph
        """
        # Convert the graph to a dictionary
        data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, attrs in self.graph.nodes(data=True):
            node_data = dict(attrs)
            node_data["id"] = node_id
            # Convert non-serializable data
            if "entity_data" in node_data:
                node_data["entity_data"] = json.dumps(node_data["entity_data"])
            data["nodes"].append(node_data)
        
        # Add edges
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = dict(attrs)
            edge_data["source"] = source
            edge_data["target"] = target
            data["edges"].append(edge_data)
        
        # Convert to JSON string
        json_data = json.dumps(data, indent=2)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_data)
        
        return json_data
    
    def to_plotly(self, include_labels: bool = True, 
                 use_communities: bool = True,
                 physics_enabled: bool = True) -> Dict:
        """
        Convert the graph to a format suitable for Plotly visualization.
        
        Args:
            include_labels: Whether to include node labels
            use_communities: Whether to color nodes by community
            physics_enabled: Whether to enable physics simulation
            
        Returns:
            Dictionary with Plotly graph data and layout
        """
        # Detect communities if not already done and use_communities is True
        if use_communities and not self.communities:
            self.detect_communities()
        
        # Use a force-directed layout algorithm
        pos = nx.spring_layout(self.graph, dim=3, seed=42)
        
        # Create a color map for communities
        unique_communities = set(self.communities.values()) if self.communities else set()
        colors = list(mcolors.TABLEAU_COLORS.values())
        if len(unique_communities) > len(colors):
            # Generate more colors if needed
            colors.extend([f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(unique_communities) - len(colors))])
        
        community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
        
        # Create node trace
        node_trace = go.Scatter3d(
            x=[pos[node][0] for node in self.graph.nodes()],
            y=[pos[node][1] for node in self.graph.nodes()],
            z=[pos[node][2] for node in self.graph.nodes()],
            mode='markers',
            name='Entities',
            marker=dict(
                size=10,
                color=[community_colors.get(self.communities.get(node, 0), '#888888') 
                       if use_communities else self.graph.nodes[node]['type']
                       for node in self.graph.nodes()],
                line=dict(width=1, color='#888'),
                opacity=0.8
            ),
            text=[f"{self.graph.nodes[node]['name']} ({self.graph.nodes[node]['type']})" 
                  for node in self.graph.nodes()],
            hoverinfo='text'
        )
        
        # Create edge traces (group by relationship type)
        edge_traces = []
        relationship_types = set()
        
        for u, v, data in self.graph.edges(data=True):
            relationship_types.add(data.get('type', 'Unknown'))
            
        for rel_type in relationship_types:
            x_lines, y_lines, z_lines = [], [], []
            edge_texts = []
            
            for u, v, data in self.graph.edges(data=True):
                if data.get('type', 'Unknown') == rel_type:
                    x_lines.extend([pos[u][0], pos[v][0], None])
                    y_lines.extend([pos[u][1], pos[v][1], None])
                    z_lines.extend([pos[u][2], pos[v][2], None])
                    edge_texts.append(f"{data.get('description', '')} (Strength: {data.get('strength', 0)})")
            
            if x_lines:  # Only add the trace if there are edges of this type
                edge_trace = go.Scatter3d(
                    x=x_lines,
                    y=y_lines,
                    z=z_lines,
                    mode='lines',
                    name=rel_type,
                    line=dict(
                        width=2,
                        color=mcolors.TABLEAU_COLORS[f'tab:{rel_type.lower()}'] 
                              if f'tab:{rel_type.lower()}' in mcolors.TABLEAU_COLORS 
                              else '#888',
                        opacity=0.6
                    ),
                    text=edge_texts,
                    hoverinfo='text'
                )
                edge_traces.append(edge_trace)
        
        # Create label trace if requested
        if include_labels:
            label_trace = go.Scatter3d(
                x=[pos[node][0] for node in self.graph.nodes()],
                y=[pos[node][1] for node in self.graph.nodes()],
                z=[pos[node][2] for node in self.graph.nodes()],
                mode='text',
                text=[self.graph.nodes[node]['name'] for node in self.graph.nodes()],
                textposition='top center',
                textfont=dict(
                    family='Arial',
                    size=10,
                    color='#000000'
                ),
                hoverinfo='none'
            )
            traces = edge_traces + [node_trace, label_trace]
        else:
            traces = edge_traces + [node_trace]
        
        # Create layout
        layout = go.Layout(
            title='Code Knowledge Graph',
            titlefont=dict(size=16),
            showlegend=True,
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title='')
            ),
            margin=dict(t=100),
            hovermode='closest'
        )
        
        return {
            'data': traces,
            'layout': layout
        }
    
    def save_plotly_html(self, output_file: str) -> None:
        """
        Save the Plotly graph as an HTML file.
        
        Args:
            output_file: Path to save the HTML file
        """
        plotly_data = self.to_plotly()
        
        # Create the figure
        fig = go.Figure(data=plotly_data['data'], layout=plotly_data['layout'])
        
        # Save as HTML
        with open(output_file, 'w') as f:
            f.write(fig.to_html(include_plotlyjs='cdn'))

    def get_files(self) -> List[str]:
        """
        Get a list of all source files referenced in the graph.
        
        Returns:
            List of file paths
        """
        # Extract all unique source files from node attributes
        files = set()
        for _, attrs in self.graph.nodes(data=True):
            if 'source_file' in attrs and attrs['source_file']:
                files.add(attrs['source_file'])
        
        # Convert to list and sort
        return sorted(list(files))


def build_graph_from_entities_and_relationships(
    entities: Dict[str, List[CodeEntity]],
    relationships: List[Relationship],
    entity_purposes: Optional[Dict[str, Dict[str, str]]] = None
) -> CodeKnowledgeGraph:
    """
    Build a knowledge graph from entities and relationships.
    
    Args:
        entities: Dictionary of code entities by type
        relationships: List of relationships between entities
        entity_purposes: Optional dictionary mapping entity IDs to their purposes
        
    Returns:
        A CodeKnowledgeGraph instance
    """
    graph = CodeKnowledgeGraph()
    graph.build_from_entities_and_relationships(entities, relationships, entity_purposes)
    
    # Apply community detection
    graph.detect_communities(resolution=MODEL_CONFIG["community_detection"]["resolution"])
    
    return graph 