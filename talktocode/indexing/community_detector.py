import networkx as nx
# Make community import optional
try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    print("Warning: 'community' module (python-louvain) not found. Community detection will be disabled.")
    COMMUNITY_AVAILABLE = False
    # Create a simple placeholder for compatibility
    class DummyCommunity:
        @staticmethod
        def best_partition(graph, **kwargs):
            # Just assign all nodes to the same community (0)
            return {node: 0 for node in graph.nodes()}
    community_louvain = DummyCommunity()
    
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import random
import colorsys

# Import local modules
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import MODEL_CONFIG


class HierarchicalCommunityDetector:
    """
    Implements hierarchical community detection for code knowledge graphs.
    Creates a multi-level representation of code structure.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize the community detector with a graph.
        
        Args:
            graph: NetworkX graph (undirected) for community detection
        """
        # Store the original graph
        self.graph = graph.to_undirected() if graph.is_directed() else graph
        
        # Store hierarchical community results
        self.hierarchical_communities = {}
        self.community_levels = []
        
        # Default resolution parameters for hierarchical detection
        self.default_resolutions = [
            0.5,   # Level 3: High-level architectural groups (coarse)
            1.0,   # Level 2: Module-level communities (medium)
            2.0    # Level 1: Fine-grained communities (detailed)
        ]
        
        # Color schemes for each level
        self.color_schemes = {}
        
        # Track if community detection is available
        self.community_available = COMMUNITY_AVAILABLE
    
    def detect_communities_at_resolution(self, resolution: float) -> Dict[Any, int]:
        """
        Detect communities using the Louvain algorithm at a specific resolution.
        
        Args:
            resolution: Resolution parameter (higher values = more communities)
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        # If community module is not available, use dummy implementation
        if not COMMUNITY_AVAILABLE:
            print(f"Warning: Using simplified community detection at resolution {resolution} as python-louvain is not installed")
            
        # Apply the Louvain algorithm
        communities = community_louvain.best_partition(
            self.graph, 
            weight='weight', 
            resolution=resolution,
            random_state=MODEL_CONFIG["community_detection"]["seed"]
        )
        
        return communities
    
    def detect_hierarchical_communities(self, 
                                       resolutions: Optional[List[float]] = None) -> Dict[float, Dict[Any, int]]:
        """
        Detect communities at multiple resolution levels to create a hierarchy.
        
        Args:
            resolutions: List of resolution parameters (if None, use defaults)
            
        Returns:
            Dictionary mapping resolution levels to community mappings
        """
        if resolutions is None:
            resolutions = self.default_resolutions
        
        # Sort resolutions from coarse to fine (ascending)
        resolutions = sorted(resolutions)
        self.community_levels = resolutions
        
        # Detect communities at each resolution
        for resolution in resolutions:
            communities = self.detect_communities_at_resolution(resolution)
            self.hierarchical_communities[resolution] = communities
            
            # Generate color scheme for this level
            self.generate_color_scheme(resolution)
        
        return self.hierarchical_communities
    
    def get_community_at_level(self, node_id: Any, level_idx: int) -> int:
        """
        Get the community ID for a node at a specific level.
        
        Args:
            node_id: The ID of the node
            level_idx: Index of the level (0 = coarsest, 2 = finest in default settings)
            
        Returns:
            Community ID at the specified level
        """
        if not self.community_levels or level_idx >= len(self.community_levels):
            raise ValueError(f"Invalid level index: {level_idx}")
        
        resolution = self.community_levels[level_idx]
        return self.hierarchical_communities[resolution].get(node_id, -1)
    
    def get_node_community_hierarchy(self, node_id: Any) -> List[int]:
        """
        Get the full hierarchy of communities for a node.
        
        Args:
            node_id: The ID of the node
            
        Returns:
            List of community IDs from coarsest to finest level
        """
        if not self.community_levels:
            return []
        
        return [self.hierarchical_communities[resolution].get(node_id, -1) 
                for resolution in self.community_levels]
    
    def get_community_sizes(self, resolution: float) -> Dict[int, int]:
        """
        Get the size of each community at a specified resolution.
        
        Args:
            resolution: The resolution level
            
        Returns:
            Dictionary mapping community IDs to their sizes
        """
        if resolution not in self.hierarchical_communities:
            raise ValueError(f"No communities detected at resolution {resolution}")
        
        communities = self.hierarchical_communities[resolution]
        sizes = {}
        
        for node_id, community_id in communities.items():
            sizes[community_id] = sizes.get(community_id, 0) + 1
        
        return sizes
    
    def get_communities_at_level(self, level_idx: int) -> Dict[int, List[Any]]:
        """
        Get all communities and their members at a specific level.
        
        Args:
            level_idx: Index of the level (0 = coarsest, 2 = finest in default settings)
            
        Returns:
            Dictionary mapping community IDs to lists of member nodes
        """
        if not self.community_levels or level_idx >= len(self.community_levels):
            raise ValueError(f"Invalid level index: {level_idx}")
        
        resolution = self.community_levels[level_idx]
        communities = self.hierarchical_communities[resolution]
        
        # Group nodes by community
        community_members = {}
        for node_id, community_id in communities.items():
            if community_id not in community_members:
                community_members[community_id] = []
            community_members[community_id].append(node_id)
        
        return community_members
    
    def get_community_hierarchy_mapping(self) -> Dict[Tuple[int, int], Set[int]]:
        """
        Create a mapping between communities at different levels.
        For each coarse community, which fine-grained communities does it contain?
        
        Returns:
            Dictionary mapping (level_idx, community_id) to set of finer-level community IDs
        """
        if len(self.community_levels) < 2:
            return {}
        
        hierarchy_map = {}
        
        # For each pair of adjacent levels
        for i in range(len(self.community_levels) - 1):
            coarse_resolution = self.community_levels[i]
            fine_resolution = self.community_levels[i + 1]
            
            coarse_communities = self.hierarchical_communities[coarse_resolution]
            fine_communities = self.hierarchical_communities[fine_resolution]
            
            # Map coarse communities to fine communities
            for node_id in self.graph.nodes():
                coarse_comm = coarse_communities.get(node_id, -1)
                fine_comm = fine_communities.get(node_id, -1)
                
                key = (i, coarse_comm)
                if key not in hierarchy_map:
                    hierarchy_map[key] = set()
                
                hierarchy_map[key].add(fine_comm)
        
        return hierarchy_map
    
    def generate_color_scheme(self, resolution: float) -> Dict[int, str]:
        """
        Generate a visually pleasing color scheme for communities at a given resolution.
        
        Args:
            resolution: The resolution level
            
        Returns:
            Dictionary mapping community IDs to hex color codes
        """
        if resolution not in self.hierarchical_communities:
            raise ValueError(f"No communities detected at resolution {resolution}")
        
        communities = self.hierarchical_communities[resolution]
        unique_communities = set(communities.values())
        num_communities = len(unique_communities)
        
        # Generate visually distinct colors
        colors = {}
        
        if num_communities <= 10:
            # Use a qualitative colormap for small number of communities
            tableau_colors = list(mcolors.TABLEAU_COLORS.values())
            for i, comm_id in enumerate(unique_communities):
                colors[comm_id] = tableau_colors[i % len(tableau_colors)]
        else:
            # Generate evenly spaced colors in HSV space (more visually distinct)
            for i, comm_id in enumerate(unique_communities):
                hue = i / num_communities
                saturation = 0.7 + 0.3 * (i % 3) / 3  # Slight variation in saturation
                value = 0.8 + 0.2 * ((i // 3) % 2)    # Slight variation in value
                
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors[comm_id] = mcolors.rgb2hex(rgb)
        
        self.color_schemes[resolution] = colors
        return colors
    
    def calculate_modularity(self, resolution: float) -> float:
        """
        Calculate the modularity of the community partition at a given resolution.
        
        Args:
            resolution: The resolution level
            
        Returns:
            Modularity score (higher is better)
        """
        if resolution not in self.hierarchical_communities:
            raise ValueError(f"No communities detected at resolution {resolution}")
        
        return community_louvain.modularity(
            self.hierarchical_communities[resolution],
            self.graph,
            weight='weight'
        )
    
    def find_optimal_resolution(self, min_res: float = 0.1, max_res: float = 3.0, 
                              steps: int = 10) -> float:
        """
        Find an optimal resolution parameter by scanning a range of values.
        
        Args:
            min_res: Minimum resolution value
            max_res: Maximum resolution value
            steps: Number of steps in the scan
            
        Returns:
            Resolution with the highest modularity
        """
        resolutions = np.linspace(min_res, max_res, steps)
        modularities = []
        
        for res in resolutions:
            # Detect communities
            communities = self.detect_communities_at_resolution(res)
            
            # Calculate modularity
            mod = community_louvain.modularity(communities, self.graph, weight='weight')
            modularities.append((res, mod))
        
        # Find resolution with the highest modularity
        optimal_res = max(modularities, key=lambda x: x[1])[0]
        return optimal_res
    
    def find_optimal_hierarchy(self, num_levels: int = 3) -> List[float]:
        """
        Find an optimal set of resolution parameters for hierarchical community detection.
        
        Args:
            num_levels: Number of hierarchy levels to detect
            
        Returns:
            List of optimal resolution parameters
        """
        # Find a first good resolution
        mid_resolution = self.find_optimal_resolution(0.5, 2.0, 10)
        
        if num_levels == 1:
            return [mid_resolution]
        
        # Find a good coarse resolution
        coarse_resolution = self.find_optimal_resolution(0.1, mid_resolution * 0.8, 5)
        
        if num_levels == 2:
            return [coarse_resolution, mid_resolution]
        
        # Find a good fine resolution
        fine_resolution = self.find_optimal_resolution(mid_resolution * 1.2, 3.0, 5)
        
        return [coarse_resolution, mid_resolution, fine_resolution]
    
    def visualize_hierarchy(self, output_file: Optional[str] = None) -> None:
        """
        Visualize the hierarchy of communities.
        
        Args:
            output_file: Optional file path to save the visualization
        """
        if not self.community_levels or len(self.community_levels) < 2:
            raise ValueError("Hierarchical community detection not performed yet")
        
        # Get hierarchy mapping
        hierarchy_map = self.get_community_hierarchy_mapping()
        
        # Create a hierarchical graph
        G = nx.DiGraph()
        
        # Add nodes for each community at each level
        for level_idx, resolution in enumerate(self.community_levels):
            community_sizes = self.get_community_sizes(resolution)
            
            for comm_id, size in community_sizes.items():
                node_id = f"L{level_idx}_C{comm_id}"
                G.add_node(node_id, level=level_idx, comm_id=comm_id, size=size)
        
        # Add edges between levels
        for (level_idx, coarse_comm), fine_comms in hierarchy_map.items():
            coarse_node = f"L{level_idx}_C{coarse_comm}"
            
            for fine_comm in fine_comms:
                fine_node = f"L{level_idx+1}_C{fine_comm}"
                G.add_edge(coarse_node, fine_node)
        
        # Create positions for nodes (hierarchical layout)
        pos = {}
        for node, attrs in G.nodes(data=True):
            level = attrs['level']
            comm_id = attrs['comm_id']
            
            # Position horizontally based on community ID
            # Position vertically based on level (top to bottom)
            pos[node] = (comm_id, -level)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        
        # Draw nodes with sizes based on community size
        node_sizes = [G.nodes[node]['size'] * 100 for node in G.nodes()]
        
        # Color nodes by level
        node_colors = []
        for node in G.nodes():
            level = G.nodes[node]['level']
            comm_id = G.nodes[node]['comm_id']
            
            if level < len(self.community_levels):
                resolution = self.community_levels[level]
                if resolution in self.color_schemes:
                    color = self.color_schemes[resolution].get(comm_id, '#888888')
                    node_colors.append(color)
                else:
                    node_colors.append('#888888')
            else:
                node_colors.append('#888888')
        
        nx.draw_networkx(
            G, pos=pos,
            node_size=node_sizes,
            node_color=node_colors,
            with_labels=True,
            font_size=8,
            edge_color='#888888',
            arrows=False
        )
        
        plt.title("Hierarchical Community Structure")
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def detect_hierarchical_communities(graph: nx.Graph, 
                                  num_levels: int = 3,
                                  auto_detect_resolutions: bool = True,
                                  resolutions: Optional[List[float]] = None) -> HierarchicalCommunityDetector:
    """
    Detect hierarchical communities in a graph.
    
    Args:
        graph: NetworkX graph
        num_levels: Number of hierarchy levels
        auto_detect_resolutions: Whether to automatically find optimal resolutions
        resolutions: Custom resolution parameters (if auto_detect_resolutions is False)
        
    Returns:
        HierarchicalCommunityDetector instance with detected communities
    """
    detector = HierarchicalCommunityDetector(graph)
    
    # Check if community detection is available
    if not COMMUNITY_AVAILABLE:
        print("Warning: Community detection is limited as python-louvain is not installed")
        print("All nodes will be assigned to the same community (0)")
        # Use simple default resolutions when community detection is not available
        if resolutions is None:
            resolutions = detector.default_resolutions
        detector.detect_hierarchical_communities(resolutions)
        return detector
    
    # Normal operation when community detection is available
    if auto_detect_resolutions:
        optimal_resolutions = detector.find_optimal_hierarchy(num_levels)
        detector.detect_hierarchical_communities(optimal_resolutions)
    else:
        if resolutions is None:
            # Use default resolutions
            detector.detect_hierarchical_communities()
        else:
            detector.detect_hierarchical_communities(resolutions)
    
    return detector


def map_nodes_to_hierarchical_communities(graph: nx.Graph, 
                                        node_attribute_prefix: str = "community_level_",
                                        num_levels: int = 3) -> nx.Graph:
    """
    Add community information to node attributes at each level.
    
    Args:
        graph: NetworkX graph
        node_attribute_prefix: Prefix for node attributes to store community IDs
        num_levels: Number of hierarchy levels
        
    Returns:
        Graph with community attributes added to nodes
    """
    # Create a deep copy of the graph to avoid modifying the original
    G = graph.copy()
    
    # Detect hierarchical communities
    detector = detect_hierarchical_communities(G, num_levels=num_levels)
    
    # Annotate nodes with community information
    for node in G.nodes():
        for level_idx in range(len(detector.community_levels)):
            resolution = detector.community_levels[level_idx]
            
            if resolution in detector.hierarchical_communities:
                comm_id = detector.hierarchical_communities[resolution].get(node, -1)
                
                # Add the community ID as a node attribute
                G.nodes[node][f"{node_attribute_prefix}{level_idx}"] = comm_id
                
                # Also add the community color if available
                if resolution in detector.color_schemes:
                    color = detector.color_schemes[resolution].get(comm_id, "#888888")
                    G.nodes[node][f"{node_attribute_prefix}{level_idx}_color"] = color
    
    return G


def get_community_color_mapping(detector: HierarchicalCommunityDetector) -> Dict[str, Dict[int, str]]:
    """
    Get the color mapping for each community level.
    
    Args:
        detector: HierarchicalCommunityDetector instance with detected communities
        
    Returns:
        Dictionary mapping resolution levels to community color dictionaries
    """
    return detector.color_schemes 