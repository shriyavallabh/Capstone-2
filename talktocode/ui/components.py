import os
import json
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objs as go
from plotly.offline import plot
import networkx as nx
import matplotlib.colors as mcolors
import random

# Import from talktocode
import sys
sys.path.append(".")  # Add the project root to the path
from talktocode.utils.config import UBS_RED, LIGHT_GREY, DARK_GREY, BORDER_GREY, MODEL_CONFIG
from talktocode.indexing.graph_builder import CodeKnowledgeGraph


def render_graph_visualization(
    graph: CodeKnowledgeGraph,
    container_id: str = "graph-container",
    width: str = "100%",
    height: str = "800px",
    view_mode: str = "full",
    focus_community: Optional[int] = None,
    focus_entity: Optional[str] = None,
    dark_mode: bool = False
) -> str:
    """
    Render a knowledge graph visualization using Plotly.
    
    Args:
        graph: CodeKnowledgeGraph instance to visualize
        container_id: HTML ID for the container element
        width: Width of the visualization
        height: Height of the visualization
        view_mode: Visualization mode ('full', 'community', 'entity')
        focus_community: Community ID to focus on (if view_mode is 'community')
        focus_entity: Entity ID to focus on (if view_mode is 'entity')
        dark_mode: Whether to use dark mode styling
        
    Returns:
        HTML string with the Plotly visualization
    """
    # Define UBS color palette for visualization
    ubs_colors = {
        "primary": UBS_RED,
        "secondary": "#000000",  # Black
        "background": DARK_GREY if dark_mode else LIGHT_GREY,
        "text": "#FFFFFF" if dark_mode else "#000000",
        "border": BORDER_GREY,
    }
    
    # Create a layout based on the graph structure
    if not hasattr(graph, 'communities') or not graph.communities:
        graph.detect_communities(resolution=MODEL_CONFIG["community_detection"]["resolution"])
    
    # Generate community colors
    community_ids = set(graph.communities.values())
    community_colors = {}
    
    # Use UBS red for the first community, then generate complementary colors
    community_colors[0] = UBS_RED
    
    # Generate colors for other communities that complement UBS_RED
    for i, comm_id in enumerate(sorted(community_ids)):
        if i == 0:
            continue  # Skip the first one as we already assigned UBS_RED
        
        # Generate a color that's visually distinct but harmonious
        hue = (0.1 + (i * 0.618033988749895) % 1.0)  # Golden ratio distribution
        saturation = 0.7
        value = 0.9
        
        # Convert HSV to RGB
        r, g, b = mcolors.hsv_to_rgb([hue, saturation, value])
        community_colors[comm_id] = mcolors.rgb2hex((r, g, b))
    
    # Filter the graph based on view mode
    if view_mode == "community" and focus_community is not None:
        # Filter to show only nodes in the selected community
        nodes_in_community = [
            node for node, comm in graph.communities.items() 
            if comm == focus_community
        ]
        # Also include immediate neighbors
        neighbors = []
        for node in nodes_in_community:
            neighbors.extend(list(graph.graph.successors(node)))
            neighbors.extend(list(graph.graph.predecessors(node)))
        
        # Create a subgraph
        display_graph = graph.graph.subgraph(set(nodes_in_community + neighbors))
    elif view_mode == "entity" and focus_entity is not None:
        # Get the ego network for the entity
        display_graph = graph.get_subgraph_for_entity(focus_entity, max_distance=2)
    else:
        # Show the full graph
        display_graph = graph.graph
    
    # Create a 3D force-directed layout
    pos = nx.spring_layout(display_graph, dim=3, seed=42)
    
    # Create node trace with hover info
    node_sizes = []
    node_colors = []
    node_texts = []
    
    for node in display_graph.nodes():
        # Node size based on degree centrality
        degree = display_graph.degree(node)
        node_size = 10 + (degree * 2)
        node_sizes.append(node_size)
        
        # Node color based on community
        community_id = graph.communities.get(node, 0)
        color = community_colors.get(community_id, "#888888")
        
        # Highlight the focused entity or community
        if (view_mode == "entity" and node == focus_entity) or \
           (view_mode == "community" and graph.communities.get(node) == focus_community):
            # Make the focused nodes brighter
            node_colors.append(UBS_RED)
        else:
            node_colors.append(color)
        
        # Create detailed hover text
        node_data = display_graph.nodes[node]
        hover_text = f"<b>{node_data.get('name', '')}</b> ({node_data.get('type', '')})<br>"
        hover_text += f"File: {node_data.get('source_file', '')}<br>"
        hover_text += f"Line: {node_data.get('lineno', '')}<br>"
        
        if 'description' in node_data and node_data['description'] != "No description available":
            hover_text += f"<br><i>{node_data.get('description', '')}</i><br>"
        
        if 'code_snippet' in node_data:
            code = node_data.get('code_snippet', '').replace('\n', '<br>')
            hover_text += f"<br><pre>{code}</pre>"
        
        node_texts.append(hover_text)
    
    node_trace = go.Scatter3d(
        x=[pos[node][0] for node in display_graph.nodes()],
        y=[pos[node][1] for node in display_graph.nodes()],
        z=[pos[node][2] for node in display_graph.nodes()],
        mode='markers',
        name='Entities',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color=ubs_colors["border"]),
            opacity=0.8
        ),
        text=node_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor=ubs_colors["background"],
            font=dict(color=ubs_colors["text"]),
            bordercolor=ubs_colors["border"],
        )
    )
    
    # Create edge traces grouped by relationship type
    edge_traces = []
    relationship_types = set()
    
    for u, v, data in display_graph.edges(data=True):
        relationship_types.add(data.get('type', 'Unknown'))
    
    for rel_type in relationship_types:
        x_lines, y_lines, z_lines = [], [], []
        edge_texts = []
        edge_widths = []
        
        for u, v, data in display_graph.edges(data=True):
            if data.get('type', 'Unknown') == rel_type:
                x_lines.extend([pos[u][0], pos[v][0], None])
                y_lines.extend([pos[u][1], pos[v][1], None])
                z_lines.extend([pos[u][2], pos[v][2], None])
                
                # Edge width based on relationship strength
                strength = data.get('strength', 5)
                edge_widths.append(strength / 2)  # Scale down for visual clarity
                
                # Edge hover text
                source_name = display_graph.nodes[u].get('name', u)
                target_name = display_graph.nodes[v].get('name', v)
                edge_text = f"<b>{source_name}</b> → <b>{target_name}</b><br>"
                edge_text += f"Type: {rel_type}<br>"
                edge_text += f"Strength: {strength}/10<br>"
                
                if 'description' in data:
                    edge_text += f"<br>{data.get('description', '')}"
                
                edge_texts.append(edge_text)
        
        if x_lines:  # Only add the trace if there are edges of this type
            edge_trace = go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode='lines',
                name=rel_type,
                line=dict(
                    width=sum(edge_widths) / len(edge_widths) if edge_widths else 2,
                    color=community_colors.get(list(relationship_types).index(rel_type) % len(community_colors), "#888888"),
                    opacity=0.7
                ),
                text=edge_texts,
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor=ubs_colors["background"],
                    font=dict(color=ubs_colors["text"]),
                    bordercolor=ubs_colors["border"],
                )
            )
            edge_traces.append(edge_trace)
    
    # Create the layout
    graph_title = "Code Knowledge Graph"
    if view_mode == "community" and focus_community is not None:
        graph_title = f"Community {focus_community} View"
    elif view_mode == "entity" and focus_entity is not None:
        entity_name = display_graph.nodes[focus_entity].get('name', focus_entity) if focus_entity in display_graph.nodes else focus_entity
        graph_title = f"Entity View: {entity_name}"
    
    layout = go.Layout(
        title=dict(
            text=graph_title,
            font=dict(
                family="Arial, sans-serif",
                size=20,
                color=ubs_colors["text"]
            ),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor=ubs_colors["background"],
            bordercolor=ubs_colors["border"]
        ),
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
            bgcolor=ubs_colors["background"]
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor=ubs_colors["background"],
        plot_bgcolor=ubs_colors["background"],
        hovermode='closest',
        uirevision='true',  # Keep view state on data updates
    )
    
    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
    
    # Add custom JavaScript to hide all Plotly controls
    hide_controls_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Hide the modebar
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length) {
                    const modebar = document.querySelector('.modebar-container');
                    if (modebar) {
                        modebar.style.display = 'none';
                        observer.disconnect();
                    }
                }
            });
        });
        
        observer.observe(document.getElementById('""" + container_id + """'), { 
            childList: true,
            subtree: true 
        });
        
        // Handle container resizing
        function resizeGraph() {
            const container = document.getElementById('""" + container_id + """');
            if (container) {
                const containerWidth = container.offsetWidth;
                const containerHeight = container.offsetHeight;
                
                if (containerWidth > 0 && containerHeight > 0) {
                    const graphDiv = container.querySelector('.js-plotly-plot');
                    if (graphDiv) {
                        Plotly.relayout(graphDiv, {
                            width: containerWidth,
                            height: containerHeight
                        });
                    }
                }
            }
        }
        
        // Initial resize
        resizeGraph();
        
        // Resize on window resize
        window.addEventListener('resize', resizeGraph);
    });
    </script>
    """
    
    # Add custom CSS
    custom_css = """
    <style>
    #""" + container_id + """ {
        width: """ + width + """;
        height: """ + height + """;
        position: relative;
        overflow: hidden;
        background-color: """ + ubs_colors["background"] + """;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide all Plotly controls */
    .modebar {
        display: none !important;
    }
    
    /* Custom tooltip styling */
    .hoverlabel {
        font-family: Arial, sans-serif;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        max-width: 300px;
        overflow-y: auto;
        max-height: 300px;
    }
    </style>
    """
    
    # Generate the HTML with hidden controls
    html = fig.to_html(include_plotlyjs=True, full_html=False)
    
    # Assemble the final HTML
    final_html = f"""
    <div id="{container_id}" class="knowledge-graph-container">
        {html}
    </div>
    {custom_css}
    {hide_controls_js}
    """
    
    return final_html


def render_community_focused_view(
    graph: CodeKnowledgeGraph,
    community_id: int,
    container_id: str = "community-graph-container",
    width: str = "100%",
    height: str = "600px",
    dark_mode: bool = False
) -> str:
    """
    Render a visualization focused on a specific community.
    
    Args:
        graph: CodeKnowledgeGraph instance
        community_id: ID of the community to focus on
        container_id: HTML ID for the container element
        width: Width of the visualization
        height: Height of the visualization
        dark_mode: Whether to use dark mode styling
        
    Returns:
        HTML string with the community-focused visualization
    """
    return render_graph_visualization(
        graph=graph,
        container_id=container_id,
        width=width,
        height=height,
        view_mode="community",
        focus_community=community_id,
        dark_mode=dark_mode
    )


def render_entity_focused_view(
    graph: CodeKnowledgeGraph,
    entity_id: str,
    container_id: str = "entity-graph-container",
    width: str = "100%",
    height: str = "600px",
    dark_mode: bool = False
) -> str:
    """
    Render a visualization focused on a specific entity.
    
    Args:
        graph: CodeKnowledgeGraph instance
        entity_id: ID of the entity to focus on
        container_id: HTML ID for the container element
        width: Width of the visualization
        height: Height of the visualization
        dark_mode: Whether to use dark mode styling
        
    Returns:
        HTML string with the entity-focused visualization
    """
    return render_graph_visualization(
        graph=graph,
        container_id=container_id,
        width=width,
        height=height,
        view_mode="entity",
        focus_entity=entity_id,
        dark_mode=dark_mode
    )


def calculate_responsive_dimensions(container_selector: str) -> str:
    """
    Generate JavaScript to calculate and set responsive dimensions for the graph.
    
    Args:
        container_selector: CSS selector for the container element
        
    Returns:
        JavaScript code as a string
    """
    return """
    <script>
    function setResponsiveDimensions() {
        const container = document.querySelector('""" + container_selector + """');
        if (!container) return;
        
        // Get container dimensions
        const rect = container.getBoundingClientRect();
        
        // Find the Plotly graph div
        const plotDiv = container.querySelector('.js-plotly-plot');
        if (!plotDiv) return;
        
        // Adjust dimensions
        const newWidth = Math.max(rect.width, 300);  // Minimum width of 300px
        const newHeight = Math.max(rect.height, 300);  // Minimum height of 300px
        
        // Update the layout
        Plotly.relayout(plotDiv, {
            width: newWidth,
            height: newHeight
        });
    }
    
    // Set dimensions when page loads
    document.addEventListener('DOMContentLoaded', setResponsiveDimensions);
    
    // Update dimensions when window resizes
    window.addEventListener('resize', setResponsiveDimensions);
    
    // Expose function for external calls (e.g., when container is resized programmatically)
    window.recalculateGraphDimensions = setResponsiveDimensions;
    </script>
    """


def embed_graph_in_container(html_content: str, target_selector: str) -> str:
    """
    Generate JavaScript to embed the graph HTML into a container element.
    
    Args:
        html_content: HTML string with the graph visualization
        target_selector: CSS selector for the target container element
        
    Returns:
        JavaScript code as a string
    """
    # Escape the HTML content for JavaScript string
    escaped_html = html_content.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    
    return f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const targetContainer = document.querySelector('{target_selector}');
        if (targetContainer) {{
            targetContainer.innerHTML = `{escaped_html}`;
            
            // Calculate dimensions after embedding
            if (window.recalculateGraphDimensions) {{
                setTimeout(window.recalculateGraphDimensions, 100);
            }}
        }}
    }});
    </script>
    """


def create_graph_export_button(
    graph_container_id: str,
    button_id: str = "export-graph-button",
    button_text: str = "Export as PNG"
) -> str:
    """
    Create a button to export the graph visualization as an image.
    
    Args:
        graph_container_id: ID of the container element with the graph
        button_id: ID for the button element
        button_text: Text to display on the button
        
    Returns:
        HTML string with the export button
    """
    return f"""
    <button id="{button_id}" class="export-button">
        {button_text}
    </button>
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const exportButton = document.getElementById('{button_id}');
        if (!exportButton) return;
        
        exportButton.addEventListener('click', function() {{
            const graphContainer = document.getElementById('{graph_container_id}');
            if (!graphContainer) return;
            
            const plotDiv = graphContainer.querySelector('.js-plotly-plot');
            if (!plotDiv) return;
            
            Plotly.downloadImage(plotDiv, {{
                format: 'png',
                filename: 'code_knowledge_graph',
                width: plotDiv.offsetWidth * 2,  // Higher resolution
                height: plotDiv.offsetHeight * 2  // Higher resolution
            }});
        }});
        
        // Style the button
        exportButton.style.backgroundColor = '{UBS_RED}';
        exportButton.style.color = 'white';
        exportButton.style.border = 'none';
        exportButton.style.padding = '8px 16px';
        exportButton.style.borderRadius = '4px';
        exportButton.style.cursor = 'pointer';
        exportButton.style.fontFamily = 'Arial, sans-serif';
        exportButton.style.fontSize = '14px';
        exportButton.style.margin = '10px 0';
        
        // Hover effect
        exportButton.addEventListener('mouseover', function() {{
            this.style.backgroundColor = '#cc0000';  // Darker shade of UBS red
        }});
        
        exportButton.addEventListener('mouseout', function() {{
            this.style.backgroundColor = '{UBS_RED}';
        }});
    }});
    </script>
    """ 