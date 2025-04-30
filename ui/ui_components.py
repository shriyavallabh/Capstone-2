#!/usr/bin/env python3
"""
UI Components for the Talk to Code application.

This module contains pure Python Streamlit components for creating
a consistent and user-friendly interface for the Talk to Code application.
"""

import os
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pathlib import Path
import tempfile
import zipfile
import base64
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from plotly.subplots import make_subplots
from typing import Dict, Any, Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime
from talktocode.utils.config import COMPANY_COLORS

# Base CSS Styles
def inject_base_css():
    """Inject base CSS styles for consistent UI appearance."""
    st.markdown("""
    <style>
        /* Base styling */
        .app-header {
            color: #EC0016;
            font-weight: bold;
        }
        .app-subheader {
            color: #0205A8;
        }
        
        /* Status boxes */
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #9A9CFF;
            color: #000066;
        }
        .success-box {
            background-color: #66BB6A;
            color: #1B5E20;
        }
        .warning-box {
            background-color: #FFCA28;
            color: #F57F17;
        }
        .error-box {
            background-color: #FF6D6A;
            color: #B30012;
        }
        
        /* Chat styling */
        .chat-container {
            margin-top: 2rem;
            border-top: 1px solid #CCCCCC;
            padding-top: 1rem;
        }
        
        /* Hide unwanted elements */
        .stPlotlyChart .modebar {
            display: none !important;
        }
        .js-plotly-plot .plotly .modebar {
            display: none !important;
        }
        .matplotlib-controls, .reportview-container .main footer {
            display: none !important;
        }
        
        /* Sidebar toggle button */
        .sidebar-toggle-btn {
            border-radius: 20px;
            padding: 0.5rem 1rem;
            background-color: #EC0016;
            color: white;
            border: none;
            cursor: pointer;
            text-align: center;
            margin-bottom: 1rem;
            transition: background-color 0.3s;
        }
        .sidebar-toggle-btn:hover {
            background-color: #B30012;
        }
        
        /* Adjust sidebar when collapsed */
        .sidebar-collapsed .css-1d391kg, .sidebar-collapsed .css-1lcbmhc {
            width: 0 !important;
            margin-left: -21rem;
            visibility: hidden;
        }
        .sidebar-collapsed .block-container {
            padding-left: 1rem;
            max-width: 100%;
        }
        
        /* Custom file uploader */
        .ubs-file-uploader {
            border: 2px dashed #CCCCCC;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .ubs-file-uploader:hover {
            border-color: #EC0016;
        }
        
        /* Graph container */
        .graph-container {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Custom button styles */
        .stButton > button {
            background-color: #EC0016;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #B30012;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #F0F0F0;
            border-radius: 4px 4px 0 0;
            padding: 0.5rem 1rem;
            border: 1px solid #CCCCCC;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0205A8;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
def initialize_ui_state():
    """Initialize session state variables for UI components."""
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "graph"
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

# Status message functions
def show_info(message: str):
    """Display an information message."""
    st.markdown(f'<div class="status-box info-box">{message}</div>', unsafe_allow_html=True)

def show_success(message: str):
    """Display a success message."""
    st.markdown(f'<div class="status-box success-box">{message}</div>', unsafe_allow_html=True)

def show_warning(message: str):
    """Display a warning message."""
    st.markdown(f'<div class="status-box warning-box">{message}</div>', unsafe_allow_html=True)

def show_error(message: str):
    """Display an error message."""
    st.markdown(f'<div class="status-box error-box">{message}</div>', unsafe_allow_html=True)

# 1. Header Component
def create_header(title: str = "Talk to Code", subtitle: str = "Explore and understand code repositories"):
    """
    Create a custom header with UBS styling.
    
    Args:
        title: The main title text
        subtitle: The subtitle text
    """
    header_container = st.container()
    with header_container:
        st.markdown(f'<h1 class="app-header" style="font-size: 2.5rem;">{title}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="app-subheader" style="font-size: 1.5rem;">{subtitle}</h3>', unsafe_allow_html=True)
        
        # Add separator line
        st.markdown('<hr style="margin: 1rem 0; border-color: #CCCCCC;">', unsafe_allow_html=True)
    
    return header_container

# 2. Sidebar Component
def create_sidebar(
    on_file_upload: Callable[[Optional[UploadedFile], Optional[str]], Any],
    on_settings_change: Callable[[Dict[str, Any]], Any]
) -> Dict[str, Any]:
    """
    Create a collapsible sidebar with file upload features.
    This version removes the settings sections and delegates processing to the callback.

    Args:
        on_file_upload: Callback function expecting either uploaded_file or url keyword argument.
        on_settings_change: Callback function when search settings change (kept for signature compatibility).

    Returns:
        An empty dictionary, as settings are removed.
    """
    # Apply sidebar collapsed class if needed
    if st.session_state.sidebar_collapsed:
        st.markdown('<style>.main .block-container{padding-left: 1rem; max-width: 100%;}</style>', unsafe_allow_html=True)
    
    # Sidebar toggle button in main content area (only visible when sidebar is collapsed)
    if st.session_state.sidebar_collapsed:
        col1, col2 = st.columns([1, 20])
        with col1:
            if st.button("≫", key="expand_sidebar"):
                st.session_state.sidebar_collapsed = False
                st.rerun()
    
    with st.sidebar:
        # Toggle button in sidebar
        if not st.session_state.sidebar_collapsed:
            if st.button("≪ Collapse Sidebar", key="collapse_sidebar"):
                st.session_state.sidebar_collapsed = True
                st.rerun()
        
        st.header("Code Upload")

        # File Uploader
        uploaded_file = st.file_uploader(
            "Upload a zip archive of your codebase",
            type=["zip"],
            key="file_uploader",
            help="Select the .zip file containing your Python project."
        )

        # URL Input
        url = st.text_input(
            "Or enter a URL to a zip file",
            key="url_input",
            placeholder="https://example.com/repo.zip"
        )

        # Process Button
        if st.button("Process Codebase", key="process_button", use_container_width=True):
            if uploaded_file is not None:
                # Directly call the callback with the uploaded file object
                with st.spinner("Processing uploaded file..."):
                    try:
                        on_file_upload(uploaded_file=uploaded_file, url=None)
                    except Exception as e:
                        show_error(f"Error triggering processing for uploaded file: {e}")
            elif url:
                # Directly call the callback with the URL
                with st.spinner("Processing URL..."):
                    try:
                        on_file_upload(uploaded_file=None, url=url)
                    except Exception as e:
                        show_error(f"Error triggering processing for URL: {e}")
            else:
                show_warning("Please upload a zip file or provide a URL.")

        # Return an empty dictionary as no settings are controlled here anymore
        return {}

# 3. Graph Container Component
def create_graph_container(
    is_processed: bool = False,
    graph: Optional[nx.Graph] = None,
    communities: Optional[Dict[str, List[str]]] = None,
    search_results: Optional[Dict[str, Any]] = None
):
    """
    Create a clean container for graph visualization.
    
    Args:
        is_processed: Whether code has been processed
        graph: Optional NetworkX graph to visualize
        communities: Optional dictionary of community ID to list of node IDs
        search_results: Optional search results to visualize
        
    Returns:
        The container for graph visualization
    """
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Code Knowledge Graph Visualization", "Search Results"])
    
    # Graph visualization tab
    with tab1:
        # Add custom CSS to hide graph controls but avoid creating excess whitespace
        st.markdown("""
        <style>
            /* Hide all matplotlib controls */
            .mp-controls-container, .mp-controls-intercept {
                display: none !important;
            }
            /* Hide Plotly modebar */
            .js-plotly-plot .plotly .modebar-btn[data-title="Download plot as a png"] {
                display: inline-block !important;
            }
            .js-plotly-plot .plotly .modebar-btn:not([data-title="Download plot as a png"]) {
                display: none !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        if not is_processed:
            show_info("Upload and process your code to see the graph visualization.")
        elif graph is not None:
            # Display visualization directly without nested containers
            create_enhanced_visualization(
                graph,
                communities=communities,
                container=None,  # Use current context
                search_results=search_results
            )
        else:
            show_info("Graph data not available.")
    
    # Search results tab
    with tab2:
        results_container = st.container()
        
        with results_container:
            if not is_processed or not search_results:
                show_info("Ask a question to see search results.")
            else:
                st.subheader("Search Results")
                
                # Determine the type of search results
                if "entities" in search_results and search_results["entities"]:
                    # Local search results
                    entities = search_results["entities"][:10]  # Limit to top 10
                    
                    # Create a table of results
                    data = []
                    for entity in entities:
                        data.append({
                            "Name": entity["name"],
                            "Type": entity["type"],
                            "Score": f"{entity['score']:.3f}",
                            "File": os.path.basename(entity["source_file"]),
                            "Description": entity["description"][:100] + "..." if len(entity["description"]) > 100 else entity["description"]
                        })
                    
                    st.table(data)
                    
                    # Add button to visualize these results in the graph
                    if st.button("Visualize These Results in Graph"):
                        # Set session state to switch to graph tab with focus view
                        if "entity_ids_to_highlight" not in st.session_state:
                            st.session_state.entity_ids_to_highlight = [entity["id"] for entity in entities]
                        st.rerun()
                
                elif "communities" in search_results and search_results["communities"]:
                    # Replace with message about communities feature being removed
                    st.info("The code communities feature has been removed from this application.")
    
    return tab1, tab2

# 4. Chat Interface Component
def create_chat_interface(
    on_message_send: Callable[[str], Any],
    is_processed: bool = False
):
    """
    Create a styled chat input and history display using pure Python/Streamlit.
    
    Args:
        on_message_send: Callback function when a message is sent
        is_processed: Whether code has been processed
    """
    # Add custom CSS for chat styling
    st.markdown("""
    <style>
    /* Enhanced chat styling */
    .chat-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .chat-icon {
        margin-right: 0.5rem;
        font-size: 1.5rem;
    }
    .stChatInput {
        border-radius: 20px !important;
    }
    .stChatInput > div {
        border-radius: 20px !important;
    }
    .stChatInput input {
        border-radius: 20px !important;
        padding-left: 1rem !important;
    }
    .stChatInput button {
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .stChatInput button:hover {
        transform: scale(1.05);
    }
    /* Message styling */
    .stChatMessage .message-content {
        border-radius: 18px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Chat container with max height and scrolling */
    .chat-message-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat header with icon
    st.markdown('<div class="chat-header"><span class="chat-icon">💬</span><h2>Chat with Your Code</h2></div>', unsafe_allow_html=True)
    
    # Display chat history in a scrollable container
    st.markdown('<div class="chat-message-container">', unsafe_allow_html=True)
    chat_container = st.container()
    
    with chat_container:
        # Initialize chat_messages in session state if not present
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Get messages from session state
        messages = st.session_state.get("chat_messages", [])
        
        # Display chat messages with custom styling
        for i, message in enumerate(messages):
            with st.chat_message(message["role"]):
                st.markdown(f'<div class="message-content">{message["content"]}</div>', unsafe_allow_html=True)
                
                # Add timestamp or message number for context
                if "timestamp" in message:
                    st.caption(f"Sent: {message['timestamp']}")
                else:
                    st.caption(f"Message #{i+1}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create a status indicator
    if not is_processed:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px; color: #666;">
            <span style="width: 8px; height: 8px; background-color: #EF5350; border-radius: 50%; margin-right: 8px;"></span>
            Upload and process code to enable chat
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px; color: #4CAF50;">
            <span style="width: 8px; height: 8px; background-color: #4CAF50; border-radius: 50%; margin-right: 8px;"></span>
            Ready to answer questions about your code
        </div>
        """, unsafe_allow_html=True)
    
    # We DO NOT include chat_input here anymore - it should be defined outside of tabs
    # in the main app.py file
    
    # Create a session state flag for handling chat messages
    if "pending_chat_message" not in st.session_state:
        st.session_state.pending_chat_message = None
    
    # Check if we have a pending message and process it
    if st.session_state.pending_chat_message:
        prompt = st.session_state.pending_chat_message
        st.session_state.pending_chat_message = None  # Clear the message
        
        if not is_processed:
            show_warning("Please upload and process your code before asking questions.")
            return
        
        # Add timestamp to message
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Display the user message
        with st.chat_message("user"):
            st.markdown(f'<div class="message-content">{prompt}</div>', unsafe_allow_html=True)
            st.caption(f"Sent: {timestamp}")
        
        # Add to history with timestamp
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Process the message via callback
        on_message_send(prompt)
    
def handle_chat_input(message, is_processed):
    """
    Handle the chat input that must be placed outside of tabs.
    This function should be called in the main app after the tabs.
    
    Args:
        message: The message from chat_input
        is_processed: Whether code has been processed
    """
    if message:
        if not is_processed:
            show_warning("Please upload and process your code before asking questions.")
            return
            
        # Store the message in session state for processing on next rerun
        st.session_state.pending_chat_message = message
        st.rerun()  # Trigger a rerun to process the message

# Helper function to extract a specific entity or community from search results
def extract_entity_details(entity_id: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract details about a specific entity from search results."""
    if "entities" in search_results:
        for entity in search_results["entities"]:
            if entity["id"] == entity_id:
                return entity
    return None

def extract_community_details(community_id: int, level: int, search_results: Dict[str, Any]) -> Dict[str, Any]:
    """This function has been deprecated as the communities feature is no longer available."""
    return {}

# Graph visualization functions
def create_plotly_graph(
    graph: nx.Graph,
    communities: Optional[Dict[str, List[str]]] = None,
    view_mode: str = "standard",
    max_nodes: int = 100,
    highlight_nodes: Optional[List[str]] = None,
    title: str = "Code Knowledge Graph"
) -> go.Figure:
    """
    Create an interactive graph visualization using Plotly with controlled display options.
    
    Args:
        graph: NetworkX graph to visualize
        communities: Optional dictionary of community ID to list of node IDs
        view_mode: Visualization mode ('standard', 'community', 'hierarchy', 'focus')
        max_nodes: Maximum number of nodes to display
        highlight_nodes: Optional list of node IDs to highlight
        title: Title for the graph visualization
        
    Returns:
        Plotly figure object with the graph visualization and configuration
    """
    # If graph is too large, sample nodes
    if len(graph.nodes) > max_nodes:
        if highlight_nodes:
            # Include highlighted nodes in the sample
            other_nodes = [n for n in graph.nodes if n not in highlight_nodes]
            num_sample = min(max_nodes - len(highlight_nodes), len(other_nodes))
            if num_sample > 0:
                sampled_nodes = list(np.random.choice(other_nodes, num_sample, replace=False))
                nodes_to_include = highlight_nodes + sampled_nodes
            else:
                nodes_to_include = highlight_nodes[:max_nodes]
        else:
            # Random sample
            nodes_to_include = list(np.random.choice(list(graph.nodes), max_nodes, replace=False))
        
        subgraph = graph.subgraph(nodes_to_include)
    else:
        subgraph = graph
    
    # Generate layout for nodes
    if view_mode == "hierarchy":
        # Hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(subgraph, prog="dot") if len(subgraph) > 0 else {}
    else:
        # Standard force-directed layout
        pos = nx.spring_layout(subgraph, seed=42)
    
    # Create figure
    fig = go.Figure()
    
    # Prepare node colors based on view mode
    if view_mode == "community" and communities:
        # Color nodes by community
        node_colors = {}
        colorscale = px.colors.qualitative.Bold
        
        for i, (comm_id, members) in enumerate(communities.items()):
            color_idx = i % len(colorscale)
            for node_id in members:
                if node_id in subgraph:
                    node_colors[node_id] = colorscale[color_idx]
    else:
        # Default color scheme
        node_colors = {
            node: COMPANY_COLORS["blue"] for node in subgraph.nodes
        }
    
    # Highlight specific nodes if provided
    if highlight_nodes:
        for node in highlight_nodes:
            if node in subgraph:
                node_colors[node] = COMPANY_COLORS["red"]
    
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in subgraph.edges():
        if edge[0] in pos and edge[1] in pos:  # Ensure nodes are in layout
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.7, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in subgraph.nodes():
        if node in pos:  # Ensure node is in layout
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text for hover info
            node_info = f"ID: {node}"
            if 'name' in subgraph.nodes[node]:
                node_info += f"<br>Name: {subgraph.nodes[node]['name']}"
            if 'type' in subgraph.nodes[node]:
                node_info += f"<br>Type: {subgraph.nodes[node]['type']}"
            node_text.append(node_info)
            
            # Node color
            node_color.append(node_colors.get(node, COMPANY_COLORS["blue"]))
            
            # Node size based on degree
            size = 10 + subgraph.degree[node]
            node_size.append(min(size, 30))  # Cap size
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#333')
        ),
        showlegend=False
    )
    
    # Add traces to figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    
    # Update layout
    fig.update_layout(
        title=title,
        titlefont=dict(size=16, color=COMPANY_COLORS["dark_gray"]),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=5, l=5, r=5, t=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=600,
    )
    
    # Configure to disable/control interactive features
    is_static = view_mode == "static"
    
    config = {
        # Complete control over mode bar display
        'displayModeBar': not is_static,
        
        # Static mode disables all interactivity
        'staticPlot': is_static,
        
        # Enable/disable scroll zoom based on mode
        'scrollZoom': not is_static,
        
        # Remove Plotly logo
        'displaylogo': False,
        
        # Remove specific buttons from mode bar
        'modeBarButtonsToRemove': [
            'pan2d', 'select2d', 'lasso2d', 'resetScale2d',
            'toggleHover', 'toggleSpikelines', 'hoverClosestCartesian',
            'hoverCompareCartesian'
        ],
        
        # Only keep download, zoom, and reset view
        'modeBarButtonsToAdd': [],
        
        # Configure download options
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'code_graph',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
    
    return fig, config

# Function to display the plotly graph in Streamlit with controls
def display_graph_with_controls(
    graph: nx.Graph,
    communities: Optional[Dict[str, List[str]]] = None,
    container: Optional[st.container] = None
):
    """
    Display a graph visualization with Python-based controls.
    
    Args:
        graph: NetworkX graph to visualize
        communities: Optional dictionary of community ID to list of node IDs
        container: Optional Streamlit container to render in
    """
    if not container:
        container = st.container()
    
    with container:
        # Controls in a form for batched execution
        with st.form(key="graph_controls"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                view_mode = st.selectbox(
                    "View mode",
                    ["standard", "hierarchy", "focus", "static"],  # Removed 'community' option
                    index=0,
                    key="view_mode",
                    on_change=lambda: update_graph(),
                    help="How to visualize the graph"
                )
            
            with col2:
                max_nodes = st.slider(
                    "Max Nodes", 
                    min_value=10, 
                    max_value=300, 
                    value=100,
                    help="Maximum number of nodes to display"
                )
            
            with col3:
                # Only show filter by type if graph has node type information
                node_types = list(set(
                    data.get('type', 'unknown') 
                    for _, data in graph.nodes(data=True)
                ))
                
                selected_types = st.multiselect(
                    "Filter by Type",
                    options=node_types,
                    default=node_types,
                    help="Select node types to display"
                )
            
            # Search box for finding specific nodes
            search_term = st.text_input(
                "Search Nodes",
                help="Search for nodes by name or ID"
            )
            
            submitted = st.form_submit_button("Update Graph")
        
        # Process the filtered graph
        if submitted or "graph_view_state" not in st.session_state:
            # Store view state
            st.session_state.graph_view_state = {
                "view_mode": view_mode,
                "max_nodes": max_nodes,
                "selected_types": selected_types,
                "search_term": search_term
            }
            
            # Filter graph based on selected types
            if selected_types and selected_types != node_types:
                filtered_nodes = [
                    node for node, data in graph.nodes(data=True)
                    if data.get('type', 'unknown') in selected_types
                ]
                filtered_graph = graph.subgraph(filtered_nodes)
            else:
                filtered_graph = graph
            
            # Find nodes matching search term
            highlight_nodes = []
            if search_term:
                for node, data in filtered_graph.nodes(data=True):
                    node_name = str(data.get('name', '')).lower()
                    node_id = str(node).lower()
                    
                    if (search_term.lower() in node_name or 
                        search_term.lower() in node_id):
                        highlight_nodes.append(node)
            
            # Create and display the graph
            fig, config = create_plotly_graph(
                filtered_graph,
                communities=communities,
                view_mode=view_mode,
                max_nodes=max_nodes,
                highlight_nodes=highlight_nodes,
                title=f"Code Knowledge Graph ({len(filtered_graph.nodes)} nodes)"
            )
            
            # Display the graph with specific config
            st.plotly_chart(fig, config=config, use_container_width=True)

# Utility function to create a download link for data
def get_download_link(data, filename, text):
    """Generate a download link for data."""
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href 

# Attempt to import pygraphviz for layout
has_graphviz = False
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    has_graphviz = True
except ImportError:
    print("INFO: PyGraphviz not found. Hierarchical layout ('dot') disabled. Falling back to Spring layout.")
    # Fallback layout function if graphviz not available
    def graphviz_layout(G, prog='spring', args=''):
         # Provide a consistent fallback using spring_layout.
         print(f"Warning: graphviz_layout called but pygraphviz not found. Using default spring_layout.")
         return nx.spring_layout(G, seed=42, dim=2, k=1.0, iterations=75)

def create_enhanced_visualization(
    graph: nx.Graph,
    communities: Optional[Dict[str, List[str]]] = None,
    container: Optional[st.container] = None,
    search_results: Optional[Dict[str, Any]] = None
):
    """
    Create an enhanced visualization using hierarchical layout if available.
    Focuses on clear edges with arrows, marker nodes, and detailed hover info.
    """
    global has_graphviz
    
    if container is None:
        pass # Render in current context
    else:
        # This context manager might cause issues if container is None
        # It's generally better to call st elements directly or check container first
        # For simplicity, assuming rendering happens in the main flow if container is None
        pass

    if graph is None or len(graph.nodes) == 0:
        st.warning("No graph data available to visualize.")
        return

    try:
        # --- Layout (Hierarchical if possible, else Spring) --- 
        progress_msg = st.empty()
        progress_msg.info("Calculating graph layout...")
        pos = None
        if has_graphviz:
            try:
                # Use 'dot' for hierarchical layout
                pos = graphviz_layout(graph, prog='dot') 
            except Exception as e_gv:
                st.warning(f"Graphviz layout failed ({e_gv}), falling back to Spring layout.")
                has_graphviz = False # Disable further attempts if it fails once
        
        if pos is None: # Fallback if graphviz failed or wasn't available
             try:
                 pos = nx.spring_layout(graph, dim=2, k=1.0, iterations=75, seed=42) 
             except Exception as e_spring:
                 st.error(f"Spring layout calculation failed: {e_spring}")
                 progress_msg.empty()
                 return
        progress_msg.empty()

        # --- Colors (Pastel based on Node Type) --- 
        node_types = sorted(list(set(d.get('type', 'Unknown') for n, d in graph.nodes(data=True))))
        pastel_colors = px.colors.qualitative.Pastel
        type_color_map = {ntype: pastel_colors[i % len(pastel_colors)] for i, ntype in enumerate(node_types)}

        # --- Edge Trace (Lines) --- 
        edge_x = []
        edge_y = []
        edge_hover_texts = []
        edge_midpoints = [] # For potential future label placement or interaction
        
        for u, v, data in graph.edges(data=True):
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                # Calculate midpoint for potential use
                edge_midpoints.append(((x0+x1)/2, (y0+y1)/2))
                
                # Create hover text for the edge
                rel_type = data.get('type', 'Unknown')
                strength = data.get('strength', '')
                description = data.get('description', '')
                hover_info = f"<b>Type: {rel_type}</b>"
                if strength: hover_info += f"<br>Strength: {strength}"
                if description: hover_info += f"<br>Desc: {description[:100]}{'...' if len(description)>100 else ''}"
                edge_hover_texts.extend([hover_info, hover_info, ''])
            else:
                 edge_hover_texts.extend(['', '', ''])
                 edge_midpoints.append((None, None)) # Add placeholder

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#888888'), # Darker, slightly thicker edges
            hoverinfo='text',
            hovertext=edge_hover_texts,
            mode='lines'
        )

        # --- Node Trace (Markers only) --- 
        node_x = []
        node_y = []
        node_hover_texts = []
        node_colors = []
        node_sizes = []
        node_labels = []  # NEW list to hold visible labels
        for node, data in graph.nodes(data=True):
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                # Node Color & Size
                node_type = data.get('type', 'Unknown')
                node_colors.append(type_color_map.get(node_type, '#ABABAB')) 
                degree = graph.degree(node)
                node_sizes.append(max(10, min(12 + degree * 2, 35))) # Adjusted size slightly
                
                # Node Hover Text (ensure it's comprehensive)
                node_name = data.get('name', str(node))
                hover_info = f"<b>{node_name}</b><br>Type: {node_type}<br>" 
                hover_info += f"File: {data.get('source_file', '?')}:{data.get('lineno', '?')}<br>"
                if 'description' in data and data['description'] and data['description'] != "No description available":
                    hover_info += f"Desc: {data.get('description', '')}<br>"
                code_snippet = data.get('code_snippet', '')
                max_snippet_len = 400 # Allow slightly longer snippet on hover
                if code_snippet:
                     hover_info += f"<br>--- Code ---<br><pre style='white-space: pre-wrap; word-wrap: break-word;'>{code_snippet[:max_snippet_len].replace('\n','<br>')}{'...' if len(code_snippet)>max_snippet_len else ''}</pre>"
                node_hover_texts.append(hover_info)
                node_labels.append(node_name)  # Store label
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',  # show labels
            text=node_labels,
            textposition='top center',
            textfont=dict(size=12, color='#000000'),
            hovertext=node_hover_texts, 
            hoverinfo='text',
            marker=dict(
                color=node_colors,
                size=node_sizes,
                opacity=0.9,
                line=dict(width=1, color='#FFFFFF')
            )
        )

        # Edge label trace
        edge_label_x = []
        edge_label_y = []
        edge_labels = []
        for (u, v, data), (mx, my) in zip(graph.edges(data=True), edge_midpoints):
            if mx is None:
                continue
            edge_label_x.append(mx)
            edge_label_y.append(my)
            edge_labels.append(data.get('type', ''))

        edge_label_trace = go.Scatter(
            x=edge_label_x,
            y=edge_label_y,
            mode='text',
            text=edge_labels,
            textposition='middle center',
            textfont=dict(size=10, color='#444444'),
            hoverinfo='none'
        )

        # --- Layout --- 
        layout = go.Layout(
            title='Code Knowledge Graph',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False), 
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False), 
            plot_bgcolor='#FFFFFF', 
            paper_bgcolor='#FFFFFF',
            height=750, # Increased height slightly
        )
        
        # --- Create List for Edge Arrows --- 
        arrow_annotations = [] # Initialize an empty LIST
        arrow_scale = 0.03 # Adjust scale based on graph size/zoom
        arrow_color = '#888888'
        arrow_width = 1.5
        
        for i, (u, v) in enumerate(graph.edges()):
             if u in pos and v in pos:
                 x0, y0 = pos[u]
                 x1, y1 = pos[v]
                 # Calculate angle and endpoint for arrowhead
                 angle = np.arctan2(y1 - y0, x1 - x0)
                 # Place arrow slightly before the target node
                 node_size_target = max(8, min(12 + graph.degree(v) * 2, 35)) / 2 # Approx radius
                 # Adjust arrow placement based on layout scale - this needs tuning!
                 layout_scale = max(np.ptp(node_x), np.ptp(node_y)) if node_x else 1 # Estimate plot scale
                 offset = min(arrow_scale * layout_scale, np.sqrt((x1-x0)**2 + (y1-y0)**2)*0.3) # Don't overshoot midpoint
                 
                 arrow_x = x1 - offset * np.cos(angle)
                 arrow_y = y1 - offset * np.sin(angle)

                 # Add arrow annotation DICTIONARY to the LIST
                 arrow_annotations.append(
                     go.layout.Annotation(
                         ax=x0, ay=y0, 
                         x=arrow_x, y=arrow_y,
                         xref='x', yref='y',
                         showarrow=True,
                         axref='x', ayref='y',
                         arrowhead=2, 
                         arrowsize=1, 
                         arrowwidth=arrow_width,
                         arrowcolor=arrow_color
                     )
                 )
        
        # --- Assign the completed list to the layout --- 
        layout.annotations = arrow_annotations

        # --- Figure --- 
        fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace], layout=layout)

        # --- Configuration --- 
        config = {
            'displayModeBar': True, 
            'scrollZoom': True,
            'displaylogo': False,
             'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toggleSpikelines']
        }
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        # Display simple legend for node types below the graph
        st.write("**Node Colors (Type):**")
        # Use columns for a cleaner legend layout
        max_legend_cols = 5 
        cols = st.columns(min(max_legend_cols, len(type_color_map))) 
        i = 0
        # Sort types for consistent legend order
        for ntype in sorted(type_color_map.keys()):
            color = type_color_map[ntype]
            with cols[i % len(cols)]:
                # Use markdown for colored circle and text
                st.markdown(f'<span style="display:inline-block; width:12px; height:12px; background-color:{color}; border-radius:50%; margin-right:5px; border: 1px solid #ccc;"></span> {ntype}', unsafe_allow_html=True)
            i += 1

    except Exception as e:
        st.error(f"Error generating enhanced visualization: {e}")
        st.exception(e) # Show full traceback for debugging

# Enhanced visualization modes
def create_artistic_graph_visualization(
    graph: nx.Graph,
    communities: Optional[Dict[str, List[str]]] = None,
    container: Optional[st.container] = None,
    max_nodes: int = 200,
    view_mode: str = "artistic"
):
    """
    Create an artistic, flowing network visualization similar to high-end graph tools.
    
    Args:
        graph: NetworkX graph to visualize
        communities: Optional dictionary of communities
        container: Streamlit container to render in
        max_nodes: Maximum number of nodes to display
        view_mode: Visualization mode
    """
    # Use st.container if no container provided to ensure there's a place to render
    if container is None:
        container = st.container()

    # Early check for empty graph
    if graph is None or len(graph.nodes) == 0:
        with container:
            st.warning("No graph data available to visualize. Please process a codebase first.")
        return None, None
    
    try:
        with container:
            # Create a spinner to indicate loading
            with st.spinner("Generating artistic visualization..."):
                # Add a progress message
                progress_msg = st.empty()
                progress_msg.info("Initializing visualization...")
                
                # Sample nodes if graph is too large
                if len(graph.nodes) > max_nodes:
                    progress_msg.info(f"Sampling {max_nodes} nodes from {len(graph.nodes)} total nodes...")
                    # Ensure we select nodes with diverse types for a better visualization
                    node_types = {}
                    for node, data in graph.nodes(data=True):
                        node_type = data.get('type', 'unknown')
                        if node_type not in node_types:
                            node_types[node_type] = []
                        node_types[node_type].append(node)
                    
                    # Sample nodes from each type to ensure diversity
                    nodes_to_include = []
                    
                    # Calculate how many to take from each type
                    num_types = len(node_types)
                    nodes_per_type = max(2, int(max_nodes / (num_types or 1)))
                    
                    for node_type, nodes in node_types.items():
                        # Take a sample from each type, ensuring at least some representation
                        sample_size = min(len(nodes), nodes_per_type)
                        nodes_to_include.extend(random.sample(nodes, sample_size))
                    
                    # If we haven't reached max_nodes, add random nodes
                    if len(nodes_to_include) < max_nodes:
                        remaining_nodes = [n for n in graph.nodes() if n not in nodes_to_include]
                        additional_sample = min(max_nodes - len(nodes_to_include), len(remaining_nodes))
                        if additional_sample > 0:
                            nodes_to_include.extend(random.sample(remaining_nodes, additional_sample))
                            
                    subgraph = graph.subgraph(nodes_to_include)
                else:
                    subgraph = graph
                
                progress_msg.info("Generating layout...")
                
                # Set a seed for deterministic layout
                random.seed(42)
                np.random.seed(42)
                
                # Generate a force-directed layout
                # Using Fruchterman-Reingold algorithm for a more natural, flowing layout
                pos = nx.fruchterman_reingold_layout(subgraph, dim=3, k=0.5, seed=42)
                
                progress_msg.info("Creating node styles...")
                
                # Prepare node colors based on communities or types
                node_colors = []
                node_sizes = []
                node_symbols = []
                
                # Use a vibrant color palette similar to the image
                color_palette = [
                    '#FF3EA5', '#FF9E00', '#00C2FF', '#00FF88', '#AA44FF', 
                    '#FFFF00', '#FF0000', '#0088FF', '#8BFF00', '#FF00FF',
                    '#00FFFF', '#FF7700', '#00FF44', '#8800FF', '#00BBFF'
                ]
                
                # Assign colors based on node types or communities
                type_to_color = {}
                
                for node in subgraph.nodes():
                    node_data = subgraph.nodes[node]
                    
                    # Determine node type (default to 'unknown')
                    node_type = node_data.get('type', 'unknown')
                    
                    # Assign color based on type
                    if node_type not in type_to_color:
                        type_to_color[node_type] = color_palette[len(type_to_color) % len(color_palette)]
                    
                    node_color = type_to_color[node_type]
                    node_colors.append(node_color)
                    
                    # Determine node size based on its importance (degree)
                    degree = subgraph.degree[node]
                    node_size = 10 + (degree * 2)  # Scale size based on degree
                    node_sizes.append(min(node_size, 40))  # Cap maximum size
                    
                    # Vary node symbols for visual interest
                    symbols = ['circle', 'diamond', 'square', 'cross']
                    if 'Class' in node_type:
                        node_symbols.append('circle')
                    elif 'Function' in node_type:
                        node_symbols.append('diamond')
                    elif 'Module' in node_type:
                        node_symbols.append('square')
                    else:
                        node_symbols.append(random.choice(symbols))
                
                progress_msg.info("Creating edge traces...")
                
                # Create edge traces with gradient colors based on connected nodes
                edge_traces = []
                
                for edge in subgraph.edges():
                    if edge[0] in pos and edge[1] in pos:  # Ensure nodes are in layout
                        x0, y0, z0 = pos[edge[0]]
                        x1, y1, z1 = pos[edge[1]]
                        
                        # Create color gradient for edge based on connected nodes
                        color_a = node_colors[list(subgraph.nodes).index(edge[0])]
                        color_b = node_colors[list(subgraph.nodes).index(edge[1])]
                        
                        # Create separate segments to create a gradient-like effect
                        # We'll use 2 segments with different colors to simulate a gradient
                        
                        # First half of the edge - color_a
                        points_first_half = 8  # Fewer points for performance
                        xs1, ys1, zs1 = [], [], []
                        
                        for i in range(points_first_half):
                            t = i / (points_first_half - 1) * 0.5  # 0 to 0.5
                            # Interpolation with slight curve
                            x = x0 + (x1 - x0) * t * 2 + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                            y = y0 + (y1 - y0) * t * 2 + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                            z = z0 + (z1 - z0) * t * 2 + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                            
                            xs1.append(x)
                            ys1.append(y)
                            zs1.append(z)
                        
                        # Second half of the edge - color_b
                        points_second_half = 8  # Fewer points for performance
                        xs2, ys2, zs2 = [], [], []
                        
                        for i in range(points_second_half):
                            t = 0.5 + i / (points_second_half - 1) * 0.5  # 0.5 to 1.0
                            # Interpolation with slight curve
                            x = x0 + (x1 - x0) * t * 2 - (x1 - x0) + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                            y = y0 + (y1 - y0) * t * 2 - (y1 - y0) + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                            z = z0 + (z1 - z0) * t * 2 - (z1 - z0) + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                            
                            xs2.append(x)
                            ys2.append(y)
                            zs2.append(z)
                        
                        # Create two edge traces with different colors
                        edge_trace1 = go.Scatter3d(
                            x=xs1, y=ys1, z=zs1,
                            mode='lines',
                            line=dict(
                                width=2,
                                color=color_a,
                            ),
                            opacity=0.6,
                            hoverinfo='none'
                        )
                        
                        edge_trace2 = go.Scatter3d(
                            x=xs2, y=ys2, z=zs2,
                            mode='lines',
                            line=dict(
                                width=2,
                                color=color_b,
                            ),
                            opacity=0.6,
                            hoverinfo='none'
                        )
                        
                        edge_traces.append(edge_trace1)
                        edge_traces.append(edge_trace2)
                
                progress_msg.info("Finalizing node properties...")
                
                # Prepare node text and hover information
                node_text = []
                hover_text = []
                
                for node in subgraph.nodes():
                    node_data = subgraph.nodes[node]
                    # Get a meaningful name for the node
                    name = node_data.get('name', str(node))
                    if len(name) > 20:  # Truncate long names for display
                        display_name = name[:18] + "..."
                    else:
                        display_name = name
                        
                    node_text.append(display_name)
                    
                    # Create detailed hover information
                    hover_info = f"<b>{name}</b><br>"
                    if 'type' in node_data:
                        hover_info += f"Type: {node_data['type']}<br>"
                    if 'source_file' in node_data:
                        file_name = node_data['source_file'].split('/')[-1] if '/' in node_data['source_file'] else node_data['source_file']
                        hover_info += f"File: {file_name}<br>"
                    if 'description' in node_data and node_data['description']:
                        desc = node_data['description']
                        if len(desc) > 100:
                            desc = desc[:97] + "..."
                        hover_info += f"Description: {desc}"
                        
                    hover_text.append(hover_info)
                
                # Create node trace
                node_trace = go.Scatter3d(
                    x=[pos[node][0] for node in subgraph.nodes()],
                    y=[pos[node][1] for node in subgraph.nodes()],
                    z=[pos[node][2] for node in subgraph.nodes()],
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    textfont=dict(
                        family="Arial, sans-serif",
                        size=14,
                        color='#333333'
                    ),
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        opacity=0.9,
                        line=dict(width=1, color='#ffffff'),
                        symbol=node_symbols
                    ),
                    hoverinfo='text',
                    hovertext=hover_text,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                
                progress_msg.info("Building visualization...")
                
                # Create figure
                fig = go.Figure(data=edge_traces + [node_trace])
                
                # Update layout for clean, aesthetic design
                fig.update_layout(
                    title=f"Code Knowledge Graph ({len(subgraph.nodes)} nodes, {len(subgraph.edges())} edges)",
                    title_font=dict(size=20, family="Arial, sans-serif"),
                    showlegend=False,
                    scene=dict(
                        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showbackground=False, showspikes=False),
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showbackground=False, showspikes=False),
                        zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showbackground=False, showspikes=False),
                        bgcolor='white',
                        aspectmode='data',  # 'data' preserves the data aspect ratio
                        dragmode='orbit',
                        camera=dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        ),
                    ),
                    margin=dict(l=0, r=0, b=0, t=50),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    height=800,  # Explicit height to ensure visibility
                )
                
                # Configure display settings
                config = {
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['resetCameraDefault3d', 'toImage'],
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'artistic_graph',
                        'height': 1000,
                        'width': 1500,
                        'scale': 2
                    }
                }
                
                # Clear progress message
                progress_msg.empty()
                
                # Display usage instructions
                st.info("💡 **Tip:** Try these interactions for a better experience:\n"
                       "- **Drag** to rotate the graph in 3D\n"
                       "- **Scroll** to zoom in/out\n"
                       "- **Hover** over nodes to see detailed information\n"
                       "- **Double-click** on a node to focus on it\n"
                       "- **Click and hold** then drag to pan the view\n"
                       "- Node names appear more clearly when you zoom in")
                
                # Display the graph
                st.plotly_chart(fig, config=config, use_container_width=True)
                
                # Add legend for node types
                st.write("### Legend")
                legend_cols = st.columns(4)
                
                for i, (node_type, color) in enumerate(type_to_color.items()):
                    col_idx = i % 4
                    with legend_cols[col_idx]:
                        st.markdown(
                            f'<div style="display:flex;align-items:center;margin-bottom:5px;">'
                            f'<div style="width:15px;height:15px;background-color:{color};'
                            f'margin-right:5px;border-radius:50%;"></div>'
                            f'<span>{node_type}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                return fig, config
        
    except Exception as e:
        if container:
            with container:
                st.error(f"Error generating artistic visualization: {str(e)}")
                st.exception(e)  # Show the full exception for easier debugging
                st.warning("Please try with a smaller graph or use the standard visualization.")
        return None, None 