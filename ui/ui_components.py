#!/usr/bin/env python3
"""
UI Components for the Talk to Code application.

This module contains pure Python Streamlit components for creating
a consistent and user-friendly interface for the Talk to Code application.
"""

import os
import streamlit as st
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

# UBS Brand Colors
UBS_COLORS = {
    "red": "#EC0016",
    "light_red": "#FF6D6A",
    "dark_red": "#B30012",
    "blue": "#0205A8",
    "light_blue": "#9A9CFF",
    "dark_blue": "#000066",
    "black": "#000000",
    "dark_gray": "#333333",
    "medium_gray": "#666666",
    "light_gray": "#CCCCCC",
    "white": "#FFFFFF",
}

# Base CSS Styles
def inject_base_css():
    """Inject base CSS styles for consistent UI appearance."""
    st.markdown("""
    <style>
        /* Base styling */
        .ubs-header {
            color: #EC0016;
            font-weight: bold;
        }
        .ubs-subheader {
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
        .ubs-button {
            background-color: #EC0016;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            cursor: pointer;
            text-align: center;
        }
        .ubs-button:hover {
            background-color: #B30012;
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
        st.markdown(f'<h1 class="ubs-header" style="font-size: 2.5rem;">{title}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="ubs-subheader" style="font-size: 1.5rem;">{subtitle}</h3>', unsafe_allow_html=True)
        
        # Add separator line
        st.markdown('<hr style="margin: 1rem 0; border-color: #CCCCCC;">', unsafe_allow_html=True)
    
    return header_container

# 2. Sidebar Component
def create_sidebar(
    on_file_upload: Callable[[str], Any],
    on_settings_change: Callable[[Dict[str, Any]], Any]
) -> Dict[str, Any]:
    """
    Create a collapsible sidebar with file upload features.
    
    Args:
        on_file_upload: Callback function when a file is uploaded and processed
        on_settings_change: Callback function when search settings change
        
    Returns:
        Dictionary containing all sidebar settings
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
        
        # File Upload Section
        st.header("Upload Code")
        
        # Custom file uploader
        st.markdown('<div class="ubs-file-uploader">Drag & Drop a ZIP file here</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a Zip file containing your code", type=["zip"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                zip_path = tmp_file.name
            
            # Create a temp directory for extraction
            temp_dir = tempfile.mkdtemp()
            
            # Process button
            if st.button("Process Code", key="process_code_btn", use_container_width=True):
                with st.spinner("Extracting and processing code..."):
                    # Extract the zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Call the callback function
                    try:
                        on_file_upload(temp_dir)
                        show_success("Code processed successfully!")
                    except Exception as e:
                        show_error(f"Error processing code: {str(e)}")
                        st.exception(e)
        
        # Search Settings Section
        st.header("Search Settings")
        
        # Strategy selector
        search_strategy = st.selectbox(
            "Search Strategy",
            ["local", "global", "drift"],
            help="Local: Entity-level search, Global: Community-level search, Drift: Exploratory search"
        )
        
        # Settings based on strategy
        search_params = {}
        
        if search_strategy == "local":
            col1, col2 = st.columns(2)
            with col1:
                search_params["max_hops"] = st.number_input("Max Hops", 1, 5, 2)
            with col2:
                search_params["top_k_entities"] = st.number_input("Top K Entities", 5, 50, 15)
            
            col3, col4 = st.columns(2)
            with col3:
                search_params["min_similarity"] = st.slider("Min Similarity", 0.0, 1.0, 0.6, 0.05)
            with col4:
                search_params["include_code"] = st.checkbox("Include Code", True)
            
        elif search_strategy == "global":
            col1, col2 = st.columns(2)
            with col1:
                search_params["top_k_communities"] = st.number_input("Top K Communities", 1, 10, 5)
            with col2:
                search_params["min_similarity"] = st.slider("Min Similarity", 0.0, 1.0, 0.5, 0.05)
            
            # Community levels
            search_params["community_levels"] = st.multiselect(
                "Community Levels",
                options=[0, 1, 2],
                default=[0, 1, 2],
                format_func=lambda x: f"Level {x+1}"
            )
            
            # Report aspect
            search_params["report_aspect"] = st.selectbox(
                "Report Aspect",
                ["title", "summary", "full"],
                index=2,
                help="Which aspect of the community report to search"
            )
            
        elif search_strategy == "drift":
            col1, col2 = st.columns(2)
            with col1:
                search_params["num_hypotheses"] = st.number_input("Num Hypotheses", 1, 5, 2)
            with col2:
                search_params["max_steps"] = st.number_input("Max Steps", 1, 5, 2)
            
            search_params["branching_factor"] = st.number_input("Branching Factor", 1, 5, 2)
        
        # Apply settings button
        if st.button("Apply Settings", use_container_width=True):
            # Call the callback function
            on_settings_change({
                "strategy": search_strategy,
                "params": search_params
            })
        
        # Add some spacing
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Visualization settings (if needed)
        if "processing_complete" in st.session_state and st.session_state.processing_complete:
            st.header("Visualization")
            
            level = st.slider("Community Level", 0, 2, 0, 
                            help="Hierarchy level (0: fine-grained, 1: module-level, 2: architectural)")
            
            max_nodes = st.slider("Max Nodes", 10, 200, 50,
                                help="Maximum nodes to display")
            
            if st.button("Update Visualization", use_container_width=True):
                # Store in session state
                st.session_state.visualization = {
                    "level": level,
                    "max_nodes": max_nodes
                }
                st.rerun()
    
    # Return settings
    return {
        "strategy": search_strategy,
        "params": search_params
    }

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
                    # Global search results
                    communities = search_results["communities"]
                    
                    # Create a table of results
                    data = []
                    for community in communities:
                        data.append({
                            "Level": community["level"] + 1,  # 1-indexed for display
                            "ID": community["id"],
                            "Title": community["title"],
                            "Similarity": f"{community['similarity']:.3f}",
                            "Nodes": community["node_count"],
                            "Summary": community["summary"][:100] + "..." if len(community["summary"]) > 100 else community["summary"]
                        })
                    
                    st.table(data)
    
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
    """Extract details about a specific community from search results."""
    if "communities" in search_results:
        for community in search_results["communities"]:
            if community["id"] == community_id and community["level"] == level:
                return community
    return None

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
            node: UBS_COLORS["blue"] for node in subgraph.nodes
        }
    
    # Highlight specific nodes if provided
    if highlight_nodes:
        for node in highlight_nodes:
            if node in subgraph:
                node_colors[node] = UBS_COLORS["red"]
    
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
            node_color.append(node_colors.get(node, UBS_COLORS["blue"]))
            
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
        titlefont=dict(size=16, color=UBS_COLORS["dark_gray"]),
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
                    "View Mode",
                    ["standard", "community", "hierarchy", "focus", "static"],
                    help="Select visualization style"
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
            
            # Show legend for community colors if in community mode
            if view_mode == "community" and communities:
                st.markdown("### Community Legend")
                
                # Create a grid layout for the legend
                legend_cols = st.columns(4)
                
                for i, comm_id in enumerate(communities.keys()):
                    col_idx = i % 4
                    color_idx = i % len(px.colors.qualitative.Bold)
                    color = px.colors.qualitative.Bold[color_idx]
                    
                    with legend_cols[col_idx]:
                        st.markdown(
                            f'<div style="display:flex;align-items:center;margin-bottom:5px;">'
                            f'<div style="width:15px;height:15px;background-color:{color};'
                            f'margin-right:5px;border-radius:50%;"></div>'
                            f'<span>Community {comm_id}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

# Utility function to create a download link for data
def get_download_link(data, filename, text):
    """Generate a download link for data."""
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href 

# Enhanced visualization modes
def create_enhanced_visualization(
    graph: nx.Graph,
    communities: Optional[Dict[str, List[str]]] = None,
    container: Optional[st.container] = None,
    search_results: Optional[Dict[str, Any]] = None
):
    """
    Create an enhanced visualization with multiple view modes and advanced configuration.
    
    Args:
        graph: NetworkX graph to visualize
        communities: Optional dictionary of communities
        container: Streamlit container to render in
        search_results: Optional search results to highlight
    """
    if container is None:
        # Use current context if no container provided
        pass
    else:
        with container:
            pass
            
    try:
        # Try to create a simplified graph display
        try:
            # First attempt - with standard layout
            fig, config = create_plotly_graph(
                graph=graph,
                communities=communities,
                max_nodes=100,
                title=f"Code Graph ({len(graph.nodes)} nodes)"
            )
        except Exception as layout_error:
            # Fallback to a simpler approach if the main visualization fails
            st.warning(f"Using simplified visualization due to: {str(layout_error)}")
            
            # Create a basic graph directly without complex layouts
            G = nx.Graph()
            
            # Add some nodes from the original graph
            node_limit = min(100, len(graph.nodes))
            nodes_to_use = list(graph.nodes)[:node_limit]
            
            # Add nodes with basic attributes
            for node_id in nodes_to_use:
                node_data = graph.nodes[node_id]
                G.add_node(
                    node_id, 
                    name=node_data.get('name', 'Unknown'),
                    type=node_data.get('type', 'Unknown')
                )
            
            # Add edges between these nodes
            for u, v in graph.edges():
                if u in nodes_to_use and v in nodes_to_use:
                    G.add_edge(u, v)
            
            # Create a simple spring layout
            pos = nx.spring_layout(G, seed=42)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.7, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{G.nodes[node]['name']} ({G.nodes[node]['type']})")
                
                # Color by node type
                if 'type' in G.nodes[node]:
                    node_type = G.nodes[node]['type']
                    # Use different colors for different types
                    if 'Class' in node_type:
                        node_color.append('blue')
                    elif 'Function' in node_type:
                        node_color.append('green')
                    elif 'Import' in node_type:
                        node_color.append('purple')
                    else:
                        node_color.append('red')
                else:
                    node_color.append('gray')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=10,
                    color=node_color,
                    line=dict(width=1, color='#333')
                ),
                text=node_text,
                hoverinfo='text'
            )
            
            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace])
            
            # Update layout
            fig.update_layout(
                title=f"Simplified Code Graph ({len(G.nodes)} nodes, {len(G.edges)} edges)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=5, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            # Basic config
            config = {'displayModeBar': True, 'scrollZoom': True}
        
        # Display the graph
        st.plotly_chart(fig, config=config, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.warning("Please try uploading a smaller codebase or using the 'Chat' and 'Code Explorer' tabs instead.")

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
    if container is None:
        # Use current context if no container provided
        pass
    else:
        with container:
            pass
    
    try:
        import numpy as np
        import plotly.graph_objects as go
        import random
        
        # Sample nodes if graph is too large
        if len(graph.nodes) > max_nodes:
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
            
        # Generate a force-directed layout
        # Using Fruchterman-Reingold algorithm for a more natural, flowing layout
        pos = nx.fruchterman_reingold_layout(subgraph, dim=3, k=0.5, seed=42)
        
        # Prepare node colors based on communities or types
        node_colors = []
        node_sizes = []
        node_opacities = []
        node_symbols = []
        
        # Use a vibrant color palette similar to the image
        color_palette = [
            '#FF3EA5', '#FF9E00', '#00C2FF', '#00FF88', '#AA44FF', 
            '#FFFF00', '#FF0000', '#0088FF', '#8BFF00', '#FF00FF',
            '#00FFFF', '#FF7700', '#00FF44', '#8800FF', '#00BBFF'
        ]
        
        # Assign colors based on node types or communities
        type_to_color = {}
        
        for node in subgraph.nodes:
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
            
            # Slightly vary node opacity for depth effect
            node_opacities.append(random.uniform(0.85, 1.0))
            
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
        
        # Create edge traces with gradient colors based on connected nodes
        edge_traces = []
        
        for edge in subgraph.edges():
            if edge[0] in pos and edge[1] in pos:  # Ensure nodes are in layout
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                
                # Create color gradient for edge based on connected nodes
                color_a = node_colors[list(subgraph.nodes).index(edge[0])]
                color_b = node_colors[list(subgraph.nodes).index(edge[1])]
                
                # Create a smooth edge with multiple points for gradient effect
                points = 15  # More points = smoother curve
                xs = []
                ys = []
                zs = []
                edge_colors = []
                
                for i in range(points):
                    # Add some curvature with sinusoidal functions
                    t = i / (points - 1)
                    # Interpolation with slight curve
                    x = x0 + (x1 - x0) * t + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                    y = y0 + (y1 - y0) * t + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                    z = z0 + (z1 - z0) * t + np.sin(np.pi * t) * 0.05 * (random.random() - 0.5)
                    
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    
                    # Color gradient along the edge
                    edge_colors.append(color_a if t < 0.5 else color_b)
                
                # Add None to create a break in the line (separate edges)
                xs.append(None)
                ys.append(None)
                zs.append(None)
                edge_colors.append(None)
                
                # Create edge trace
                edge_trace = go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(
                        width=2,
                        color=edge_colors,
                        opacity=0.6,
                    ),
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
        
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
                opacity=node_opacities,
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
        st.error(f"Error generating artistic visualization: {str(e)}")
        st.exception(e)  # Show the full exception for easier debugging
        st.warning("Please try with a smaller graph or use the standard visualization.")
        return None, None 