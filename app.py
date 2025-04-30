#!/usr/bin/env python3
"""
Talk to Code - Streamlit Application

This application provides a user interface for exploring and understanding code
repositories using semantic search and graph-based visualization.
"""

import os
import sys
import time
import tempfile
import streamlit as st
from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import queue
import threading
import re

# Import OpenAI client (using the modern approach)
from openai import OpenAI
import httpx

# For preventing proxies issue in the OpenAI client
original_httpx_client = httpx.Client

# Patch the httpx.Client constructor to filter out problematic parameters
def patched_httpx_client(*args, **kwargs):
    # Remove proxies parameter if present to avoid errors
    if 'proxies' in kwargs:
        print(f"Removing proxies config: {kwargs['proxies']}")
        del kwargs['proxies']
    return original_httpx_client(*args, **kwargs)

# Apply the patch
httpx.Client = patched_httpx_client

# Use lazy initialization to avoid recursion issues
def get_openai_client():
    """Get or initialize the OpenAI client."""
    # Do not pass any proxy or http_client settings to avoid recursion issues
    # The modern OpenAI client does not accept proxies parameter like the old one did
    client = OpenAI()
    return client

# Set OpenAI API key directly from environment or .env file
# Force reload the environment variable to ensure it's picked up
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Function to check and prompt for OpenAI API key
def check_api_key():
    """
    Check if OpenAI API key is set and prompt the user if it's missing.
    This function should be called early in the application startup.
    """
    from talktocode.utils.config import OPENAI_API_KEY, is_api_key_valid
    
    if not OPENAI_API_KEY:
        st.warning("OpenAI API key is not set. You need to provide an API key to use most features.")
        
        with st.form("openai_api_key_form"):
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            submitted = st.form_submit_button("Save API Key")
            
            if submitted and api_key:
                # Set environment variable
                os.environ["OPENAI_API_KEY"] = api_key
                
                # Update the module variable
                import talktocode.utils.config
                talktocode.utils.config.OPENAI_API_KEY = api_key
                
                # Validate the key
                if is_api_key_valid():
                    st.success("API key is valid and has been saved for this session.")
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error("The provided API key is invalid. Please check and try again.")
        
        st.info("You can also create a .env file in the project root with OPENAI_API_KEY=your_key")
        return False
    
    return True

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import UI components
from ui.ui_components import (
    initialize_ui_state,
    inject_base_css,
    create_header,
    create_sidebar,
    create_graph_container,
    create_chat_interface,
    create_enhanced_visualization,
    show_info,
    show_success,
    show_warning,
    show_error
)

# Import TalkToCode modules
from talktocode.utils.config import MODEL_CONFIG, set_embedding_model
from talktocode.indexing.entity_extractor import extract_entities_from_directory
from talktocode.indexing.relationship_extractor import extract_all_relationships
from talktocode.indexing.graph_builder import CodeKnowledgeGraph
from talktocode.indexing.community_detector import detect_hierarchical_communities
from talktocode.retrieval.search import search_codebase
from talktocode.rag.graph_rag_engine import GraphRAGEngine
from talktocode.utils.code_reference import CodeReference, format_code_references

# Load Custom CSS Styles
def load_ubs_styles():
    """
    Load and return custom CSS styling for UBS theme and components.
    REMOVED: This function now returns an empty string as custom CSS is disabled.
    """
    # Return empty string to remove all custom styles defined here
    return """ """

# Page Configuration
st.set_page_config(
    page_title="Talk to Code",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
def apply_custom_styling():
    """Apply custom styling to the app."""
    # The call below now injects an empty string, removing the old styles
    st.markdown(load_ubs_styles(), unsafe_allow_html=True)
    
    # Inject base CSS from the UI components module (check if inject_base_css uses UBS classes)
    inject_base_css()
    
    # Add conditional CSS based on sidebar state
    # Check if var(--ubs-red) is used here
    if "sidebar_collapsed" in st.session_state and st.session_state.sidebar_collapsed:
        # ... (CSS for collapsed sidebar - check for UBS vars)
        pass # Assuming no direct UBS refs here, might use standard colors
    else:
        # ... (CSS for expanded sidebar - check for UBS vars)
        pass # Assuming no direct UBS refs here
    
    # Highlight active tab if set - Check if var(--ubs-red) is used
    if "current_tab" in st.session_state:
        tab_name = st.session_state.current_tab
        # Replace var(--ubs-red) with a direct color like '#EC0016' or COMPANY_COLORS['red'] if needed
        st.markdown(f"""
        <style>
            /* Highlight the active tab */
            [data-baseweb="tab"][id$="{tab_name}"] {{
                position: relative;
            }}
            [data-baseweb="tab"][id$="{tab_name}"]::after {{
                content: '';
                position: absolute;
                bottom: -1px;
                left: 0;
                width: 100%;
                height: 3px;
                background-color: #EC0016; /* Replaced var(--ubs-red) */
                border-radius: 3px 3px 0 0;
            }}
        </style>
        """, unsafe_allow_html=True)

# Initialize Session State
def initialize_app_state():
    """Initialize application state variables."""
    # Initialize file upload state
    if "uploading_file" not in st.session_state:
        st.session_state.uploading_file = False
    
    # Initialize code directory
    if "code_dir" not in st.session_state:
        st.session_state.code_dir = None
    
    # Initialize processing state and results
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "community_detector" not in st.session_state:
        st.session_state.community_detector = None
    # Community reports state removed
    
    # Initialize UI state
    initialize_ui_state()
    
    # Initialize search settings
    if "current_search_settings" not in st.session_state:
        st.session_state.current_search_settings = {
            "strategy": "local",
            "params": {
                "max_results": 20,
                "include_code": True,
                "max_hops": 2
            }
        }
    if "visualization" not in st.session_state:
        st.session_state.visualization = {
            "level": 0,
            "max_nodes": 100
        }
    # Set default visualization mode to Artistic
    if "visualization_mode" not in st.session_state:
        st.session_state.visualization_mode = "Artistic"
    # Set default max nodes for display
    if "max_display_nodes" not in st.session_state:
        st.session_state.max_display_nodes = 200

# Initialize session state
initialize_app_state()

# File Processing Functions
def extract_zip(zip_file, extract_to):
    """Extract a zip file to the specified directory."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def handle_url_download(url, download_to):
    """
    Download a file from a URL and save it to the specified location.
    
    Args:
        url: URL to download from
        download_to: Path to save the downloaded file
        
    Returns:
        Path to the downloaded file
    """
    import requests
    from urllib.parse import urlparse
    
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:
            filename = "downloaded_file.zip"
        
        # Create the full path
        file_path = os.path.join(download_to, filename)
        
        # Download the file with progress
        with st.status(f"Downloading {filename} from URL...") as status:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Create a progress bar if size is known
            if total_size > 0:
                progress_bar = st.progress(0)
            
            # Download the file in chunks
            downloaded = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress if total size is known
                        if total_size > 0:
                            progress = int(downloaded / total_size * 100)
                            progress_bar.progress(progress / 100)
                            status.update(label=f"Downloading: {progress}% complete")
            
            status.update(label="Download complete!", state="complete")
        
        return file_path
    
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        raise e

def process_codebase(code_dir):
    """
    Process a codebase to build a knowledge graph and detect communities.
    
    Args:
        code_dir: Path to the code directory
        
    Returns:
        Tuple containing:
        - CodeKnowledgeGraph instance
        - HierarchicalCommunityDetector instance
    """
    # Set embedding model to text-embedding-ada-002
    set_embedding_model("text-embedding-ada-002")
    
    # Prepare progress tracking
    total_steps = 4  # Updated from 3 to 4 to match the actual number of steps
    current_step = 0
    
    def update_progress(step_name):
        nonlocal current_step
        current_step += 1
        progress = current_step / total_steps
        progress_bar.progress(progress, text=f"Step {current_step}/{total_steps}: {step_name}")
    
    # Create progress bar
    progress_bar = st.progress(0, text="Starting processing...")
    
    try:
        # Extract entities from code
        with st.status("Extracting code entities...") as status:
            update_progress("Extracting entities")
            all_entities = extract_entities_from_directory(code_dir)
            entity_counts = {entity_type: len(entities) for entity_type, entities in all_entities.items()}
            status.update(label=f"Extracted {sum(entity_counts.values())} entities", state="complete")
            st.write(f"Extracted entities: {entity_counts}")
        
        # Extract relationships
        with st.status("Extracting relationships...") as status:
            update_progress("Extracting relationships")
            relationships = extract_all_relationships(
                all_entities, 
                use_llm=False,  # Faster without LLM
                use_embeddings=True  # Use embeddings for semantic relationships
            )
            status.update(label=f"Extracted {len(relationships)} relationships", state="complete")
            st.write(f"Extracted {len(relationships)} relationships")
        
        # Build knowledge graph
        with st.status("Building knowledge graph...") as status:
            update_progress("Building knowledge graph")
            graph = CodeKnowledgeGraph()
            graph.build_from_entities_and_relationships(all_entities, relationships)
            status.update(label=f"Graph built with {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges", state="complete")
            st.write(f"Graph built with {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges")
        
        # Detect communities
        with st.status("Detecting hierarchical communities...") as status:
            update_progress("Detecting communities")
            community_detector = detect_hierarchical_communities(
                graph.graph, 
                num_levels=3  # 3 levels: fine-grained, module, architecture
            )
            
            # Print community statistics
            community_counts = []
            for level_idx in range(3):
                communities = community_detector.get_communities_at_level(level_idx)
                community_counts.append(len(communities))
                st.write(f"Level {level_idx+1}: {len(communities)} communities")
            
            status.update(label=f"Detected communities: {', '.join([str(c) for c in community_counts])}", state="complete")
        
        # Complete progress bar
        progress_bar.progress(1.0, text="Processing complete!")
        
        return graph, community_detector
    
    except Exception as e:
        # Update progress bar to show error
        progress_bar.progress(1.0, text=f"Error: {str(e)}")
        raise e

# Callback Functions
def handle_file_upload(uploaded_file=None, url=None):
    """
    Handle file upload and processing.
    
    Args:
        uploaded_file: Uploaded file object or directory path (optional)
        url: URL to download from (optional)
    """
    # Check if we have valid input
    if not uploaded_file and not url:
        show_error("Please upload a file or provide a URL.")
        return
    
    # Set uploading state for CSS styling
    st.session_state.uploading_file = True
    
    try:
        # Check if uploaded_file is already a directory path (string)
        if isinstance(uploaded_file, str) and os.path.isdir(uploaded_file):
            # Use the provided directory directly
            code_dir = uploaded_file
            show_info(f"Using provided directory for code analysis.")
        else:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            if uploaded_file:
                # Save the uploaded file to the temporary directory
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                show_info(f"File '{uploaded_file.name}' saved temporarily.")
            else:
                # Download the file from the URL
                file_path = handle_url_download(url, temp_dir)
                show_info(f"File downloaded from URL and saved temporarily.")
            
            # Extract the file if it's a zip
            if file_path.endswith('.zip'):
                extract_dir = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)
                code_dir = extract_zip(file_path, extract_dir)
                show_info(f"Zip file extracted to temporary directory.")
            else:
                code_dir = temp_dir
        
        # Store code directory in session state
        st.session_state.code_dir = code_dir
        
        # Display a header for processing details
        st.subheader("Processing details")
        
        # Process the codebase with interactive progress
        graph, community_detector = process_codebase(code_dir)
        
        # Store results in session state
        st.session_state.graph = graph
        st.session_state.community_detector = community_detector
        st.session_state.processing_complete = True
        
        show_success("Code processing complete! You can now search and explore the codebase.")
        
        # Save workspace to allow for reloading
        os.makedirs("temp", exist_ok=True)
        graph.to_json("temp/graph.json")
        
        # Suggest initial queries
        st.markdown("""
        ### Try asking:
        - What are the main components of this codebase?
        - How are the modules organized?
        - What are the key classes and their relationships?
        """)
        
        # Switch to graph tab after processing
        st.session_state.current_tab = "graph"
        st.rerun()
        
    except Exception as e:
        show_error(f"Error processing code: {str(e)}")
        st.exception(e)
    
    # Reset uploading state
    st.session_state.uploading_file = False

def handle_settings_change(settings):
    """
    Handle search settings changes.
    
    Args:
        settings: Dictionary with search strategy and parameters
    """
    st.session_state.current_search_settings = settings
    show_info(f"Search settings updated to {settings['strategy']} strategy.")

def format_response_with_references(response_text: str, code_references: List[CodeReference]) -> str:
    """
    Format the response text with code references.
    
    Args:
        response_text: The raw response text
        code_references: List of CodeReference objects
        
    Returns:
        Formatted response with proper code blocks and references
    """
    import re
    from talktocode.utils.code_reference import format_code_references
    
    # Process code references to include proper formatting
    formatted_text = response_text
    
    # Handle existing markdown code blocks (```...```)
    code_block_pattern = r'```([a-zA-Z]*)\n(.*?)```'
    
    # Store all code blocks to maintain proper formatting
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        return f"```{lang}\n{code}```"
    
    # Replace all code blocks with proper formatting
    formatted_text = re.sub(code_block_pattern, replace_code_block, formatted_text, flags=re.DOTALL)
    
    # Add formatted code references
    if code_references:
        # Use the utility function to format code references
        code_ref_text = format_code_references(code_references, include_snippets=True)
        
        # Add to the formatted text with appropriate separator
        if code_ref_text:
            formatted_text += "\n\n" + code_ref_text
    
    return formatted_text

def process_query(
    query: str, 
    graph: Optional[CodeKnowledgeGraph] = None,
    community_detector: Optional[Any] = None,
    community_reports: Optional[Dict[str, Any]] = None,
    search_settings: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Process a user query and generate a response.
    
    This function searches the codebase for relevant information,
    and generates a response based on the search results.
    
    Args:
        query: The user's query
        graph: The code knowledge graph
        community_detector: Optional community detector for structure info
        community_reports: Optional community reports
        search_settings: Settings for the search
        timeout: Maximum time to spend on the query
        
    Returns:
        A dictionary containing the response text, code references, and metadata
    """
    if graph is None:
        return {
            "response_text": "No codebase has been processed yet. Please upload and process a codebase first.",
            "code_references": [],
            "status": "error",
            "error_message": "No codebase processed",
            "metadata": {}
        }
    
    if search_settings is None:
        # Use default settings if none provided
        search_settings = {
            "strategy": "local",
            "params": {
                "max_results": 20,  # Increased from default of 10
                "min_similarity": 0.6,
                "max_distance": 2
            }
        }
    
    try:
        # Initialize the GraphRAGEngine
        rag_engine = GraphRAGEngine(
            graph=graph
        )
        
        # Apply minimal optimizations to search params to maintain quality
        params = search_settings.get("params", {})
        
        # Check if this is a structure-related query
        structure_keywords = ["structure", "architecture", "organization", "layout", "overview", "components"]
        query_lower = query.lower()
        is_structure_query = any(keyword in query_lower for keyword in structure_keywords)
        
        # --- Structure Query Handling ---
        if is_structure_query:
            print(f"DEBUG: Detected structure query: '{query}'") # DEBUG LINE
            if community_detector:
                print("DEBUG: Community detector found.") # DEBUG LINE
                # Get community information at the architectural level (level 2)
                level = 2  # Architectural level
                try:
                    communities = community_detector.get_communities_at_level(level)
                    print(f"DEBUG: Found {len(communities)} communities at level {level}") # DEBUG LINE
                    
                    if communities:
                        # Get the entities in each community
                        community_data = []
                        for comm_id, node_ids in communities.items():
                            nodes = [graph.graph.nodes[nid] for nid in node_ids if nid in graph.graph.nodes]
                            # Filter to only include nodes with names (exclude anonymous nodes)
                            named_nodes = [n for n in nodes if 'name' in n and n.get('name')]
                            
                            # Group by type
                            types = {}
                            for node in named_nodes:
                                node_type = node.get('type', 'Unknown')
                                if node_type not in types:
                                    types[node_type] = []
                                types[node_type].append(node.get('name', ''))
                            
                            # Determine community name from files or dominant node types
                            community_name = f"Community {comm_id}"
                            source_files = set()
                            for node in named_nodes:
                                if 'source_file' in node and node['source_file']:
                                    parts = node['source_file'].split('/')
                                    if len(parts) > 1:
                                        source_files.add(parts[0])  # Top-level dir
                            
                            if source_files:
                                community_name = f"Module: {', '.join(sorted(source_files)[:3])}"
                                if len(source_files) > 3:
                                    community_name += " and others"
                            
                            community_data.append({
                                "id": comm_id,
                                "name": community_name,
                                "size": len(named_nodes),
                                "types": types,
                                "nodes": named_nodes[:10]  # Limit to top 10 nodes for context
                            })
                        
                        # Sort communities by size
                        community_data.sort(key=lambda x: x["size"], reverse=True)
                        
                        # Structure the community information for the RAG response
                        structured_info = []
                        for i, comm in enumerate(community_data[:5]):  # Top 5 communities
                            section = f"Component {i+1}: {comm['name']}\n"
                            section += f"Size: {comm['size']} entities\n"
                            section += "Main entity types:\n"
                            
                            # List the main types and count
                            for t, entities in comm['types'].items():
                                if len(entities) > 0:
                                    section += f"- {t}: {len(entities)} entities\n"
                                    # List a few examples
                                    examples = entities[:3]
                                    if examples:
                                        section += f"  Examples: {', '.join(examples)}"
                                        if len(entities) > 3:
                                            section += f" and {len(entities) - 3} more"
                                        section += "\n"
                        
                        structured_info.append(section)
                        
                        # Create a special context for structure queries
                        structure_context = "--- Codebase Structure Overview ---\n\n"
                        structure_context += f"This codebase has {len(communities)} main components at the architectural level.\n\n"
                        structure_context += "\n\n".join(structured_info)
                        
                        # Add this structure overview to the context
                        context_entities = []
                        
                        # Still do a standard search but with modified parameters
                        print(f"Searching codebase with strategy: {search_settings['strategy']}")
                        search_results = search_codebase(
                            query=query,
                            graph=graph,
                            community_detector=community_detector,
                            community_reports=community_reports,
                            strategy=search_settings["strategy"],
                            params=params
                        )
                        
                        # Add the standard search results to our context
                        if search_results.get("entities"):
                            context_entities = search_results.get("entities")
                        
                        # Format the structure context as an entity
                        structure_entity = {
                            "id": "structure_overview",
                            "name": "Codebase Structure Overview",
                            "type": "Documentation",
                            "source_file": "",
                            "lineno": 0,
                            "end_lineno": 0,
                            "description": "An overview of the main components and modules in the codebase.",
                            "code_snippet": structure_context,
                            "community": None,
                            "distance": 0
                        }
                        
                        # Prepend the structure overview to ensure it gets priority
                        context_entities.insert(0, structure_entity)
                        
                        # Get the response from the RAG engine
                        print(f"Generating response with structure context and {len(context_entities)} context items")
                        response = rag_engine.generate_response(
                            query_text=query,
                            context=context_entities,
                            timeout=timeout
                        )
                        
                        return {
                            "response_text": response.get("content", ""),
                            "code_references": response.get("references", []),
                            "status": "success",
                            "error_message": "",
                            "metadata": {
                                "query_time": response.get("query_time", 0),
                                "context_items": len(context_entities),
                                "token_count": response.get("token_count", 0),
                                "is_structure_query": True
                            }
                        }
                    else:
                        print("DEBUG: No communities found at level 2, falling back to standard search.") # DEBUG LINE
                        is_structure_query = False # Force fallback
                except Exception as e:
                    print(f"DEBUG: Error getting/processing communities: {e}. Falling back.") # DEBUG LINE
                    is_structure_query = False # Force fallback
            else:
                print("DEBUG: Community detector NOT found, falling back to standard search.") # DEBUG LINE
                is_structure_query = False # Force fallback
        
        # --- Standard Search (or fallback) ---
        # This block now runs if it wasn't a structure query OR if structure handling failed
        if not is_structure_query: 
             print(f"DEBUG: Entering standard search path for query: '{query}'") # DEBUG LINE
             # Standard search for non-structure queries or if community info failed
             search_results = search_codebase(
                 query=query,
                 graph=graph,
                 community_detector=community_detector,
                 community_reports=community_reports,
                 strategy=search_settings["strategy"],
                 params=params
             )
        
        # Get the response from the RAG engine
        print(f"Generating response with {len(search_results.get('entities', []))} context items")
        response = rag_engine.generate_response(
            query_text=query,
            context=search_results.get("entities", []),
            timeout=timeout
        )
        
        return {
            "response_text": response.get("content", ""),
            "code_references": response.get("references", []),
            "status": "success",
            "error_message": "",
            "metadata": {
                "query_time": response.get("query_time", 0),
                "context_items": len(search_results.get("entities", [])),
                "token_count": response.get("token_count", 0)
            }
        }
    except Exception as e:
        print(f"Error in process_query: {e}")
        # traceback.print_exc()
        return {
            "response_text": f"I encountered an error while processing your query. Please try rephrasing or ask a different question.",
            "code_references": [],
            "status": "error",
            "error_message": str(e),
            "metadata": {}
        }

def handle_message(prompt):
    """
    Handle incoming chat messages and generate responses.
    
    Args:
        prompt: User's message text
    """
    # Check if we have a processed codebase
    if not st.session_state.get("processing_complete", False):
        response_text = "Please upload and process a codebase first before asking questions."
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.setdefault("chat_messages", []).append({
            "role": "assistant", 
            "content": response_text,
            "timestamp": timestamp
        })
        return
    
    # Get current search settings (use defaults if not set)
    search_settings = st.session_state.get("current_search_settings", {
        "strategy": "local",
        "params": {
            "max_results": 20,
            "include_code": True,
            "max_hops": 2
    }
    })
    
    # Standard timeout for queries
    timeout_seconds = 60  # Standard timeout for queries

    with st.status("Analyzing your question...", expanded=True) as status:
        # Create progress updates using milestones that provide more detail
        milestones = [
            ("Parsing question and identifying key concepts...", 0),
            ("Searching codebase for relevant entities...", 0.2),
            ("Analyzing code relationships...", 0.4),
            ("Retrieving context from similar code sections...", 0.6),
            ("Generating detailed response...", 0.8)
        ]
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Display initial milestone
        progress_text.write(milestones[0][0])
        
        # Function to update progress in a separate thread
        def update_progress():
            for i, (message, progress_value) in enumerate(milestones):
                time.sleep(0.3)  # Faster updates (reduced from 0.4)
                try:
                    progress_bar.progress(progress_value)
                    progress_text.write(message)
                except Exception as e:
                    # Silently handle missing context errors when running in thread
                    print(f"Progress update ignored: {str(e)}")
                
                # Don't sleep after the last milestone
                if i < len(milestones) - 1:
                    time.sleep(0.5)  # Reduced sleep time (from 0.6)
        
        # Start progress updates in a separate thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True  # Ensure thread terminates with main thread
        progress_thread.start()
        
        # Process the query with timeout
        response_data = process_query(
            query=prompt,
            graph=st.session_state.get("graph"),
            community_detector=st.session_state.get("community_detector"),
            community_reports=None,  # Setting to None since we removed community reports
            search_settings=search_settings,
            timeout=timeout_seconds
        )
        
        # Update final progress
        progress_bar.progress(1.0)
        
        # Different progress messages based on status
        if response_data["status"] == "success":
            progress_text.write("Response complete!")
            status.update(label="Response ready!", state="complete", expanded=False)
        elif response_data["status"] == "partial":
            progress_text.write("Partial results retrieved (timeout occurred)")
            status.update(label="Partial results available", state="running", expanded=True)
            st.warning("The full analysis couldn't complete in time, but I've provided some initial findings.")
        elif response_data["status"] == "timeout":
            progress_text.write("Analysis timed out")
            status.update(label="Query timed out", state="error", expanded=True)
            st.warning("The query took too long to process. Please try a more specific question.")
        else:
            progress_text.write("Error processing query")
            status.update(label="Error processing query", state="error", expanded=True)
            st.error(f"Error: {response_data['error_message']}")
    
    # --- Reference Filtering Logic --- 
    response_text = response_data.get("response_text", "")
    code_references = response_data.get("code_references", [])
    
    # Define simple greetings and generic responses
    simple_greetings = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay"]
    generic_responses = [
        "how can i assist", 
        "how can i help", 
        "couldn't find any relevant information",
        "don't have specific information about that",
        "based on the provided context"
        # Add other phrases if needed
    ]

    is_simple_greeting = prompt.strip().lower() in simple_greetings
    is_generic_response = any(phrase in response_text.lower() for phrase in generic_responses)

    # If it's a simple greeting OR the response seems generic/unrelated to specific code,
    # clear the code references.
    if is_simple_greeting or is_generic_response:
        print("INFO: Clearing code references for generic query/response.") # Optional logging
        code_references = [] 
    # --- End of Reference Filtering Logic --- 

    # Format the response (potentially with empty references now)
    formatted_response = format_response_with_references(response_text, code_references)
    
    # Add assistant message to chat history
    response_timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.setdefault("chat_messages", []).append({
        "role": "assistant", 
        "content": formatted_response,
        "raw_content": response_text, # Store raw text separately if needed
        "code_references": code_references, # Store the (potentially empty) list
        "timestamp": response_timestamp,
        "metadata": response_data.get("metadata", {})
    })

# Main Application
def main():
    """Main function to render the Streamlit app."""
    # Check if API key is valid before proceeding
    # Assuming check_api_key() is defined and works
    if not check_api_key():
        st.stop()  # Stop execution if API key setup is needed

    # Initialize state if not already done
    if "initialized" not in st.session_state:
        initialize_app_state() # Assuming this function initializes required states
        st.session_state.initialized = True
        # Initialize the current page - removing current_page since we no longer need navigation
        # st.session_state.current_page = "chat"

    # Load and apply custom styles
    # Ensure st.set_page_config is called only once here or within apply_custom_styling
    apply_custom_styling()
    
    # --- Header ---
    create_header() # Assuming this function is defined in ui_components
    
    # --- Sidebar ---
    # Assuming create_sidebar will be modified separately to remove settings
    create_sidebar(
        on_file_upload=handle_file_upload,
        on_settings_change=handle_settings_change # Callback might be unused now
    )
    
    # Removed sidebar navigation with Chat Interface and Code Communities buttons
    
    # --- Main Content Area ---
    # Always show the chat interface directly (no conditional page rendering)
    main_container = st.container()
    with main_container:
        # == Content previously in "Knowledge Graph" tab ==
        st.subheader("Code Knowledge Graph") # Added subheader for clarity
        graph_placeholder = st.empty() # Placeholder for the graph
        if st.session_state.get("processing_complete") and st.session_state.get("graph"):
            try:
                # Use the function that renders the graph visualization HTML
                from ui.ui_components import create_enhanced_visualization # Adjust import if needed

                # Access the internal NetworkX graph object
                networkx_graph_obj = st.session_state.graph.graph # Assuming .graph holds the nx object

                visualization_html = create_enhanced_visualization(
                    graph=networkx_graph_obj, # Pass the internal nx graph
                    # Pass communities if needed and available (adjust based on your structure)
                    communities=st.session_state.get("community_detector").get_communities_at_level(0) if st.session_state.get("community_detector") else None
                    # Community reports removed
                )
                graph_placeholder.markdown(visualization_html, unsafe_allow_html=True)
            except ImportError:
                graph_placeholder.error("Graph visualization component not found.")
            except AttributeError as ae:
                # More specific error handling if the attribute name is wrong
                if "'CodeKnowledgeGraph' object has no attribute 'graph'" in str(ae):
                    graph_placeholder.error("Internal Error: Cannot access the graph object within CodeKnowledgeGraph. Check attribute name (e.g., self.graph).")
                else:
                    graph_placeholder.error(f"Failed to render graph visualization (AttributeError): {ae}")
            except Exception as e:
                graph_placeholder.error(f"Failed to render graph visualization: {e}")
        elif st.session_state.get("code_dir") and not st.session_state.get("processing_complete"):
            st.info("Processing codebase... Please wait.")
        else:
            st.info("Upload a codebase using the sidebar to view the graph.")
        # == End of former "Knowledge Graph" tab content ==

        # Divider
        st.divider()

        # == Content previously in "Chat" tab ==
        st.subheader("Chat with Code") # Added subheader for clarity
        chat_container = st.container()
        with chat_container:
            # Use create_chat_interface or directly display messages
            # create_chat_interface(on_message_send=handle_message, is_processed=st.session_state.get("processing_complete", False))
            # OR, directly display messages:
            messages = st.session_state.get("chat_messages", [])
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    references = msg.get("references")
                    if references:
                        try:
                            with st.expander("Show References"):
                                from talktocode.utils.code_reference import format_code_references
                                st.markdown(format_code_references(references), unsafe_allow_html=True)
                        except ImportError:
                            st.warning("Could not display references (formatting function not found).")
                        except Exception as e:
                            st.error(f"Error displaying references: {e}")
        # == End of former "Chat" tab content ==

        # Note: Content from "Code Explorer" and "Settings" tabs is intentionally omitted.
    
    # --- Chat Input (Remains at the bottom, outside the main_container) ---
    chat_input_disabled = not st.session_state.get("processing_complete", False)
    prompt = st.chat_input("Ask a question about the code...", disabled=chat_input_disabled, key="chat_input")

    # --- Chat Processing Logic (Handles input and updates state) ---
    if prompt:
        st.session_state.setdefault("chat_messages", []).append(
            {"role": "user", "content": prompt, "timestamp": datetime.now()}
        )
        st.session_state.prompt_to_process = prompt
        st.rerun()
        
    if "prompt_to_process" in st.session_state and st.session_state.prompt_to_process:
        prompt_to_process = st.session_state.prompt_to_process
        st.session_state.prompt_to_process = None
        handle_message(prompt_to_process)
        st.rerun()
    
if __name__ == "__main__":
    main() 