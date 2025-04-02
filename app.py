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

# Monkey patch for OpenAI client to handle proxies compatibility issue
# This is needed because the newer OpenAI Python library handles proxies differently
import openai
from openai import OpenAI

# Set OpenAI API key directly from environment or .env file
# Force reload the environment variable to ensure it's picked up
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Double-check and set API key if available in environment
if os.environ.get("OPENAI_API_KEY"):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    print("API key loaded from environment")

# We need a more comprehensive approach to handle the proxy configuration
# The issue is that the API tries to pass proxies to the httpx Client directly
import openai._base_client
import httpx

# Store the original Client initialization
original_httpx_client = httpx.Client

# Create a wrapper for httpx.Client that filters out proxies parameter
def patched_httpx_client(*args, **kwargs):
    # Remove 'proxies' from kwargs if present
    if 'proxies' in kwargs:
        proxies = kwargs.pop('proxies')
        print(f"Removing proxies config: {proxies}")
    
    # Call the original Client with the modified kwargs
    return original_httpx_client(*args, **kwargs)

# Apply the monkey patch to httpx.Client
httpx.Client = patched_httpx_client

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
    UBS_COLORS,
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
from talktocode.indexing.report_generator import generate_community_reports
from talktocode.retrieval.search import search_codebase
from talktocode.rag.graph_rag_engine import GraphRAGEngine
from talktocode.utils.code_reference import CodeReference, format_code_references

# Load Custom CSS Styles
def load_ubs_styles():
    """
    Load and return custom CSS styling for UBS theme and components.
    """
    return """
    <style>
        /* ==================== UBS THEME VARIABLES ==================== */
        :root {
            --ubs-red: #EC0016;
            --ubs-light-red: #FF6D6A;
            --ubs-dark-red: #B30012;
            --ubs-blue: #0205A8;
            --ubs-light-blue: #9A9CFF;
            --ubs-dark-blue: #000066;
            --ubs-black: #000000;
            --ubs-dark-gray: #333333;
            --ubs-medium-gray: #666666;
            --ubs-light-gray: #CCCCCC;
            --ubs-white: #FFFFFF;
            --ubs-font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            --ubs-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --ubs-border-radius: 8px;
            --ubs-transition: all 0.3s ease;
        }

        /* ==================== BASE STYLING ==================== */
        body {
            font-family: var(--ubs-font-family);
            color: var(--ubs-dark-gray);
        }

        /* Override Streamlit's base styling */
        .stApp {
            background-color: var(--ubs-white);
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: var(--ubs-font-family);
            font-weight: 600;
        }

        /* Links in UBS colors */
        a {
            color: var(--ubs-blue);
            text-decoration: none;
        }
        a:hover {
            color: var(--ubs-dark-blue);
            text-decoration: underline;
        }

        /* Button styling */
        .stButton > button {
            background-color: var(--ubs-red);
            color: white;
            border: none;
            border-radius: var(--ubs-border-radius);
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: var(--ubs-transition);
        }
        .stButton > button:hover {
            background-color: var(--ubs-dark-red);
            box-shadow: var(--ubs-shadow);
        }
        .stButton > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(236, 0, 22, 0.25);
        }

        /* ==================== HEADER STYLING ==================== */
        .ubs-header-container {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--ubs-light-gray);
        }
        .ubs-main-header {
            font-size: 2.5rem;
            color: var(--ubs-red);
            font-weight: bold;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        .ubs-sub-header {
            font-size: 1.25rem;
            color: var(--ubs-blue);
            font-weight: 400;
        }

        /* ==================== SIDEBAR STYLING ==================== */
        /* Base sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            border-right: 1px solid var(--ubs-light-gray);
        }
        
        /* Sidebar headers */
        .sidebar .sidebar-content h1,
        .sidebar .sidebar-content h2,
        .sidebar .sidebar-content h3 {
            color: var(--ubs-red);
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--ubs-light-gray);
        }

        /* File uploader styling */
        .ubs-file-uploader {
            margin: 1rem 0;
            padding: 1.5rem;
            border: 2px dashed var(--ubs-light-gray);
            border-radius: var(--ubs-border-radius);
            background-color: #fafafa;
            text-align: center;
            transition: var(--ubs-transition);
        }
        .ubs-file-uploader:hover {
            border-color: var(--ubs-blue);
            background-color: #f0f4ff;
        }
        .ubs-file-uploader-icon {
            font-size: 2rem;
            color: var(--ubs-light-blue);
            margin-bottom: 0.5rem;
        }

        /* Sidebar toggle button */
        .sidebar-toggle-btn {
            position: absolute;
            top: 10px;
            right: -15px;
            width: 30px;
            height: 30px;
            background-color: var(--ubs-red);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 100;
            box-shadow: var(--ubs-shadow);
            font-weight: bold;
            transition: var(--ubs-transition);
        }
        .sidebar-toggle-btn:hover {
            background-color: var(--ubs-dark-red);
            transform: scale(1.1);
        }

        /* Collapsed sidebar */
        .sidebar-collapsed .css-1d391kg,
        .sidebar-collapsed .css-1lcbmhc {
            width: 0 !important;
            margin-left: -21rem;
            visibility: hidden;
        }
        .sidebar-collapsed .block-container {
            padding-left: 1rem;
            max-width: 100%;
        }

        /* ==================== CONTAINER STYLING ==================== */
        /* Clean container style */
        .ubs-container {
            background-color: white;
            border-radius: var(--ubs-border-radius);
            padding: 1.5rem;
            box-shadow: var(--ubs-shadow);
            margin-bottom: 1.5rem;
            border: 1px solid var(--ubs-light-gray);
        }

        /* Graph container specific styling */
        .graph-container {
            background-color: white;
            border-radius: var(--ubs-border-radius);
            padding: 1.5rem;
            box-shadow: var(--ubs-shadow);
            margin-bottom: 1.5rem;
            position: relative;
            min-height: 400px;
        }
        .graph-container-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--ubs-light-gray);
            padding-bottom: 0.5rem;
        }
        .graph-container-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--ubs-dark-blue);
        }
        .graph-controls {
            display: flex;
            gap: 0.5rem;
        }

        /* Hide Plotly and Matplotlib controls */
        .js-plotly-plot .plotly .modebar {
            display: none !important;
        }
        .js-plotly-plot .plotly .modebar-btn[data-title="Download plot as a png"] {
            display: inline-block !important;
        }
        .js-plotly-plot .plotly .modebar-btn:not([data-title="Download plot as a png"]) {
            display: none !important;
        }
        .mp-controls-container, .mp-controls-intercept, 
        .matplotlib-controls, .reportview-container .main footer {
            display: none !important;
        }

        /* ==================== STATUS MESSAGES ==================== */
        .status-box {
            padding: 1rem;
            border-radius: var(--ubs-border-radius);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        .status-box::before {
            content: '';
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border-radius: 50%;
            margin-right: 0.75rem;
        }
        .info-box {
            background-color: #e6f0ff;
            color: var(--ubs-blue);
            border-left: 5px solid var(--ubs-blue);
        }
        .info-box::before {
            background-color: var(--ubs-blue);
        }
        .success-box {
            background-color: #e6fff0;
            color: #1B5E20;
            border-left: 5px solid #1B5E20;
        }
        .success-box::before {
            background-color: #1B5E20;
        }
        .warning-box {
            background-color: #fff9e6;
            color: #F57F17;
            border-left: 5px solid #F57F17;
        }
        .warning-box::before {
            background-color: #F57F17;
        }
        .error-box {
            background-color: #ffe6e6;
            color: var(--ubs-dark-red);
            border-left: 5px solid var(--ubs-dark-red);
        }
        .error-box::before {
            background-color: var(--ubs-dark-red);
        }

        /* ==================== TAB STYLING ==================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: #f8f9fa;
            border-radius: var(--ubs-border-radius) var(--ubs-border-radius) 0 0;
            padding: 0.25rem 0.25rem 0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f0f0;
            border-radius: var(--ubs-border-radius) var(--ubs-border-radius) 0 0;
            padding: 0.5rem 1rem;
            border: 1px solid var(--ubs-light-gray);
            border-bottom: none;
            margin-right: 0.25rem;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--ubs-blue);
            color: white;
            border-color: var(--ubs-blue);
        }
        .stTabs [aria-selected="true"]:hover {
            background-color: var(--ubs-dark-blue);
        }
        .stTabs [aria-selected="false"]:hover {
            background-color: #e0e0e0;
        }
        
        /* Tab content container */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: white;
            border: 1px solid var(--ubs-light-gray);
            border-top: none;
            border-radius: 0 0 var(--ubs-border-radius) var(--ubs-border-radius);
            padding: 1rem;
        }

        /* ==================== CHAT INTERFACE ==================== */
        .chat-container {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--ubs-light-gray);
        }
        .chat-message {
            padding: 1rem;
            border-radius: var(--ubs-border-radius);
            margin-bottom: 1rem;
            max-width: 85%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        /* User messages */
        .stChatMessage[data-testid*="stChatMessage-user"] .chat-message {
            background-color: #f0f0f0;
            border-top-right-radius: 0;
            margin-left: auto;
            color: var(--ubs-dark-gray);
        }
        /* Assistant messages */
        .stChatMessage[data-testid*="stChatMessage-assistant"] .chat-message {
            background-color: var(--ubs-light-blue);
            border-top-left-radius: 0;
            margin-right: auto;
            color: var(--ubs-dark-blue);
        }
        /* Chat input */
        .stChatInput > div {
            padding: 0.5rem;
            border-radius: var(--ubs-border-radius);
            border: 1px solid var(--ubs-light-gray);
            background-color: white;
            transition: var(--ubs-transition);
        }
        .stChatInput > div:focus-within {
            border-color: var(--ubs-blue);
            box-shadow: 0 0 0 2px rgba(2, 5, 168, 0.2);
        }
        .stChatInput input {
            font-size: 1rem;
        }
        /* Send button */
        .stChatInput button {
            background-color: var(--ubs-red);
        }
        .stChatInput button:hover {
            background-color: var(--ubs-dark-red);
        }

        /* ==================== RESPONSIVE DESIGN ==================== */
        /* Different styles for smaller screens */
        @media (max-width: 768px) {
            .ubs-main-header {
                font-size: 2rem;
            }
            .ubs-sub-header {
                font-size: 1rem;
            }
            .graph-container {
                padding: 1rem;
                min-height: 300px;
            }
            .chat-message {
                max-width: 95%;
            }
        }
        
        /* Very small screens */
        @media (max-width: 480px) {
            .ubs-main-header {
                font-size: 1.75rem;
            }
            .sidebar-collapsed .css-1d391kg {
                margin-left: -100%;
            }
            .graph-container {
                padding: 0.75rem;
                min-height: 250px;
            }
        }

        /* ==================== ACTIVE ELEMENT HIGHLIGHTING ==================== */
        /* Highlight active elements */
        .highlight-element {
            border: 2px solid var(--ubs-red);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(236, 0, 22, 0.4);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(236, 0, 22, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(236, 0, 22, 0);
            }
        }
        
        /* Active tab indicator */
        .active-tab {
            position: relative;
        }
        .active-tab::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 3px;
            background-color: var(--ubs-red);
            border-radius: 3px 3px 0 0;
        }
        
        /* File upload status */
        .file-upload-in-progress .ubs-file-uploader {
            border-color: var(--ubs-blue);
            background-color: #f0f4ff;
            animation: pulse-blue 2s infinite;
        }
        
        @keyframes pulse-blue {
            0% {
                box-shadow: 0 0 0 0 rgba(2, 5, 168, 0.4);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(2, 5, 168, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(2, 5, 168, 0);
            }
        }

        /* ==================== TOGGLE BUTTONS ==================== */
        /* Sidebar toggle buttons */
        [data-testid="baseButton-secondary"]:has([key="collapse_sidebar"]) {
            position: fixed;
            top: 75px;
            right: calc(21rem - 30px);
            width: 30px;
            height: 30px;
            padding: 0 !important;
            border-radius: 50% !important;
            background-color: var(--ubs-red) !important;
            color: white !important;
            box-shadow: var(--ubs-shadow);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--ubs-transition);
        }
        
        [data-testid="baseButton-secondary"]:has([key="collapse_sidebar"]):hover {
            background-color: var(--ubs-dark-red) !important;
            transform: scale(1.1);
        }
        
        [data-testid="baseButton-secondary"]:has([key="expand_sidebar"]) {
            position: fixed;
            top: 75px;
            left: 10px;
            width: 30px;
            height: 30px;
            padding: 0 !important;
            border-radius: 50% !important;
            background-color: var(--ubs-red) !important;
            color: white !important;
            box-shadow: var(--ubs-shadow);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--ubs-transition);
        }
        
        [data-testid="baseButton-secondary"]:has([key="expand_sidebar"]):hover {
            background-color: var(--ubs-dark-red) !important;
            transform: scale(1.1);
        }
        
        /* Shift main content when sidebar is collapsed */
        .sidebar-collapsed .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            transition: all 0.3s ease;
        }
    </style>
    <!-- Keyboard shortcuts -->
    <script>
        document.addEventListener('keydown', function(e) {
            // Ctrl+/ or Cmd+/ - Focus on chat input
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                document.querySelector('.stChatInput input').focus();
            }
            
            // Ctrl+Shift+F or Cmd+Shift+F - Toggle sidebar
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'F') {
                e.preventDefault();
                const sidebarToggleButton = document.querySelector('[key="main_collapse_sidebar"], [key="main_expand_sidebar"], [key="collapse_sidebar"], [key="expand_sidebar"]');
                if (sidebarToggleButton) sidebarToggleButton.click();
            }
            
            // Ctrl+G or Cmd+G - Switch to graph visualization tab
            if ((e.ctrlKey || e.metaKey) && e.key === 'g') {
                e.preventDefault();
                // Find and click the graph tab
                const graphTab = document.querySelector('[data-baseweb="tab"][id$="graph"]');
                if (graphTab) graphTab.click();
            }
            
            // Ctrl+R or Cmd+R - Switch to search results tab
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                // Find and click the results tab
                const resultsTab = document.querySelector('[data-baseweb="tab"][id$="results"]');
                if (resultsTab) resultsTab.click();
            }
        });
    </script>
    """

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
    # Add UBS custom styles
    st.markdown(load_ubs_styles(), unsafe_allow_html=True)
    
    # Inject base CSS from the UI components module
    inject_base_css()
    
    # Add conditional CSS based on sidebar state
    if "sidebar_collapsed" in st.session_state and st.session_state.sidebar_collapsed:
        st.markdown("""
        <style>
            /* Add custom body class */
            body {
                overflow-x: hidden;
            }
            /* Make sidebar invisible when collapsed */
            [data-testid="stSidebar"] {
                width: 0px !important;
                margin-left: -21rem !important;
                visibility: hidden !important;
                transition: width 0.3s, margin-left 0.3s, visibility 0.3s;
            }
            /* Expand main content when sidebar is collapsed */
            [data-testid="stSidebar"] ~ .main .block-container {
                max-width: 100% !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
                transition: max-width 0.3s, padding 0.3s;
            }
            /* Show expand button */
            .sidebar-expand-btn {
                display: block;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            /* Normal sidebar styling */
            [data-testid="stSidebar"] {
                width: 21rem !important;
                margin-left: 0 !important;
                visibility: visible !important;
                transition: width 0.3s, margin-left 0.3s, visibility 0.3s;
            }
            /* Normal main content */
            [data-testid="stSidebar"] ~ .main .block-container {
                transition: max-width 0.3s, padding 0.3s;
            }
            /* Hide expand button when sidebar is visible */
            .sidebar-expand-btn {
                display: none;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Highlight active tab if set
    if "current_tab" in st.session_state:
        tab_name = st.session_state.current_tab
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
                background-color: var(--ubs-red);
                border-radius: 3px 3px 0 0;
            }}
        </style>
        """, unsafe_allow_html=True)

# Initialize Session State
def initialize_app_state():
    """Initialize session state for the application."""
    # UI state
    initialize_ui_state()
    
    # Application state
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "community_detector" not in st.session_state:
        st.session_state.community_detector = None
    if "community_reports" not in st.session_state:
        st.session_state.community_reports = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "code_dir" not in st.session_state:
        st.session_state.code_dir = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "current_search_settings" not in st.session_state:
        st.session_state.current_search_settings = {
            "strategy": "local",
            "params": {
                "max_hops": 2,
                "top_k_entities": 15,
                "min_similarity": 0.6,
                "include_code": True
            }
        }
    if "visualization" not in st.session_state:
        st.session_state.visualization = {
            "level": 0,
            "max_nodes": 100
        }
    if "uploading_file" not in st.session_state:
        st.session_state.uploading_file = False
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
    Process a codebase to build a knowledge graph, detect communities, and generate reports.
    
    Args:
        code_dir: Path to the code directory
        
    Returns:
        Tuple containing:
        - CodeKnowledgeGraph instance
        - HierarchicalCommunityDetector instance
        - Dictionary of community reports
    """
    # Set embedding model to text-embedding-ada-002
    set_embedding_model("text-embedding-ada-002")
    
    # Prepare progress tracking
    total_steps = 4
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
        
        # Generate community reports
        with st.status("Generating community reports with embeddings...") as status:
            reports = generate_community_reports(
                graph, 
                community_detector
            )
            status.update(label=f"Generated {len(reports)} community reports", state="complete")
            st.write(f"Generated {len(reports)} community reports")
        
        # Complete progress bar
        progress_bar.progress(1.0, text="Processing complete!")
        
        return graph, community_detector, reports
    
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
        graph, community_detector, community_reports = process_codebase(code_dir)
        
        # Store results in session state
        st.session_state.graph = graph
        st.session_state.community_detector = community_detector
        st.session_state.community_reports = community_reports
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
    
    # Convert code references to markdown format
    if code_references:
        formatted_text += "\n\n**Code References:**\n"
        
        for i, ref in enumerate(code_references):
            file_path = ref.file_path
            # Get just the filename if it's a full path
            if "/" in file_path:
                file_name = file_path.split("/")[-1]
            else:
                file_name = file_path
                
            start_line = ref.start_line
            end_line = ref.end_line
            
            # Add reference with line numbers
            formatted_text += f"\n**[{i+1}] `{file_name}` (Lines {start_line}-{end_line}):**\n"
            
            # Determine language from file name for syntax highlighting
            language = get_language_from_file(file_name)
            
            # Add code block with syntax highlighting - using snippet instead of code
            formatted_text += f"```{language}\n{ref.snippet}\n```\n"
    
    return formatted_text

def get_language_from_file(file_name: str) -> str:
    """
    Determine the programming language based on file extension for syntax highlighting.
    
    Args:
        file_name: Name of the file
        
    Returns:
        Language identifier for syntax highlighting
    """
    extension = file_name.split('.')[-1].lower() if '.' in file_name else ''
    
    language_map = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'jsx': 'jsx',
        'tsx': 'tsx',
        'html': 'html',
        'css': 'css',
        'json': 'json',
        'md': 'markdown',
        'sql': 'sql',
        'java': 'java',
        'c': 'c',
        'cpp': 'cpp',
        'h': 'c',
        'hpp': 'cpp',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'php': 'php',
        'sh': 'bash',
        'yaml': 'yaml',
        'yml': 'yaml',
    }
    
    return language_map.get(extension, '')

def process_query(
    query: str, 
    graph: Optional[CodeKnowledgeGraph] = None,
    community_detector: Optional[Any] = None,
    community_reports: Optional[Dict[str, Any]] = None,
    search_settings: Optional[Dict[str, Any]] = None,
    timeout: int = 30  # Reduced from 60 to 30 seconds
) -> Dict[str, Any]:
    """
    Process a natural language query about the code.
    
    Args:
        query: The natural language query
        graph: The code knowledge graph
        community_detector: The community detector
        community_reports: The community reports
        search_settings: Search settings
        timeout: Timeout in seconds (default: 30)
        
    Returns:
        Dictionary containing the response, code references, and other metadata
    """
    # Default search settings if not provided
    if not search_settings:
        search_settings = {
            "strategy": "local",
            "params": {
                "max_results": 10,
                "include_code": True,
                "max_hops": 2
            }
        }
    
    # Initialize results with default values
    results = {
        "response_text": "",
        "code_references": [],
        "status": "success",
        "error_message": "",
        "metadata": {
            "query_time": 0,
            "strategy": search_settings["strategy"],
            "search_params": search_settings["params"]
        }
    }
    
    # Start timing the query
    start_time = time.time()
    
    try:
        # Initialize the GraphRAGEngine
        rag_engine = GraphRAGEngine(
            graph=graph
        )
        
        # Optimize search parameters for faster response
        if "params" in search_settings:
            # Reduce max_results for faster search
            if "max_results" in search_settings["params"] and search_settings["params"]["max_results"] > 10:
                search_settings["params"]["max_results"] = 10
                
            # Limit max_hops to reduce graph traversal time
            if "max_hops" in search_settings["params"] and search_settings["params"]["max_hops"] > 2:
                search_settings["params"]["max_hops"] = 2
        
        # Process the query using the engine with optimized settings
        search_results = search_codebase(
            query=query,
            graph=graph,
            community_detector=community_detector,
            community_reports=community_reports,
            strategy=search_settings["strategy"],
            params=search_settings["params"]
        )
        
        # Get the response from the RAG engine
        response = rag_engine.generate_response(
            query_text=query,
            context=search_results.get("entities", []),
            search_params=search_settings["params"]
        )
        
        # Extract code references from the response
        code_references = extract_code_references_from_response(response)
        
        # Format code blocks in the response text
        response_text = response.get("response", "No response generated.")
        
        # Update the results with the RAG output
        results["response_text"] = response_text
        results["code_references"] = code_references
        results["search_results"] = search_results
        results["metadata"]["entities"] = len(search_results.get("entities", []))
        results["metadata"]["communities"] = len(search_results.get("communities", []))
        
    except Exception as e:
        results["status"] = "error"
        results["error_message"] = str(e)
        traceback.print_exc()
    
    # Calculate query time
    end_time = time.time()
    results["metadata"]["query_time"] = end_time - start_time
    
    return results

# Helper function to extract code references from response
def extract_code_references_from_response(response):
    """
    Extract code references from the RAG engine response.
    
    Args:
        response: The response dictionary from the RAG engine
        
    Returns:
        List of CodeReference objects
    """
    code_references = []
    references = response.get("references", [])
    
    for ref in references:
        # Skip invalid references
        if not ref or not isinstance(ref, dict):
            continue
            
        # Only process references with required fields
        if "file" not in ref or not ref["file"]:
            continue
            
        # Try to create a CodeReference object
        try:
            # Get line numbers with appropriate defaults
            start_line = int(ref.get("line", 1))
            # Use end_line if provided, otherwise default to start_line + a small number
            end_line = int(ref.get("end_line", start_line + 10))
            
            # Ensure we have a valid file path
            file_path = ref["file"]
            if not file_path or not isinstance(file_path, str):
                continue
                
            # Extract code snippet, with fallback
            code_snippet = ref.get("code", "")
            if not code_snippet and "code_snippet" in ref:
                code_snippet = ref["code_snippet"]
                
            # Create the reference object
            code_ref = CodeReference(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                entity_name=ref.get("name", ""),
                entity_type=ref.get("type", ""),
                snippet=code_snippet
            )
            code_references.append(code_ref)
        except Exception as e:
            # Log error details but continue processing other references
            print(f"Error creating code reference: {str(e)}")
            print(f"Reference data: {ref}")
    
    return code_references

def process_query_with_timeout(
    query: str,
    graph: Optional[CodeKnowledgeGraph] = None,
    community_detector: Optional[Any] = None,
    community_reports: Optional[Dict[str, Any]] = None,
    search_settings: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Process a query with a timeout to prevent hanging.
    
    Args:
        query: The query string
        graph: Code knowledge graph
        community_detector: Community detector
        community_reports: Community reports
        search_settings: Search settings
        timeout: Timeout in seconds (default: 30)
        
    Returns:
        Dictionary with the response data
    """
    # Queue for the result
    result_queue = queue.Queue()
    
    # Function to process the query and put the result in the queue
    def process_and_queue():
        try:
            result = process_query(
                query=query,
                graph=graph,
                community_detector=community_detector,
                community_reports=community_reports,
                search_settings=search_settings
            )
            result_queue.put(result)
        except Exception as e:
            result_queue.put({
                "response_text": f"Error processing query: {str(e)}",
                "code_references": [],
                "status": "error",
                "error_message": str(e),
                "metadata": {}
            })
    
    # Start the processing thread
    thread = threading.Thread(target=process_and_queue)
    thread.daemon = True  # Allow the thread to be terminated when the main thread exits
    thread.start()
    
    try:
        # Wait for the result with timeout
        result = result_queue.get(timeout=timeout)
        return result
    except queue.Empty:
        # Timeout occurred
        return {
            "response_text": "I'm sorry, but your query took too long to process. Please try asking a more specific question about the codebase.",
            "code_references": [],
            "status": "timeout",
            "error_message": "Query processing timed out",
            "metadata": {"query_time": timeout}
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
        
        # Add assistant message to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": response_text,
            "timestamp": timestamp
        })
        return
    
    # Get current search settings
    search_settings = {
        "strategy": st.session_state.get("search_strategy", "local"),
        "params": st.session_state.get("search_params", {
            "max_results": 10,
            "include_code": True,
            "max_hops": 2
        })
    }
    
    # Show processing status with a more detailed, dynamically updating indicator
    with st.status("Analyzing your question...", expanded=True) as status:
        # Create progress updates using milestones
        milestones = [
            ("Searching codebase...", 0),
            ("Analyzing code entities...", 0.3),
            ("Retrieving relevant context...", 0.6),
            ("Generating response...", 0.8)
        ]
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Display initial milestone
        progress_text.write(milestones[0][0])
        
        # Function to update progress in a separate thread
        def update_progress():
            for i, (message, progress_value) in enumerate(milestones):
                time.sleep(0.5)  # Small delay between updates
                progress_bar.progress(progress_value)
                progress_text.write(message)
                
                # Don't sleep after the last milestone
                if i < len(milestones) - 1:
                    time.sleep(0.8)
        
        # Start progress updates in a separate thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()
        
        # Process the query with reduced timeout (30 seconds)
        response_data = process_query_with_timeout(
            query=prompt,
            graph=st.session_state.graph,
            community_detector=st.session_state.community_detector,
            community_reports=st.session_state.community_reports,
            search_settings=search_settings,
            timeout=30  # Reduced timeout for faster response
        )
        
        # Update final progress
        progress_bar.progress(1.0)
        progress_text.write("Response complete!")
        
        # Update status based on response
        if response_data["status"] == "success":
            status.update(label="Response ready!", state="complete", expanded=False)
        elif response_data["status"] == "timeout":
            status.update(label="Query timed out", state="error", expanded=True)
            st.warning("The query took too long to process. Please try a more specific question.")
        else:
            status.update(label="Error processing query", state="error", expanded=True)
            st.error(f"Error: {response_data['error_message']}")
    
    # Format the response with code references
    response_text = response_data["response_text"]
    code_references = response_data.get("code_references", [])
    
    # Create a formatted response with code blocks and references
    formatted_response = format_response_with_references(response_text, code_references)
    
    # Add assistant message to chat history
    response_timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_messages.append({
        "role": "assistant", 
        "content": formatted_response,
        "raw_content": response_text,
        "code_references": code_references,
        "timestamp": response_timestamp,
        "metadata": response_data.get("metadata", {})
    })

# Handle tab change
def handle_tab_change(tab_name):
    """Update the current tab in session state."""
    st.session_state.current_tab = tab_name

# Toggle sidebar visibility
def toggle_sidebar():
    """Toggle sidebar visibility by updating session state."""
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False
    
    # Toggle the state
    st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
    
    # Force rerun to apply changes
    st.rerun()

def check_required_packages():
    """
    Check if all required packages are installed.
    Provides guidance if any are missing.
    
    Returns:
        bool: True if all packages are installed, False otherwise
    """
    required_packages = {
        "streamlit": "Streamlit UI framework",
        "networkx": "Graph operations and structure",
        "plotly": "Interactive visualizations",
        "requests": "HTTP client for API calls",
        "numpy": "Numerical computations",
        "matplotlib": "Basic plotting capabilities"
    }
    
    optional_packages = {
        "community": "Python-Louvain package for community detection (optional)",
        "watchdog": "Improved file watching for Streamlit (optional)"
    }
    
    missing_packages = []
    missing_optional_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append((package, description))
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional_packages.append((package, description))
    
    if missing_packages:
        st.error("### Missing Required Packages")
        st.markdown("The following packages are required but not installed:")
        
        for package, description in missing_packages:
            st.markdown(f"- **{package}**: {description}")
        
        st.markdown("""
        ### Installation Instructions
        
        Run the following command to install all required packages:
        ```
        pip install streamlit networkx plotly requests numpy matplotlib
        ```
        
        Then restart the application.
        """)
        return False
    
    if missing_optional_packages:
        st.warning("### Optional Packages Not Installed")
        st.markdown("The following optional packages are not installed:")
        
        for package, description in missing_optional_packages:
            st.markdown(f"- **{package}**: {description}")
        
        st.markdown("""
        Some features may be limited. You can install optional packages with:
        ```
        pip install python-louvain watchdog
        ```
        """)
    
    return True

# Add fullscreen toggle JavaScript
def add_fullscreen_toggle():
    """Add a fullscreen toggle button using JavaScript."""
    st.markdown("""
    <style>
        .fullscreen-button {
            position: fixed;
            top: 15px;
            right: 15px;
            background-color: var(--ubs-red);
            color: white;
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 9999;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .fullscreen-button:hover {
            background-color: var(--ubs-dark-red);
            transform: scale(1.1);
        }
        .fullscreen-button svg {
            width: 18px;
            height: 18px;
        }
    </style>
    
    <button onclick="toggleFullscreen()" class="fullscreen-button" title="Toggle Fullscreen (F11)">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
        </svg>
    </button>
    
    <script>
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                // Go fullscreen
                if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen();
                } else if (document.documentElement.mozRequestFullScreen) { /* Firefox */
                    document.documentElement.mozRequestFullScreen();
                } else if (document.documentElement.webkitRequestFullscreen) { /* Chrome, Safari & Opera */
                    document.documentElement.webkitRequestFullscreen();
                } else if (document.documentElement.msRequestFullscreen) { /* IE/Edge */
                    document.documentElement.msRequestFullscreen();
                }
            } else {
                // Exit fullscreen
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.mozCancelFullScreen) { /* Firefox */
                    document.mozCancelFullScreen();
                } else if (document.webkitExitFullscreen) { /* Chrome, Safari & Opera */
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) { /* IE/Edge */
                    document.msExitFullscreen();
                }
            }
        }
        
        // Also listen for F11 key to toggle fullscreen
        document.addEventListener('keydown', function(e) {
            if (e.key === 'F11') {
                e.preventDefault();
                toggleFullscreen();
            }
        });
    </script>
    """, unsafe_allow_html=True)

# Main Application
def main():
    """Main function to render the Streamlit app."""
    # Initialize app state and styles
    initialize_app_state()
    apply_custom_styling()
    
    # Create header
    create_header()
    
    # Create sidebar with callbacks
    settings = create_sidebar(
        on_file_upload=handle_file_upload,
        on_settings_change=handle_settings_change
    )
    
    # Move debug information to a collapsed expander in the sidebar
    with st.sidebar:
        # Add debug information in a collapsed expander for developers
        with st.expander("Developer Debug Info", expanded=False):
            st.write("### Debug Information")
            
            # Check if graph is loaded
            if "graph" in st.session_state and st.session_state.graph is not None:
                st.success(f"✓ Graph loaded: {len(st.session_state.graph.graph.nodes) if hasattr(st.session_state.graph, 'graph') else 'No nodes'} nodes")
            else:
                st.warning("⚠️ No graph loaded")
                
            # Check if community detector is loaded
            if "community_detector" in st.session_state and st.session_state.community_detector is not None:
                st.success("✓ Community detector loaded")
            else:
                st.warning("⚠️ No community detector")
                
            # Check if community reports are available
            if "community_reports" in st.session_state and st.session_state.community_reports is not None:
                st.success(f"✓ Community reports available")
            else:
                st.warning("⚠️ No community reports")
                
            # Check if processing is complete
            if "processing_complete" in st.session_state and st.session_state.processing_complete:
                st.success("✓ Processing complete")
            else:
                st.warning("⚠️ Processing not complete")
                
            # Check current tab
            if "current_tab" in st.session_state:
                st.info(f"Current tab: {st.session_state.current_tab}")
        
        # Add visualization mode selector
        if "graph" in st.session_state and st.session_state.graph is not None:
            st.write("### Visualization Settings")
            viz_mode = st.selectbox(
                "Visualization Mode",
                options=["Standard", "Artistic"],
                index=0,
                help="Choose the graph visualization style"
            )
            if "visualization_mode" not in st.session_state or st.session_state.visualization_mode != viz_mode:
                st.session_state.visualization_mode = viz_mode
    
    # Store the current settings
    if "search_settings" not in st.session_state:
        st.session_state.search_settings = settings
    
    # Check if page should be in fullscreen mode
    if st.session_state.get("fullscreen", False):
        add_fullscreen_toggle()
    
    # Initialize the appropriate RAG engine based on settings
    # Ensure we have all required objects before initializing
    if "graph" in st.session_state and st.session_state.graph is not None:
        # Check if we're using global search (requires community detector)
        using_global = settings["strategy"] == "global"
        
        if (using_global and 
            "community_detector" in st.session_state and 
            st.session_state.community_detector is not None and
            "community_reports" in st.session_state and 
            st.session_state.community_reports is not None):
            
            # Initialize global search engine
            if "rag_engine" not in st.session_state or not isinstance(st.session_state.rag_engine, GraphRAGEngine):
                try:
                    # Create the RAG engine with just the graph parameter
                    st.session_state.rag_engine = GraphRAGEngine(
                        graph=st.session_state.graph
                    )
                    
                    # Separately create a search engine for search operations if needed
                    if "search_engine" not in st.session_state:
                        from talktocode.retrieval.search import GraphSearchEngine
                        st.session_state.search_engine = GraphSearchEngine(
                            graph=st.session_state.graph,
                            community_detector=st.session_state.community_detector,
                            community_reports=st.session_state.community_reports
                        )
                except Exception as e:
                    st.error(f"Error initializing RAG engine: {str(e)}")
                    st.exception(e)
        
        elif not using_global:
            # Initialize local search engine (doesn't require community detector)
            if "rag_engine" not in st.session_state or not isinstance(st.session_state.rag_engine, GraphRAGEngine):
                try:
                    # Create the RAG engine with just the graph parameter
                    st.session_state.rag_engine = GraphRAGEngine(
                        graph=st.session_state.graph
                    )
                    
                    # Separately create a search engine for search operations if needed
                    if "search_engine" not in st.session_state:
                        from talktocode.retrieval.search import GraphSearchEngine
                        st.session_state.search_engine = GraphSearchEngine(
                            graph=st.session_state.graph
                        )
                except Exception as e:
                    st.error(f"Error initializing RAG engine: {str(e)}")
                    st.exception(e)
    
    # Define tab names for easy reference
    tab_names = ["Knowledge Graph", "Chat", "Code Explorer", "Settings"]
    
    # Create main content tabs
    tabs = st.tabs(tab_names)
    
    # Track which tab is selected (index-based)
    if "active_tab_index" not in st.session_state:
        st.session_state.active_tab_index = 0
    
    # Knowledge Graph tab
    with tabs[0]:
        if st.session_state.active_tab_index != 0:
            st.session_state.active_tab_index = 0
            st.session_state.current_tab = tab_names[0]
            
        # Check if processing is complete
        is_processed = st.session_state.get("processing_complete", False)
        
        graph_data = None
        if is_processed and "graph" in st.session_state and st.session_state.graph is not None:
            # Use the graph from session state
            graph_data = st.session_state.graph.graph
        
        # Get community data if available
        communities = None
        if is_processed and "community_detector" in st.session_state and st.session_state.community_detector is not None:
            # Get communities at the appropriate level
            level = st.session_state.get("visualization", {}).get("level", 0)
            communities = st.session_state.community_detector.get_communities_at_level(level)
        
        # Get search results if available
        search_results = st.session_state.get("search_results", None)
        
        # Add prominent visualization mode selector directly in the graph tab
        if is_processed and graph_data is not None:
            st.markdown("### Visualization Style")
            cols = st.columns([3, 3, 2])
            
            with cols[0]:
                viz_mode = st.radio(
                    "Select Visualization Style",
                    options=["Standard", "Artistic"],
                    index=1 if st.session_state.get("visualization_mode") == "Artistic" else 0,
                    horizontal=True
                )
                if viz_mode != st.session_state.get("visualization_mode"):
                    st.session_state.visualization_mode = viz_mode
                    st.rerun()
            
            with cols[1]:
                max_nodes = st.slider(
                    "Max Nodes to Display",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Higher values show more nodes but may slow down visualization"
                )
                if "max_display_nodes" not in st.session_state or st.session_state.max_display_nodes != max_nodes:
                    st.session_state.max_display_nodes = max_nodes
            
            with cols[2]:
                if st.button("🔄 Refresh Visualization", use_container_width=True):
                    st.rerun()
        
        # Create graph container with visualization mode
        if is_processed and graph_data is not None:
            # Get visualization mode
            viz_mode = st.session_state.get("visualization_mode", "Artistic")  # Default to Artistic now
            max_nodes = st.session_state.get("max_display_nodes", 200)
            
            if viz_mode == "Artistic":
                # Import the artistic visualization function
                from ui.ui_components import create_artistic_graph_visualization
                
                # Create the artistic visualization
                st.markdown("## Artistic Code Knowledge Graph")
                st.markdown("This visualization shows your code as an interactive 3D network with flowing connections.")
                st.markdown("**Instructions:** Drag to rotate, scroll to zoom, and hover over nodes to see details. Node names become visible when you zoom in.")
                
                create_artistic_graph_visualization(
                    graph=graph_data,
                    communities=communities,
                    max_nodes=max_nodes,
                    view_mode="artistic"
                )
            else:
                # Create standard graph container
                create_graph_container(
                    is_processed=is_processed,
                    graph=graph_data,
                    communities=communities,
                    search_results=search_results
                )
    
    # Chat tab
    with tabs[1]:
        if st.session_state.active_tab_index != 1:
            st.session_state.active_tab_index = 1
            st.session_state.current_tab = tab_names[1]
            
        # Create chat interface that calls handle_message but without chat_input
        create_chat_interface(
            on_message_send=handle_message,
            is_processed=st.session_state.get("processing_complete", False)
        )
    
    # Code Explorer tab
    with tabs[2]:
        if st.session_state.active_tab_index != 2:
            st.session_state.active_tab_index = 2
            st.session_state.current_tab = tab_names[2]
            
        st.header("Code Explorer")
        
        # Check if processing is complete
        if not st.session_state.get("processing_complete", False):
            st.info("Process your code to enable the Code Explorer.")
        else:
            # Create file browser
            if "graph" in st.session_state and st.session_state.graph is not None:
                # Get file list
                files = st.session_state.graph.get_files()
                
                if files:
                    # Create file selector
                    selected_file = st.selectbox(
                        "Select a file to view",
                        options=files,
                        format_func=lambda x: os.path.basename(x)
                    )
                    
                    # Show file content
                    if selected_file:
                        try:
                            with open(selected_file, "r") as f:
                                content = f.read()
                            
                            language = get_language_from_file(selected_file)
                            
                            # Show the file content with syntax highlighting
                            st.code(content, language=language)
                            
                            # Add download button
                            st.download_button(
                                "Download File",
                                content,
                                file_name=os.path.basename(selected_file),
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Error loading file: {str(e)}")
                else:
                    st.warning("No files found in the graph.")
            else:
                st.warning("Graph not available. Please process your code first.")
    
    # Settings tab
    with tabs[3]:
        if st.session_state.active_tab_index != 3:
            st.session_state.active_tab_index = 3
            st.session_state.current_tab = tab_names[3]
            
        st.header("Application Settings")
        
        # API key settings
        api_key = st.text_input(
            "OpenAI API Key (optional)",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Enter your OpenAI API key to use ChatGPT API"
        )
        
        if st.button("Save API Key"):
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved for this session.")
            else:
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]
                st.info("API key removed.")
        
        # UI settings
        st.subheader("UI Settings")
        
        # Toggle fullscreen mode
        fullscreen = st.checkbox(
            "Fullscreen Mode",
            value=st.session_state.get("fullscreen", False),
            help="Remove Streamlit's default header and footer"
        )
        
        if fullscreen != st.session_state.get("fullscreen", False):
            st.session_state.fullscreen = fullscreen
            st.rerun()
        
        # Advanced settings
        st.subheader("Advanced Settings")
        
        # Timeout setting
        query_timeout = st.slider(
            "Query Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.get("query_timeout", 60),
            step=10,
            help="Maximum time allowed for processing queries"
        )
        
        if query_timeout != st.session_state.get("query_timeout", 60):
            st.session_state.query_timeout = query_timeout
            
        # Reset application button
        if st.button("Reset Application", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
                
            # Reinitialize
            initialize_app_state()
            st.success("Application has been reset.")
            st.rerun()
    
    # Create custom chat display if enabled
    create_custom_chat_display()
    
    # Initialize current_tab if it hasn't been set yet
    if "current_tab" not in st.session_state and "active_tab_index" in st.session_state:
        st.session_state.current_tab = tab_names[st.session_state.active_tab_index]
    
    # Add chat input OUTSIDE of tabs and call the handle_chat_input function
    is_processed = st.session_state.get("processing_complete", False)
    placeholder = "Ask a question about your code..." if is_processed else "Upload and process code to enable chat"
    
    from ui.ui_components import handle_chat_input
    message = st.chat_input(placeholder, disabled=not is_processed, key="main_chat_input")
    handle_chat_input(message, is_processed)

def create_custom_chat_display():
    """
    Create a custom chat display that supports code formatting and references.
    
    This function replaces the standard create_chat_interface function with enhanced
    styling and support for code blocks and references.
    """
    # Add custom CSS for enhanced chat styling
    st.markdown("""
    <style>
    /* Enhanced chat styling */
    .chat-container {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--ubs-light-gray);
    }
    
    /* Chat header */
    .chat-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .chat-icon {
        margin-right: 0.5rem;
        font-size: 1.5rem;
    }
    
    /* Message styling */
    .code-reference {
        background-color: #f8f9fa;
        border-left: 3px solid var(--ubs-blue);
        padding: 10px;
        margin-top: 5px;
        border-radius: 0 4px 4px 0;
        font-family: monospace;
    }
    
    /* Code blocks */
    pre {
        background-color: #f6f8fa;
        border-radius: 6px;
        padding: 16px;
        overflow-x: auto;
    }
    
    code {
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.9em;
    }
    
    /* File references */
    .file-reference {
        color: var(--ubs-blue);
        font-weight: bold;
        cursor: pointer;
    }
    
    /* Message metadata */
    .message-metadata {
        font-size: 0.8em;
        color: var(--ubs-medium-gray);
        margin-top: 5px;
        font-style: italic;
    }
    
    /* Error message styling */
    .error-message {
        background-color: #ffebee;
        border-left: 3px solid var(--ubs-red);
        padding: 10px;
        margin-top: 5px;
        border-radius: 0 4px 4px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat header with icon
    st.markdown('<div class="chat-header"><span class="chat-icon">💬</span><h2>Chat with Your Code</h2></div>', unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        # Initialize chat_messages in session state if not present
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Get messages from session state
        messages = st.session_state.get("chat_messages", [])
        
        # Display chat messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            timestamp = message.get("timestamp", "")
            
            with st.chat_message(role):
                # Display message content
                st.markdown(content, unsafe_allow_html=True)
                
                # Display metadata if it exists
                if "metadata" in message and message["metadata"]:
                    metadata = message["metadata"]
                    with st.expander("Query Details", expanded=False):
                        st.markdown(f"**Query time:** {metadata.get('query_time', 0):.2f} seconds")
                        st.markdown(f"**Strategy:** {metadata.get('strategy', 'Unknown')}")
                        
                        # Show entities/communities if available
                        if "entities" in metadata:
                            st.markdown(f"**Entities found:** {metadata['entities']}")
                        if "communities" in metadata:
                            st.markdown(f"**Communities found:** {metadata['communities']}")
                
                # Show timestamp
                if timestamp:
                    st.caption(f"Sent: {timestamp}")
    
    # Create status indicator for chat availability
    if not st.session_state.get("processing_complete", False):
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px; color: #666;">
            <span style="width: 8px; height: 8px; background-color: #EF5350; border-radius: 50%; margin-right: 8px;"></span>
            Please upload and process code to enable chat
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px; color: #4CAF50;">
            <span style="width: 8px; height: 8px; background-color: #4CAF50; border-radius: 50%; margin-right: 8px;"></span>
            Ready to answer questions about your code
        </div>
        """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main() 