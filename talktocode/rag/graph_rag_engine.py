"""
Graph-based RAG (Retrieval Augmented Generation) engine for the TalkToCode project.
This module integrates the code knowledge graph with the retrieval and generation process.
"""

import os
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Import OpenAI client
from openai import OpenAI

# Import local modules
from talktocode.indexing.graph_builder import CodeKnowledgeGraph
from talktocode.retrieval.search import GraphSearchEngine
from talktocode.utils.config import MODEL_CONFIG


class GraphRAGEngine:
    """
    Graph-based Retrieval Augmented Generation engine.
    Combines graph traversal with embedding-based retrieval for improved context retrieval.
    """
    
    def __init__(self, graph: Optional[CodeKnowledgeGraph] = None, 
                 search_engine: Optional[GraphSearchEngine] = None):
        """
        Initialize the Graph RAG engine.
        
        Args:
            graph: Code knowledge graph
            search_engine: Graph search engine
        """
        self.graph = graph
        self.search_engine = search_engine
        self.client = OpenAI()
        self.chat_model = MODEL_CONFIG["models"].get("chat", "gpt-3.5-turbo")
        
    def query(self, query_text: str, 
              max_results: int = 5,
              search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform a query using the Graph RAG approach.
        
        Args:
            query_text: The query text
            max_results: Maximum number of results to return
            search_params: Additional search parameters
            
        Returns:
            Dictionary containing search results and context
        """
        if not self.search_engine:
            return {"error": "Search engine not initialized", "results": []}
            
        if not search_params:
            search_params = {}
            
        # Set default parameters
        params = {
            "max_results": max_results,
            "traverse_graph": True,
            "use_embeddings": True,
            "include_code_context": True
        }
        params.update(search_params)
        
        # Use the search engine to find relevant entities
        try:
            results = self.search_engine.search(
                query=query_text,
                **params
            )
            
            return {
                "query": query_text,
                "results": results,
                "count": len(results) if results else 0,
                "search_params": params
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "query": query_text,
                "results": [],
                "count": 0
            }
    
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
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating chat completion: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}. Please try again with a different question."
    
    def generate_response(self, query_text: str, 
                        context: Optional[List[Dict[str, Any]]] = None,
                        search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response using retrieved context.
        
        Args:
            query_text: The query text
            context: Optional pre-retrieved context
            search_params: Additional search parameters
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        # If no context is provided, retrieve it
        if context is None:
            search_results = self.query(query_text, search_params=search_params)
            if "error" in search_results:
                return search_results
                
            context = search_results.get("results", [])
            
        # If search returned no results, return an error
        if not context:
            return {
                "query": query_text,
                "response": "I couldn't find any relevant information in the codebase to answer your question.",
                "context": [],
                "references": []
            }
        
        try:
            # Format context for the LLM
            formatted_context = "Here is information from the codebase:\n\n"
            
            # Extract information from the context
            for i, item in enumerate(context[:10]):  # Limit to first 10 items to avoid token limits
                if isinstance(item, dict):
                    # Get entity information
                    entity_type = item.get("type", "Unknown")
                    entity_name = item.get("name", "Unknown")
                    source_file = item.get("source_file", "Unknown")
                    description = item.get("description", "No description available")
                    code_snippet = item.get("code_snippet", "")
                    
                    # Format the entity information
                    formatted_context += f"--- Entity {i+1}: {entity_name} ({entity_type}) ---\n"
                    formatted_context += f"File: {source_file}\n"
                    formatted_context += f"Description: {description}\n"
                    
                    # Include code snippet if available and not too long
                    if code_snippet and len(code_snippet) < 1000:
                        formatted_context += f"Code:\n{code_snippet}\n\n"
                    else:
                        formatted_context += "\n"
            
            # Create messages for the chat completion
            messages = [
                {"role": "system", "content": 
                 "You are an AI assistant specialized in explaining code. Analyze the provided code entities and answer questions about them. "
                 "Be precise and technical, but explain concepts clearly. If you're not sure about something, say so instead of guessing. "
                 "When referring to code entities, include their type and file location to help the user locate them."
                },
                {"role": "user", "content": f"Here is information about code from a codebase:\n\n{formatted_context}\n\nMy question is: {query_text}"}
            ]
            
            # Generate response using the LLM
            response_text = self._generate_chat_completion(messages)
            
            # Extract references to include in the response
            references = []
            for item in context:
                if isinstance(item, dict) and "source_file" in item and "lineno" in item:
                    references.append({
                        "file": item.get("source_file", ""),
                        "line": item.get("lineno", 0),
                        "name": item.get("name", ""),
                        "type": item.get("type", ""),
                        "code": item.get("code_snippet", "")
                    })
            
            return {
                "query": query_text,
                "response": response_text,
                "context": context,
                "references": references
            }
            
        except Exception as e:
            # Return error message if something went wrong
            return {
                "query": query_text,
                "response": f"I encountered an error while processing your query: {str(e)}. Please try again with a different question.",
                "context": context,
                "references": []
            } 