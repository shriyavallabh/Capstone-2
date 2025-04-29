"""
Graph-based RAG (Retrieval Augmented Generation) engine for the TalkToCode project.
This module integrates the code knowledge graph with the retrieval and generation process.
"""

import os
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json
import hashlib
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
        self._client = None
        # Use a faster model for chat completions
        self.chat_model = MODEL_CONFIG["models"].get("chat", "gpt-3.5-turbo-1106")
        # Add response cache to avoid redundant API calls
        self.response_cache = {}
    
    def get_client(self):
        """Get or initialize the OpenAI client."""
        if self._client is None:
            try:
                # Import the shared client function from the main app
                from app import get_openai_client
                self._client = get_openai_client()
            except ImportError:
                # Fallback if the import fails
                self._client = OpenAI()
        return self._client
    
    def query(self, query_text: str, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search query using the graph search engine.
        
        Args:
            query_text: The query text
            search_params: Additional search parameters
            
        Returns:
            Dictionary containing search results and metadata
        """
        if self.search_engine is None:
            if self.graph is None:
                return {"error": "No graph available for search."}
                
            # Create a search engine using the graph
            from talktocode.retrieval.search import GraphSearchEngine
            self.search_engine = GraphSearchEngine(self.graph)
        
        # Execute search
        try:
            results = self.search_engine.search(query_text, params=search_params)
            return {
                "query": query_text,
                "results": results,
                "metadata": {
                    "strategy": search_params.get("strategy", "local") if search_params else "local",
                    "num_results": len(results)
                }
            }
        except Exception as e:
            return {
                "error": f"Error executing search: {str(e)}",
                "query": query_text,
                "results": []
            }
    
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
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
            response = self.get_client().chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.2,  # Lower temperature for more deterministic responses
                max_tokens=800    # Limit tokens for faster responses
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
        # Check cache first to avoid redundant API calls
        cache_key = self._generate_cache_key(query_text, context)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
            
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
            # Format context for the LLM, allowing for a much larger context
            formatted_context = "Here is information from the codebase:\n\n"
            current_char_count = len(formatted_context)
            # Increased character limit significantly for gpt-4o
            MAX_CONTEXT_CHARS = 100000 # Allow up to 100k characters (approx 25k tokens)

            # Include more context items
            # Limit based on character count instead of a fixed number of items
            for i, item in enumerate(context):
                if not isinstance(item, dict):
                    continue

                # Get entity information
                entity_type = item.get("type", "Unknown")
                entity_name = item.get("name", "Unknown")
                source_file = item.get("source_file", "Unknown")
                description = item.get("description", "No description available")
                code_snippet = item.get("code_snippet", "")

                # Format the entity information for the context string
                entry_text = f"--- Entity {i+1}: {entity_name} ({entity_type}) ---\n"
                entry_text += f"File: {source_file}\n"
                entry_text += f"Description: {description}\n"

                # Include code snippet if available
                if code_snippet:
                    # Limit snippet length individually if necessary, but prioritize overall context length
                    # max_snippet_len = 2000
                    # entry_text += f"Code:\n{code_snippet[:max_snippet_len]}{'...' if len(code_snippet) > max_snippet_len else ''}\n\n"
                    entry_text += f"Code:\n{code_snippet}\n\n"
                else:
                    entry_text += "\n"

                # Check if adding this entry exceeds the character limit
                if current_char_count + len(entry_text) > MAX_CONTEXT_CHARS:
                    print(f"Context limit ({MAX_CONTEXT_CHARS} chars) reached. Stopping context inclusion.")
                    break # Stop adding more context

                # Add the entry to the context and update the count
                formatted_context += entry_text
                current_char_count += len(entry_text)

            # Create messages for the chat completion
            messages = [
                {"role": "system", "content": 
                 "You are an AI assistant specialized in explaining code. Analyze the provided code entities and answer questions about them. "
                 "Be precise and technical, but explain concepts clearly. If you're not sure about something, say so instead of guessing. "
                 "When referring to code entities, include their type and file location to help the user locate them. "
                 "Keep your responses concise and focused on answering the user's question directly."
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
                        "end_line": item.get("end_lineno", item.get("lineno", 0) + 5),  # Ensure end_line is provided
                        "name": item.get("name", ""),
                        "type": item.get("type", ""),
                        "snippet": item.get("code_snippet", "")
                    })
            
            result = {
                "query": query_text,
                "response": response_text,
                "context": context,
                "references": references
            }
            
            # Cache the result
            self.response_cache[cache_key] = result
            return result
            
        except Exception as e:
            # Return error message if something went wrong
            return {
                "query": query_text,
                "response": f"I encountered an error while processing your query: {str(e)}. Please try again with a different question.",
                "context": context,
                "references": []
            }
    
    def _generate_cache_key(self, query_text: str, context: Optional[List[Dict[str, Any]]]) -> str:
        """Generate a cache key for the query and context."""
        if context is None:
            context_str = "none"
        else:
            # Use a stable representation of the context
            context_ids = sorted([item.get("id", str(i)) for i, item in enumerate(context)])
            context_str = ",".join(context_ids)
        
        # Create a hash of the query and context
        key_str = f"{query_text}:{context_str}"
        return hashlib.md5(key_str.encode()).hexdigest() 