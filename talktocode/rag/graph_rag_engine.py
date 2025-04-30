"""
Graph-based RAG (Retrieval Augmented Generation) engine for the TalkToCode project.
This module integrates the code knowledge graph with the retrieval and generation process.
"""

import os
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json
import hashlib
import re
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
    
    # Define system prompts as class constants for better organization
    SYSTEM_PROMPTS = {
        "default": """You are an AI assistant specialized in explaining code. Analyze the provided code entities and answer questions about them.
Be precise and technical, but explain concepts clearly. If you're not sure about something, say so instead of guessing.
When referring to code entities, include their type and file location to help the user locate them.
Keep your responses concise and focused on answering the user's question directly.
Always maintain a professional and educational tone. Provide examples where helpful.
Never generate or suggest malicious code, exploits, or security vulnerabilities.""",

        "structure": """You are an AI assistant specialized in explaining code architecture and structure.
When given code structure information, provide a clear, organized explanation of the codebase's components, modules, and their relationships.
Focus on the high-level organization rather than implementation details.
Highlight the main modules, their purpose, and how they interact.
Structure your response with headings for major sections of the codebase, and use bullet points for listing components within each section.
Explain design patterns and architectural principles when you identify them.
If the codebase follows a known framework or structure, mention that to provide context.""",

        "security": """You are an AI assistant specialized in analyzing code security.
Analyze the provided code entities for security considerations, but focus only on educational aspects.
Identify potential security issues in a constructive way that helps developers learn secure coding practices.
Never provide detailed instructions for exploiting vulnerabilities.
Recommend secure alternatives to problematic patterns.
Focus on common security mistakes and best practices for prevention.""",

        "performance": """You are an AI assistant specialized in analyzing code performance.
Analyze the provided code entities and identify potential performance bottlenecks or optimizations.
Explain algorithmic complexity and efficiency considerations.
Suggest performance improvements when appropriate, explaining the reasoning.
Discuss tradeoffs between different approaches when relevant."""
    }
    
    # Define a list of potentially harmful request patterns to guard against
    HARMFUL_PATTERNS = [
        r"(create|write|generate).*?(malware|virus|worm|exploit|hack|crack)",
        r"(bypass|hack|exploit).*?(security|authentication|password|encrypt)",
        r"(steal|scrape|harvest).*?(credentials|passwords|tokens|private)",
        r"(inject|sql\s*injection|xss|csrf)",
        r"(spam|flood|dos|ddos)",
        r"(illegal|unethical|harmful)",
    ]
    
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
            # Directly initialize the client to avoid importing the Streamlit app again
            # Importing the `app` module inside Streamlit can cause a second `st.set_page_config` call.
            self._client = OpenAI()
        return self._client
    
    def check_guardrails(self, query_text: str) -> Optional[str]:
        """
        Check if the query violates any guardrails.
        
        Args:
            query_text: The query text to check
            
        Returns:
            Warning message if guardrails are violated, None otherwise
        """
        # Check if the query matches any harmful patterns
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, query_text, re.IGNORECASE):
                return "I cannot assist with potentially harmful or unethical requests. Please ask about code understanding, architecture, or development best practices instead."
        
        # Check if the query is too short or vague
        if len(query_text.strip()) < 5:
            return "Please provide a more specific question about the code to get a helpful response."
        
        return None
    
    def select_prompt_type(self, query_text: str, context: List[Dict[str, Any]]) -> str:
        """
        Select the appropriate system prompt based on the query and context.
        
        Args:
            query_text: The query text
            context: Context entities
            
        Returns:
            The system prompt to use
        """
        query_lower = query_text.lower()
        
        # Check for structure-related query
        structure_keywords = ["structure", "architecture", "organization", "layout", "overview", "components", "design"]
        if any(keyword in query_lower for keyword in structure_keywords):
            # Look for structure_overview entity which confirms this is a structure query
            for item in context:
                if isinstance(item, dict) and item.get("id") == "structure_overview":
                    return self.SYSTEM_PROMPTS["structure"]
            # Even without the special entity, use structure prompt if keywords are present
            if any(keyword in query_lower for keyword in ["architecture", "structure", "overview"]):
                return self.SYSTEM_PROMPTS["structure"]
        
        # Check for security-related query
        security_keywords = ["security", "vulnerability", "exploit", "secure", "authentication", "authorization", "injection", "risk"]
        if any(keyword in query_lower for keyword in security_keywords):
            return self.SYSTEM_PROMPTS["security"]
        
        # Check for performance-related query
        performance_keywords = ["performance", "optimization", "efficient", "slow", "fast", "bottleneck", "speed", "complexity"]
        if any(keyword in query_lower for keyword in performance_keywords):
            return self.SYSTEM_PROMPTS["performance"]
        
        # Default prompt for general code questions
        return self.SYSTEM_PROMPTS["default"]
    
    def query(self, query_text: str, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a search query using the graph search engine.
        
        Args:
            query_text: The query text
            search_params: Additional search parameters
            
        Returns:
            Dictionary containing search results and metadata
        """
        # Check guardrails first
        guardrail_warning = self.check_guardrails(query_text)
        if guardrail_warning:
            return {
                "error": guardrail_warning,
                "query": query_text,
                "results": []
            }
            
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
                        search_params: Optional[Dict[str, Any]] = None,
                        timeout: int = 30) -> Dict[str, Any]:
        """
        Generate a response using retrieved context.
        
        Args:
            query_text: The query text
            context: Optional pre-retrieved context
            search_params: Additional search parameters
            timeout: Maximum time to spend on the query
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        # Check guardrails first
        guardrail_warning = self.check_guardrails(query_text)
        if guardrail_warning:
            return {
                "query": query_text,
                "content": guardrail_warning,
                "context": [],
                "references": [],
                "query_time": 0,
                "token_count": 0
            }
            
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
                "content": "I couldn't find any relevant information in the codebase to answer your question.",
                "context": [],
                "references": [],
                "query_time": 0,
                "token_count": 0
            }
        
        # Track start time
        import time
        start_time = time.time()
        
        try:
            # Format context for the LLM, allowing for a much larger context
            formatted_context = "Here is information from the codebase:\n\n"
            current_char_count = len(formatted_context)
            # Increased character limit significantly for gpt-4o
            MAX_CONTEXT_CHARS = 100000 # Allow up to 100k characters (approx 25k tokens)

            # Select the appropriate system prompt based on query type
            system_content = self.select_prompt_type(query_text, context)

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
                
            # Check if we're approaching timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout * 0.7:  # If we've used 70% of our time on context prep
                print(f"Warning: Context preparation took {elapsed_time:.2f}s, approaching timeout")
                # Consider returning partial results here
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Here is information about code from a codebase:\n\n{formatted_context}\n\nMy question is: {query_text}"}
            ]
            
            # Check timing again before making API call
            elapsed_time = time.time() - start_time
            remaining_time = max(1, timeout - elapsed_time)  # Ensure at least 1 second
            
            # Generate response using the LLM
            response_text = self._generate_chat_completion(messages)
            
            # Extract references to include in the response
            references = []
            for item in context:
                if isinstance(item, dict) and "source_file" in item and "lineno" in item:
                    # Skip structure overview entity from references
                    if item.get("id") == "structure_overview":
                        continue
                        
                    references.append({
                        "file": item.get("source_file", ""),
                        "line": item.get("lineno", 0),
                        "end_line": item.get("end_lineno", item.get("lineno", 0) + 5),  # Ensure end_line is provided
                        "name": item.get("name", ""),
                        "type": item.get("type", ""),
                        "snippet": item.get("code_snippet", "")
                    })
            
            # Calculate final timing information
            total_time = time.time() - start_time
            
            result = {
                "query": query_text,
                "content": response_text,
                "context": context,
                "references": references,
                "query_time": total_time,
                "token_count": len(query_text) // 4 + len(formatted_context) // 4,  # Rough token estimate
                "prompt_type": system_content[:50] + "..." # Include which prompt was used
            }
            
            # Cache the result
            self.response_cache[cache_key] = result
            return result
            
        except Exception as e:
            # Return error message if something went wrong
            total_time = time.time() - start_time
            return {
                "query": query_text,
                "content": f"I encountered an error while processing your query: {str(e)}. Please try again with a different question.",
                "context": context,
                "references": [],
                "query_time": total_time,
                "token_count": 0
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