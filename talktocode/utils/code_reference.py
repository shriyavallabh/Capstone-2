"""
Code reference utilities for TalkToCode.

This module provides classes and functions for handling code references and their formatting.
"""

from typing import List, Dict, Any, Optional, Union
import re
import os
from pathlib import Path


class CodeReference:
    """
    Represents a reference to a specific section of code in a file.
    Used to link responses to the relevant parts of the codebase.
    """
    
    def __init__(self, 
                file_path: str, 
                start_line: int, 
                end_line: Optional[int] = None,
                entity_name: Optional[str] = None,
                entity_type: Optional[str] = None,
                relevance_score: Optional[float] = None,
                snippet: Optional[str] = None):
        """
        Initialize a code reference.
        
        Args:
            file_path: Path to the file containing the code
            start_line: Starting line number of the reference
            end_line: Ending line number of the reference (optional)
            entity_name: Name of the entity being referenced (optional)
            entity_type: Type of the entity (e.g., function, class) (optional)
            relevance_score: Score indicating how relevant this reference is to the query (optional)
            snippet: Code snippet for the reference (optional)
        """
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line or start_line
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.relevance_score = relevance_score
        self.snippet = snippet
    
    def __str__(self) -> str:
        """String representation of the code reference."""
        if self.entity_name:
            if self.start_line == self.end_line:
                return f"{self.entity_name} ({self.entity_type}) in {self.file_path}:{self.start_line}"
            else:
                return f"{self.entity_name} ({self.entity_type}) in {self.file_path}:{self.start_line}-{self.end_line}"
        else:
            if self.start_line == self.end_line:
                return f"{self.file_path}:{self.start_line}"
            else:
                return f"{self.file_path}:{self.start_line}-{self.end_line}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the code reference to a dictionary."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "relevance_score": self.relevance_score,
            "snippet": self.snippet
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeReference':
        """Create a CodeReference from a dictionary."""
        return cls(
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data.get("end_line"),
            entity_name=data.get("entity_name"),
            entity_type=data.get("entity_type"),
            relevance_score=data.get("relevance_score"),
            snippet=data.get("snippet")
        )
    
    def get_formatted_reference(self) -> str:
        """Get a formatted string reference for use in markdown."""
        filename = os.path.basename(self.file_path)
        if self.start_line == self.end_line:
            return f"`{filename}:{self.start_line}`"
        else:
            return f"`{filename}:{self.start_line}-{self.end_line}`"
    
    def get_language(self) -> str:
        """Determine the programming language from the file extension."""
        ext = os.path.splitext(self.file_path)[1].lower()
        languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.sh': 'bash',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.sql': 'sql'
        }
        return languages.get(ext, '')


def format_code_references(references: List[CodeReference], 
                          include_snippets: bool = True) -> str:
    """
    Format a list of code references into a markdown string.
    
    Args:
        references: List of CodeReference objects
        include_snippets: Whether to include code snippets in the output
        
    Returns:
        Markdown formatted string with code references
    """
    if not references:
        return ""
    
    formatted = ["### Code References", ""]
    
    for i, ref in enumerate(references, 1):
        # Add entity name and type if available
        if ref.entity_name and ref.entity_type:
            formatted.append(f"**{i}. {ref.entity_name}** ({ref.entity_type})")
        else:
            formatted.append(f"**{i}. Code Reference**")
        
        # Add file and line info
        formatted.append(f"File: {ref.file_path}")
        if ref.start_line == ref.end_line:
            formatted.append(f"Line: {ref.start_line}")
        else:
            formatted.append(f"Lines: {ref.start_line}-{ref.end_line}")
        
        # Add relevance score if available
        if ref.relevance_score is not None:
            formatted.append(f"Relevance: {ref.relevance_score:.2f}")
        
        # Add code snippet if available and requested
        if include_snippets and ref.snippet:
            # Get language for syntax highlighting
            lang = ref.get_language()
            formatted.append("")
            formatted.append(f"```{lang}")
            formatted.append(ref.snippet)
            formatted.append("```")
        
        # Add a separator between references
        formatted.append("")
    
    return "\n".join(formatted)


def extract_code_references_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract code references from a text string.
    Detects patterns like filename:line or filename:line-line.
    
    Args:
        text: The text to extract code references from
        
    Returns:
        List of dictionaries with file_path, start_line, and end_line
    """
    # Pattern for filename:line or filename:line-line
    pattern = r'`([^`]+):(\d+)(?:-(\d+))?`'
    references = []
    
    for match in re.finditer(pattern, text):
        file_path = match.group(1)
        start_line = int(match.group(2))
        end_line = int(match.group(3)) if match.group(3) else start_line
        
        references.append({
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line
        })
    
    return references


def load_referenced_code(references: List[CodeReference], base_dir: str = "") -> List[CodeReference]:
    """
    Load the actual code snippets for the given references.
    
    Args:
        references: List of CodeReference objects
        base_dir: Base directory to resolve relative paths
        
    Returns:
        Updated list of CodeReference objects with snippets included
    """
    updated_refs = []
    
    for ref in references:
        # Resolve the file path
        full_path = os.path.join(base_dir, ref.file_path)
        
        try:
            # Check if file exists
            if not os.path.isfile(full_path):
                # Try to find a matching file by basename
                basename = os.path.basename(ref.file_path)
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        if file == basename:
                            full_path = os.path.join(root, file)
                            break
                    if os.path.isfile(full_path):
                        break
            
            # If file exists, read the lines
            if os.path.isfile(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    all_lines = f.readlines()
                
                # Adjust line numbers to be 0-indexed
                start_idx = max(0, ref.start_line - 1)
                end_idx = min(len(all_lines), ref.end_line)
                
                # Extract the relevant lines
                if start_idx < len(all_lines) and end_idx >= start_idx:
                    snippet_lines = all_lines[start_idx:end_idx]
                    ref.snippet = ''.join(snippet_lines).rstrip()
        
        except Exception as e:
            # If there's an error, just skip adding the snippet
            print(f"Error loading code for {ref.file_path}: {str(e)}")
        
        updated_refs.append(ref)
    
    return updated_refs 