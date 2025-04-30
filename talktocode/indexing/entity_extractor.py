import ast
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import traceback

class CodeEntity:
    """Base class for code entities extracted from source files."""
    
    def __init__(self, name: str, lineno: int, end_lineno: int, source_file: str):
        self.name = name
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.source_file = source_file
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "source_file": self.source_file,
        }


class FunctionEntity(CodeEntity):
    """Represents a function or method in the code."""
    
    def __init__(self, name: str, lineno: int, end_lineno: int, source_file: str, 
                 parameters: List[Dict[str, Any]], returns: Optional[str] = None, 
                 docstring: Optional[str] = None, is_method: bool = False, 
                 parent_class: Optional[str] = None):
        super().__init__(name, lineno, end_lineno, source_file)
        self.parameters = parameters
        self.returns = returns
        self.docstring = docstring
        self.is_method = is_method
        self.parent_class = parent_class
        self.called_functions: List[str] = []
        self.used_variables: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "parameters": self.parameters,
            "returns": self.returns,
            "docstring": self.docstring,
            "is_method": self.is_method,
            "parent_class": self.parent_class,
            "called_functions": self.called_functions,
            "used_variables": self.used_variables,
        })
        return result


class ClassEntity(CodeEntity):
    """Represents a class in the code."""
    
    def __init__(self, name: str, lineno: int, end_lineno: int, source_file: str,
                 base_classes: List[str], docstring: Optional[str] = None):
        super().__init__(name, lineno, end_lineno, source_file)
        self.base_classes = base_classes
        self.docstring = docstring
        self.methods: List[FunctionEntity] = []
        self.properties: List[VariableEntity] = []
        
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "base_classes": self.base_classes,
            "docstring": self.docstring,
            "methods": [method.to_dict() for method in self.methods],
            "properties": [prop.to_dict() for prop in self.properties],
        })
        return result


class VariableEntity(CodeEntity):
    """Represents a variable or constant in the code."""
    
    def __init__(self, name: str, lineno: int, end_lineno: int, source_file: str,
                 var_type: Optional[str] = None, value: Optional[str] = None,
                 is_class_property: bool = False, parent_class: Optional[str] = None):
        super().__init__(name, lineno, end_lineno, source_file)
        self.var_type = var_type
        self.value = value
        self.is_class_property = is_class_property
        self.parent_class = parent_class
        self.references: List[Tuple[int, str]] = []  # (line number, context)
        
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "var_type": self.var_type,
            "value": self.value,
            "is_class_property": self.is_class_property,
            "parent_class": self.parent_class,
            "references": self.references
        })
        return result


class ImportEntity(CodeEntity):
    """Represents a *static* import statement (``import x`` or ``from y import x``)."""
    
    def __init__(self, name: str, lineno: int, end_lineno: int, source_file: str,
                 import_from: Optional[str] = None, alias: Optional[str] = None,
                 description: str = "", code_snippet: str = ""):
        super().__init__(name, lineno, end_lineno, source_file)
        self.import_from = import_from
        self.alias = alias
        self.description = description or "No description available"
        self.code_snippet = code_snippet
        
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "import_from": self.import_from,
            "alias": self.alias,
            "description": self.description,
            "code_snippet": self.code_snippet,
        })
        return result


# New entity type: dynamic import via importlib/__import__
class DynamicImportEntity(CodeEntity):
    """Represents a dynamic import like ``importlib.import_module('pkg')``."""
    
    def __init__(self, name: str, lineno: int, end_lineno: int, source_file: str,
                 call_text: str = ""):
        super().__init__(name, lineno, end_lineno, source_file)
        self.description = f"Dynamic import of module '{name}'"
        self.code_snippet = call_text
        
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "description": self.description,
            "code_snippet": self.code_snippet,
        })
        return d


class CodeEntityExtractor(ast.NodeVisitor):
    """Extracts code entities from Python source code using AST."""
    
    def __init__(self, source_code: str, source_file: str):
        self.source_code = source_code
        self.source_file = source_file
        self.source_lines = source_code.splitlines()
        
        self.functions: List[FunctionEntity] = []
        self.classes: List[ClassEntity] = []
        self.variables: List[VariableEntity] = []
        self.imports: List[ImportEntity] = []
        
        self.current_class: Optional[ClassEntity] = None
        self.current_function: Optional[FunctionEntity] = None
        self.scope_stack: List[str] = []
        
    def extract_entities(self) -> Dict[str, List[CodeEntity]]:
        """Parse the source code and extract all code entities."""
        try:
            tree = ast.parse(self.source_code, filename=self.source_file)
            self.visit(tree)
            
            return {
                "functions": self.functions,
                "classes": self.classes,
                "variables": self.variables,
                "imports": self.imports,
            }
        except SyntaxError as e:
            # Handle incomplete or invalid code
            print(f"Syntax error in {self.source_file}:{e.lineno}:{e.offset} - {e.msg}")
            
            # Try to perform partial extraction by recovering at the line level
            return self.recover_from_syntax_error(e)
    
    def recover_from_syntax_error(self, error: SyntaxError) -> Dict[str, List[CodeEntity]]:
        """Attempt to recover and extract entities from code with syntax errors."""
        valid_lines = self.source_lines[:error.lineno - 1]
        
        try:
            # Try to parse the code up to the error
            if valid_lines:
                valid_code = "\n".join(valid_lines)
                tree = ast.parse(valid_code, filename=self.source_file)
                self.visit(tree)
        except Exception:
            pass  # If partial parsing fails, we'll return what we've extracted so far
        
        return {
            "functions": self.functions,
            "classes": self.classes,
            "variables": self.variables,
            "imports": self.imports,
        }
    
    def get_docstring(self, node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract docstring from a node if present."""
        if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Str)):
            return node.body[0].value.s
        return None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions."""
        docstring = self.get_docstring(node)
        base_classes = [base.__class__.__name__ for base in node.bases]
        
        end_lineno = getattr(node, 'end_lineno', node.lineno)
        class_entity = ClassEntity(
            name=node.name,
            lineno=node.lineno,
            end_lineno=end_lineno,
            source_file=self.source_file,
            base_classes=base_classes,
            docstring=docstring
        )
        
        # Save the current class context and set the new one
        old_class = self.current_class
        self.current_class = class_entity
        self.scope_stack.append(node.name)
        
        # Visit all child nodes to extract methods and properties
        for child in node.body:
            self.visit(child)
        
        # Restore the previous class context
        self.current_class = old_class
        self.scope_stack.pop()
        
        self.classes.append(class_entity)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function and method definitions."""
        self._process_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract async function and method definitions."""
        self._process_function(node)
    
    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        """Process function and method definitions."""
        docstring = self.get_docstring(node)
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                param["annotation"] = ast.unparse(arg.annotation)
            parameters.append(param)
        
        # Extract return type
        returns = None
        if node.returns:
            returns = ast.unparse(node.returns)
        
        end_lineno = getattr(node, 'end_lineno', node.lineno)
        is_method = self.current_class is not None
        parent_class_name = self.current_class.name if self.current_class else None
        
        function_entity = FunctionEntity(
            name=node.name,
            lineno=node.lineno,
            end_lineno=end_lineno,
            source_file=self.source_file,
            parameters=parameters,
            returns=returns,
            docstring=docstring,
            is_method=is_method,
            parent_class=parent_class_name
        )
        
        # Save the current function context and set the new one
        old_function = self.current_function
        self.current_function = function_entity
        self.scope_stack.append(node.name)
        
        # Visit all child nodes to extract function body details
        for child in node.body:
            self.visit(child)
        
        # Restore the previous function context
        self.current_function = old_function
        self.scope_stack.pop()
        
        if is_method:
            assert self.current_class is not None
            self.current_class.methods.append(function_entity)
        else:
            self.functions.append(function_entity)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Simple variable assignment
                end_lineno = getattr(node, 'end_lineno', node.lineno)
                
                try:
                    value = ast.unparse(node.value)
                except:
                    value = None
                
                var_entity = VariableEntity(
                    name=target.id,
                    lineno=node.lineno,
                    end_lineno=end_lineno,
                    source_file=self.source_file,
                    value=value,
                    is_class_property=self.current_function is None and self.current_class is not None,
                    parent_class=self.current_class.name if self.current_class else None
                )
                
                if var_entity.is_class_property and self.current_class:
                    self.current_class.properties.append(var_entity)
                else:
                    self.variables.append(var_entity)
            
            # Handle attribute assignments, etc.
            self.visit(target)
        
        self.visit(node.value)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Extract annotated variable assignments."""
        if isinstance(node.target, ast.Name):
            end_lineno = getattr(node, 'end_lineno', node.lineno)
            
            annotation = ast.unparse(node.annotation) if node.annotation else None
            value = ast.unparse(node.value) if node.value else None
            
            var_entity = VariableEntity(
                name=node.target.id,
                lineno=node.lineno,
                end_lineno=end_lineno,
                source_file=self.source_file,
                var_type=annotation,
                value=value,
                is_class_property=self.current_function is None and self.current_class is not None,
                parent_class=self.current_class.name if self.current_class else None
            )
            
            if var_entity.is_class_property and self.current_class:
                self.current_class.properties.append(var_entity)
            else:
                self.variables.append(var_entity)
        
        # Visit annotation and value
        if node.annotation:
            self.visit(node.annotation)
        if node.value:
            self.visit(node.value)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Extract import statements."""
        for name in node.names:
            end_lineno = getattr(node, 'end_lineno', node.lineno)
            
            full_line = self.source_lines[node.lineno - 1]
            import_entity = ImportEntity(
                name=name.name,
                lineno=node.lineno,
                end_lineno=end_lineno,
                source_file=self.source_file,
                alias=name.asname,
                description=f"Import statement: {full_line.strip()}",
                code_snippet=full_line.rstrip()
            )
            
            self.imports.append(import_entity)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from-import statements."""
        module = node.module or ""
        for name in node.names:
            end_lineno = getattr(node, 'end_lineno', node.lineno)
            
            full_line = self.source_lines[node.lineno - 1]
            import_entity = ImportEntity(
                name=name.name,
                lineno=node.lineno,
                end_lineno=end_lineno,
                source_file=self.source_file,
                import_from=module,
                alias=name.asname,
                description=f"Import statement: {full_line.strip()}",
                code_snippet=full_line.rstrip()
            )
            
            self.imports.append(import_entity)
    
    def visit_Name(self, node: ast.Name) -> None:
        """Track variable usage."""
        if isinstance(node.ctx, ast.Load) and self.current_function:
            # Variable is being used/referenced
            if node.id not in self.current_function.used_variables:
                self.current_function.used_variables.append(node.id)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Track function calls."""
        if isinstance(node.func, ast.Name) and self.current_function:
            # Direct function call
            if node.func.id not in self.current_function.called_functions:
                self.current_function.called_functions.append(node.func.id)
        elif isinstance(node.func, ast.Attribute) and self.current_function:
            # Method call
            if node.func.attr not in self.current_function.called_functions:
                # Only store the method name, not the full path
                self.current_function.called_functions.append(node.func.attr)
        
        # Detect dynamic imports: importlib.import_module('pkg') or __import__('pkg')
        try:
            if isinstance(node.func, ast.Attribute):
                if (getattr(node.func, 'attr', '') == 'import_module' and
                        isinstance(node.func.value, ast.Name) and node.func.value.id == 'importlib'):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        mod_name = str(node.args[0].value)
                    else:
                        mod_name = '<dynamic>'
                    full_line = self.source_lines[node.lineno - 1]
                    dyn_ent = DynamicImportEntity(
                        name=mod_name,
                        lineno=node.lineno,
                        end_lineno=getattr(node, 'end_lineno', node.lineno),
                        source_file=self.source_file,
                        call_text=full_line.rstrip()
                    )
                    self.imports.append(dyn_ent)
            elif isinstance(node.func, ast.Name) and node.func.id == '__import__':
                mod_name = '<dynamic>'
                if node.args and isinstance(node.args[0], ast.Constant):
                    mod_name = str(node.args[0].value)
                full_line = self.source_lines[node.lineno - 1]
                dyn_ent = DynamicImportEntity(
                    name=mod_name,
                    lineno=node.lineno,
                    end_lineno=getattr(node, 'end_lineno', node.lineno),
                    source_file=self.source_file,
                    call_text=full_line.rstrip()
                )
                self.imports.append(dyn_ent)
        except Exception:
            pass  # safety – don't break extraction on edge cases
        
        # Visit all child nodes
        for child in ast.iter_child_nodes(node):
            self.visit(child)


def extract_code_snippet(file_path: str, start_line: int, end_line: int, context_lines: int = 2) -> str:
    """
    Extract a code snippet from a file with optional context lines.
    
    Args:
        file_path: Path to the source file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        context_lines: Number of context lines to include before and after
    
    Returns:
        String containing the code snippet with line numbers
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_lines = f.readlines()
        
        # Adjust line numbers to 0-indexed
        start_idx = max(0, start_line - 1 - context_lines)
        end_idx = min(len(file_lines), end_line + context_lines)
        
        # Build the snippet with line numbers
        snippet_lines = []
        for i in range(start_idx, end_idx):
            line_num = i + 1
            prefix = "→ " if start_line <= line_num <= end_line else "  "
            snippet_lines.append(f"{prefix}{line_num:4d}: {file_lines[i].rstrip()}")
        
        return "\n".join(snippet_lines)
    
    except Exception as e:
        return f"Error extracting code snippet: {str(e)}"


def extract_entities_from_file(file_path: str) -> Dict[str, List[CodeEntity]]:
    """
    Extract all code entities from a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        Dictionary of extracted entities by type
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        extractor = CodeEntityExtractor(source_code, file_path)
        return extractor.extract_entities()
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        traceback.print_exc()
        return {
            "functions": [],
            "classes": [],
            "variables": [],
            "imports": [],
        }


def extract_entities_from_directory(directory_path: str, 
                                   file_extensions: List[str] = ['.py'],
                                   exclude_dirs: List[str] = ['__pycache__', '.git', 'venv', 'env']) -> Dict[str, List[CodeEntity]]:
    """
    Extract all code entities from Python files in a directory.
    
    Args:
        directory_path: Path to the directory
        file_extensions: List of file extensions to process
        exclude_dirs: Directories to exclude
    
    Returns:
        Dictionary of extracted entities by type
    """
    all_entities = {
        "functions": [],
        "classes": [],
        "variables": [],
        "imports": [],
    }
    
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                entities = extract_entities_from_file(file_path)
                
                # Merge entities
                for entity_type, entity_list in entities.items():
                    all_entities[entity_type].extend(entity_list)
    
    return all_entities


def extract_code_with_context(entity: CodeEntity, context_lines: int = 2) -> str:
    """
    Extract code with context for a given entity.
    
    Args:
        entity: The code entity to extract
        context_lines: Number of context lines to include
    
    Returns:
        String containing the code snippet with context
    """
    return extract_code_snippet(
        entity.source_file,
        entity.lineno,
        entity.end_lineno,
        context_lines
    ) 