system_introspection_manager.py - System Introspection Module with Kotlin interoperability

This module allows for introspection of Python code, particularly focused on 
examining implementation details of swarm intelligence algorithms.
"""

import inspect
import importlib
import importlib.util
import ast
import os
import sys
import types
import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/introspection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('system_introspection')

# Constants for default paths
DEFAULT_APP_ROOT = os.environ.get('APP_ROOT_DIR', os.getcwd())
DEFAULT_CODEBASE_DIR = os.path.join(DEFAULT_APP_ROOT, 'assets', 'codebase')
DEFAULT_SWARM_DIR = os.path.join(DEFAULT_CODEBASE_DIR, 'swarm_intelligence')
DEFAULT_MAPPING_FILE = os.path.join(DEFAULT_CODEBASE_DIR, "introspection_mapping.json")


class CodeParser:
    """
    Parses Python code files to extract classes, methods, functions, and their relationships.
    """
    
    def __init__(self):
        """Initialize the code parser."""
        self.module_cache = {}  # Maps module names to parsed AST
        self.class_info = {}    # Maps class names to class info
        self.function_info = {} # Maps function names to function info
        self.method_info = {}   # Maps method names (class.method) to method info
        self.module_paths = {}  # Maps module names to file paths
        self.parse_errors = {}  # Maps file paths to parsing errors
    
    def to_json(self) -> str:
        """
        Convert the current state to JSON for Kotlin interoperability.
        
        Returns:
            JSON string representation of the current state
        """
        result = self._prepare_result_for_kotlin_bridge()
        return json.dumps(result)
    
    @classmethod
    def from_json(cls, json_data: str) -> 'CodeParser':
        """
        Create a CodeParser instance from JSON data.
        
        Args:
            json_data: JSON string with module configuration
            
        Returns:
            New CodeParser instance
        """
        try:
            data = json.loads(json_data)
            instance = cls()
            
            # Restore paths if provided
            if "module_paths" in data:
                for module_name, path in data["module_paths"].items():
                    instance.module_paths[module_name] = path
            
            return instance
        except Exception as e:
            error_msg = f"Failed to create CodeParser from JSON: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
        """
        Prepare results in a format optimized for Kotlin bridge transmission.
        
        Returns:
            Dictionary with results formatted for Kotlin
        """
        # Create a simplified view of the parsed data
        return {
            "status": "success",
            "module_paths": self.module_paths,
            "classes_count": len(self.class_info),
            "functions_count": len(self.function_info),
            "methods_count": len(self.method_info),
            "errors": self.parse_errors
        }
    
    def parse_file(self, file_path: str) -> Optional[ast.Module]:
        """
        Parse a Python file and return its AST.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            ast_module: The AST of the parsed file or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code = file.read()
            
            return ast.parse(code, filename=file_path)
        except Exception as e:
            self.parse_errors[file_path] = str(e)
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
    
    def discover_modules(self, base_path: str) -> List[str]:
        """
        Discover all Python modules in a directory.
        
        Args:
            base_path: Base directory to search
            
        Returns:
            module_paths: List of discovered module file paths
        """
        module_paths = []
        
        try:
            for root, _, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        module_path = os.path.join(root, file)
                        module_paths.append(module_path)
        except Exception as e:
            logger.error(f"Error discovering modules in {base_path}: {e}")
        
        return module_paths
    
    def analyze_module(self, module_path: str, module_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a Python module to extract its classes and functions.
        
        Args:
            module_path: Path to the module file
            module_name: Optional name for the module
            
        Returns:
            module_info: Dictionary with module information
        """
        if module_name is None:
            module_name = os.path.basename(module_path).replace('.py', '')
        
        ast_module = self.parse_file(module_path)
        if not ast_module:
            return {"error": f"Failed to parse module {module_path}"}
            
        self.module_cache[module_name] = ast_module
        self.module_paths[module_name] = module_path
        
        module_info = {
            'name': module_name,
            'path': module_path,
            'docstring': ast.get_docstring(ast_module),
            'classes': [],
            'functions': [],
            'imports': []
        }
        
        # Process module content
        self._process_module_content(ast_module, module_info, module_name)
        
        return module_info
    
    def _process_module_content(self, ast_module: ast.Module, module_info: Dict[str, Any], module_name: str) -> None:
        """Process the content of a module's AST."""
        for node in ast_module.body:
            # Extract imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_info['imports'].append(self._get_import_info(node))
            
            # Extract classes
            elif isinstance(node, ast.ClassDef):
                try:
                    class_info = self._extract_class_info(node, module_name)
                    module_info['classes'].append(class_info)
                    self.class_info[node.name] = class_info
                except Exception as e:
                    logger.error(f"Error extracting class {node.name}: {e}")
            
            # Extract functions
            elif isinstance(node, ast.FunctionDef):
                try:
                    function_info = self._extract_function_info(node, module_name)
                    module_info['functions'].append(function_info)
                    self.function_info[node.name] = function_info
                except Exception as e:
                    logger.error(f"Error extracting function {node.name}: {e}")
    
    def _extract_class_info(self, node: ast.ClassDef, module_name: str) -> Dict[str, Any]:
        """Extract information about a class from its AST node."""
        class_info = {
            'name': node.name,
            'module': module_name,
            'docstring': ast.get_docstring(node),
            'methods': [],
            'attributes': [],
            'base_classes': [base.id if isinstance(base, ast.Name) else None for base in node.bases],
            'decorators': [decorator.id if isinstance(decorator, ast.Name) else None for decorator in node.decorator_list],
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, module_name)
                method_info['class'] = node.name
                class_info['methods'].append(method_info)
                self.method_info[f"{node.name}.{item.name}"] = method_info
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info['attributes'].append({
                            'name': target.id,
                            'value': self._get_attribute_value(item.value)
                        })
        
        return class_info
    
    def _extract_function_info(self, node: ast.FunctionDef, module_name: str) -> Dict[str, Any]:
        """Extract information about a function from its AST node."""
        function_info = {
            'name': node.name,
            'module': module_name,
            'docstring': ast.get_docstring(node),
            'parameters': [],
            'return_type': None,
            'decorators': [decorator.id if isinstance(decorator, ast.Name) else None for decorator in node.decorator_list],
            'body': self._get_function_body(node)
        }
        
        # Extract return type annotation if present
        if node.returns:
            function_info['return_type'] = self._get_annotation_name(node.returns)
        
        # Extract parameters
        self._extract_function_parameters(node, function_info)
        
        return function_info
    
    def _extract_function_parameters(self, node: ast.FunctionDef, function_info: Dict[str, Any]) -> None:
        """Extract function parameters and their default values."""
        # Extract parameters
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': self._get_annotation_name(arg.annotation) if arg.annotation else None,
                'default': None
            }
            function_info['parameters'].append(param_info)
        
        # Extract default values for parameters
        if node.args.defaults:
            non_default_count = len(node.args.args) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                param_index = non_default_count + i
                if param_index < len(function_info['parameters']):  # Safety check
                    function_info['parameters'][param_index]['default'] = self._get_attribute_value(default)
    
    def _get_function_body(self, node: ast.FunctionDef) -> List[str]:
        """Extract the lines of code in a function body."""
        body_lines = []
        for statement in node.body:
            # Skip the docstring
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Str):
                continue
            
            try:
                body_lines.append(ast.unparse(statement))
            except Exception:
                # Fallback for older Python versions
                body_lines.append("# Could not parse statement")
        
        return body_lines
    
    def _get_annotation_name(self, annotation) -> Optional[str]:
        """Extract the name of a type annotation."""
        if annotation is None:
            return None
        
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Subscript):
                if isinstance(annotation.value, ast.Name):
                    return f"{annotation.value.id}[...]"
            
            return ast.unparse(annotation)
        except Exception:
            return "Unknown"
    
    def _get_attribute_value(self, node) -> Any:
        """Extract the value of an attribute from its AST node."""
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Dict):
                return "{...}"
            elif isinstance(node, ast.List):
                return "[...]"
            elif isinstance(node, ast.Set):
                return "{...}"
            elif isinstance(node, ast.Tuple):
                return "(...)"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    return f"{node.func.id}(...)"
                return "function_call(...)"
            
            try:
                return ast.unparse(node) if node else None
            except Exception:
                return "Complex expression"
        except Exception:
            return "Unknown value"
    
    def _get_import_info(self, node) -> Dict[str, Any]:
        """Extract information about an import statement."""
        try:
            if isinstance(node, ast.Import):
                return {
                    'type': 'import',
                    'modules': [alias.name for alias in node.names],
                    'aliases': {alias.name: alias.asname for alias in node.names if alias.asname}
                }
            elif isinstance(node, ast.ImportFrom):
                return {
                    'type': 'from_import',
                    'module': node.module,
                    'names': [alias.name for alias in node.names],
                    'aliases': {alias.name: alias.asname for alias in node.names if alias.asname}
                }
        except Exception as e:
            logger.error(f"Error extracting import info: {e}")
        
        return {}
    
    def analyze_project(self, base_path: str) -> Dict[str, Any]:
        """
        Analyze all Python modules in a project.
        
        Args:
            base_path: Base directory of the project
            
        Returns:
            project_info: Dictionary with project information
        """
        try:
            module_paths = self.discover_modules(base_path)
            
            project_info = {
                'base_path': base_path,
                'modules': {},
                'errors': {}
            }
            
            for module_path in module_paths:
                module_name = os.path.basename(module_path).replace('.py', '')
                try:
                    module_info = self.analyze_module(module_path, module_name)
                    project_info['modules'][module_name] = module_info
                except Exception as e:
                    logger.error(f"Error analyzing module {module_path}: {e}")
                    project_info['errors'][module_name] = str(e)
            
            # Include any parse errors
            if self.parse_errors:
                project_info['parse_errors'] = self.parse_errors
            
            return project_info
        except Exception as e:
            logger.error(f"Error analyzing project {base_path}: {e}")
            return {'error': str(e)}
    
    def find_class_definition(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Find the definition of a class by name.
        
        Args:
            class_name: Name of the class to find
            
        Returns:
            class_info: Dictionary with class information, or None if not found
        """
        return self.class_info.get(class_name)
    
    def find_method_definition(self, class_name: str, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Find the definition of a method by class and method name.
        
        Args:
            class_name: Name of the class containing the method
            method_name: Name of the method to find
            
        Returns:
            method_info: Dictionary with method information, or None if not found
        """
        return self.method_info.get(f"{class_name}.{method_name}")
    
    def find_function_definition(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Find the definition of a function by name.
        
        Args:
            function_name: Name of the function to find
            
        Returns:
            function_info: Dictionary with function information, or None if not found
        """
        return self.function_info.get(function_name)
    
    def safe_execute(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """
        Safely execute a function with error handling.
        
        Args:
            function_name: Name of the method to execute
            **kwargs: Arguments for the method
            
        Returns:
            Dictionary with execution results
        """
        try:
            method = getattr(self, function_name)
            result = method(**kwargs)
            return {"status": "success", "data": result}
        except Exception as e:
            error_msg = f"Error executing {function_name}: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
    
    def cleanup(self) -> None:
        """Release resources to free memory."""
        # Clear caches
        self.module_cache.clear()
        # Keep only essential data
        # We keep class_info, function_info, and method_info as they're needed for queries


class ConceptMapper:
    """
    Maps abstract concepts to concrete code implementations.
    """
    
    def __init__(self, code_parser: CodeParser):
        """
        Initialize the concept mapper.
        
        Args:
            code_parser: CodeParser instance for code analysis
        """
        self.code_parser = code_parser
        self.concept_map = {}  # Maps concept names to implementation details
        self.reverse_map = defaultdict(list)  # Maps code elements to concepts
    
    def to_json(self) -> str:
        """
        Convert the current state to JSON for Kotlin interoperability.
        
        Returns:
            JSON string representation of the current state
        """
        result = self._prepare_result_for_kotlin_bridge()
        return json.dumps(result)
    
    @classmethod
    def from_json(cls, json_data: str, code_parser: CodeParser) -> 'ConceptMapper':
        """
        Create a ConceptMapper instance from JSON data.
        
        Args:
            json_data: JSON string with module configuration
            code_parser: CodeParser instance
            
        Returns:
            New ConceptMapper instance
        """
        try:
            data = json.loads(json_data)
            instance = cls(code_parser)
            
            # Restore concepts if provided
            if "concepts" in data:
                for concept_name, implementation in data["concepts"].items():
                    instance.register_concept(concept_name, implementation)
            
            return instance
        except Exception as e:
            error_msg = f"Failed to create ConceptMapper from JSON: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
        """
        Prepare results in a format optimized for Kotlin bridge transmission.
        
        Returns:
            Dictionary with results formatted for Kotlin
        """
        # Create a simplified view of the concept mappings
        simplified_concepts = {}
        
        for concept, implementation in self.concept_map.items():
            # Simplify the implementation for transmission
            simplified_impl = {
                "description": implementation.get("description", ""),
                "classes_count": len(implementation.get("classes", [])),
                "methods_count": len(implementation.get("methods", [])),
                "functions_count": len(implementation.get("functions", []))
            }
            
            # Include parameters if they exist as they're small and valuable
            if "parameters" in implementation:
                simplified_impl["parameters"] = implementation["parameters"]
                
            simplified_concepts[concept] = simplified_impl
        
        return {
            "status": "success",
            "concepts_count": len(self.concept_map),
            "concepts": simplified_concepts
        }
    
    def register_concept(self, concept_name: str, implementation_details: Dict[str, Any]) -> None:
        """
        Register a concept and its implementation details.
        
        Args:
            concept_name: Name of the concept
            implementation_details: Dictionary with implementation details
        """
        self.concept_map[concept_name] = implementation_details
        
        # Update reverse map
        for impl_type, impl_items in implementation_details.items():
            if impl_type in ['classes', 'methods', 'functions']:
                for item in impl_items:
                    self.reverse_map[item].append(concept_name)
    
    def get_implementation(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Get implementation details for a concept.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            implementation: Dictionary with implementation details, or None if not found
        """
        return self.concept_map.get(concept_name)
    
    def get_concepts_for_implementation(self, implementation_name: str) -> List[str]:
        """
        Get concepts associated with an implementation.
        
        Args:
            implementation_name: Name of the implementation (class, method, or function)
            
        Returns:
            concepts: List of concept names associated with the implementation
        """
        return self.reverse_map.get(implementation_name, [])
    
    def search_concepts(self, query: str) -> List[str]:
        """
        Search for concepts matching a query.
        
        Args:
            query: Search query
            
        Returns:
            matching_concepts: List of concept names matching the query
        """
        query = query.lower()
        return [name for name in self.concept_map if query in name.lower()]
    
    def load_concept_mapping(self, mapping_file: str) -> None:
        """
        Load concept mapping from a JSON file.
        
        Args:
            mapping_file: Path to the JSON file with concept mapping
        """
        try:
            if not os.path.exists(mapping_file):
                logger.warning(f"Mapping file not found: {mapping_file}")
                return
                
            with open(mapping_file, 'r', encoding='utf-8') as file:
                mapping_data = json.load(file)
            
            for concept_name, implementation_details in mapping_data.items():
                self.register_concept(concept_name, implementation_details)
                
            logger.info(f"Loaded {len(mapping_data)} concepts from {mapping_file}")
        
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading concept mapping: {e}")
    
    def save_concept_mapping(self, mapping_file: str) -> None:
        """
        Save concept mapping to a JSON file.
        
        Args:
            mapping_file: Path to save the JSON file with concept mapping
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
            
            with open(mapping_file, 'w', encoding='utf-8') as file:
                json.dump(self.concept_map, file, indent=2)
                
            logger.info(f"Saved {len(self.concept_map)} concepts to {mapping_file}")
        
        except IOError as e:
            logger.error(f"Error saving concept mapping: {e}")
    
    def generate_default_mapping(self, base_path: str) -> None:
        """
        Generate a default concept mapping for a project.
        
        Args:
            base_path: Base directory of the project
        """
        # Build mapping from docstrings
        self.build_mapping_from_docstrings()
        
        # Load SwarmIntelligence specific mappings
        self._add_swarm_intelligence_mappings()
            
        logger.info(f"Generated default concept mapping for {base_path}")
    
    def _add_swarm_intelligence_mappings(self) -> None:
        """Add SwarmIntelligence specific concept mappings."""
        # Define specific mappings for SwarmIntelligence
        swarm_mappings = {
            "AgentDecisionMaking": {
                "description": "Decision making process for agents in the swarm",
                "classes": ["Agent"],
                "methods": [
                    "Agent.decide_action",
                    "Agent._calc_exploration_utility",
                    "Agent._calc_gathering_utility",
                    "Agent._calc_avoidance_utility",
                    "Agent._calc_communication_utility",
                    "Agent._calc_rest_utility",
                    "Agent._calc_cooperation_utility",
                    "Agent._get_action_parameters"
                ],
                "functions": []
            },
            "AgentCommunication": {
                "description": "Communication mechanisms between agents",
                "classes": ["Agent", "Swarm"],
                "methods": [
                    "Agent.process_messages",
                    "Agent._process_message_content",
                    "Agent._update_trust",
                    "Agent.receive_message",
                    "Swarm.send_message",
                    "Swarm._deliver_messages"
                ],
                "functions": []
            },
            "EnvironmentModel": {
                "description": "Environmental representation and dynamics",
                "classes": ["Environment"],
                "methods": [
                    "Environment._initialize_environment",
                    "Environment.is_valid_position",
                    "Environment.get_cell_state",
                    "Environment.collect_resource",
                    "Environment.add_signal",
                    "Environment.get_signal_strength",
                    "Environment.get_signal_type",
                    "Environment.update_signals",
                    "Environment.diffuse_signals"
                ],
                "functions": []
            },
            "TaskAllocation": {
                "description": "Task allocation and management in the swarm",
                "classes": ["Task", "Swarm"],
                "methods": [
                    "Task.assign_agent",
                    "Task.unassign_agent",
                    "Task.add_contribution",
                    "Task.get_completion_percentage",
                    "Task.is_cooperative",
                    "Swarm.get_cooperative_tasks",
                    "Swarm.contribute_to_task",
                    "Swarm._create_random_task",
                    "Swarm._update_tasks"
                ],
                "functions": []
            },
            "EnergyManagement": {
                "description": "Energy consumption and management for agent actions",
                "classes": ["Agent"],
                "methods": [
                    "Agent.execute_action"
                ],
                "parameters": {
                    "explore": 1.0,
                    "gather": 1.0,
                    "avoid": 1.5,
                    "communicate": 0.5,
                    "rest": -5.0,
                    "cooperate": 1.0
                },
                "functions": []
            },
            "AgentLearning": {
                "description": "Learning and adaptation mechanisms for agents",
                "classes": ["Agent"],
                "methods": [
                    "Agent.update_learning"
                ],
                "functions": []
            },
            "SignalSystem": {
                "description": "Pheromone-like signaling system for indirect communication",
                "classes": ["Environment"],
                "methods": [
                    "Environment.add_signal",
                    "Environment.update_signals",
                    "Environment.diffuse_signals",
                    "Environment._get_neighbors"
                ],
                "functions": []
            }
        }
        
        # Register the mappings
        for concept_name, details in swarm_mappings.items():
            self.register_concept(concept_name, details)
    
    def build_mapping_from_docstrings(self) -> None:
        """Build concept mapping by analyzing docstrings in the code."""
        # Process classes
        for class_name, class_info in self.code_parser.class_info.items():
            docstring = class_info.get('docstring', '')
            if docstring:
                concepts = self._extract_concepts_from_docstring(docstring)
                
                for concept in concepts:
                    if concept not in self.concept_map:
                        self.concept_map[concept] = {'classes': [], 'methods': [], 'functions': []}
                    
                    self.concept_map[concept]['classes'].append(class_name)
                    self.reverse_map[class_name].append(concept)
        
        # Process methods
        for method_key, method_info in self.code_parser.method_info.items():
            docstring = method_info.get('docstring', '')
            if docstring:
                concepts = self._extract_concepts_from_docstring(docstring)
                
                for concept in concepts:
                    if concept not in self.concept_map:
                        self.concept_map[concept] = {'classes': [], 'methods': [], 'functions': []}
                    
                    self.concept_map[concept]['methods'].append(method_key)
                    self.reverse_map[method_key].append(concept)
        
        # Process functions
        for function_name, function_info in self.code_parser.function_info.items():
            docstring = function_info.get('docstring', '')
            if docstring:
                concepts = self._extract_concepts_from_docstring(docstring)
                
                for concept in concepts:
                    if concept not in self.concept_map:
                        self.concept_map[concept] = {'classes': [], 'methods': [], 'functions': []}
                    
                    self.concept_map[concept]['functions'].append(function_name)
                    self.reverse_map[function_name].append(concept)
                    
        logger.info(f"Built concept mapping from docstrings: found {len(self.concept_map)} concepts")
    
    def _extract_concepts_from_docstring(self, docstring: str) -> List[str]:
        """
        Extract concept keywords from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            concepts: List of concept keywords found in the docstring
        """
        concepts = []
        
        # Common concept-related terms
        concept_indicators = [
            'concept', 'framework', 'model', 'theory', 'paradigm',
            'approach', 'methodology', 'algorithm', 'process', 'system'
        ]
        
        # Look for sentences or phrases mentioning concepts
        lines = docstring.split('\n')
        for line in lines:
            line = line.strip().lower()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for concept indicators
            for indicator in concept_indicators:
                if indicator in line:
                    # Extract concept name using regex patterns
                    patterns = [
                        rf"{indicator} of (\w+)",
                        rf"{indicator} for (\w+)",
                        rf"(\w+) {indicator}",
                        rf"(\w+)'s {indicator}"
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, line)
                        concepts.extend(matches)
        
        # Deduplicate and normalize
        return list(set(concept.capitalize() for concept in concepts))
    
    def safe_execute(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """
        Safely execute a function with error handling.
        
        Args:
            function_name: Name of the method to execute
            **kwargs: Arguments for the method
            
        Returns:
            Dictionary with execution results
        """
        try:
            method = getattr(self, function_name)
            result = method(**kwargs)
            return {"status": "success", "data": result}
        except Exception as e:
            error_msg = f"Error executing {function_name}: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
    
    def cleanup(self) -> None
