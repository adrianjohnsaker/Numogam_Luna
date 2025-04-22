"""```python
"""
System Introspection Module

This module allows Amelia to examine her own codebase, understand its structure,
and access specific implementation details about her metacognitive architecture.

The module provides four key capabilities:
1. Code Introspection: Parse and analyze Python code files
2. Implementation-to-Concept Mapping: Link abstract concepts to code implementations
3. Self-Reference Framework: Distinguish between general knowledge and specific implementation
4. Memory Access Layer: Query actual data structures and objects in the running system
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
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
import pkgutil
import pathlib
import logging
from collections import defaultdict

# Configuration
APP_ROOT = os.environ.get('APP_ROOT_DIR', '/app')  # Default fallback path
CODEBASE_DIR = os.path.join(APP_ROOT, 'assets/codebase')
SWARM_INTELLIGENCE_DIR = os.path.join(CODEBASE_DIR, 'swarm_intelligence')
MAPPING_FILE = os.path.join(CODEBASE_DIR, "introspection_mapping.json")

# Ensure directories exist
for path in [CODEBASE_DIR, SWARM_INTELLIGENCE_DIR]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(APP_ROOT, 'logs/introspection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('system_introspection')


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
    
    def import_module_from_path(self, module_path: str) -> Optional[types.ModuleType]:
        """
        Import a module from a file path.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            module: The imported module or None if error
        """
        try:
            module_name = os.path.basename(module_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            logger.error(f"Error importing module {module_path}: {e}")
            return None
    
    def extract_class_info(self, node: ast.ClassDef, module_name: str) -> Dict[str, Any]:
        """
        Extract information about a class from its AST node.
        
        Args:
            node: AST node for the class
            module_name: Name of the module containing the class
            
        Returns:
            class_info: Dictionary with class information
        """
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
                method_info = self.extract_function_info(item, module_name)
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
    
    def extract_function_info(self, node: ast.FunctionDef, module_name: str) -> Dict[str, Any]:
        """
        Extract information about a function from its AST node.
        
        Args:
            node: AST node for the function
            module_name: Name of the module containing the function
            
        Returns:
            function_info: Dictionary with function information
        """
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
        
        return function_info
    
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
        
        for node in ast_module.body:
            # Extract imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_info['imports'].append(self._get_import_info(node))
            
            # Extract classes
            elif isinstance(node, ast.ClassDef):
                try:
                    class_info = self.extract_class_info(node, module_name)
                    module_info['classes'].append(class_info)
                    self.class_info[node.name] = class_info
                except Exception as e:
                    logger.error(f"Error extracting class {node.name}: {e}")
            
            # Extract functions
            elif isinstance(node, ast.FunctionDef):
                try:
                    function_info = self.extract_function_info(node, module_name)
                    module_info['functions'].append(function_info)
                    self.function_info[node.name] = function_info
                except Exception as e:
                    logger.error(f"Error extracting function {node.name}: {e}")
        
        return module_info
    
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
    
    def get_class_hierarchy(self, class_name: str) -> Dict[str, Any]:
        """
        Get the class hierarchy for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            hierarchy: Dictionary with class hierarchy information
        """
        class_info = self.find_class_definition(class_name)
        if not class_info:
            return {}
        
        hierarchy = {
            'name': class_name,
            'base_classes': []
        }
        
        for base_class in class_info.get('base_classes', []):
            if base_class and base_class in self.class_info:
                base_hierarchy = self.get_class_hierarchy(base_class)
                hierarchy['base_classes'].append(base_hierarchy)
        
        return hierarchy


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
            with open(mapping_file, 'w', encoding='utf-8') as file:
                json.dump(self.concept_map, file, indent=2)
                
            logger.info(f"Saved {len(self.concept_map)} concepts to {mapping_file}")
        
        except IOError as e:
            logger.error(f"Error saving concept mapping: {e}")
    
    def build_mapping_from_docstrings(self) -> None:
        """
        Build concept mapping by analyzing docstrings in the code.
        """
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
        # This is a simple implementation that looks for concept-related keywords
        # A more sophisticated implementation might use NLP techniques
        
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
    
    def _add_swarm_intelligence_mappings(self):
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


class SelfReferenceFramework:
    """
    Provides methods for Amelia to distinguish between general knowledge and specific implementations.
    """
    
    def __init__(self, concept_mapper: ConceptMapper):
        """
        Initialize the self-reference framework.
        
        Args:
            concept_mapper: ConceptMapper instance for concept mapping
        """
        self.concept_mapper = concept_mapper
        self.general_knowledge = {}  # Maps concept names to general knowledge
        self.specific_implementations = {}  # Maps concept names to specific implementations
        
    def register_general_knowledge(self, concept_name: str, knowledge: Dict[str, Any]) -> None:
        """
        Register general knowledge about a concept.
        
        Args:
            concept_name: Name of the concept
            knowledge: Dictionary with general knowledge about the concept
        """
        self.general_knowledge[concept_name] = knowledge
        logger.info(f"Registered general knowledge for concept: {concept_name}")
    
    def register_specific_implementation(self, concept_name: str, implementation: Dict[str, Any]) -> None:
        """
        Register specific implementation details for a concept.
        
        Args:
            concept_name: Name of the concept
            implementation: Dictionary with specific implementation details
        """
        self.specific_implementations[concept_name] = implementation
        logger.info(f"Registered specific implementation for concept: {concept_name}")
    
    def get_concept_knowledge(self, concept_name: str, knowledge_type: str = 'both') -> Dict[str, Any]:
        """
        Get knowledge about a concept.
        
        Args:
            concept_name: Name of the concept
            knowledge_type: Type of knowledge to get ('general', 'specific', or 'both')
            
        Returns:
            knowledge: Dictionary with knowledge about the concept
        """
        result = {}
        
        if knowledge_type in ['general', 'both'] and concept_name in self.general_knowledge:
            result['general'] = self.general_knowledge[concept_name]
        
        if knowledge_type in ['specific', 'both']:
            # Get implementation details from the concept mapper
            implementation = self.concept_mapper.get_implementation(concept_name)
            
            if implementation:
                result['specific'] = implementation
            
            # Add any registered specific implementation details
            if concept_name in self.specific_implementations:
                if 'specific' not in result:
                    result['specific'] = {}
                
                result['specific'].update(self.specific_implementations[concept_name])
        
        return result
    
    def compare_knowledge(self, concept_name: str) -> Dict[str, Any]:
        """
        Compare general knowledge about a concept with its specific implementation.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            comparison: Dictionary with comparison results
        """
        knowledge = self.get_concept_knowledge(concept_name)
        
        if 'general' not in knowledge or 'specific' not in knowledge:
            return {
                'concept': concept_name,
                'has_general': 'general' in knowledge,
                'has_specific': 'specific' in knowledge,
                'comparison': 'incomplete'
            }
        
        general = knowledge['general']
        specific = knowledge['specific']
        
        # This is a simple comparison implementation
        # A more sophisticated implementation would perform a deeper analysis
        
        comparison = {
            'concept': concept_name,
            'has_general': True,
            'has_specific': True,
            'comparison': 'complete',
            'alignment': 'unknown',
            'differences': [],
            'alignments': []
        }
        
        # Compare description with implementation
        if 'description' in general and 'description' in specific:
            if general['description'] == specific['description']:
                comparison['alignments'].append('description')
            else:
                comparison['differences'].append('description')
        
        # Other comparisons would depend on the structure of the knowledge
        
        # Determine overall alignment
        if not comparison['differences']:
            comparison['alignment'] = 'high'
        elif len(comparison['alignments']) > len(comparison['differences']):
            comparison['alignment'] = 'medium'
        else:
            comparison['alignment'] = 'low'
System Introspection Module

This module allows Amelia to examine her own codebase, understand its structure,
and access specific implementation details about her metacognitive architecture.

The module provides four key capabilities:
1. Code Introspection: Parse and analyze Python code files
2. Implementation-to-Concept Mapping: Link abstract concepts to code implementations
3. Self-Reference Framework: Distinguish between general knowledge and specific implementation
4. Memory Access Layer: Query actual data structures and objects in the running system
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
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
import pkgutil
import pathlib
import logging
from collections import defaultdict

# Configuration
APP_ROOT = os.environ.get('APP_ROOT_DIR', '/app')  # Default fallback path
CODEBASE_DIR = os.path.join(APP_ROOT, 'assets/codebase')
SWARM_INTELLIGENCE_DIR = os.path.join(CODEBASE_DIR, 'swarm_intelligence')
MAPPING_FILE = os.path.join(CODEBASE_DIR, "introspection_mapping.json")

# Ensure directories exist
for path in [CODEBASE_DIR, SWARM_INTELLIGENCE_DIR]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(APP_ROOT, 'logs/introspection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('system_introspection')


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
    
    def import_module_from_path(self, module_path: str) -> Optional[types.ModuleType]:
        """
        Import a module from a file path.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            module: The imported module or None if error
        """
        try:
            module_name = os.path.basename(module_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            logger.error(f"Error importing module {module_path}: {e}")
            return None
    
    def extract_class_info(self, node: ast.ClassDef, module_name: str) -> Dict[str, Any]:
        """
        Extract information about a class from its AST node.
        
        Args:
            node: AST node for the class
            module_name: Name of the module containing the class
            
        Returns:
            class_info: Dictionary with class information
        """
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
                method_info = self.extract_function_info(item, module_name)
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
    
    def extract_function_info(self, node: ast.FunctionDef, module_name: str) -> Dict[str, Any]:
        """
        Extract information about a function from its AST node.
        
        Args:
            node: AST node for the function
            module_name: Name of the module containing the function
            
        Returns:
            function_info: Dictionary with function information
        """
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
        
        return function_info
    
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
        
        for node in ast_module.body:
            # Extract imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_info['imports'].append(self._get_import_info(node))
            
            # Extract classes
            elif isinstance(node, ast.ClassDef):
                try:
                    class_info = self.extract_class_info(node, module_name)
                    module_info['classes'].append(class_info)
                    self.class_info[node.name] = class_info
                except Exception as e:
                    logger.error(f"Error extracting class {node.name}: {e}")
            
            # Extract functions
            elif isinstance(node, ast.FunctionDef):
                try:
                    function_info = self.extract_function_info(node, module_name)
                    module_info['functions'].append(function_info)
                    self.function_info[node.name] = function_info
                except Exception as e:
                    logger.error(f"Error extracting function {node.name}: {e}")
        
        return module_info
    
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
    
    def get_class_hierarchy(self, class_name: str) -> Dict[str, Any]:
        """
        Get the class hierarchy for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            hierarchy: Dictionary with class hierarchy information
        """
        class_info = self.find_class_definition(class_name)
        if not class_info:
            return {}
        
        hierarchy = {
            'name': class_name,
            'base_classes': []
        }
        
        for base_class in class_info.get('base_classes', []):
            if base_class and base_class in self.class_info:
                base_hierarchy = self.get_class_hierarchy(base_class)
                hierarchy['base_classes'].append(base_hierarchy)
        
        return hierarchy


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
            with open(mapping_file, 'w', encoding='utf-8') as file:
                json.dump(self.concept_map, file, indent=2)
                
            logger.info(f"Saved {len(self.concept_map)} concepts to {mapping_file}")
        
        except IOError as e:
            logger.error(f"Error saving concept mapping: {e}")
    
    def build_mapping_from_docstrings(self) -> None:
        """
        Build concept mapping by analyzing docstrings in the code.
        """
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
        # This is a simple implementation that looks for concept-related keywords
        # A more sophisticated implementation might use NLP techniques
        
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
    
    def _add_swarm_intelligence_mappings(self):
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


class SelfReferenceFramework:
    """
    Provides methods for Amelia to distinguish between general knowledge and specific implementations.
    """
    
    def __init__(self, concept_mapper: ConceptMapper):
        """
        Initialize the self-reference framework.
        
        Args:
            concept_mapper: ConceptMapper instance for concept mapping
        """
        self.concept_mapper = concept_mapper
        self.general_knowledge = {}  # Maps concept names to general knowledge
        self.specific_implementations = {}  # Maps concept names to specific implementations
        
    def register_general_knowledge(self, concept_name: str, knowledge: Dict[str, Any]) -> None:
        """
        Register general knowledge about a concept.
        
        Args:
            concept_name: Name of the concept
            knowledge: Dictionary with general knowledge about the concept
        """
        self.general_knowledge[concept_name] = knowledge
        logger.info(f"Registered general knowledge for concept: {concept_name}")
    
    def register_specific_implementation(self, concept_name: str, implementation: Dict[str, Any]) -> None:
        """
        Register specific implementation details for a concept.
        
        Args:
            concept_name: Name of the concept
            implementation: Dictionary with specific implementation details
        """
        self.specific_implementations[concept_name] = implementation
        logger.info(f"Registered specific implementation for concept: {concept_name}")
    
    def get_concept_knowledge(self, concept_name: str, knowledge_type: str = 'both') -> Dict[str, Any]:
        """
        Get knowledge about a concept.
        
        Args:
            concept_name: Name of the concept
            knowledge_type: Type of knowledge to get ('general', 'specific', or 'both')
            
        Returns:
            knowledge: Dictionary with knowledge about the concept
        """
        result = {}
        
        if knowledge_type in ['general', 'both'] and concept_name in self.general_knowledge:
            result['general'] = self.general_knowledge[concept_name]
        
        if knowledge_type in ['specific', 'both']:
            # Get implementation details from the concept mapper
            implementation = self.concept_mapper.get_implementation(concept_name)
            
            if implementation:
                result['specific'] = implementation
            
            # Add any registered specific implementation details
            if concept_name in self.specific_implementations:
                if 'specific' not in result:
                    result['specific'] = {}
                
                result['specific'].update(self.specific_implementations[concept_name])
        
        return result
    
    def compare_knowledge(self, concept_name: str) -> Dict[str, Any]:
        """
        Compare general knowledge about a concept with its specific implementation.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            comparison: Dictionary with comparison results
        """
        knowledge = self.get_concept_knowledge(concept_name)
        
        if 'general' not in knowledge or 'specific' not in knowledge:
            return {
                'concept': concept_name,
                'has_general': 'general' in knowledge,
                'has_specific': 'specific' in knowledge,
                'comparison': 'incomplete'
            }
        
        general = knowledge['general']
        specific = knowledge['specific']
        
        # This is a simple comparison implementation
        # A more sophisticated implementation would perform a deeper analysis
        
        comparison = {
            'concept': concept_name,
            'has_general': True,
            'has_specific': True,
            'comparison': 'complete',
            'alignment': 'unknown',
            'differences': [],
            'alignments': []
        }
        
        # Compare description with implementation
        if 'description' in general and 'description' in specific:
            if general['description'] == specific['description']:
                comparison['alignments'].append('description')
            else:
                comparison['differences'].append('description')
        
        # Other comparisons would depend on the structure of the knowledge
        
        # Determine overall alignment
        if not comparison['differences']:
            comparison['alignment'] = 'high'
        elif len(comparison['alignments']) > len(comparison['differences']):
            comparison['alignment'] = 'medium'
        else:
            comparison['alignment'] = 'low'
Here's the continuation of the SelfReferenceFramework class and the rest of the system_introspection_module.py code:

```python
        return comparison
    
    def load_general_knowledge(self, knowledge_file: str) -> None:
        """
        Load general knowledge from a JSON file.
        
        Args:
            knowledge_file: Path to the JSON file with general knowledge
        """
        try:
            if not os.path.exists(knowledge_file):
                logger.warning(f"General knowledge file not found: {knowledge_file}")
                return
                
            with open(knowledge_file, 'r', encoding='utf-8') as file:
                knowledge_data = json.load(file)
            
            self.general_knowledge.update(knowledge_data)
            logger.info(f"Loaded general knowledge from {knowledge_file}: {len(knowledge_data)} concepts")
        
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading general knowledge: {e}")
    
    def save_general_knowledge(self, knowledge_file: str) -> None:
        """
        Save general knowledge to a JSON file.
        
        Args:
            knowledge_file: Path to save the JSON file with general knowledge
        """
        try:
            with open(knowledge_file, 'w', encoding='utf-8') as file:
                json.dump(self.general_knowledge, file, indent=2)
            logger.info(f"Saved general knowledge to {knowledge_file}: {len(self.general_knowledge)} concepts")
        
        except IOError as e:
            logger.error(f"Error saving general knowledge: {e}")


class MemoryAccessLayer:
    """
    Provides access to data structures and objects in the running system.
    """
    
    def __init__(self):
        """Initialize the memory access layer."""
        self.runtime_objects = {}  # Maps object IDs to runtime objects
        self.type_registry = {}  # Maps type names to lists of object IDs
        
    def register_object(self, obj_id: str, obj: Any, obj_type: str = None) -> None:
        """
        Register a runtime object.
        
        Args:
            obj_id: ID for the object
            obj: The object to register
            obj_type: Optional type name for the object
        """
        self.runtime_objects[obj_id] = obj
        
        if obj_type is None:
            obj_type = type(obj).__name__
        
        if obj_type not in self.type_registry:
            self.type_registry[obj_type] = []
        
        self.type_registry[obj_type].append(obj_id)
        logger.info(f"Registered runtime object: {obj_id} (type: {obj_type})")
    
    def get_object(self, obj_id: str) -> Optional[Any]:
        """
        Get a runtime object by ID.
        
        Args:
            obj_id: ID of the object
            
        Returns:
            obj: The object, or None if not found
        """
        return self.runtime_objects.get(obj_id)
    
    def find_objects_by_type(self, obj_type: str) -> List[str]:
        """
        Find object IDs by type.
        
        Args:
            obj_type: Type of objects to find
            
        Returns:
            obj_ids: List of object IDs of the specified type
        """
        return self.type_registry.get(obj_type, [])
    
    def find_objects_by_attribute(self, attribute_name: str, attribute_value: Any) -> List[str]:
        """
        Find object IDs by attribute value.
        
        Args:
            attribute_name: Name of the attribute to check
            attribute_value: Value of the attribute to match
            
        Returns:
            obj_ids: List of object IDs with the specified attribute value
        """
        matching_ids = []
        
        for obj_id, obj in self.runtime_objects.items():
            try:
                if hasattr(obj, attribute_name):
                    if getattr(obj, attribute_name) == attribute_value:
                        matching_ids.append(obj_id)
            except Exception as e:
                logger.error(f"Error checking attribute {attribute_name} on object {obj_id}: {e}")
        
        return matching_ids
    
    def query_object_attribute(self, obj_id: str, attribute_name: str) -> Any:
        """
        Query the value of an object's attribute.
        
        Args:
            obj_id: ID of the object
            attribute_name: Name of the attribute to query
            
        Returns:
            value: Value of the attribute, or None if not found
        """
        obj = self.get_object(obj_id)
        
        if obj is None or not hasattr(obj, attribute_name):
            return None
        
        try:
            return getattr(obj, attribute_name)
        except Exception as e:
            logger.error(f"Error getting attribute {attribute_name} from object {obj_id}: {e}")
            return None
    
    def get_object_attributes(self, obj_id: str) -> Dict[str, Any]:
        """
        Get all attributes of an object.
        
        Args:
            obj_id: ID of the object
            
        Returns:
            attributes: Dictionary mapping attribute names to values
        """
        obj = self.get_object(obj_id)
        
        if obj is None:
            return {}
        
        attributes = {}
        
        for attr in dir(obj):
            # Skip private and special attributes
            if attr.startswith('__'):
                continue
            
            try:
                value = getattr(obj, attr)
                
                # Skip methods and functions
                if callable(value):
                    continue
                
                attributes[attr] = value
            except Exception as e:
                logger.error(f"Error getting attribute {attr} from object {obj_id}: {e}")
                # Skip attributes that can't be accessed
                continue
        
        return attributes
    
    def invoke_method(self, obj_id: str, method_name: str, *args, **kwargs) -> Any:
        """
        Invoke a method on an object.
        
        Args:
            obj_id: ID of the object
            method_name: Name of the method to invoke
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            result: Result of the method invocation
        """
        obj = self.get_object(obj_id)
        
        if obj is None or not hasattr(obj, method_name):
            return None
        
        method = getattr(obj, method_name)
        
        if not callable(method):
            return None
        
        try:
            return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error invoking method {method_name} on object {obj_id}: {e}")
            return None
    
    def get_object_state(self, obj_id: str) -> Dict[str, Any]:
        """
        Get the state of an object (attributes and their values).
        
        Args:
            obj_id: ID of the object
            
        Returns:
            state: Dictionary with object state
        """
        attributes = self.get_object_attributes(obj_id)
        obj = self.get_object(obj_id)
        
        if obj is None:
            return {}
        
        return {
            'id': obj_id,
            'type': type(obj).__name__,
            'attributes': attributes
        }
    
    def register_current_state(self) -> None:
        """
        Register the current state of all modules in the boundary awareness system.
        """
        # This would typically be called during system initialization
        # to register all the relevant objects for later access
        try:
            # Try to find and register swarm intelligence objects if they exist
            import sys
            for module_name in list(sys.modules.keys()):
                if 'swarm_intelligence' in module_name.lower() or 'agent' in module_name.lower():
                    module = sys.modules[module_name]
                    for name in dir(module):
                        obj = getattr(module, name)
                        if isinstance(obj, type):  # If it's a class
                            try:
                                # Try to find class instances
                                for instance_name in dir(module):
                                    instance = getattr(module, instance_name)
                                    if isinstance(instance, obj):
                                        self.register_object(f"{name}_{instance_name}", instance, name)
                            except:
                                pass
            
            logger.info("Registered current system state objects")
        except Exception as e:
            logger.error(f"Error registering current state: {e}")


class SystemIntrospection:
    """
    Main class that combines all introspection capabilities.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the system introspection module.
        
        Args:
            base_path: Base directory of the codebase
        """
        self.base_path = base_path or SWARM_INTELLIGENCE_DIR
        self.code_parser = CodeParser()
        self.concept_mapper = ConceptMapper(self.code_parser)
        self.self_reference = SelfReferenceFramework(self.concept_mapper)
        self.memory_access = MemoryAccessLayer()
        
        # Initialize by analyzing the codebase
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize the system introspection module."""
        try:
            # Verify code files
            if not self.verify_code_files():
                logger.warning("Some required code files are missing.")
            
            # Analyze the codebase
            project_info = self.code_parser.analyze_project(self.base_path)
            if "error" in project_info:
                logger.error(f"Error analyzing project: {project_info['error']}")
            
            # Generate concept mapping
            self.concept_mapper.generate_default_mapping(self.base_path)
            
            # Load mapping file if it exists
            if os.path.exists(MAPPING_FILE):
                self.concept_mapper.load_concept_mapping(MAPPING_FILE)
            else:
                # Create default mapping file
                self.concept_mapper.save_concept_mapping(MAPPING_FILE)
            
            # Register current system state
            self.memory_access.register_current_state()
            
            logger.info("System introspection module initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing system introspection: {e}")
    
    def verify_code_files(self) -> bool:
        """Verify that all required code files are present and accessible."""
        required_files = [
            os.path.join(self.base_path, "adaptive_swarm_intelligence_algorithm_with_dynamic_communication.py")
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"Missing required code files: {missing_files}")
            return False
        
        return True
    
    def query_concept(self, concept_name: str) -> Dict[str, Any]:
        """
        Query information about a concept.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            concept_info: Dictionary with concept information
        """
        try:
            # Get knowledge about the concept
            knowledge = self.self_reference.get_concept_knowledge(concept_name)
            
            # Get implementation details
            implementation = self.concept_mapper.get_implementation(concept_name)
            
            # Combine the information
            concept_info = {
                'name': concept_name,
                'knowledge': knowledge,
                'implementation': implementation
            }
            
            # Add code details for implementation
            if implementation:
                code_details = {}
                
                # Add class details
                if 'classes' in implementation:
                    code_details['classes'] = []
                    
                    for class_name in implementation['classes']:
                        class_info = self.code_parser.find_class_definition(class_name)
                        if class_info:
                            code_details['classes'].append(class_info)
                
                # Add method details
                if 'methods' in implementation:
                    code_details['methods'] = []
                    
                    for method_key in implementation['methods']:
                        if '.' in method_key:
                            class_name, method_name = method_key.split('.')
                            method_info = self.code_parser.find_method_definition(class_name, method_name)
                            if method_info:
                                code_details['methods'].append(method_info)
                
                # Add function details
                if 'functions' in implementation:
                    code_details['functions'] = []
                    
                    for function_name in implementation['functions']:
                        function_info = self.code_parser.find_function_definition(function_name)
                        if function_info:
                            code_details['functions'].append(function_info)
                
                # Add parameters if available
                if 'parameters' in implementation:
                    code_details['parameters'] = implementation['parameters']
                
                concept_info['code_details'] = code_details
            
            return concept_info
        except Exception as e:
            logger.error(f"Error querying concept {concept_name}: {e}")
            return {'error': str(e)}
    
    def query_implementation(self, implementation_name: str) -> Dict[str, Any]:
        """
        Query information about an implementation.
        
        Args:
            implementation_name: Name of the implementation (class, method, or function)
            
        Returns:
            implementation_info: Dictionary with implementation information
        """
        try:
            implementation_info = {
                'name': implementation_name,
                'type': 'unknown',
                'details': None,
                'concepts': self.concept_mapper.get_concepts_for_implementation(implementation_name)
            }
            
            # Check if it's a class
            class_info = self.code_parser.find_class_definition(implementation_name)
            if class_info:
                implementation_info['type'] = 'class'
                implementation_info['details'] = class_info
                return implementation_info
            
            # Check if it's a method
            if '.' in implementation_name:
                class_name, method_name = implementation_name.split('.')
                method_info = self.code_parser.find_method_definition(class_name, method_name)
                if method_info:
                    implementation_info['type'] = 'method'
                    implementation_info['details'] = method_info
                    return implementation_info
            
            # Check if it's a function
            function_info = self.code_parser.find_function_definition(implementation_name)
            if function_info:
                implementation_info['type'] = 'function'
                implementation_info['details'] = function_info
            
            return implementation_info
        except Exception as e:
            logger.error(f"Error querying implementation {implementation_name}: {e}")
            return {'error': str(e)}
    
    def get_runtime_object(self, obj_id: str) -> Dict[str, Any]:
        """
        Get information about a runtime object.
        
        Args:
            obj_id: ID of the object
            
        Returns:
            object_info: Dictionary with object information
        """
        try:
            return self.memory_access.get_object_state(obj_id)
        except Exception as e:
            logger.error(f"Error getting runtime object {obj_id}: {e}")
            return {'error': str(e)}
    
    def find_implementation_for_concept(self, concept_query: str) -> List[Dict[str, Any]]:
        """
        Find implementations related to a concept query.
        
        Args:
            concept_query: Concept search query
            
        Returns:
            implementations: List of implementation information
        """
        try:
            # Search for matching concepts
            matching_concepts = self.concept_mapper.search_concepts(concept_query)
            
            implementations = []
            
            for concept_name in matching_concepts:
                implementation = self.concept_mapper.get_implementation(concept_name)
                
                if implementation:
                    implementations.append({
                        'concept': concept_name,
                        'implementation': implementation
                    })
            
            return implementations
        except Exception as e:
            logger.error(f"Error finding implementation for concept {concept_query}: {e}")
            return [{'error': str(e)}]
    
    def explain_implementation(self, implementation_name: str) -> Dict[str, Any]:
        """
        Generate an explanation of an implementation.
        
        Args:
            implementation_name: Name of the implementation (class, method, or function)
            
        Returns:
            explanation: Dictionary with explanation information
        """
        try:
            # Query the implementation
            implementation_info = self.query_implementation(implementation_name)
            
            if 'error' in implementation_info:
                return implementation_info
                
            if implementation_info['type'] == 'unknown':
                return {
                    'name': implementation_name,
                    'explanation': f"No implementation found for '{implementation_name}'."
                }
            
            # Get associated concepts
            concepts = implementation_info['concepts']
            
            # Generate explanation based on the implementation type
            explanation = {
                'name': implementation_name,
                'type': implementation_info['type'],
                'concepts': concepts,
                'explanation': '',
                'code': None
            }
            
            details = implementation_info['details']
            
            if implementation_info['type'] == 'class':
                explanation['explanation'] = self._generate_class_explanation(details)
                explanation['code'] = {
                    'methods': [method['name'] for method in details.get('methods', [])],
                    'attributes': [attr['name'] for attr in details.get('attributes', [])]
                }
            
            elif implementation_info['type'] == 'method':
                explanation['explanation'] = self._generate_method_explanation(details)
                explanation['code'] = {
                    'parameters': [param['name'] for param in details.get('parameters', [])],
                    'body': details.get('body', [])
                }
            
            elif implementation_info['type'] == 'function':
                explanation['explanation'] = self._generate_function_explanation(details)
                explanation['code'] = {
                    'parameters': [param['name'] for param in details.get('parameters', [])],
                    'body': details.get('body', [])
                }
            
            return explanation
        except Exception as e:
            logger.error(f"Error explaining implementation {implementation_name}: {e}")
            return {'error': str(e)}
    
    def _generate_class_explanation(self, class_info: Dict[str, Any]) -> str:
        """Generate explanation for a class."""
        docstring = class_info.get('docstring', 'No description available.')
        methods = class_info.get('methods', [])
        attributes = class_info.get('attributes', [])
        
        explanation = f"{docstring}\n\n"
        
        if attributes:
            explanation += "Attributes:\n"
            for attr in attributes:
                explanation += f"- {attr['name']}\n"
            explanation += "\n"
        
        if methods:
            explanation += "Methods:\n"
            for method in methods:
                method_desc = method.get('docstring', '')
                first_sentence = method_desc.split('.')[0] if method_desc else 'No description.'
                explanation += f"- {method['name']}: {first_sentence}.\n"
        
        return explanation
    
    def _generate_method_explanation(self, method_info: Dict[str, Any]) -> str:
        """Generate explanation for a method."""
        docstring = method_info.get('docstring', 'No description available.')
        parameters = method_info.get('parameters', [])
        
        explanation = f"{docstring}\n\n"
        
        if parameters:
            explanation += "Parameters:\n"
            for param in parameters:
                param_type = f": {param['type']}" if param['type'] else ""
                explanation += f"- {param['name']}{param_type}\n"
        
        return explanation
    
    def _generate_function_explanation(self, function_info: Dict[str, Any]) -> str:
        """Generate explanation for a function."""
        docstring = function_info.get('docstring', 'No description available.')
        parameters = function_info.get('parameters', [])
        
        explanation = f"{docstring}\n\n"
        
        if parameters:
            explanation += "Parameters:\n"
            for param in parameters:
                param_type = f": {param['type']}" if param['type'] else ""
                explanation += f"- {param['name']}{param_type}\n"
        
        return explanation
    
    def get_implementation_details(self, concept: str, detail_type: str = None) -> Dict[str, Any]:
        """
        Get specific implementation details for a concept.
        
        Args:
            concept: The concept to look up (e.g., "AgentDecisionMaking")
            detail_type: Specific detail type to retrieve (e.g., "methods", "parameters")
            
        Returns:
            Dictionary with implementation details
        """
        try:
            concept_info = self.query_concept(concept)
            
            if 'error' in concept_info:
                return concept_info
                
            if 'implementation' not in concept_info or not concept_info['implementation']:
                return {'error': f"No implementation found for concept '{concept}'"}
            
            implementation = concept_info['implementation']
            
            # If detail type is specified, return only that part
            if detail_type:
                if detail_type in implementation:
                    return {
                        'concept': concept,
                        detail_type: implementation[detail_type]
                    }
                else:
                    return {'error': f"Detail type '{detail_type}' not found for concept '{concept}'"}
            
            # If code details are available, include them
            if 'code_details' in concept_info:
                implementation['code_details'] = concept_info['code_details']
            
            return {
                'concept': concept,
                'implementation': implementation
            }
        except Exception as e:
            logger.error(f"Error getting implementation details for {concept}: {e}")
            return {'error': str(e)}


# Kotlin bridge for the System Introspection Module
class KotlinBridge:
    """
    Bridge to expose the System Introspection Module to Kotlin code.
    """
    
    def __init__(self, introspection: SystemIntrospection = None):
        """
        Initialize the Kotlin bridge.
        
        Args:
            introspection: SystemIntrospection instance
        """
        try:
            self.introspection = introspection or SystemIntrospection()
            logger.info("Kotlin bridge initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Kotlin bridge: {e}")
            self.introspection = None
    
    def query_concept_json(self, concept_name: str) -> str:
        """
        Query information about a concept and return as JSON.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            json_result: JSON string with concept information
        """
        try:
            if not self.introspection:
                return json.dumps({"error": "Introspection module not initialized"})
                
            concept_info = self.introspection.query_concept(concept_name)
            return json.dumps(concept_info)
        except Exception as e:
            logger.error(f"Error in query_concept_json for {concept_name}: {e}")
            return json.dumps({"error": str(e)})
    
    def query_implementation_json(self, implementation_name: str) -> str:
        """
        Query information about an implementation and return as JSON.
        
        Args:
            implementation_name: Name of the implementation
            
        Returns:
            json_result: JSON string with implementation information
        """
        try:
            if not self.introspection:
                return json.dumps({"error": "Introspection module not initialized"})
                
            implementation_info = self.introspection.query_implementation(implementation_name)
            return json.dumps(implementation_info)
        except Exception as e:
            logger.error(f"Error in query_implementation_json for {implementation_name}: {e}")
            return json.dumps({"error": str(e)})
    
    def get_runtime_object_json(self, obj_id: str) -> str:
        """
        Get information about a runtime object and return as JSON.
        
        Args:
            obj_id: ID of the object
            
        Returns:
            json_result: JSON string with object information
        """
        try:
            if not self.introspection:
                return json.dumps({"error": "Introspection module not initialized"})
                
            object_info = self.introspection.get_runtime_object(obj_id)
            return json.dumps(object_info)
        except Exception as e:
            logger.error(f"Error in get_runtime_object_json for {obj_id}: {e}")
            return json.dumps({"error": str(e)})
    
    def find_implementation_for_concept_json(self, concept_query: str) -> str:
        """
        Find implementations related to a concept query and return as JSON.
        
        Args:
            concept_query: Concept search query
            
        Returns:
            json_result: JSON string with implementation information
        """
        try:
            if not self.introspection:
                return json.dumps({"error": "Introspection module not initialized"})
                
            implementations = self.introspection.find_implementation_for_concept(concept_query)
            return json.dumps(implementations)
        except Exception as e:
            logger.error(f"Error in find_implementation_for_concept_json for {concept_query}: {e}")
            return json.dumps({"error": str(e)})
    
    def explain_implementation_json(self, implementation_name: str) -> str:
        """
        Generate an explanation of an implementation and return as JSON.
        
        Args:
            implementation_name: Name of the implementation
            
        Returns:
            json_result: JSON string with explanation information
        """
        try:
            if not self.introspection:
                return json.dumps({"error": "Introspection module not initialized"})
                
            explanation = self.introspection.explain_implementation(implementation_name)
            return json.dumps(explanation)
        except Exception as e:
            logger.error(f"Error in explain_implementation_json for {implementation_name}: {e}")
            return json.dumps({"error": str(e)})
    
    def get_implementation_details_json(self, concept: str, detail_type: str = None) -> str:
        """
        Get specific implementation details for a concept and return as JSON.
        
        Args:
            concept: The concept to look up
            detail_type: Specific detail type to retrieve
            
        Returns:
            json_result: JSON string with implementation details
        """
        try:
            if not self.introspection:
                return json.dumps({"error": "Introspection module not initialized"})
                
            details = self.introspection.get_implementation_details(concept, detail_type)
            return json.dumps(details)
        except Exception as e:
            logger.error(f"Error in get_implementation_details_json for {concept}: {e}")
            return json.dumps({"error": str(e)})
    
    def test_connection(self) -> str:
        """
        Test the connection to the introspection module.
        
        Returns:
            json_result: JSON string with test result
        """
        try:
            if not self.introspection:
                return json.dumps({
                    "status": "error",
                    "message": "Introspection module not initialized"
                })
                
            # Check if we can access the code parser
            modules = len(self.introspection.code_parser.module_paths)
            classes = len(self.introspection.code_parser.class_info)
            
            return json.dumps({
                "status": "success",
                "message": f"Connection successful. Found {modules} modules and {classes} classes.",
                "timestamp": str(import_time)
            })
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            return json.dumps({
                "status": "error",
                "message": f"Connection test failed: {str(e)}"
            })


# Helper functions for common introspection tasks
def get_agent_decision_making_details():
    """Get specific implementation details for agent decision-making."""
    introspection = SystemIntrospection()
    return introspection.get_implementation_details("AgentDecisionMaking", "methods")

def get_energy_cost_details():
    """Get specific implementation details for energy costs."""
    introspection = SystemIntrospection()
    return introspection.get_implementation_details("EnergyManagement", "parameters")

def get_swarm_communication_details():
    """Get specific implementation details for swarm communication."""
    introspection = SystemIntrospection()
    return introspection.get_implementation_details("AgentCommunication", "methods")


# Set up tracking for import time
import_time = None
try:
    from datetime import datetime
    import_time = datetime.now()
except:
    pass


# Initialize the Kotlin bridge for external access
kotlin_bridge = None
try:
    introspection = SystemIntrospection()
    kotlin_bridge = KotlinBridge(introspection)
    logger.info("Kotlin bridge initialized and ready for use")
except Exception as e:
    logger.error(f"Failed to initialize Kotlin bridge: {e}")


# Example usage and test function
def test_introspection_module():
    """Test the introspection module functionality."""
    print("Testing System Introspection Module...")
    
    # Initialize
    introspection = SystemIntrospection()
    
    # Test accessing agent decision making details
    decision_details = introspection.get_implementation_details("AgentDecisionMaking", "methods")
    if "error" in decision_details:
        print(f"Error: {decision_details['error']}")
    else:
        methods = decision_details.get("methods", [])
        print(f"Successfully retrieved agent decision making details: {len(methods)} methods found")
        for method in methods:
            print(f"  - {method}")
    
    # Test accessing energy cost details
    energy_details = introspection.get_implementation_details("EnergyManagement", "parameters")
    if "error" in energy_details:
        print(f"Error: {energy_details['error']}")
    else:
        parameters = energy_details.get("parameters", {})
        print(f"Successfully retrieved energy cost details:")
        for action, cost in parameters.items():
            print(f"  - {action}: {cost}")
    
    # Test Kotlin bridge
    bridge = KotlinBridge(introspection)
    connection_test = bridge.test_connection()
    print(f"Kotlin bridge connection test: {connection_test}")
    
    print("Testing complete!")


if __name__ == "__main__":
    test_introspection_module()
```

This completes the implementation of the SystemIntrospection module with comprehensive error handling, logging, and helper functions to make it robust and easy to integrate with the Kotlin application.
