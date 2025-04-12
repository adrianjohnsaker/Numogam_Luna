"""
ModuleDocumentationRepository.py

This module serves as a comprehensive repository of documentation for Amelia's internal
modules, allowing for detailed self-awareness and the ability to explain her own
functionality at multiple levels of technical detail.

Features:
- Centralized storage for all module documentation
- Version tracking for documentation updates
- Module relationship management
- Technical concepts indexing
- Documentation completeness metrics
- Search and retrieval capabilities
- Serialization/deserialization for persistence
"""

import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core system types
from core_types import (
    ModuleReference,
    ModuleDocumentation,
    AlgorithmDescription,
    CodeFragment,
    DiagramReference,
    ImplementationDetail,
    TechnicalConcept,
    DocumentationCategory,
    PerformanceMetric
)

@dataclass
class DocumentationVersion:
    """Class to track versions of module documentation"""
    content: ModuleDocumentation
    timestamp: datetime
    version_hash: str
    changes: Optional[List[str]] = None
    author: Optional[str] = None

@dataclass
class ModuleRelationships:
    """Dataclass to store module relationship information"""
    depends_on: Set[str] = field(default_factory=set)
    required_by: Set[str] = field(default_factory=set)
    related_to: Set[str] = field(default_factory=set)

@dataclass
class DocumentationMetrics:
    """Dataclass to store documentation metrics"""
    total_modules: int = 0
    fully_documented_modules: int = 0
    partially_documented_modules: int = 0
    undocumented_modules: int = 0
    avg_documentation_completeness: float = 0.0
    last_update_time: Optional[datetime] = None
    categories: Dict[str, int] = field(default_factory=dict)

class ModuleDocumentationRepository:
    """
    Central repository for comprehensive documentation of Amelia's internal modules,
    enabling detailed self-awareness and technical explanations.
    """
    
    def __init__(
        self,
        documentation_store: Optional[Dict[str, ModuleDocumentation]] = None,
        auto_discover_modules: bool = True,
        load_from_file: Optional[str] = None,
        enable_version_tracking: bool = True,
        version_history_limit: int = 10
    ):
        """
        Initialize the Module Documentation Repository.
        
        Args:
            documentation_store: Optional pre-loaded documentation
            auto_discover_modules: Whether to automatically discover modules
            load_from_file: Optional file path to load documentation from
            enable_version_tracking: Whether to track documentation versions
            version_history_limit: Maximum number of versions to keep per module
        """
        self.documentation_store = documentation_store or {}
        self.enable_version_tracking = enable_version_tracking
        self.version_history_limit = version_history_limit
        
        # Version history for documentation
        self.documentation_versions: Dict[str, List[DocumentationVersion]] = {}
        
        # Module relationships graph
        self.module_relationships: Dict[str, ModuleRelationships] = {}
        
        # Technical concepts index
        self.technical_concepts_index: Dict[str, TechnicalConcept] = {}
        
        # Documentation metrics
        self.metrics = DocumentationMetrics()
        
        # Load documentation if specified
        if load_from_file and os.path.exists(load_from_file):
            self.load_documentation(load_from_file)
        
        # Auto-discover modules if enabled
        if auto_discover_modules:
            self.discover_modules()
        
        logger.info("ModuleDocumentationRepository initialized with %d modules", len(self.documentation_store))
    
    def discover_modules(self) -> List[str]:
        """
        Discover modules in the system and create basic documentation entries.
        
        Returns:
            List of discovered module IDs
        """
        # This would typically scan the system for modules
        # For now, we'll return an empty list as this would be implementation-specific
        discovered_modules = []
        
        # Update metrics
        self._update_documentation_metrics()
        
        return discovered_modules
    
    def get_module_documentation(self, module_id: str) -> Optional[ModuleDocumentation]:
        """
        Get documentation for a specific module.
        
        Args:
            module_id: ID of the module to retrieve documentation for
            
        Returns:
            ModuleDocumentation if found, None otherwise
            
        Raises:
            ValueError: If module_id is empty
        """
        if not module_id:
            raise ValueError("Module ID cannot be empty")
            
        return self.documentation_store.get(module_id)
    
    def register_module_documentation(
        self, 
        documentation: ModuleDocumentation, 
        update_existing: bool = True,
        author: Optional[str] = None,
        changes: Optional[List[str]] = None
    ) -> bool:
        """
        Register or update documentation for a module.
        
        Args:
            documentation: Documentation to register
            update_existing: Whether to update existing documentation
            author: Optional author of the changes
            changes: List of changes made in this version
            
        Returns:
            True if registration was successful, False otherwise
            
        Raises:
            ValueError: If documentation is invalid
        """
        if not documentation.module_id:
            raise ValueError("Documentation must have a module_id")
            
        module_id = documentation.module_id
        
        # Check if documentation already exists
        if module_id in self.documentation_store:
            if not update_existing:
                logger.warning("Not updating existing documentation for module %s (update_existing=False)", module_id)
                return False
            
            # Store previous version if version tracking is enabled
            if self.enable_version_tracking:
                self._store_documentation_version(module_id, author, changes)
        
        # Update documentation
        self.documentation_store[module_id] = documentation
        
        # Update relationships and indexes
        self._update_module_relationships(documentation)
        self._update_technical_concepts_index(documentation)
        
        # Update metrics
        self._update_documentation_metrics()
        
        logger.info("Registered documentation for module %s", module_id)
        return True
    
    def _store_documentation_version(
        self,
        module_id: str,
        author: Optional[str] = None,
        changes: Optional[List[str]] = None
    ) -> None:
        """Store a version of the documentation in version history"""
        if module_id not in self.documentation_versions:
            self.documentation_versions[module_id] = []
        
        current_doc = self.documentation_store[module_id]
        content_hash = self._generate_documentation_hash(current_doc)
        
        version = DocumentationVersion(
            content=current_doc,
            timestamp=datetime.now(),
            version_hash=content_hash,
            changes=changes,
            author=author
        )
        
        self.documentation_versions[module_id].append(version)
        
        # Maintain version history limit
        if len(self.documentation_versions[module_id]) > self.version_history_limit:
            self.documentation_versions[module_id] = self.documentation_versions[module_id][-self.version_history_limit:]
    
    def _generate_documentation_hash(self, doc: ModuleDocumentation) -> str:
        """Generate a hash of the documentation content for version comparison"""
        serialized = self._serialize_documentation(doc)
        json_str = json.dumps(serialized, sort_keys=True).encode('utf-8')
        return hashlib.sha256(json_str).hexdigest()
    
    def _update_documentation_metrics(self) -> None:
        """Update documentation metrics"""
        total_modules = len(self.documentation_store)
        fully_documented = 0
        partially_documented = 0
        undocumented = 0
        completeness_sum = 0.0
        
        # Reset category counts
        self.metrics.categories.clear()
        
        for doc in self.documentation_store.values():
            completeness = self._calculate_documentation_completeness(doc)
            completeness_sum += completeness
            
            if completeness >= 0.9:
                fully_documented += 1
            elif completeness > 0:
                partially_documented += 1
            else:
                undocumented += 1
            
            # Update category counts
            if hasattr(doc, 'categories') and doc.categories:
                for category in doc.categories:
                    self.metrics.categories[category] = self.metrics.categories.get(category, 0) + 1
        
        self.metrics.total_modules = total_modules
        self.metrics.fully_documented_modules = fully_documented
        self.metrics.partially_documented_modules = partially_documented
        self.metrics.undocumented_modules = undocumented
        self.metrics.avg_documentation_completeness = (
            completeness_sum / total_modules if total_modules > 0 else 0.0
        )
        self.metrics.last_update_time = datetime.now()
    
    def _calculate_documentation_completeness(self, doc: ModuleDocumentation) -> float:
        """Calculate completeness score for module documentation (0.0 - 1.0)"""
        # Define sections and their weights
        sections = {
            "basic_info": 0.1,  # Name, description, purpose
            "conceptual": 0.2,  # Conceptual explanation
            "technical": 0.3,   # Technical explanations
            "algorithms": 0.15, # Algorithm descriptions
            "code": 0.1,        # Code examples
            "diagrams": 0.05,   # Visual diagrams
            "performance": 0.1  # Performance metrics
        }
        
        score = 0.0
        
        # Check basic info
        if doc.name and doc.description and doc.purpose:
            score += sections["basic_info"]
        elif doc.name and (doc.description or doc.purpose):
            score += sections["basic_info"] * 0.7
        elif doc.name:
            score += sections["basic_info"] * 0.3
        
        # Check conceptual explanation
        if doc.conceptual_explanation:
            score += sections["conceptual"]
        
        # Check technical explanations
        if doc.technical_explanation:
            if doc.implementation_details:
                score += sections["technical"]
            else:
                score += sections["technical"] * 0.7
        elif doc.technical_overview:
            score += sections["technical"] * 0.4
        
        # Check algorithm descriptions
        if hasattr(doc, 'algorithms') and doc.algorithms:
            algo_completeness = sum(1 for a in doc.algorithms if a.detailed_description) / len(doc.algorithms)
            score += sections["algorithms"] * algo_completeness
        
        # Check code examples
        if hasattr(doc, 'code_examples') and doc.code_examples:
            score += sections["code"]
        
        # Check diagrams
        if hasattr(doc, 'diagrams') and doc.diagrams:
            score += sections["diagrams"]
        
        # Check performance metrics
        if hasattr(doc, 'performance_metrics') and doc.performance_metrics:
            score += sections["performance"]
        
        return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
    
    def _update_module_relationships(self, doc: ModuleDocumentation) -> None:
        """Update module relationship graph based on documentation"""
        module_id = doc.module_id
        
        # Initialize relationships for this module if not exists
        if module_id not in self.module_relationships:
            self.module_relationships[module_id] = ModuleRelationships()
        
        # Update dependencies
        if hasattr(doc, 'dependencies') and doc.dependencies:
            self.module_relationships[module_id].depends_on = set(doc.dependencies)
            
            # Update reverse dependencies
            for dep_id in doc.dependencies:
                if dep_id not in self.module_relationships:
                    self.module_relationships[dep_id] = ModuleRelationships()
                
                self.module_relationships[dep_id].required_by.add(module_id)
        
        # Update related modules
        if hasattr(doc, 'related_modules') and doc.related_modules:
            self.module_relationships[module_id].related_to = set(doc.related_modules)
            
            # Update reverse relationships
            for rel_id in doc.related_modules:
                if rel_id not in self.module_relationships:
                    self.module_relationships[rel_id] = ModuleRelationships()
                
                self.module_relationships[rel_id].related_to.add(module_id)
    
    def _update_technical_concepts_index(self, doc: ModuleDocumentation) -> None:
        """Update technical concepts index based on documentation"""
        # Extract technical concepts from documentation
        if hasattr(doc, 'technical_concepts') and doc.technical_concepts:
            for concept in doc.technical_concepts:
                self.technical_concepts_index[concept.key] = concept
                # Ensure the concept has a related_modules list
                if not hasattr(concept, 'related_modules'):
                    concept.related_modules = []
                # Add this module to the concept's related modules if not already present
                if doc.module_id not in concept.related_modules:
                    concept.related_modules.append(doc.module_id)
        
        # Extract concepts from algorithms
        if hasattr(doc, 'algorithms') and doc.algorithms:
            for algo in doc.algorithms:
                if hasattr(algo, 'related_concepts') and algo.related_concepts:
                    for concept_key in algo.related_concepts:
                        if concept_key in self.technical_concepts_index:
                            concept = self.technical_concepts_index[concept_key]
                            if not hasattr(concept, 'related_modules'):
                                concept.related_modules = []
                            if doc.module_id not in concept.related_modules:
                                concept.related_modules.append(doc.module_id)
    
    def load_documentation(self, file_path: str) -> bool:
        """
        Load documentation from a JSON file.
        
        Args:
            file_path: Path to JSON file containing documentation
            
        Returns:
            True if loading was successful, False otherwise
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Documentation file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            loaded_count = 0
            for module_id, doc_data in data.items():
                try:
                    doc = self._deserialize_documentation(doc_data)
                    if self.register_module_documentation(doc, update_existing=True):
                        loaded_count += 1
                except Exception as e:
                    logger.error("Error loading documentation for module %s: %s", module_id, str(e))
            
            logger.info("Loaded documentation for %d modules from %s", loaded_count, file_path)
            return True
            
        except Exception as e:
            logger.error("Error loading documentation from file %s: %s", file_path, str(e))
            return False
    
    def save_documentation(self, file_path: str) -> bool:
        """
        Save documentation to a JSON file.
        
        Args:
            file_path: Path to save documentation to
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            data = {}
            for module_id, doc in self.documentation_store.items():
                data[module_id] = self._serialize_documentation(doc)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("Saved documentation for %d modules to %s", len(data), file_path)
            return True
            
        except Exception as e:
            logger.error("Error saving documentation to file %s: %s", file_path, str(e))
            return False
    
    def _serialize_documentation(self, doc: ModuleDocumentation) -> Dict[str, Any]:
        """Convert ModuleDocumentation to serializable dictionary"""
        result = {
            "module_id": doc.module_id,
            "name": doc.name,
            "description": doc.description,
            "purpose": doc.purpose,
            "version": getattr(doc, 'version', '1.0'),
            "last_updated": (
                doc.last_updated.isoformat() 
                if hasattr(doc, 'last_updated') and doc.last_updated 
                else None
            )
        }
        
        # Add optional fields if they exist
        optional_fields = [
            'conceptual_explanation', 'technical_overview', 'technical_explanation',
            'implementation_details', 'algorithms', 'code_examples', 'diagrams',
            'performance_metrics', 'dependencies', 'related_modules', 'categories',
            'technical_concepts'
        ]
        
        for field in optional_fields:
            if hasattr(doc, field) and getattr(doc, field):
                result[field] = getattr(doc, field)
        
        return result
    
    def _deserialize_documentation(self, data: Dict[str, Any]) -> ModuleDocumentation:
        """Convert dictionary to ModuleDocumentation object"""
        # Create basic ModuleDocumentation
        doc = ModuleDocumentation(
            module_id=data["module_id"],
            name=data["name"],
            description=data["description"],
            purpose=data["purpose"],
            version=data.get("version", "1.0"),
        )
        
        # Add last_updated if present
        if "last_updated" in data and data["last_updated"]:
            try:
                doc.last_updated = datetime.fromisoformat(data["last_updated"])
            except ValueError:
                doc.last_updated = datetime.now()
        
        # Add optional fields if present in data
        optional_fields = [
            'conceptual_explanation', 'technical_overview', 'technical_explanation',
            'implementation_details', 'algorithms', 'code_examples', 'diagrams',
            'performance_metrics', 'dependencies', 'related_modules', 'categories',
            'technical_concepts'
        ]
        
        for field in optional_fields:
            if field in data and data[field]:
                setattr(doc, field, data[field])
        
        return doc
    
    def get_documentation_for_category(self, category: str) -> List[ModuleDocumentation]:
        """
        Get all documentation for modules in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of ModuleDocumentation objects in the category
        """
        return [
            doc for doc in self.documentation_store.values()
            if hasattr(doc, 'categories') and category in doc.categories
        ]
    
    def get_technical_concept(self, concept_key: str) -> Optional[TechnicalConcept]:
        """
        Get a technical concept by its key.
        
        Args:
            concept_key: Key of the concept to retrieve
            
        Returns:
            TechnicalConcept if found, None otherwise
        """
        return self.technical_concepts_index.get(concept_key)
    
    def search_documentation(
        self, 
        query: str,
        search_fields: Optional[List[str]] = None,
        min_relevance: float = 0.1
    ) -> List[Tuple[ModuleDocumentation, float]]:
        """
        Search documentation for a query string.
        
        Args:
            query: Query string to search for
            search_fields: Optional list of fields to search (defaults to all)
            min_relevance: Minimum relevance score to include in results
            
        Returns:
            List of (ModuleDocumentation, relevance_score) tuples, sorted by relevance
        """
        if not query:
            return []
            
        query = query.lower()
        default_search_fields = [
            'name', 'description', 'purpose', 'conceptual_explanation',
            'technical_overview', 'technical_explanation'
        ]
        search_fields = search_fields or default_search_fields
        
        results = []
        
        for doc in self.documentation_store.values():
            relevance = 0.0
            
            # Search in specified fields
            for field in search_fields:
                if hasattr(doc, field):
                    field_value = str(getattr(doc, field)).lower()
                    if query in field_value:
                        # Higher weight for matches in name and purpose
                        weight = 0.5 if field in ['name', 'purpose'] else 0.2
                        relevance += weight
            
            # Additional algorithm-specific search
            if hasattr(doc, 'algorithms') and doc.algorithms:
                for algo in doc.algorithms:
                    if query in algo.name.lower() or query in algo.short_description.lower():
                        relevance += 0.1
            
            # Add to results if meets minimum relevance
            if relevance >= min_relevance:
                results.append((doc, relevance))
        
        # Sort by relevance (highest first)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def get_related_modules(self, module_id: str) -> ModuleRelationships:
        """
        Get all modules related to the specified module.
        
        Args:
            module_id: ID of the module to get related modules for
            
        Returns:
            ModuleRelationships object with depends_on, required_by, and related_to sets
            
        Raises:
            ValueError: If module_id is invalid
        """
        if not module_id:
            raise ValueError("Module ID cannot be empty")
            
        return self.module_relationships.get(
            module_id, 
            ModuleRelationships()
        )
    
    def get_documentation_metrics(self) -> DocumentationMetrics:
        """
        Get metrics about the documentation.
        
        Returns:
            DocumentationMetrics object with current metrics
        """
        return self.metrics
    
    def generate_documentation_report(self) -> str:
        """
        Generate a human-readable report about the documentation status.
        
        Returns:
            String containing the formatted report
        """
        metrics = self.metrics
        total = metrics.total_modules
        
        # Calculate percentages
        def calc_pct(count: int) -> float:
            return (count / total * 100) if total > 0 else 0.0
        
        report = [
            "# Module Documentation Status Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            f"- Total modules: {total}",
            f"- Fully documented: {metrics.fully_documented_modules} ({calc_pct(metrics.fully_documented_modules):.1f}%)",
            f"- Partially documented: {metrics.partially_documented_modules} ({calc_pct(metrics.partially_documented_modules):.1f}%)",
            f"- Undocumented: {metrics.undocumented_modules} ({calc_pct(metrics.undocumented_modules):.1f}%)",
            f"- Average completeness: {metrics.avg_documentation_completeness * 100:.1f}%",
            f"- Last updated: {metrics.last_update_time.strftime('%Y-%m-%d %H:%M:%S') if metrics.last_update_time else 'Never'}",
            "",
            "## Module Categories"
        ]
        
        # Add category information
        for category, count in sorted(metrics.categories.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {category}: {count} modules ({calc_pct(count):.1f}%)")
        
        # Add technical concepts section
        report.extend([
            "",
            "## Technical Concepts",
            f"- Total concepts: {len(self.technical_concepts_index)}",
            f"- Concepts with related modules: {sum(1 for c in self.technical_concepts_index.values() if hasattr(c, 'related_modules') and c.related_modules)}"
        ])
        
        return "\n".join(report)
    
    def get_module_version_history(self, module_id: str) -> List[DocumentationVersion]:
        """
        Get version history for a module.
        
        Args:
            module_id: ID of the module to get history for
            
        Returns:
            List of DocumentationVersion objects, newest first
            
        Raises:
            ValueError: If module_id is invalid or version tracking is disabled
        """
        if not self.enable_version_tracking:
            raise ValueError("Version tracking is disabled")
            
        if not module_id:
            raise ValueError("Module ID cannot be empty")
            
        return self.documentation_versions.get(module_id, [])[::-1]  # Return newest first
    
    def register_technical_concept(self, concept: TechnicalConcept) -> None:
        """
        Register a new technical concept.
        
        Args:
            concept: TechnicalConcept to register
            
        Raises:
            ValueError: If concept is invalid
        """
        if not concept.key or not concept.short_description:
            raise ValueError("Technical concept must have a key and short description")
            
        self.technical_concepts_index[concept.key] = concept
        logger.info("Registered technical concept: %s", concept.key)
    
    def get_undocumented_modules(self) -> List[str]:
        """
        Get a list of module IDs with no or minimal documentation.
        
        Returns:
            List of module IDs with documentation completeness < 0.1
        """
        return [
            module_id for module_id, doc in self.documentation_store.items()
            if self._calculate_documentation_completeness(doc) < 0.1
        ]
    
    def get_module_dependencies(self, module_id: str, recursive: bool = False) -> Set[str]:
        """
        Get all dependencies for a module, optionally recursively.
        
        Args:
            module_id: ID of the module to get dependencies for
            recursive: Whether to include transitive dependencies
            
        Returns:
            Set of module IDs that the specified module depends on
        """
        if module_id not in self.module_relationships:
            return set()
            
        if not recursive:
            return self.module_relationships[module_id].depends_on
            
        # Recursive dependency resolution
        dependencies = set()
        visited = set()
        
        def _resolve_deps(current_id: str):
            if current_id in visited:
                return
                
            visited.add(current_id)
            current_deps = self.module_relationships.get(current_id, ModuleRelationships()).depends_on
            
            for dep_id in current_deps:
                dependencies.add(dep_id)
                _resolve_deps(dep_id)
        
        _resolve_deps(module_id)
        return dependencies
