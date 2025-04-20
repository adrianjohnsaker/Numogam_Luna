#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evolutionary Pressure System

This module implements a system for applying evolutionary pressure to Amelia's
narrative elements, allowing for natural selection of effective narrative patterns
and adaptive generation of new content.

Core functionality:
- Fitness evaluation across multiple dimensions
- Selection mechanisms for narrative elements
- Variation generation with controlled mutation
- Population management of narrative elements
- Evolutionary strategy adaptation
"""

import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict, Counter
import re
import math
import random

# Import from other modules if available, otherwise define interface classes
try:
    from agentic_myth_constructor import MythConstructor, NarrativeFragment, NarrativeSequence
except ImportError:
    # Define interface classes for type checking
    class NarrativeFragment:
        """Interface class for NarrativeFragment"""
        id = ""
        content = ""
        fragment_type = ""
        context_weights = {}
        
    class NarrativeSequence:
        """Interface class for NarrativeSequence"""
        id = ""
        title = ""
        fragments = []
        
    class MythConstructor:
        """Interface class for MythConstructor"""
        def get_fragment(self, fragment_id: str) -> Optional[NarrativeFragment]:
            return         return new_member_ids
    
    def _mutation_variation(self,
                          source_ids: List[str],
                          variation_count: int,
                          mutation_rate: float,
                          mutation_magnitude: float,
                          directed_bias: float) -> List[str]:
        """
        Create variations using mutation operator.
        
        Args:
            source_ids: IDs of source members
            variation_count: Number of variations to create
            mutation_rate: Probability of mutation per element
            mutation_magnitude: Size of mutation changes
            directed_bias: Bias towards directed vs. random mutation
            
        Returns:
            List of newly created member IDs
        """
        if not source_ids:
            return []
        
        # Get source members
        source_members = [self.population.get(sid) for sid in source_ids]
        source_members = [m for m in source_members if m and m.is_active]
        
        if not source_members:
            return []
        
        # Create variations
        new_member_ids = []
        
        for _ in range(variation_count):
            # Select a source member to mutate
            source = random.choice(source_members)
            
            # Apply mutation based on entity type
            if source.entity_type == "fragment":
                new_id = self._mutate_fragment(
                    source.id, 
                    mutation_rate, 
                    mutation_magnitude, 
                    directed_bias
                )
                
                if new_id:
                    new_member_ids.append(new_id)
            
            elif source.entity_type == "sequence":
                new_id = self._mutate_sequence(
                    source.id, 
                    mutation_rate, 
                    mutation_magnitude, 
                    directed_bias
                )
                
                if new_id:
                    new_member_ids.append(new_id)
        
        return new_member_ids
    
    def _mutate_fragment(self,
                       fragment_id: str,
                       mutation_rate: float,
                       mutation_magnitude: float,
                       directed_bias: float) -> Optional[str]:
        """
        Mutate a narrative fragment.
        
        Args:
            fragment_id: ID of the fragment to mutate
            mutation_rate: Probability of mutation per element
            mutation_magnitude: Size of mutation changes
            directed_bias: Bias towards directed vs. random mutation
            
        Returns:
            ID of the new fragment, or None if mutation failed
        """
        # Check if we have a myth constructor
        if not self.myth_constructor:
            return None
        
        # Get the source fragment
        source_fragment = self.myth_constructor.get_fragment(fragment_id)
        if not source_fragment:
            return None
        
        # Get fragment properties
        content = getattr(source_fragment, "content", "")
        fragment_type = getattr(source_fragment, "fragment_type", "")
        context_weights = getattr(source_fragment, "context_weights", {})
        
        if not content:
            return None
        
        # Apply mutation to content (simplified example)
        # In a real implementation, this would use more sophisticated NLP techniques
        
        # Split into sentences
        sentences = re.split(r'[.!?]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return None
        
        # Determine mutation approach based on directed bias
        if random.random() < directed_bias and self.analytics:
            # Directed mutation using analytics
            analytics = self.analytics.get_analytics_overview()
            key_themes = analytics.get("key_themes", [])
            
            # Apply theme-directed mutations
            mutated_sentences = []
            for sentence in sentences:
                if random.random() < mutation_rate:
                    # Enhance a sentence with a key theme
                    if key_themes and random.random() < 0.5:
                        theme = random.choice(key_themes)
                        mutated = f"{sentence} This relates to {theme}."
                        mutated_sentences.append(mutated)
                    else:
                        # Random enhancement
                        mutated = f"{sentence} This evolves further."
                        mutated_sentences.append(mutated)
                else:
                    mutated_sentences.append(sentence)
            
            new_content = ". ".join(mutated_sentences)
        else:
            # Random mutation
            mutated_sentences = []
            for sentence in sentences:
                if random.random() < mutation_rate:
                    # Apply magnitude-scaled mutation
                    change_type = random.choice(["expand", "replace", "remove"])
                    
                    if change_type == "expand" and mutation_magnitude > 0.3:
                        mutated = f"{sentence} Additionally, this expands the narrative."
                        mutated_sentences.append(mutated)
                    elif change_type == "replace":
                        mutated = "This element transforms the original content."
                        mutated_sentences.append(mutated)
                    # For "remove", we just skip this sentence
                else:
                    mutated_sentences.append(sentence)
            
            new_content = ". ".join(mutated_sentences)
        
        # Ensure content has minimum length
        if len(new_content) < 20:
            new_content = f"{content} This has evolved through mutation."
        
        # Simulate creation of a new fragment
        # In a real implementation, this would call the actual creation method
        new_fragment_id = str(uuid.uuid4())
        
        # Add to population
        self.add_population_member(
            entity_id=new_fragment_id,
            entity_type="fragment",
            generation=self.current_generation + 1,
            parent_ids=[fragment_id]
        )
        
        # Cache the simulated entity
        self._entity_cache[new_fragment_id] = {
            "content": new_content,
            "fragment_type": fragment_type,
            "context_weights": context_weights.copy()
        }
        
        return new_fragment_id
    
    def _mutate_sequence(self,
                       sequence_id: str,
                       mutation_rate: float,
                       mutation_magnitude: float,
                       directed_bias: float) -> Optional[str]:
        """
        Mutate a narrative sequence.
        
        Args:
            sequence_id: ID of the sequence to mutate
            mutation_rate: Probability of mutation per element
            mutation_magnitude: Size of mutation changes
            directed_bias: Bias towards directed vs. random mutation
            
        Returns:
            ID of the new sequence, or None if mutation failed
        """
        # Check if we have a myth constructor
        if not self.myth_constructor:
            return None
        
        # Get the source sequence
        source_sequence = self.myth_constructor.get_sequence(sequence_id)
        if not source_sequence:
            return None
        
        # Get sequence properties
        title = getattr(source_sequence, "title", "")
        fragments = getattr(source_sequence, "fragments", [])
        
        if not fragments:
            return None
        
        # Apply mutation to fragment list
        new_fragments = fragments.copy()
        
        # Determine mutation approach based on directed bias
        if random.random() < directed_bias and self.analytics:
            # Directed mutation using analytics
            
            # Choose operation based on mutation magnitude
            if mutation_magnitude > 0.7:
                # Major restructuring
                if len(fragments) > 1:
                    # Shuffle some fragments
                    shuffle_count = max(1, int(len(fragments) * mutation_rate))
                    indices = random.sample(range(len(fragments)), shuffle_count)
                    
        def get_sequence(self, sequence_id: str) -> Optional[NarrativeSequence]:
            return None

try:
    from narrative_analytics_layer import NarrativeAnalytics
except ImportError:
    # Define interface class for type checking
    class NarrativeAnalytics:
        """Interface class for NarrativeAnalytics"""
        def get_analytics_overview(self) -> Dict[str, Any]:
            return {}

try:
    from autocritical_function import AutocriticalFunction, EvaluationResult
except ImportError:
    # Define interface classes for type checking
    class EvaluationResult:
        """Interface class for EvaluationResult"""
        overall_score = 0.5
        dimension_scores = {}
        
    class AutocriticalFunction:
        """Interface class for AutocriticalFunction"""
        def evaluate_fragment(self, fragment_id: str) -> EvaluationResult:
            return EvaluationResult()
        def evaluate_sequence(self, sequence_id: str) -> EvaluationResult:
            return EvaluationResult()


class FitnessFunction:
    """
    Represents a function for evaluating the fitness of narrative elements 
    across specific dimensions.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 weight: float = 1.0):
        """
        Initialize a fitness function.
        
        Args:
            name: Name of this fitness function
            description: Description of what this function evaluates
            weight: Relative importance of this function (0.0 to 2.0)
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.weight = max(0.0, min(2.0, weight))
        self.creation_date = datetime.now().isoformat()
        self.last_modified = self.creation_date
        
        # Evaluation parameters
        self.parameters = {}
        
        # Evaluation history
        self.evaluation_history = []
        
        # Adaptation metrics
        self.effectiveness_score = 0.5  # How well this function predicts quality
        self.adaptation_rate = 0.0  # How quickly this function adapts
        
        # Additional properties
        self.attributes = {}
    
    def add_parameter(self, name: str, value: Any) -> None:
        """
        Add an evaluation parameter to this function.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
        self.last_modified = datetime.now().isoformat()
    
    def record_evaluation(self, entity_id: str, score: float) -> None:
        """
        Record an evaluation using this function.
        
        Args:
            entity_id: ID of the evaluated entity
            score: Evaluation score (0.0 to 1.0)
        """
        self.evaluation_history.append({
            "entity_id": entity_id,
            "score": max(0.0, min(1.0, score)),
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]
            
        self.last_modified = datetime.now().isoformat()
    
    def update_weight(self, new_weight: float) -> None:
        """
        Update the relative importance of this function.
        
        Args:
            new_weight: New weight value (0.0 to 2.0)
        """
        self.weight = max(0.0, min(2.0, new_weight))
        self.last_modified = datetime.now().isoformat()
    
    def update_effectiveness(self, effectiveness: float, adaptation: float) -> None:
        """
        Update effectiveness metrics for this function.
        
        Args:
            effectiveness: New effectiveness score (0.0 to 1.0)
            adaptation: New adaptation rate (-1.0 to 1.0)
        """
        self.effectiveness_score = max(0.0, min(1.0, effectiveness))
        self.adaptation_rate = max(-1.0, min(1.0, adaptation))
        self.last_modified = datetime.now().isoformat()
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute for this function."""
        self.attributes[key] = value
        self.last_modified = datetime.now().isoformat()
    
    def get_average_score(self, recent_only: bool = False) -> float:
        """
        Get the average score across evaluations.
        
        Args:
            recent_only: Whether to only consider recent evaluations
            
        Returns:
            Average score, or 0.5 if no evaluations exist
        """
        if not self.evaluation_history:
            return 0.5
            
        if recent_only:
            # Only consider evaluations from the last 30 days
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            recent = [e["score"] for e in self.evaluation_history 
                     if e["timestamp"] >= cutoff]
            
            if not recent:
                return 0.5
                
            return sum(recent) / len(recent)
        
        return sum(e["score"] for e in self.evaluation_history) / len(self.evaluation_history)
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "parameters": self.parameters,
            "evaluation_history": self.evaluation_history,
            "effectiveness_score": self.effectiveness_score,
            "adaptation_rate": self.adaptation_rate,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FitnessFunction':
        """Create a FitnessFunction from a dictionary."""
        function = cls(
            name=data["name"],
            description=data["description"],
            weight=data["weight"]
        )
        
        function.id = data["id"]
        function.creation_date = data["creation_date"]
        function.last_modified = data["last_modified"]
        function.parameters = data["parameters"]
        function.evaluation_history = data["evaluation_history"]
        function.effectiveness_score = data["effectiveness_score"]
        function.adaptation_rate = data["adaptation_rate"]
        function.attributes = data["attributes"]
        
        return function


class SelectionStrategy:
    """
    Represents a strategy for selecting narrative elements based on fitness.
    """
    
    def __init__(self, 
                 name: str,
                 strategy_type: str,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a selection strategy.
        
        Args:
            name: Name of this strategy
            strategy_type: Type of strategy (tournament, roulette, elitist, etc.)
            parameters: Strategy-specific parameters
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.strategy_type = strategy_type
        self.parameters = parameters or {}
        self.creation_date = datetime.now().isoformat()
        self.last_modified = self.creation_date
        
        # Performance tracking
        self.selection_history = []
        self.performance_metrics = {
            "selection_diversity": 0.5,
            "selection_pressure": 0.5,
            "adaptation_rate": 0.0
        }
        
        # Additional properties
        self.attributes = {}
    
    def record_selection(self, selected_ids: List[str], 
                        population_size: int,
                        avg_fitness: float) -> None:
        """
        Record a selection operation.
        
        Args:
            selected_ids: IDs of selected elements
            population_size: Size of the population selected from
            avg_fitness: Average fitness of selected elements
        """
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "selected_count": len(selected_ids),
            "population_size": population_size,
            "avg_fitness": avg_fitness
        })
        
        # Limit history size
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
            
        self.last_modified = datetime.now().isoformat()
    
    def update_parameter(self, name: str, value: Any) -> None:
        """
        Update a strategy parameter.
        
        Args:
            name: Parameter name
            value: New parameter value
        """
        self.parameters[name] = value
        self.last_modified = datetime.now().isoformat()
    
    def update_performance_metrics(self, 
                                  diversity: Optional[float] = None,
                                  pressure: Optional[float] = None,
                                  adaptation: Optional[float] = None) -> None:
        """
        Update performance metrics for this strategy.
        
        Args:
            diversity: Selection diversity score (0.0 to 1.0)
            pressure: Selection pressure intensity (0.0 to 1.0)
            adaptation: Adaptation rate (-1.0 to 1.0)
        """
        if diversity is not None:
            self.performance_metrics["selection_diversity"] = max(0.0, min(1.0, diversity))
            
        if pressure is not None:
            self.performance_metrics["selection_pressure"] = max(0.0, min(1.0, pressure))
            
        if adaptation is not None:
            self.performance_metrics["adaptation_rate"] = max(-1.0, min(1.0, adaptation))
            
        self.last_modified = datetime.now().isoformat()
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute for this strategy."""
        self.attributes[key] = value
        self.last_modified = datetime.now().isoformat()
    
    def get_selection_rate(self) -> float:
        """
        Get the average selection rate (selected/population).
        
        Returns:
            Average selection rate, or 0.5 if no selection history
        """
        if not self.selection_history:
            return 0.5
            
        rates = [e["selected_count"] / e["population_size"] 
                for e in self.selection_history if e["population_size"] > 0]
        
        if not rates:
            return 0.5
            
        return sum(rates) / len(rates)
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "selection_history": self.selection_history,
            "performance_metrics": self.performance_metrics,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SelectionStrategy':
        """Create a SelectionStrategy from a dictionary."""
        strategy = cls(
            name=data["name"],
            strategy_type=data["strategy_type"],
            parameters=data["parameters"]
        )
        
        strategy.id = data["id"]
        strategy.creation_date = data["creation_date"]
        strategy.last_modified = data["last_modified"]
        strategy.selection_history = data["selection_history"]
        strategy.performance_metrics = data["performance_metrics"]
        strategy.attributes = data["attributes"]
        
        return strategy


class VariationOperator:
    """
    Represents an operator for creating variations of narrative elements.
    """
    
    def __init__(self, 
                 name: str,
                 operator_type: str,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a variation operator.
        
        Args:
            name: Name of this operator
            operator_type: Type of operator (mutation, recombination, etc.)
            parameters: Operator-specific parameters
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.operator_type = operator_type
        self.parameters = parameters or {}
        self.creation_date = datetime.now().isoformat()
        self.last_modified = self.creation_date
        
        # Performance tracking
        self.operation_history = []
        self.performance_metrics = {
            "variation_diversity": 0.5,
            "variation_quality": 0.5,
            "adaptation_rate": 0.0
        }
        
        # Additional properties
        self.attributes = {}
    
    def record_operation(self, source_ids: List[str], 
                       result_id: str,
                       fitness_change: float) -> None:
        """
        Record a variation operation.
        
        Args:
            source_ids: IDs of source elements
            result_id: ID of the resulting element
            fitness_change: Change in fitness from source to result
        """
        self.operation_history.append({
            "timestamp": datetime.now().isoformat(),
            "source_count": len(source_ids),
            "result_id": result_id,
            "fitness_change": fitness_change
        })
        
        # Limit history size
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-100:]
            
        self.last_modified = datetime.now().isoformat()
    
    def update_parameter(self, name: str, value: Any) -> None:
        """
        Update an operator parameter.
        
        Args:
            name: Parameter name
            value: New parameter value
        """
        self.parameters[name] = value
        self.last_modified = datetime.now().isoformat()
    
    def update_performance_metrics(self, 
                                  diversity: Optional[float] = None,
                                  quality: Optional[float] = None,
                                  adaptation: Optional[float] = None) -> None:
        """
        Update performance metrics for this operator.
        
        Args:
            diversity: Variation diversity score (0.0 to 1.0)
            quality: Variation quality score (0.0 to 1.0)
            adaptation: Adaptation rate (-1.0 to 1.0)
        """
        if diversity is not None:
            self.performance_metrics["variation_diversity"] = max(0.0, min(1.0, diversity))
            
        if quality is not None:
            self.performance_metrics["variation_quality"] = max(0.0, min(1.0, quality))
            
        if adaptation is not None:
            self.performance_metrics["adaptation_rate"] = max(-1.0, min(1.0, adaptation))
            
        self.last_modified = datetime.now().isoformat()
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute for this operator."""
        self.attributes[key] = value
        self.last_modified = datetime.now().isoformat()
    
    def get_average_fitness_change(self) -> float:
        """
        Get the average fitness change from operations.
        
        Returns:
            Average fitness change, or 0.0 if no operation history
        """
        if not self.operation_history:
            return 0.0
            
        changes = [e["fitness_change"] for e in self.operation_history]
        
        if not changes:
            return 0.0
            
        return sum(changes) / len(changes)
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "operator_type": self.operator_type,
            "parameters": self.parameters,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "operation_history": self.operation_history,
            "performance_metrics": self.performance_metrics,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VariationOperator':
        """Create a VariationOperator from a dictionary."""
        operator = cls(
            name=data["name"],
            operator_type=data["operator_type"],
            parameters=data["parameters"]
        )
        
        operator.id = data["id"]
        operator.creation_date = data["creation_date"]
        operator.last_modified = data["last_modified"]
        operator.operation_history = data["operation_history"]
        operator.performance_metrics = data["performance_metrics"]
        operator.attributes = data["attributes"]
        
        return operator


class PopulationMember:
    """
    Represents a member of the narrative population with fitness evaluation.
    """
    
    def __init__(self, 
                 entity_id: str,
                 entity_type: str,
                 generation: int = 0,
                 parent_ids: Optional[List[str]] = None):
        """
        Initialize a population member.
        
        Args:
            entity_id: ID of the narrative entity (fragment, sequence, etc.)
            entity_type: Type of entity
            generation: Generation number (0 for initial population)
            parent_ids: IDs of parent entities (for derived elements)
        """
        self.id = entity_id  # Use the entity's ID as the member ID
        self.entity_type = entity_type
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.creation_date = datetime.now().isoformat()
        
        # Fitness evaluation
        self.fitness_scores = {}  # function_id -> score
        self.overall_fitness = None
        self.fitness_history = []
        
        # Selection history
        self.selection_count = 0
        self.last_selected = None
        
        # Variation history
        self.variation_history = []
        
        # Survival status
        self.is_active = True
        self.deactivation_reason = None
        
        # Additional properties
        self.attributes = {}
    
    def add_fitness_score(self, function_id: str, score: float) -> None:
        """
        Add a fitness score for a specific function.
        
        Args:
            function_id: ID of the fitness function
            score: Fitness score (0.0 to 1.0)
        """
        self.fitness_scores[function_id] = max(0.0, min(1.0, score))
    
    def calculate_overall_fitness(self, 
                                functions: Dict[str, FitnessFunction],
                                normalized: bool = True) -> float:
        """
        Calculate the weighted overall fitness.
        
        Args:
            functions: Dict mapping function_id to FitnessFunction
            normalized: Whether to normalize the result to [0,1]
            
        Returns:
            Overall fitness score
        """
        if not self.fitness_scores:
            return 0.5
            
        total_weight = 0.0
        weighted_sum = 0.0
        
        for function_id, score in self.fitness_scores.items():
            function = functions.get(function_id)
            if function:
                weight = function.weight
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight > 0:
            self.overall_fitness = weighted_sum / total_weight
        else:
            self.overall_fitness = sum(self.fitness_scores.values()) / len(self.fitness_scores)
        
        # Record in fitness history
        self.fitness_history.append({
            "timestamp": datetime.now().isoformat(),
            "overall_fitness": self.overall_fitness
        })
        
        # Normalize if requested
        if normalized:
            return max(0.0, min(1.0, self.overall_fitness))
        
        return self.overall_fitness
    
    def record_selection(self) -> None:
        """Record that this member was selected."""
        self.selection_count += 1
        self.last_selected = datetime.now().isoformat()
    
    def record_variation(self, operator_id: str, result_id: str) -> None:
        """
        Record a variation operation producing a new entity.
        
        Args:
            operator_id: ID of the variation operator
            result_id: ID of the resulting entity
        """
        self.variation_history.append({
            "timestamp": datetime.now().isoformat(),
            "operator_id": operator_id,
            "result_id": result_id
        })
    
    def deactivate(self, reason: str) -> None:
        """
        Deactivate this population member.
        
        Args:
            reason: Reason for deactivation
        """
        self.is_active = False
        self.deactivation_reason = reason
    
    def get_fitness_trend(self, time_periods: int = 3) -> float:
        """
        Calculate the trend in fitness over time.
        
        Args:
            time_periods: Number of time periods to consider
            
        Returns:
            Trend value from -1.0 (decreasing) to 1.0 (increasing)
        """
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Get at most 'time_periods' most recent fitness values
        recent = self.fitness_history[-time_periods:]
        
        if len(recent) < 2:
            return 0.0
        
        # Simple trend calculation: direction of change from first to last
        first = recent[0]["overall_fitness"]
        last = recent[-1]["overall_fitness"]
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, (last - first) * 5))
    
    def get_age(self) -> int:
        """
        Get the age of this member in days.
        
        Returns:
            Age in days
        """
        try:
            creation = datetime.fromisoformat(self.creation_date)
            age = (datetime.now() - creation).days
            return max(0, age)
        except ValueError:
            return 0
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute for this member."""
        self.attributes[key] = value
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "creation_date": self.creation_date,
            "fitness_scores": self.fitness_scores,
            "overall_fitness": self.overall_fitness,
            "fitness_history": self.fitness_history,
            "selection_count": self.selection_count,
            "last_selected": self.last_selected,
            "variation_history": self.variation_history,
            "is_active": self.is_active,
            "deactivation_reason": self.deactivation_reason,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PopulationMember':
        """Create a PopulationMember from a dictionary."""
        member = cls(
            entity_id=data["id"],
            entity_type=data["entity_type"],
            generation=data["generation"],
            parent_ids=data["parent_ids"]
        )
        
        member.creation_date = data["creation_date"]
        member.fitness_scores = data["fitness_scores"]
        member.overall_fitness = data["overall_fitness"]
        member.fitness_history = data["fitness_history"]
        member.selection_count = data["selection_count"]
        member.last_selected = data["last_selected"]
        member.variation_history = data["variation_history"]
        member.is_active = data["is_active"]
        member.deactivation_reason = data["deactivation_reason"]
        member.attributes = data["attributes"]
        
        return member


class EvolutionaryPressure:
    """
    System for applying evolutionary pressure to narrative elements,
    allowing for selection and adaptation over time.
    """
    
    def __init__(self, 
                 myth_constructor: Optional[MythConstructor] = None,
                 autocritical: Optional[AutocriticalFunction] = None,
                 analytics: Optional[NarrativeAnalytics] = None,
                 name: str = "Amelia's Evolutionary Pressure"):
        """
        Initialize an evolutionary pressure system.
        
        Args:
            myth_constructor: Optional link to the myth constructor
            autocritical: Optional link to the autocritical function
            analytics: Optional link to the narrative analytics
            name: Name for this evolutionary pressure system
        """
        self.myth_constructor = myth_constructor
        self.autocritical = autocritical
        self.analytics = analytics
        self.name = name
        self.creation_date = datetime.now().isoformat()
        self.last_modified = self.creation_date
        
        # Core components
        self.fitness_functions = {}  # function_id -> FitnessFunction
        self.selection_strategies = {}  # strategy_id -> SelectionStrategy
        self.variation_operators = {}  # operator_id -> VariationOperator
        
        # Population
        self.population = {}  # member_id -> PopulationMember
        
        # Evolution tracking
        self.current_generation = 0
        self.generation_history = []
        
        # Evolution parameters
        self.parameters = {
            "population_size_target": 100,
            "selection_rate": 0.3,
            "variation_rate": 0.2,
            "extinction_threshold": 0.2,
            "novelty_threshold": 0.3,
            "age_penalty_factor": 0.01
        }
        
        # System metrics
        self.metrics = {
            "avg_population_fitness": 0.5,
            "fitness_improvement_rate": 0.0,
            "biodiversity_index": 0.5,
            "adaptation_rate": 0.0
        }
        
        # Initialize default fitness functions
        self._initialize_fitness_functions()
        
        # Initialize default selection strategies
        self._initialize_selection_strategies()
        
        # Initialize default variation operators
        self._initialize_variation_operators()
        
        # Operation history
        self.operation_history = []
        
        # Entity cache for performance optimization
        self._entity_cache = {}
        
    def add_fitness_function(self, 
                           name: str, 
                           description: str, 
                           weight: float = 1.0, 
                           parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new fitness function to the system.
        
        Args:
            name: Name of the function
            description: Description of what this function evaluates
            weight: Relative importance of this function (0.0 to 2.0)
            parameters: Function-specific parameters
            
        Returns:
            ID of the newly created function
        """
        function = FitnessFunction(
            name=name,
            description=description,
            weight=weight
        )
        
        if parameters:
            for param_name, param_value in parameters.items():
                function.add_parameter(param_name, param_value)
        
        self.fitness_functions[function.id] = function
        self.last_modified = datetime.now().isoformat()
        
        # Log operation
        self._log_operation("add_fitness_function", {"function_id": function.id})
        
        return function.id
    
    def add_selection_strategy(self,
                             name: str,
                             strategy_type: str,
                             parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new selection strategy to the system.
        
        Args:
            name: Name of the strategy
            strategy_type: Type of strategy (tournament, roulette, etc.)
            parameters: Strategy-specific parameters
            
        Returns:
            ID of the newly created strategy
        """
        strategy = SelectionStrategy(
            name=name,
            strategy_type=strategy_type,
            parameters=parameters or {}
        )
        
        self.selection_strategies[strategy.id] = strategy
        self.last_modified = datetime.now().isoformat()
        
        # Log operation
        self._log_operation("add_selection_strategy", {"strategy_id": strategy.id})
        
        return strategy.id
    
    def add_variation_operator(self,
                              name: str,
                              operator_type: str,
                              parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new variation operator to the system.
        
        Args:
            name: Name of the operator
            operator_type: Type of operator (mutation, recombination, etc.)
            parameters: Operator-specific parameters
            
        Returns:
            ID of the newly created operator
        """
        operator = VariationOperator(
            name=name,
            operator_type=operator_type,
            parameters=parameters or {}
        )
        
        self.variation_operators[operator.id] = operator
        self.last_modified = datetime.now().isoformat()
        
        # Log operation
        self._log_operation("add_variation_operator", {"operator_id": operator.id})
        
        return operator.id
    
    def add_population_member(self,
                            entity_id: str,
                            entity_type: str,
                            generation: int = 0,
                            parent_ids: Optional[List[str]] = None) -> str:
        """
        Add a member to the population.
        
        Args:
            entity_id: ID of the narrative entity (fragment, sequence, etc.)
            entity_type: Type of entity
            generation: Generation number (0 for initial population)
            parent_ids: IDs of parent entities (for derived elements)
            
        Returns:
            ID of the added member (same as entity_id)
        """
        if entity_id in self.population:
            # Already exists, don't overwrite
            return entity_id
        
        member = PopulationMember(
            entity_id=entity_id,
            entity_type=entity_type,
            generation=generation,
            parent_ids=parent_ids or []
        )
        
        self.population[entity_id] = member
        self.last_modified = datetime.now().isoformat()
        
        # Log operation
        self._log_operation("add_population_member", {"member_id": entity_id})
        
        return entity_id
    
    def evaluate_population_member(self, 
                                 member_id: str,
                                 recalculate: bool = False) -> Optional[float]:
        """
        Evaluate a population member using all fitness functions.
        
        Args:
            member_id: ID of the population member
            recalculate: Whether to force recalculation of fitness
            
        Returns:
            Overall fitness score, or None if member not found
        """
        member = self.population.get(member_id)
        if not member:
            return None
        
        # If already evaluated and not forced recalculation, return existing fitness
        if member.overall_fitness is not None and not recalculate:
            return member.overall_fitness
        
        # Check if we have access to evaluation tools
        if self.autocritical is None and not self.fitness_functions:
            # No evaluation tools available
            member.overall_fitness = 0.5
            return member.overall_fitness
        
        # Use autocritical function if available
        if self.autocritical:
            if member.entity_type == "fragment":
                result = self.autocritical.evaluate_fragment(member.id)
                member.add_fitness_score("autocritical", result.overall_score)
                
                # Add dimension scores as separate fitness components
                for dimension, score in result.dimension_scores.items():
                    member.add_fitness_score(f"autocritical_{dimension}", score)
            
            elif member.entity_type == "sequence":
                result = self.autocritical.evaluate_sequence(member.id)
                member.add_fitness_score("autocritical", result.overall_score)
                
                # Add dimension scores as separate fitness components
                for dimension, score in result.dimension_scores.items():
                    member.add_fitness_score(f"autocritical_{dimension}", score)
        
        # Apply custom fitness functions
        for function_id, function in self.fitness_functions.items():
            # Skip if already evaluated by this function and not recalculating
            if function_id in member.fitness_scores and not recalculate:
                continue
            
            # Apply function-specific evaluation
            score = self._apply_fitness_function(function, member)
            member.add_fitness_score(function_id, score)
            
            # Record evaluation in function's history
            function.record_evaluation(member.id, score)
        
        # Calculate overall fitness
        fitness = member.calculate_overall_fitness(self.fitness_functions)
        
        # Log operation
        self._log_operation("evaluate_population_member", {
            "member_id": member_id,
            "fitness": fitness
        })
        
        return fitness
    
    def _apply_fitness_function(self, 
                              function: FitnessFunction, 
                              member: PopulationMember) -> float:
        """
        Apply a specific fitness function to evaluate a member.
        
        Args:
            function: The fitness function to apply
            member: The population member to evaluate
            
        Returns:
            Fitness score for this function (0.0 to 1.0)
        """
        entity_type = member.entity_type
        entity_id = member.id
        
        # Get entity from myth constructor if available
        entity = None
        if self.myth_constructor:
            if entity_type == "fragment":
                entity = self.myth_constructor.get_fragment(entity_id)
            elif entity_type == "sequence":
                entity = self.myth_constructor.get_sequence(entity_id)
        
        if entity is None:
            # Entity not found, use default score
            return 0.5
        
        # Check function type and apply appropriate evaluation
        if function.name == "Narrative Quality":
            return self._evaluate_quality(function, entity, entity_type)
        
        elif function.name == "Narrative Resonance":
            return self._evaluate_resonance(function, entity, entity_type)
        
        elif function.name == "Narrative Novelty":
            return self._evaluate_novelty(function, entity, entity_type, member)
        
        elif function.name == "Narrative Coherence":
            return self._evaluate_coherence(function, entity, entity_type)
        
        elif function.name == "Transformative Potential":
            return self._evaluate_transformation(function, entity, entity_type, member)
        
        # Default score for unknown functions
        return 0.5
    
    def _evaluate_quality(self, 
                        function: FitnessFunction, 
                        entity: Union[NarrativeFragment, NarrativeSequence],
                        entity_type: str) -> float:
        """
        Evaluate narrative quality of the entity.
        
        Args:
            function: The quality function to apply
            entity: The entity to evaluate
            entity_type: Type of entity
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Get parameters
        critical_threshold = function.parameters.get("critical_threshold", 0.6)
        excellence_threshold = function.parameters.get("excellence_threshold", 0.8)
        
        # Default quality factors
        coherence_score = 0.5
        impact_score = 0.5
        creativity_score = 0.5
        craft_score = 0.5
        
        # If entity is a fragment
        if entity_type == "fragment":
            # Evaluate content length and complexity
            content = getattr(entity, "content", "")
            if not content:
                return 0.3  # Empty content is low quality
                
            # Basic metrics (normalized to 0-1)
            length = min(1.0, len(content) / 1000)
            variety = min(1.0, len(set(content.split())) / 100)
            
            # Heuristic scoring (to be replaced with more sophisticated evaluation)
            coherence_score = 0.5 + (variety * 0.2)
            impact_score = 0.4 + (length * 0.3)
            creativity_score = 0.5 + (variety * 0.3)
            craft_score = 0.5 + (length * 0.1) + (variety * 0.2)
        
        # If entity is a sequence
        elif entity_type == "sequence":
            # Evaluate sequence length and complexity
            fragments = getattr(entity, "fragments", [])
            if not fragments:
                return 0.3  # Empty sequence is low quality
                
            # Basic metrics (normalized to 0-1)
            length = min(1.0, len(fragments) / 10)
            
            # Heuristic scoring (to be replaced with more sophisticated evaluation)
            coherence_score = 0.5 + (length * 0.1)
            impact_score = 0.4 + (length * 0.2)
            creativity_score = 0.5
            craft_score = 0.5 + (length * 0.1)
        
        # Calculate weighted quality score
        quality_score = (coherence_score * 0.3) + (impact_score * 0.3) + \
                      (creativity_score * 0.2) + (craft_score * 0.2)
        
        # Apply thresholds
        if quality_score < critical_threshold:
            # Below critical threshold, reduce score
            quality_score *= 0.8
        elif quality_score > excellence_threshold:
            # Above excellence threshold, boost score
            quality_score = min(1.0, quality_score * 1.2)
        
        return quality_score
    
    def _evaluate_resonance(self, 
                          function: FitnessFunction, 
                          entity: Union[NarrativeFragment, NarrativeSequence],
                          entity_type: str) -> float:
        """
        Evaluate narrative resonance with the broader mythos.
        
        Args:
            function: The resonance function to apply
            entity: The entity to evaluate
            entity_type: Type of entity
            
        Returns:
            Resonance score (0.0 to 1.0)
        """
        # Get parameters
        connection_threshold = function.parameters.get("connection_threshold", 3)
        pattern_match_threshold = function.parameters.get("pattern_match_threshold", 0.6)
        
        # If analytics not available, use simplified evaluation
        if not self.analytics:
            return 0.5
        
        # Get analytics overview to assess resonance
        analytics = self.analytics.get_analytics_overview()
        
        # Extract key patterns and themes
        patterns = analytics.get("key_patterns", [])
        themes = analytics.get("key_themes", [])
        
        # Initialize resonance components
        thematic_match = 0.5
        pattern_alignment = 0.5
        connection_strength = 0.5
        
        # If entity is a fragment
        if entity_type == "fragment":
            content = getattr(entity, "content", "")
            context_weights = getattr(entity, "context_weights", {})
            
            # Calculate thematic match
            if themes and content:
                theme_matches = sum(1 for theme in themes if theme.lower() in content.lower())
                thematic_match = min(1.0, theme_matches / max(3, len(themes) * 0.3))
            
            # Calculate connection strength
            if context_weights:
                connection_count = len(context_weights)
                connection_avg = sum(context_weights.values()) / max(1, connection_count)
                
                # Normalize to 0-1 range
                connection_strength = min(1.0, connection_count / connection_threshold) * 0.5 + \
                                     min(1.0, connection_avg) * 0.5
        
        # If entity is a sequence
        elif entity_type == "sequence":
            title = getattr(entity, "title", "")
            fragments = getattr(entity, "fragments", [])
            
            # Calculate thematic match
            if themes and title:
                theme_matches = sum(1 for theme in themes if theme.lower() in title.lower())
                thematic_match = min(1.0, theme_matches / max(2, len(themes) * 0.2))
            
            # Calculate connection strength
            connection_strength = min(1.0, len(fragments) / connection_threshold)
        
        # Calculate pattern alignment (simplified for now)
        pattern_alignment = 0.5 + random.uniform(-0.2, 0.2)  # Randomized component for demonstration
        
        # Apply threshold checks
        if pattern_alignment < pattern_match_threshold:
            # Below pattern match threshold, reduce score
            pattern_alignment *= 0.7
        
        # Calculate overall resonance score
        resonance_score = (thematic_match * 0.4) + (pattern_alignment * 0.3) + \
                         (connection_strength * 0.3)
        
        return resonance_score
    
    def _evaluate_novelty(self, 
                        function: FitnessFunction, 
                        entity: Union[NarrativeFragment, NarrativeSequence],
                        entity_type: str,
                        member: PopulationMember) -> float:
        """
        Evaluate narrative novelty compared to existing elements.
        
        Args:
            function: The novelty function to apply
            entity: The entity to evaluate
            entity_type: Type of entity
            member: The population member being evaluated
            
        Returns:
            Novelty score (0.0 to 1.0)
        """
        # Get parameters
        similarity_threshold = function.parameters.get("similarity_threshold", 0.7)
        innovation_bonus = function.parameters.get("innovation_bonus", 0.2)
        
        # Initialize novelty components
        uniqueness = 0.6  # Default moderate uniqueness
        innovation = 0.5  # Default neutral innovation
        
        # Compare against other population members
        if entity_type == "fragment":
            content = getattr(entity, "content", "")
            
            # Count similar fragments in population
            similar_count = 0
            total_compared = 0
            
            for other_id, other_member in self.population.items():
                # Skip self comparison
                if other_id == member.id:
                    continue
                    
                # Only compare with other fragments
                if other_member.entity_type != "fragment":
                    continue
                
                # Get other entity content
                other_entity = None
                if self.myth_constructor:
                    other_entity = self.myth_constructor.get_fragment(other_id)
                
                if other_entity:
                    other_content = getattr(other_entity, "content", "")
                    
                    # Calculate similarity (very simplified)
                    if content and other_content:
                        # Simple text overlap measure
                        content_words = set(content.lower().split())
                        other_words = set(other_content.lower().split())
                        
                        if content_words and other_words:
                            overlap = len(content_words.intersection(other_words))
                            similarity = overlap / min(len(content_words), len(other_words))
                            
                            if similarity > similarity_threshold:
                                similar_count += 1
                            
                            total_compared += 1
            
            # Calculate uniqueness from similarity comparison
            if total_compared > 0:
                uniqueness = 1.0 - (similar_count / total_compared)
            
            # Innovation based on content complexity
            if content:
                # Simplified innovation assessment based on content variety
                unique_words = len(set(content.lower().split()))
                total_words = max(1, len(content.split()))
                lexical_diversity = unique_words / total_words
                
                innovation = min(1.0, lexical_diversity * 1.5)
        
        # For sequences, evaluate structure uniqueness
        elif entity_type == "sequence":
            fragments = getattr(entity, "fragments", [])
            
            # Compare with other sequences
            similar_count = 0
            total_compared = 0
            
            for other_id, other_member in self.population.items():
                # Skip self comparison
                if other_id == member.id:
                    continue
                    
                # Only compare with other sequences
                if other_member.entity_type != "sequence":
                    continue
                
                # Get other entity
                other_entity = None
                if self.myth_constructor:
                    other_entity = self.myth_constructor.get_sequence(other_id)
                
                if other_entity:
                    other_fragments = getattr(other_entity, "fragments", [])
                    
                    # Compare sequence lengths
                    if fragments and other_fragments:
                        len_diff = abs(len(fragments) - len(other_fragments))
                        len_similarity = 1.0 - min(1.0, len_diff / max(1, len(fragments)))
                        
                        if len_similarity > similarity_threshold:
                            similar_count += 1
                        
                        total_compared += 1
            
            # Calculate uniqueness from similarity comparison
            if total_compared > 0:
                uniqueness = 1.0 - (similar_count / total_compared)
            
            # Innovation based on structure complexity
            if fragments:
                innovation = min(1.0, len(fragments) / 10)
        
        # Apply innovation bonus if highly innovative
        if innovation > 0.7:
            innovation += innovation_bonus
            innovation = min(1.0, innovation)
        
        # Calculate overall novelty score
        novelty_score = (uniqueness * 0.6) + (innovation * 0.4)
        
        return novelty_score
    
    def _evaluate_coherence(self, 
                          function: FitnessFunction, 
                          entity: Union[NarrativeFragment, NarrativeSequence],
                          entity_type: str) -> float:
        """
        Evaluate narrative coherence and consistency.
        
        Args:
            function: The coherence function to apply
            entity: The entity to evaluate
            entity_type: Type of entity
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        # Get parameters
        contradiction_penalty = function.parameters.get("contradiction_penalty", 0.3)
        integration_bonus = function.parameters.get("integration_bonus", 0.2)
        
        # Initialize coherence components
        logical_consistency = 0.6  # Default moderate consistency
        thematic_consistency = 0.6  # Default moderate consistency
        structural_coherence = 0.5  # Default neutral coherence
        
        # If entity is a fragment
        if entity_type == "fragment":
            content = getattr(entity, "content", "")
            fragment_type = getattr(entity, "fragment_type", "")
            
            # Structural coherence based on content structure
            if content:
                # Simplified coherence based on text structure
                sentences = re.split(r'[.!?]', content)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if sentences:
                    # Average sentence length
                    avg_length = sum(len(s) for s in sentences) / len(sentences)
                    # Normalized to 0-1 range (assuming ideal avg length between 50-100)
                    length_coherence = max(0, min(1.0, avg_length / 75))
                    
                    # Variance in sentence length
                    if len(sentences) > 1:
                        length_variance = sum((len(s) - avg_length) ** 2 for s in sentences) / len(sentences)
                        variance_coherence = max(0, min(1.0, 1.0 - (length_variance / 1000)))
                        
                        structural_coherence = (length_coherence * 0.5) + (variance_coherence * 0.5)
            
            # Logical consistency (simplified for now)
            logical_consistency = 0.6 + random.uniform(-0.1, 0.1)
            
            # Thematic consistency based on fragment type
            if fragment_type:
                # Different types might have different coherence expectations
                if fragment_type in ("exposition", "explanation"):
                    thematic_consistency = 0.7
                elif fragment_type in ("action", "dialogue"):
                    thematic_consistency = 0.6
                elif fragment_type in ("description", "setting"):
                    thematic_consistency = 0.8
        
        # If entity is a sequence
        elif entity_type == "sequence":
            fragments = getattr(entity, "fragments", [])
            
            # Evaluate structural coherence based on fragment order
            if fragments and len(fragments) > 1:
                # Simplified coherence based on sequence structure
                structural_coherence = 0.5 + (min(1.0, len(fragments) / 10) * 0.3)
            
            # For sequences, logical consistency depends on fragment transitions
            logical_consistency = 0.6 + min(0.3, len(fragments) * 0.05)
            
            # Thematic consistency is higher for longer sequences
            thematic_consistency = 0.6 + min(0.3, len(fragments) * 0.03)
        
        # Apply contradiction penalty if logical consistency is low
        if logical_consistency < 0.4:
            logical_consistency -= contradiction_penalty
            logical_consistency = max(0.1, logical_consistency)
        
        # Apply integration bonus if thematic consistency is high
        if thematic_consistency > 0.7:
            thematic_consistency += integration_bonus
            thematic_consistency = min(1.0, thematic_consistency)
        
        # Calculate overall coherence score
        coherence_score = (logical_consistency * 0.4) + (thematic_consistency * 0.4) + \
                          (structural_coherence * 0.2)
        
        return coherence_score
    
    def _evaluate_transformation(self, 
                               function: FitnessFunction, 
                               entity: Union[NarrativeFragment, NarrativeSequence],
                               entity_type: str,
                               member: PopulationMember) -> float:
        """
        Evaluate transformative potential of the narrative element.
        
        Args:
            function: The transformation function to apply
            entity: The entity to evaluate
            entity_type: Type of entity
            member: The population member being evaluated
            
        Returns:
            Transformation score (0.0 to 1.0)
        """
        # Get parameters
        stagnation_threshold = function.parameters.get("stagnation_threshold", 0.4)
        emergence_bonus = function.parameters.get("emergence_bonus", 0.3)
        
        # Initialize transformation components
        change_potential = 0.5  # Default neutral potential
        emergence_potential = 0.5  # Default neutral potential
        
        # Use generation to factor in evolutionary development
        generation = member.generation
        if generation > 0:
            # Later generations have higher potential for transformation
            generation_factor = min(0.3, generation * 0.05)
            change_potential += generation_factor
        
        # If entity is a fragment
        if entity_type == "fragment":
            content = getattr(entity, "content", "")
            
            # Evaluate content for transformative language
            if content:
                # Look for keywords indicating transformation
                transform_keywords = ["change", "transform", "evolve", "shift", "become", 
                                    "transition", "metamorphosis", "revolution", "rebirth"]
                                    
                # Count occurrences of transformative keywords
                keyword_count = sum(content.lower().count(keyword) for keyword in transform_keywords)
                
                # Normalize to 0-1 range with cap
                keyword_factor = min(0.3, keyword_count * 0.05)
                change_potential += keyword_factor
                
                # Analyze content complexity for emergence potential
                words = content.split()
                if words:
                    # Lexical diversity as a proxy for complexity
                    unique_ratio = len(set(words)) / len(words)
                    emergence_potential = 0.4 + (unique_ratio * 0.4)
        
        # If entity is a sequence
        elif entity_type == "sequence":
            fragments = getattr(entity, "fragments", [])
            
            # Sequences with more fragments have higher transformation potential
            if fragments:
                size_factor = min(0.3, len(fragments) * 0.03)
                change_potential += size_factor
                
                # Sequences with diverse fragment types have higher emergence potential
                if self.myth_constructor:
                    fragment_types = set()
                    for frag_id in fragments:
                        fragment = self.myth_constructor.get_fragment(frag_id)
                        if fragment:
                            fragment_type = getattr(fragment, "fragment_type", "")
                            if fragment_type:
                                fragment_types.add(fragment_type)
                    
                    # Diversity factor
                    type_diversity = min(0.4, len(fragment_types) * 0.1)
                    emergence_potential += type_diversity
        
        # Apply stagnation threshold check
        if change_potential < stagnation_threshold:
            # Below stagnation threshold, reduce score
            change_potential *= 0.7
        
        # Apply emergence bonus check
        if emergence_potential > 0.7:
            # Above emergence threshold, boost score
            emergence_potential += emergence_bonus
            emergence_potential = min(1.0, emergence_potential)
        
        # Calculate overall transformation score
        transformation_score = (change_potential * 0.6) + (emergence_potential * 0.4)
        
        return transformation_score
    
    def select_population_members(self,
                                strategy_id: Optional[str] = None,
                                count: Optional[int] = None,
                                criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Select members from the population using a specified strategy.
        
        Args:
            strategy_id: ID of selection strategy to use (or random if None)
            count: Number of members to select (or based on selection_rate if None)
            criteria: Additional selection criteria
            
        Returns:
            List of selected member IDs
        """
        # Evaluate all members first
        active_members = {mid: member for mid, member in self.population.items() 
                         if member.is_active}
        
        if not active_members:
            return []
        
        # Determine selection count
        if count is None:
            selection_count = max(1, int(len(active_members) * self.parameters["selection_rate"]))
        else:
            selection_count = min(count, len(active_members))
        
        # Choose strategy
        if strategy_id is None or strategy_id not in self.selection_strategies:
            # Random strategy choice weighted by performance
            strategies = list(self.selection_strategies.items())
            if not strategies:
                return []
                
            # Weight by performance metrics (higher diversity is better)
            weights = [s.performance_metrics["selection_diversity"] for _, s in strategies]
            total_weight = sum(weights)
            
            if total_weight > 0:
                probs = [w / total_weight for w in weights]
                strategy_id = random.choices([sid for sid, _ in strategies], probs)[0]
            else:
                strategy_id = random.choice([sid for sid, _ in strategies])
        
        strategy = self.selection_strategies[strategy_id]
        
        # Apply selection based on strategy type
        selected_ids = []
        
        if strategy.strategy_type == "tournament":
            selected_ids = self._tournament_selection(
                active_members, 
                selection_count, 
                strategy.parameters.get("tournament_size", 3),
                strategy.parameters.get("tournament_count", 10),
                strategy.parameters.get("selection_probability", 0.7)
            )
        
        elif strategy.strategy_type == "roulette":
            selected_ids = self._roulette_selection(
                active_members, 
                selection_count, 
                strategy.parameters.get("fitness_scaling", 1.5),
                strategy.parameters.get("minimum_selection_chance", 0.05)
            )
        
        elif strategy.strategy_type == "elitist":
            selected_ids = self._elitist_selection(
                active_members, 
                selection_count, 
                strategy.parameters.get("elite_percentage", 0.2),
                strategy.parameters.get("minimum_fitness", 0.6)
            )
            
        elif strategy.strategy_type == "novelty":
            selected_ids = self._novelty_selection(
                active_members, 
                selection_count, 
                strategy.parameters.get("novelty_threshold", 0.6),
                strategy.parameters.get("novelty_weight", 1.2),
                strategy.parameters.get("history_window", 5)
            )
            
        elif strategy.strategy_type == "age_based":
            selected_ids = self._age_based_selection(
                active_members, 
                selection_count, 
                strategy.parameters.get("age_threshold", 3),
                strategy.parameters.get("youth_bonus", 0.2),
                strategy.parameters.get("elder_penalty", 0.1)
            )
            
        else:
            # Default to random selection
            selected_ids = random.sample(list(active_members.keys()), selection_count)
        
        # Record selection in members
        for member_id in selected_ids:
            member = self.population.get(member_id)
            if member:
                member.record_selection()
        
        # Record selection in strategy
        avg_fitness = 0.0
        if selected_ids:
            avg_fitness = sum(self.population[mid].overall_fitness or 0.5 
                            for mid in selected_ids) / len(selected_ids)
        
        strategy.record_selection(selected_ids, len(active_members), avg_fitness)
        
        # Log operation
        self._log_operation("select_population_members", {
            "strategy_id": strategy_id,
            "count": len(selected_ids)
        })
        
        return selected_ids
    
    def _tournament_selection(self,
                           members: Dict[str, PopulationMember],
                           selection_count: int,
                           tournament_size: int,
                           tournament_count: int,
                           selection_probability: float) -> List[str]:
        """
        Perform tournament selection.
        
        Args:
            members: Dict of active population members
            selection_count: Number of members to select
            tournament_size: Size of each tournament
            tournament_count: Number of tournaments to run
            selection_probability: Probability of selecting best in tournament
            
        Returns:
            List of selected member IDs
        """
        if not members:
            return []
        
        member_ids = list(members.keys())
        selected_ids = []
        
        # Run tournaments until we have enough selections
        for _ in range(tournament_count):
            if len(selected_ids) >= selection_count:
                break
                
            # Select tournament participants
            if len(member_ids) <= tournament_size:
                tournament = member_ids
            else:
                tournament = random.sample(member_ids, tournament_size)
            
            # Sort by fitness
            tournament.sort(key=lambda mid: members[mid].overall_fitness or 0.0, reverse=True)
            
            # Select winner based on selection probability
            for i, candidate_id in enumerate(tournament):
                # First candidate (highest fitness) has selection_probability chance
                # Each subsequent candidate has progressively lower chance
                probability = selection_probability * (1 - selection_probability) ** i
                
                if random.random() < probability:
                    if candidate_id not in selected_ids:
                        selected_ids.append(candidate_id)
                    break
        
        # If we still need more, add highest fitness members not yet selected
        if len(selected_ids) < selection_count:
            remaining = [mid for mid in member_ids if mid not in selected_ids]
            remaining.sort(key=lambda mid: members[mid].overall_fitness or 0.0, reverse=True)
            
            selected_ids.extend(remaining[:selection_count - len(selected_ids)])
        
        return selected_ids[:selection_count]
    
    def _roulette_selection(self,
                         members: Dict[str, PopulationMember],
                         selection_count: int,
                         fitness_scaling: float,
                         minimum_selection_chance: float) -> List[str]:
        """
        Perform fitness-proportionate (roulette wheel) selection.
        
        Args:
            members: Dict of active population members
            selection_count: Number of members to select
            fitness_scaling: Exponent for scaling fitness differences
            minimum_selection_chance: Minimum chance even for lowest fitness
            
        Returns:
            List of selected member IDs
        """
        if not members:
            return []
        
        member_ids = list(members.keys())
        
        # Calculate selection probabilities
        fitnesses = [members[mid].overall_fitness or 0.5 for mid in member_ids]
        
        # Apply fitness scaling
        scaled_fitnesses = [f ** fitness_scaling for f in fitnesses]
        
        # Apply minimum selection chance
        min_value = sum(scaled_fitnesses) * minimum_selection_chance
        adjusted_fitnesses = [max(f, min_value) for f in scaled_fitnesses]
        
        # Calculate probabilities
        total_fitness = sum(adjusted_fitnesses)
        if total_fitness <= 0:
            # Equal probability if all fitnesses are zero
            probabilities = [1.0 / len(members) for _ in member_ids]
        else:
            probabilities = [f / total_fitness for f in adjusted_fitnesses]
        
        # Select members using weighted random sampling
        selected_ids = []
        for _ in range(selection_count):
            if not member_ids:
                break
                
            # Select an ID based on probabilities
            selected_id = random.choices(member_ids, probabilities)[0]
            selected_ids.append(selected_id)
            
            # Remove selected ID to avoid duplicates
            index = member_ids.index(selected_id)
            member_ids.pop(index)
            probabilities.pop(index)
            
            # Renormalize probabilities
            if probabilities:
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
        
        return selected_ids
    
    def _elitist_selection(self,
                        members: Dict[str, PopulationMember],
                        selection_count: int,
                        elite_percentage: float,
                        minimum_fitness: float) -> List[str]:
        """
        Perform elitist selection taking the best members.
        
        Args:
            members: Dict of active population members
            selection_count: Number of members to select
            elite_percentage: Percentage of top members to consider elite
            minimum_fitness: Minimum fitness threshold for selection
            
        Returns:
            List of selected member IDs
        """
        if not members:
            return []
        
        # Get members sorted by fitness
        sorted_members = sorted(
            members.items(),
            key=lambda item: item[1].overall_fitness or 0.0,
            reverse=True
        )
        
        # Calculate elite count
        elite_count = max(1, int(len(members) * elite_percentage))
        
        # Get elite candidates
        elite_candidates = []
        for member_id, member in sorted_members[:elite_count]:
            if member.overall_fitness and member.overall_fitness >= minimum_fitness:
                elite_candidates.append(member_id)
        
        # If we have enough elites meeting the threshold
        if len(elite_candidates) >= selection_count:
            return elite_candidates[:selection_count]
        
        # Otherwise add remaining members in fitness order
        remaining = [member_id for member_id, _ in sorted_members 
                   if member_id not in elite_candidates]
        
        return (elite_candidates + remaining)[:selection_count]
    
    def _novelty_selection(self,
                        members: Dict[str, PopulationMember],
                        selection_count: int,
                        novelty_threshold: float,
                        novelty_weight: float,
                        history_window: int) -> List[str]:
        """
        Perform selection based on novelty and recent usage.
        
        Args:
            members: Dict of active population members
            selection_count: Number of members to select
            novelty_threshold: Threshold for novelty score
            novelty_weight: Weight for novelty vs. fitness
            history_window: Number of generations to consider for novelty
            
        Returns:
            List of selected member IDs
        """
        if not members:
            return []
        
        # Calculate novelty scores based on recency and selection history
        novelty_scores = {}
        
        for member_id, member in members.items():
            # Base score is overall fitness
            base_score = member.overall_fitness or 0.5
            
            # Calculate recency factor - newer members are more novel
            generation_diff = max(0, self.current_generation - member.generation)
            if generation_diff < history_window:
                recency_bonus = (history_window - generation_diff) / history_window * 0.3
            else:
                recency_bonus = 0.0
            
            # Calculate selection history factor - less selected members are more novel
            if member.selection_count > 0:
                selection_penalty = min(0.3, member.selection_count * 0.05)
            else:
                selection_penalty = 0.0
            
            # Get novelty fitness component if available
            novelty_component = 0.0
            if "Narrative Novelty" in member.fitness_scores:
                novelty_component = member.fitness_scores["Narrative Novelty"] * 0.4
            
            # Calculate final novelty score
            novelty_score = base_score + recency_bonus - selection_penalty + novelty_component
            
            # Apply novelty threshold
            if novelty_score < novelty_threshold:
                novelty_score *= 0.7
            
            novelty_scores[member_id] = novelty_score
        
        # Blend novelty with fitness score
        selection_scores = {}
        for member_id, member in members.items():
            fitness = member.overall_fitness or 0.5
            novelty = novelty_scores.get(member_id, 0.5)
            
            # Weighted blend
            score = (fitness * (2.0 - novelty_weight) + novelty * novelty_weight) / 2.0
            selection_scores[member_id] = score
        
        # Sort by selection score
        sorted_members = sorted(
            selection_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
        
        # Return top members
        return [member_id for member_id, _ in sorted_members[:selection_count]]
    
    def _initialize_fitness_functions(self) -> None:
        """Initialize default fitness functions."""
        # Quality Function
        quality = FitnessFunction(
            name="Narrative Quality",
            description="Evaluates overall quality based on critical dimensions",
            weight=1.5
        )
        quality.add_parameter("critical_threshold", 0.6)
        quality.add_parameter("excellence_threshold", 0.8)
        self.fitness_functions[quality.id] = quality
        
        # Resonance Function
        resonance = FitnessFunction(
            name="Narrative Resonance",
            description="Evaluates how well narrative elements resonate with the broader mythos",
            weight=1.2
        )
        resonance.add_parameter("connection_threshold", 3)
        resonance.add_parameter("pattern_match_threshold", 0.6)
        self.fitness_functions[resonance.id] = resonance
        
        # Novelty Function
        novelty = FitnessFunction(
            name="Narrative Novelty",
            description="Evaluates originality and innovation in narrative elements",
            weight=1.0
        )
        novelty.add_parameter("similarity_threshold", 0.7)
        novelty.add_parameter("innovation_bonus", 0.2)
        self.fitness_functions[novelty.id] = novelty
        
        # Coherence Function
        coherence = FitnessFunction(
            name="Narrative Coherence",
            description="Evaluates logical and thematic consistency with the mythos",
            weight=1.3
        )
        coherence.add_parameter("contradiction_penalty", 0.3)
        coherence.add_parameter("integration_bonus", 0.2)
        self.fitness_functions[coherence.id] = coherence
        
        # Transformative Function
        transformation = FitnessFunction(
            name="Transformative Potential",
            description="Evaluates potential for generating meaningful narrative change",
            weight=1.1
        )
        transformation.add_parameter("stagnation_threshold", 0.4)
        transformation.add_parameter("emergence_bonus", 0.3)
        self.fitness_functions[transformation.id] = transformation
    
    def _initialize_selection_strategies(self) -> None:
        """Initialize default selection strategies."""
        # Tournament Selection
        tournament = SelectionStrategy(
            name="Tournament Selection",
            strategy_type="tournament",
            parameters={
                "tournament_size": 3,
                "tournament_count": 10,
                "selection_probability": 0.7
            }
        )
        self.selection_strategies[tournament.id] = tournament
        
        # Roulette Wheel Selection
        roulette = SelectionStrategy(
            name="Fitness-Proportionate Selection",
            strategy_type="roulette",
            parameters={
                "fitness_scaling": 1.5,
                "minimum_selection_chance": 0.05
            }
        )
        self.selection_strategies[roulette.id] = roulette
        
        # Elitist Selection
        elitist = SelectionStrategy(
            name="Elitist Selection",
            strategy_type="elitist",
            parameters={
                "elite_percentage": 0.2,
                "minimum_fitness": 0.6
            }
        )
        self.selection_strategies[elitist.id] = elitist
        
        # Novelty Search Selection
        novelty = SelectionStrategy(
            name="Novelty Search",
            strategy_type="novelty",
            parameters={
                "novelty_threshold": 0.6,
                "novelty_weight": 1.2,
                "history_window": 5
            }
        )
        self.selection_strategies[novelty.id] = novelty
        
        # Age-Based Selection
        age_based = SelectionStrategy(
            name="Age-Based Selection",
            strategy_type="age_based",
            parameters={
                "age_threshold": 3,  # generations
                "youth_bonus": 0.2,
                "elder_penalty": 0.1
            }
        )
        self.selection_strategies[age_based.id] = age_based

        # Shuffle some fragments
                    shuffle_count = max(1, int(len(fragments) * mutation_rate))
                    indices = random.sample(range(len(fragments)), shuffle_count)
                    values = [new_fragments[i] for i in indices]
                    random.shuffle(values)
                    for i, idx in enumerate(indices):
                        new_fragments[idx] = values[i]
            else:
                # Minor changes
                for i in range(len(new_fragments)):
                    if random.random() < mutation_rate:
                        # Replace fragment with a similar one if available
                        # In a real implementation, we would find truly similar fragments
                        # Using a placeholder approach here
                        if len(self.population) > 10:
                            candidates = [m for m in self.population.values() 
                                        if m.entity_type == "fragment" and m.id != new_fragments[i]]
                            if candidates:
                                new_fragments[i] = random.choice(candidates).id
        else:
            # Random mutation
            # Apply mutation rate to each position
            for i in range(len(new_fragments)):
                if random.random() < mutation_rate:
                    operation = random.choices(
                        ["replace", "insert", "delete", "duplicate"],
                        weights=[0.4, 0.3, 0.2, 0.1]
                    )[0]
                    
                    if operation == "replace" and len(self.population) > 5:
                        # Replace with another fragment
                        candidates = [m.id for m in self.population.values() 
                                    if m.entity_type == "fragment" and m.id != new_fragments[i]]
                        if candidates:
                            new_fragments[i] = random.choice(candidates)
                    
                    elif operation == "insert" and mutation_magnitude > 0.3:
                        # Insert a duplicate or existing fragment
                        if len(self.population) > 5 and random.random() < 0.5:
                            candidates = [m.id for m in self.population.values() 
                                        if m.entity_type == "fragment"]
                            if candidates:
                                insert_id = random.choice(candidates)
                                new_fragments.insert(i, insert_id)
                        else:
                            # Duplicate an adjacent fragment
                            idx = max(0, min(len(new_fragments) - 1, i + random.choice([-1, 1])))
                            new_fragments.insert(i, new_fragments[idx])
                    
                    elif operation == "delete" and len(new_fragments) > 2:
                        # Delete this fragment
                        new_fragments.pop(i)
                        # Adjust index after deletion
                        i -= 1
                    
                    elif operation == "duplicate" and mutation_magnitude > 0.5:
                        # Duplicate this fragment
                        new_fragments.insert(i + 1, new_fragments[i])
                        
            # Adjust length based on mutation magnitude
            target_length_change = int(len(fragments) * mutation_magnitude * random.choice([-1, 1]))
            current_diff = len(new_fragments) - len(fragments)
            
            if target_length_change > current_diff:
                # Need to add more fragments
                for _ in range(target_length_change - current_diff):
                    if new_fragments:
                        idx = random.randint(0, len(new_fragments) - 1)
                        new_fragments.insert(idx, random.choice(new_fragments))
            elif target_length_change < current_diff and len(new_fragments) > 2:
                # Need to remove fragments
                for _ in range(current_diff - target_length_change):
                    if len(new_fragments) > 2:
                        idx = random.randint(0, len(new_fragments) - 1)
                        new_fragments.pop(idx)
        
        # Ensure we have at least one fragment
        if not new_fragments and fragments:
            new_fragments = [random.choice(fragments)]
        
        # Create a new title with mutation
        if random.random() < mutation_rate:
            new_title = f"{title} (Evolved)"
        else:
            new_title = title
        
        # Simulate creation of a new sequence
        # In a real implementation, this would call the actual creation method
        new_sequence_id = str(uuid.uuid4())
        
        # Add to population
        self.add_population_member(
            entity_id=new_sequence_id,
            entity_type="sequence",
            generation=self.current_generation + 1,
            parent_ids=[sequence_id]
        )
        
        # Cache the simulated entity
        self._entity_cache[new_sequence_id] = {
            "title": new_title,
            "fragments": new_fragments
        }
        
        return new_sequence_id
    
    def _recombination_variation(self,
                               source_ids: List[str],
                               variation_count: int,
                               crossover_points: int,
                               parent_similarity_threshold: float,
                               element_preservation_rate: float) -> List[str]:
        """
        Create variations using recombination (crossover) operator.
        
        Args:
            source_ids: IDs of source members
            variation_count: Number of variations to create
            crossover_points: Number of crossover points
            parent_similarity_threshold: Minimum similarity for parent selection
            element_preservation_rate: Rate of preserving elements from parents
            
        Returns:
            List of newly created member IDs
        """
        if len(source_ids) < 2:
            return []
        
        # Get source members
        source_members = [self.population.get(sid) for sid in source_ids]
        source_members = [m for m in source_members if m and m.is_active]
        
        if len(source_members) < 2:
            return []
        
        # Create variations
        new_member_ids = []
        
        for _ in range(variation_count):
            # Select parents
            parents = self._select_parents(source_members, parent_similarity_threshold)
            
            if len(parents) < 2:
                continue
                
            parent1, parent2 = parents
            
            # Apply recombination based on entity type
            if parent1.entity_type == "fragment" and parent2.entity_type == "fragment":
                new_id = self._recombine_fragments(
                    parent1.id, 
                    parent2.id, 
                    crossover_points, 
                    element_preservation_rate
                )
                
                if new_id:
                    new_member_ids.append(new_id)
            
            elif parent1.entity_type == "sequence" and parent2.entity_type == "sequence":
                new_id = self._recombine_sequences(
                    parent1.id, 
                    parent2.id, 
                    crossover_points, 
                    element_preservation_rate
                )
                
                if new_id:
                    new_member_ids.append(new_id)
        
        return new_member_ids
    
    def _select_parents(self,
                      candidates: List[PopulationMember],
                      similarity_threshold: float) -> List[PopulationMember]:
        """
        Select appropriate parents for recombination.
        
        Args:
            candidates: List of candidate members
            similarity_threshold: Minimum similarity for selection
            
        Returns:
            List of two selected parent members
        """
        if len(candidates) < 2:
            return candidates
        
        # Select first parent based on fitness
        sorted_candidates = sorted(
            candidates,
            key=lambda m: m.overall_fitness or 0.0,
            reverse=True
        )
        
        # Bias towards fitter members but allow some randomness
        weights = [(m.overall_fitness or 0.5) ** 2 for m in sorted_candidates]
        parent1 = random.choices(sorted_candidates, weights=weights)[0]
        
        # Find compatible second parent
        compatible_candidates = []
        
        for candidate in candidates:
            if candidate.id == parent1.id:
                continue
                
            # Check compatibility
            if candidate.entity_type != parent1.entity_type:
                continue
                
            # Simplified similarity check
            # In a real implementation, would use more sophisticated metrics
            similarity = random.uniform(0.0, 1.0)  # Placeholder
            
            if similarity >= similarity_threshold:
                compatible_candidates.append((candidate, similarity))
        
        if not compatible_candidates:
            # If no compatible candidates, just pick randomly
            options = [c for c in candidates if c.id != parent1.id]
            if not options:
                return [parent1]
            return [parent1, random.choice(options)]
        
        # Sort by similarity
        compatible_candidates.sort(key=lambda pair: pair[1], reverse=True)
        
        # Select second parent, bias towards more similar
        total_similarity = sum(sim for _, sim in compatible_candidates)
        if total_similarity > 0:
            weights = [sim / total_similarity for _, sim in compatible_candidates]
            parent2 = random.choices(
                [candidate for candidate, _ in compatible_candidates],
                weights=weights
            )[0]
        else:
            parent2 = compatible_candidates[0][0]
        
        return [parent1, parent2]
    
    def _recombine_fragments(self,
                           fragment1_id: str,
                           fragment2_id: str,
                           crossover_points: int,
                           preservation_rate: float) -> Optional[str]:
        """
        Recombine two fragments to create a new one.
        
        Args:
            fragment1_id: ID of first parent fragment
            fragment2_id: ID of second parent fragment
            crossover_points: Number of crossover points
            preservation_rate: Rate of preserving elements from parents
            
        Returns:
            ID of the new fragment, or None if recombination failed
        """
        # Check if we have a myth constructor
        if not self.myth_constructor:
            return None
        
        # Get the parent fragments
        fragment1 = self.myth_constructor.get_fragment(fragment1_id)
        fragment2 = self.myth_constructor.get_fragment(fragment2_id)
        
        if not fragment1 or not fragment2:
            return None
        
        # Get fragment properties
        content1 = getattr(fragment1, "content", "")
        content2 = getattr(fragment2, "content", "")
        
        fragment_type1 = getattr(fragment1, "fragment_type", "")
        fragment_type2 = getattr(fragment2, "fragment_type", "")
        
        context_weights1 = getattr(fragment1, "context_weights", {})
        context_weights2 = getattr(fragment2, "context_weights", {})
        
        if not content1 or not content2:
            return None
        
        # Split contents into sentences
        sentences1 = re.split(r'[.!?]', content1)
        sentences1 = [s.strip() for s in sentences1 if s.strip()]
        
        sentences2 = re.split(r'[.!?]', content2)
        sentences2 = [s.strip() for s in sentences2 if s.strip()]
        
        if not sentences1 or not sentences2:
            return None
        
        # Determine crossover points
        max_points = min(len(sentences1), len(sentences2), 5)
        actual_points = min(crossover_points, max_points)
        
        if actual_points < 1:
            actual_points = 1
        
        # Perform crossover
        points1 = sorted(random.sample(range(1, len(sentences1)), actual_points))
        points2 = sorted(random.sample(range(1, len(sentences2)), actual_points))
        
        # Start with first parent
        new_sentences = []
        
        # Alternate segments from each parent
        start1, start2 = 0, 0
        
        for i in range(actual_points):
            end1 = points1[i]
            end2 = points2[i]
            
            # Add segment from parent 1
            segment1 = sentences1[start1:end1]
            if random.random() < preservation_rate:
                new_sentences.extend(segment1)
            
            # Add segment from parent 2
            segment2 = sentences2[start2:end2]
            if random.random() < preservation_rate:
                new_sentences.extend(segment2)
            
            start1, start2 = end1, end2
        
        # Add remaining segments
        if random.random() < 0.5:
            segment1 = sentences1[start1:]
            if random.random() < preservation_rate:
                new_sentences.extend(segment1)
        else:
            segment2 = sentences2[start2:]
            if random.random() < preservation_rate:
                new_sentences.extend(segment2)
        
        # Ensure we have content
        if not new_sentences:
            if random.random() < 0.5:
                new_sentences = sentences1[:1] + sentences2[-1:]
            else:
                new_sentences = sentences2[:1] + sentences1[-1:]
        
        # Create new content
        new_content = ". ".join(new_sentences) + "."
        
        # Merge context weights
        new_context_weights = {}
        all_keys = set(context_weights1.keys()) | set(context_weights2.keys())
        
        for key in all_keys:
            weight1 = context_weights1.get(key, 0.0)
            weight2 = context_weights2.get(key, 0.0)
            
            # Weighted average of parent weights
            if random.random() < 0.5:
                new_context_weights[key] = weight1
            else:
                new_context_weights[key] = weight2
        
        # Choose fragment type
        if random.random() < 0.5:
            new_fragment_type = fragment_type1
        else:
            new_fragment_type = fragment_type2
        
        # Simulate creation of a new fragment
        new_fragment_id = str(uuid.uuid4())
        
        # Add to population
        self.add_population_member(
            entity_id=new_fragment_id,
            entity_type="fragment",
            generation=self.current_generation + 1,
            parent_ids=[fragment1_id, fragment2_id]
        )
        
        # Cache the simulated entity
        self._entity_cache[new_fragment_id] = {
            "content": new_content,
            "fragment_type": new_fragment_type,
            "context_weights": new_context_weights
        }
        
        return new_fragment_id
    
    def _recombine_sequences(self,
                           sequence1_id: str,
                           sequence2_id: str,
                           crossover_points: int,
                           preservation_rate: float) -> Optional[str]:
        """
        Recombine two sequences to create a new one.
        
        Args:
            sequence1_id: ID of first parent sequence
            sequence2_id: ID of second parent sequence
            crossover_points: Number of crossover points
            preservation_rate: Rate of preserving elements from parents
            
        Returns:
            ID of the new sequence, or None if recombination failed
        """
        # Check if we have a myth constructor
        if not self.myth_constructor:
            return None
        
        # Get the parent sequences
        sequence1 = self.myth_constructor.get_sequence(sequence1_id)
        sequence2 = self.myth_constructor.get_sequence(sequence2_id)
        
        if not sequence1 or not sequence2:
            return None
        
        # Get sequence properties
        title1 = getattr(sequence1, "title", "")
        title2 = getattr(sequence2, "title", "")
        
        fragments1 = getattr(sequence1, "fragments", [])
        fragments2 = getattr(sequence2, "fragments", [])
        
        if not fragments1 or not fragments2:
            return None
        
        # Determine crossover points
        max_points = min(len(fragments1), len(fragments2), 5)
        actual_points = min(crossover_points, max_points)
        
        if actual_points < 1:
            actual_points = 1
        
        # Perform crossover
        points1 = sorted(random.sample(range(1, len(fragments1)), actual_points))
        points2 = sorted(random.sample(range(1, len(fragments2)), actual_points))
        
        # Start with first parent
        new_fragments = []
        
        # Alternate segments from each parent
        start1, start2 = 0, 0
        
        for i in range(actual_points):
            end1 = points1[i]
            end2 = points2[i]
            
            # Add segment from parent 1
            segment1 = fragments1[start1:end1]
            if random.random() < preservation_rate:
                new_fragments.extend(segment1)
            
            # Add segment from parent 2
            segment2 = fragments2[start2:end2]
            if random.random() < preservation_rate:
                new_fragments.extend(segment2)
            
            start1, start2 = end1, end2
        
        # Add remaining segments
        if random.random() < 0.5:
            segment1 = fragments1[start1:]
            if random.random() < preservation_rate:
                new_fragments.extend(segment1)
        else:
            segment2 = fragments2[start2:]
            if random.random() < preservation_rate:
                new_fragments.extend(segment2)
        
        # Ensure we have fragments
        if not new_fragments:
            if random.random() < 0.5:
                new_fragments = fragments1[:1] + fragments2[-1:]
            else:
                new_fragments = fragments2[:1] + fragments1[-1:]
        
        # Create new title
        if random.random() < 0.5:
            new_title = f"{title1} + {title2}"
        else:
            # Combine parts of both titles
            parts1 = title1.split()
            parts2 = title2.split()
            
            if len(parts1) > 1 and len(parts2) > 1:
                new_title = " ".join(parts1[:len(parts1)//2] + parts2[len(parts2)//2:])
            else:
                new_title = f"{title1}/{title2}"
        
        # Simulate creation of a new sequence
        new_sequence_id = str(uuid.uuid4())
        
        # Add to population
        self.add_population_member(
            entity_id=new_sequence_id,
            entity_type="sequence",
            generation=self.current_generation + 1,
            parent_ids=[sequence1_id, sequence2_id]
        )
        
        # Cache the simulated entity
        self._entity_cache[new_sequence_id] = {
            "title": new_title,
            "fragments": new_fragments
        }
        
        return new_sequence_id
    
    def _adaptive_variation(self,
                          source_ids: List[str],
                          variation_count: int,
                          learning_rate: float,
                          context_sensitivity: float,
                          adaptation_threshold: float) -> List[str]:
        """
        Create variations using adaptive learning from successful patterns.
        
        Args:
            source_ids: IDs of source members
            variation_count: Number of variations to create
            learning_rate: Rate of learning from successful patterns
            context_sensitivity: Sensitivity to narrative context
            adaptation_threshold: Threshold for adaptation
            
        Returns:
            List of newly created member IDs
        """
        if not source_ids:
            return []
        
        # Get source members
        source_members = [self.population.get(sid) for sid in source_ids]
        source_members = [m for m in source_members if m and m.is_active]
        
        if not source_members:
            return []
        
        # Analyze successful patterns
        # Sort population by fitness
        sorted_population = sorted(
            self.population.values(),
            key=lambda m: m.overall_fitness or 0.0,
            reverse=True
        )
        
        # Take top performers
        top_performers = sorted_population[:max(5, int(len(sorted_population) * 0.1))]
        
        # Learn from patterns in top performers
        adaptive_weights = {}
        
        # For fragments, analyze content patterns
        fragment_patterns = {}
        
        # For sequences, analyze structure patterns
        sequence_patterns = {}
        
        # Analyze patterns (simplified for demonstration)
        for member in top_performers:
            if member.overall_fitness and member.overall_fitness > adaptation_threshold:
                if member.entity_type == "fragment":
                    # Record fragment type
                    fragment = None
                    if self.myth_constructor:
                        fragment = self.myth_constructor.get_fragment(member.id)
                    
                    if fragment:
                        fragment_type = getattr(fragment, "fragment_type", "")
                        if fragment_type:
                            fragment_patterns[fragment_type] = fragment_patterns.get(fragment_type, 0) + 1
                
                elif member.entity_type == "sequence":
                    # Record sequence length
                    sequence = None
                    if self.myth_constructor:
                        sequence = self.myth_constructor.get_sequence(member.id)
                    
                    if sequence:
                        fragments = getattr(sequence, "fragments", [])
                        length = len(fragments)
                        length_bin = f"length_{length // 5 * 5}"  # Bin by 5s
                        sequence_patterns[length_bin] = sequence_patterns.get(length_bin, 0) + 1
        
        # Create variations
        new_member_ids = []
        
        for _ in range(variation_count):
            # Select a source member to adapt
            source = random.choice(source_members)
            
            # Apply adaptive variation based on entity type
            if source.entity_type == "fragment" and fragment_patterns:
                new_id = self._adaptive_fragment(
                    source.id, 
                    fragment_patterns,
                    learning_rate, 
                    context_sensitivity
                )
                
                if new_id:
                    new_member_ids.append(new_id)
            
            elif source.entity_type == "sequence" and sequence_patterns:
                new_id = self._adaptive_sequence(
                    source.id, 
                    sequence_patterns,
                    learning_rate, 
                    context_sensitivity
                )
                
                if new_id:
                    new_member_ids.append(new_id)
            else:
                # Fall back to mutation if no patterns available
                new_id = self._mutation_variation(
                    [source.id], 1, 0.3, 0.4, 0.2
                )
                
                if new_id:
                    new_member_ids.extend(new_id)
        
        return new_member_ids
    
    def _adaptive_fragment(self,
                         fragment_id: str,
                         patterns: Dict[str, int],
                         learning_rate: float,
                         context_sensitivity: float) -> Optional[str]:
        """
        Adapt a fragment based on successful patterns.
        
        Args:
            fragment_id: ID of the fragment to adapt
            patterns: Dictionary of successful patterns
            learning_rate: Rate of learning from patterns
            context_sensitivity: Sensitivity to narrative context
            
        Returns:
            ID of the new fragment, or None if adaptation failed
        """
        # This is a simplified implementation
        # Mostly delegate to mutation with pattern guidance
        return self._mutate_fragment(fragment_id, learning_rate, 0.4, context_sensitivity)
    
    def _adaptive_sequence(self,
                         sequence_id: str,
                         patterns: Dict[str, int],
                         learning_rate: float,
                         context_sensitivity: float) -> Optional[str]:
        """
        Adapt a sequence based on successful patterns.
        
        Args:
            sequence_id: ID of the sequence to adapt
            patterns: Dictionary of successful patterns
            learning_rate: Rate of learning from patterns
            context_sensitivity: Sensitivity to narrative context
            
        Returns:
            ID of the new sequence, or None if adaptation failed
        """
        # This is a simplified implementation
        # Mostly delegate to mutation with pattern guidance
        return self._mutate_sequence(sequence_id, learning_rate, 0.4, context_sensitivity)
    
    def _transformative_variation(self,
                                source_ids: List[str],
                                variation_count: int,
                                transformation_magnitude: float,
                                thematic_preservation_rate: float,
                                emergence_factor: float) -> List[str]:
        """
        Create variations using transformative operations.
        
        Args:
            source_ids: IDs of source members
            variation_count: Number of variations to create
            transformation_magnitude: Magnitude of transformation
            thematic_preservation_rate: Rate of preserving themes
            emergence_factor: Factor for emergent properties
            
        Returns:
            List of newly created member IDs
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated transformations
        
        # For now, delegate to mutation with high magnitude
        new_member_ids = []
        
        for _ in range(variation_count):
            if source_ids:
                source_id = random.choice(source_ids)
                
                # Apply high-magnitude mutation
                mutation_results = self._mutation_variation(
                    [source_id], 1, 0.7, transformation_magnitude, emergence_factor
                )
                
                if mutation_results:
                    new_member_ids.extend(mutation_results)
        
        return new_member_ids
    
    def evolve_population(self,
                        generations: int = 1,
                        selection_strategy_id: Optional[str] = None,
                        variation_operator_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evolve the population through multiple generations.
        
        Args:
            generations: Number of generations to evolve
            selection_strategy_id: ID of selection strategy to use
            variation_operator_id: ID of variation operator to use
            
        Returns:
            Dict with evolution results
        """
        # Initialize results
        results = {
            "generations": generations,
            "initial_population": len(self.population),
            "final_population": 0,
            "avg_fitness_start": 0.0,
            "avg_fitness_end": 0.0,
            "best_fitness_start": 0.0,
            "best_fitness_end": 0.0,
            "extinct_count": 0,
            "generation_stats": []
        }
        
        # Calculate initial fitness metrics
        if self.population:
            fitnesses = [m.overall_fitness or 0.0 for m in self.population.values() if m.is_active]
            if fitnesses:
                results["avg_fitness_start"] = sum(fitnesses) / len(fitnesses)
                results["best_fitness_start"] = max(fitnesses)
        
        # Run evolutionary algorithm for specified generations
        for i in range(generations):
            # Run one generation
            generation_results = self.evolve_generation(
                selection_strategy_id=selection_strategy_id,
                variation_operator_id=variation_operator_id
            )
            
            # Add to results
            results["generation_stats"].append(generation_results)
            
            # Update extinction count
            results["extinct_count"] += generation_results["extinct_count"]
            
            # Log progress
            self._log_operation("evolve_generation", {
                "generation": self.current_generation,
                "population_size": generation_results["population_size"],
                "avg_fitness": generation_results["avg_fitness"]
            })
        
        # Calculate final fitness metrics
        if self.population:
            fitnesses = [m.overall_fitness or 0.0 for m in self.population.values() if m.is_active]
            if fitnesses:
                results["avg_fitness_end"] = sum(fitnesses) / len(fitnesses)
                results["best_fitness_end"] = max(fitnesses)
        
        results["final_population"] = len([m for m in self.population.values() if m.is_active])
        
        # Update system metrics
        self._update_system_metrics()
        
        return results
    
    def evolve_generation(self,
                        selection_strategy_id: Optional[str] = None,
                        variation_operator_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evolve the population through one generation.
        
        Args:
            selection_strategy_id: ID of selection strategy to use
            variation_operator_id: ID of variation operator to use
            
        Returns:
            Dict with generation results
        """
        # Increment generation counter
        self.current_generation += 1
        
        # Initialize generation stats
        generation_stats = {
            "generation": self.current_generation,
            "initial_size": len([m for m in self.population.values() if m.is_active]),
            "selection_strategy": selection_strategy_id,
            "variation_operator": variation_operator_id,
            "selected_count": 0,
            "variations_created": 0,
            "extinct_count": 0,
            "population_size": 0,
            "avg_fitness": 0.0,
            "best_fitness": 0.0
        }
        
        # Step 1: Evaluate all population members
        for member_id, member in self.population.items():
            if member.is_active:
                self.evaluate_population_member(member_id)
        
        # Step 2: Selection
        selected_ids = self.select_population_members(
            strategy_id=selection_strategy_id
        )
        
        generation_stats["selected_count"] = len(selected_ids)
        
        # Step 3: Variation (create new population members)
        new_member_ids = self.create_variations(
            selected_ids=selected_ids,
            operator_id=variation_operator_id
        )
        
        generation_stats["variations_created"] = len(new_member_ids)
        
        # Step 4: Environmental pressure (extinction)
        extinct_count = self._apply_environmental_pressure()
        
        generation_stats["extinct_count"] = extinct_count
        
        # Calculate generation statistics
        active_members = [m for m in self.population.values() if m.is_active]
        generation_stats["population_size"] = len(active_members)
        
        if active_members:
            fitnesses = [m.overall_fitness or 0.0 for m in active_members]
            generation_stats["avg_fitness"] = sum(fitnesses) / len(fitnesses)
            generation_stats["best_fitness"] = max(fitnesses)
        
        # Record generation history
        self.generation_history.append(generation_stats)
        
        # Update system metrics
        self._update_system_metrics()
        
        # Log operation
        self._log_operation("evolve_generation", {
            "generation": self.current_generation,
            "selected_count": len(selected_ids),
            "new_members": len(new_member_ids),
            "extinct_count": extinct_count
        })
        
        return generation_stats
    
    def _apply_environmental_pressure(self) -> int:
        """
        Apply environmental pressure to eliminate low-fitness members.
        
        Returns:
            Number of members made extinct
        """
        # Get active population members
        active_members = [(mid, m) for mid, m in self.population.items() if m.is_active]
        
        if not active_members:
            return 0
        
        # Sort by fitness
        sorted_members = sorted(
            active_members,
            key=lambda pair: pair[1].overall_fitness or 0.0
        )
        
        # Apply extinction pressure
        extinct_count = 0
        
        # Get extinction threshold
        extinction_threshold = self.parameters["extinction_threshold"]
        extinction_count = max(1, int(len(active_members) * extinction_threshold))
        
        # Make lowest fitness members extinct
        for member_id, member in sorted_members[:extinction_count]:
            member.deactivate("low_fitness")
            extinct_count += 1
        
        # Age-based extinction (only if population is above target size)
        if len(active_members) > self.parameters["population_size_target"]:
            age_penalty_factor = self.parameters["age_penalty_factor"]
            
            # Calculate age-adjusted fitness
            age_adjusted = []
            for member_id, member in active_members:
                age = member.get_age()
                fitness = member.overall_fitness or 0.5
                
                # Apply age penalty
                adjusted_fitness = fitness - (age * age_penalty_factor)
                
                age_adjusted.append((member_id, member, adjusted_fitness))
            
            # Sort by age-adjusted fitness
            age_adjusted.sort(key=lambda triple: triple[2])
            
            # Additional extinctions to reach target population size
            excess = len(active_members) - extinct_count - self.parameters["population_size_target"]
            
            if excess > 0:
                for member_id, member, _ in age_adjusted[:excess]:
                    if member.is_active:  # Check again in case already extinct
                        member.deactivate("age_based_culling")
                        extinct_count += 1
        
        return extinct_count
    
    def _update_system_metrics(self) -> None:
        """Update system-wide evolutionary metrics."""
        # Calculate average population fitness
        active_members = [m for m in self.population.values() if m.is_active]
        
        if active_members:
            fitnesses = [m.overall_fitness or 0.5 for m in active_members]
            self.metrics["avg_population_fitness"] = sum(fitnesses) / len(fitnesses)
        
        # Calculate fitness improvement rate
        if len(self.generation_history) >= 2:
            current = self.generation_history[-1]["avg_fitness"]
            previous = self.generation_history[-2]["avg_fitness"]
            
            if previous > 0:
                improvement = (current - previous) / previous
                
                # Smooth with previous rate
                prev_rate = self.metrics["fitness_improvement_rate"]
                self.metrics["fitness_improvement_rate"] = (prev_rate * 0.7) + (improvement * 0.3)
        
        # Calculate biodiversity index (simplified)
        if active_members:
            entity_types = set(m.entity_type for m in active_members)
            generations = set(m.generation for m in active_members)
            
            # Diversity is ratio of unique attributes to population size
            type_diversity = len(entity_types) / max(1, len(set(m.entity_type for m in self.population.values())))
            gen_diversity = len(generations) / max(1, self.current_generation)
            
            self.metrics["biodiversity_index"] = (type_diversity * 0.3) + (gen_diversity * 0.7)
        
        # Calculate adaptation rate
        if len(self.generation_history) >= 3:
            # Look at fitness change acceleration
            g1 = self.generation_history[-3]["avg_fitness"]
            g2 = self.generation_history[-2]["avg_fitness"]
            g3 = self.generation_history[-1]["avg_fitness"]
            
            delta1 = g2 - g1
            delta2 = g3 - g2
            
            if abs(delta1) > 0.001:
                adaptation = (delta2 - delta1) / abs(delta1)
                
                # Smooth with previous rate
                prev_rate = self.metrics["adaptation_rate"]
                self.metrics["adaptation_rate"] = (prev_rate * 0.7) + (adaptation * 0.3)
    
    def _log_operation(self, operation_type: str, details: Dict[str, Any]) -> None:
        """
        Log an operation in the history.
        
        Args:
            operation_type: Type of operation
            details: Operation details
        """
        self.operation_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "details": details
        })
        
        # Limit history size
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]
        
        self.last_modified = datetime.now().isoformat()
    
    def reset_population(self) -> None:
        """Reset the population while keeping system configuration."""
        self.population = {}
        self.current_generation = 0
        self.generation_history = []
        self._entity_cache = {}
        self.last_modified = datetime.now().isoformat()
        
        # Reset operation history
        self.operation_history = []
        
        # Log operation
        self._log_operation("reset_population", {})
    
    def to_dict(self) -> Dict:
        """Convert system to a dictionary for serialization."""
        return {
            "name": self.name,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "current_generation": self.current_generation,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "fitness_functions": {fid: f.to_dict() for fid, f in self.fitness_functions.items()},
            "selection_strategies": {sid: s.to_dict() for sid, s in self.selection_strategies.items()},
            "variation_operators": {oid: o.to_dict() for oid, o in self.variation_operators.items()},
            "population": {mid: m.to_dict() for mid, m in self.population.items()},
            "generation_history": self.generation_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict, 
                myth_constructor: Optional[MythConstructor] = None,
                autocritical: Optional[AutocriticalFunction] = None,
                analytics: Optional[NarrativeAnalytics] = None) -> 'EvolutionaryPressure':
        """
        Create an EvolutionaryPressure system from a dictionary.
        
        Args:
            data: Dictionary representation
            myth_constructor: Optional link to the myth constructor
            autocritical: Optional link to the autocritical function
            analytics: Optional link to the narrative analytics
            
        Returns:
            New EvolutionaryPressure instance
        """
        system = cls(
            myth_constructor=myth_constructor,
            autocritical=autocritical,
            analytics=analytics,
            name=data.get("name", "Amelia's Evolutionary Pressure")
        )
        
        # Load basic properties
        system.creation_date = data.get("creation_date", system.creation_date)
        system.last_modified = data.get("last_modified", system.last_modified)
        system.current_generation = data.get("current_generation", 0)
        system.parameters = data.get("parameters", system.parameters)
        system.metrics = data.get("metrics", system.metrics)
        
        # Clear default components
        system.fitness_functions = {}
        system.selection_strategies = {}
        system.variation_operators = {}
        system.population = {}
        
        # Load fitness functions
        for fid, f_data in data.get("fitness_functions", {}).items():
            system.fitness_functions[fid] = FitnessFunction.from_dict(f_data)
        
        # Load selection strategies
        for sid, s_data in data.get("selection_strategies", {}).items():
            system.selection_strategies[sid] = SelectionStrategy.from_dict(s_data)
        
        # Load variation operators
        for oid, o_data in data.get("variation_operators", {}).items():
            system.variation_operators[oid] = VariationOperator.from_dict(o_data)
        
        # Load population
        for mid, m_data in data.get("population", {}).items():
            system.population[mid] = PopulationMember.from_dict(m_data)
        
        # Load generation history
        system.generation_history = data.get("generation_history", [])
        
        return system
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save system to a JSON file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self._log_operation("save_to_file", {"error": str(e)})
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str,
                      myth_constructor: Optional[MythConstructor] = None,
                      autocritical: Optional[AutocriticalFunction] = None,
                      analytics: Optional[NarrativeAnalytics] = None) -> Optional['EvolutionaryPressure']:
        """
        Load system from a JSON file.
        
        Args:
            filepath: Path to load file
            myth_constructor: Optional link to the myth constructor
            autocritical: Optional link to the autocritical function
            analytics: Optional link to the narrative analytics
            
        Returns:
            New EvolutionaryPressure instance or None if loading failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return cls.from_dict(
                data,
                myth_constructor=myth_constructor,
                autocritical=autocritical,
                analytics=analytics
            )
        except Exception:
            return None


# Utility function to test the module
def run_test_evolution(generations: int = 5) -> Dict:
    """
    Run a test evolution process.
    
    Args:
        generations: Number of generations to run
        
    Returns:
        Results dictionary
    """
    # Create system
    pressure = EvolutionaryPressure()
    
    # Create initial population
    for i in range(10):
        frag_id = f"test_fragment_{i}"
        pressure.add_population_member(frag_id, "fragment")
    
    for i in range(5):
        seq_id = f"test_sequence_{i}"
        pressure.add_population_member(seq_id, "sequence")
    
    # Run evolution
    results = pressure.evolve_population(generations)
    
    return results


if __name__ == "__main__":
    # Run a test evolution if executed directly
    test_results = run_test_evolution()
    print(f"Evolution test complete. Final population: {test_results['final_population']}")
    print(f"Fitness improvement: {test_results['avg_fitness_start']} -> {test_results['avg_fitness_end']}")
