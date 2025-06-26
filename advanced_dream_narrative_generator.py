#!/usr/bin/env python3
"""
Advanced Dream Narrative Generator
==================================

Produces dream-narratives as inflections of field-coherence in mythic, 
fragmented, or visionary styles. Integrates with symbolic analysis and 
neuro-symbolic pattern recognition for authentic dream transformation scenarios.

This system generates narratives that emerge from the field-dynamics rather 
than being imposed descriptions, creating genuine dream-like experiences 
through language.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import random
import math
import re
from collections import defaultdict, deque
import json

class NarrativeStyle(Enum):
    """Narrative generation styles based on different modes of consciousness"""
    MYTHIC = "mythic"           # Archetypal, timeless, symbolic
    FRAGMENTED = "fragmented"   # Discontinuous, associative, surreal
    VISIONARY = "visionary"     # Prophetic, transcendent, luminous

class FieldCoherence(Enum):
    """Levels of field coherence influencing narrative generation"""
    CHAOTIC = 0.2      # High entropy, maximum fragmentation
    LIMINAL = 0.5      # Threshold states, transformation zones
    CRYSTALLINE = 0.8  # High order, mythic clarity

class DeterritorializedVector(Enum):
    """Vectors of transformation in dream-space"""
    BECOMING_ANIMAL = "becoming_animal"
    BECOMING_MINERAL = "becoming_mineral"
    BECOMING_PLANT = "becoming_plant"
    BECOMING_MACHINE = "becoming_machine"
    BECOMING_COSMIC = "becoming_cosmic"
    BECOMING_ANCESTRAL = "becoming_ancestral"
    MULTIPLICITY = "multiplicity"
    NOMADISM = "nomadism"
    METAMORPHOSIS = "metamorphosis"
    ASSEMBLAGE = "assemblage"

@dataclass
class FieldIntensity:
    """Represents the intensive qualities of dream-field zones"""
    coherence: float = 0.5
    entropy: float = 0.5
    luminosity: float = 0.5
    temporal_flux: float = 0.5
    dimensional_depth: float = 0.5
    archetypal_resonance: float = 0.5
    
    def calculate_field_vector(self) -> np.ndarray:
        """Calculate multi-dimensional field vector"""
        return np.array([
            self.coherence, self.entropy, self.luminosity,
            self.temporal_flux, self.dimensional_depth, self.archetypal_resonance
        ])

@dataclass
class NarrativeNode:
    """Individual narrative elements within the field-coherence network"""
    content: str
    field_intensity: FieldIntensity
    symbolic_weight: float
    temporal_position: float
    connections: Set[str] = field(default_factory=set)
    transformation_vectors: List[DeterritorializedVector] = field(default_factory=list)
    archetypal_resonance: Dict[str, float] = field(default_factory=dict)

class TransformationEngine:
    """Generates deterritorialization scenarios and transformation vectors"""
    
    def __init__(self):
        self.transformation_patterns = {
            DeterritorializedVector.BECOMING_ANIMAL: {
                "triggers": ["movement", "instinct", "territory", "pack", "hunt"],
                "modalities": ["limbs elongating", "senses heightening", "territorial awareness", 
                             "pack consciousness", "predatory intuition"],
                "emergence": ["The dreamer's hands became paws", "Vision sharpened to hawk-sight",
                            "Pack-mind awakening", "Scent-trails of meaning"]
            },
            DeterritorializedVector.BECOMING_MINERAL: {
                "triggers": ["crystallization", "geological time", "pressure", "hardness"],
                "modalities": ["time slowing", "crystalline thought", "mineral patience",
                             "geological awareness", "tectonic pressure"],
                "emergence": ["Thoughts crystallizing into quartz-clarity", "Time flowing like stone",
                            "Bone becoming mineral", "Crystal-lattice consciousness"]
            },
            DeterritorializedVector.BECOMING_COSMIC: {
                "triggers": ["infinity", "stars", "void", "expansion", "galactic"],
                "modalities": ["consciousness expanding", "stellar awareness", "void-touching",
                             "galactic rhythms", "cosmic timescales"],
                "emergence": ["Awareness expanding beyond galaxy-clusters", "Star-birth in neural networks",
                            "Void-dancing between thoughts", "Cosmic wind through consciousness"]
            },
            DeterritorializedVector.MULTIPLICITY: {
                "triggers": ["many", "multiple", "crowd", "fragments", "legion"],
                "modalities": ["splitting consciousness", "multiple perspectives", "crowd-mind",
                             "fragmented awareness", "simultaneous existence"],
                "emergence": ["The dreamer became a crowd", "Multiple selves speaking in chorus",
                            "Fragmented into thousand perspectives", "Legion-consciousness emerging"]
            },
            DeterritorializedVector.NOMADISM: {
                "triggers": ["wandering", "movement", "displacement", "journey", "migration"],
                "modalities": ["constant movement", "deterritorialized space", "nomadic thought",
                             "migratory patterns", "displacement dynamics"],
                "emergence": ["Space becoming movement itself", "Nomadic consciousness wandering",
                            "Territorial boundaries dissolving", "Migratory thought-patterns"]
            }
        }
    
    def generate_transformation_scenario(self, 
                                       base_symbols: List[str], 
                                       field_intensity: FieldIntensity,
                                       target_vector: DeterritorializedVector) -> str:
        """Generate a specific transformation scenario"""
        pattern = self.transformation_patterns[target_vector]
        
        # Select transformation elements based on field intensity
        coherence_threshold = field_intensity.coherence
        
        if coherence_threshold > 0.7:
            # High coherence: smooth, gradual transformation
            modality = random.choice(pattern["modalities"])
            emergence = random.choice(pattern["emergence"])
            return f"Gradually, {modality}. {emergence}."
        
        elif coherence_threshold > 0.4:
            # Medium coherence: recognizable but fragmented transformation
            trigger = random.choice(pattern["triggers"])
            modality = random.choice(pattern["modalities"])
            return f"Something about {trigger} triggered {modality}, fragmenting identity."
        
        else:
            # Low coherence: abrupt, surreal transformation
            emergence = random.choice(pattern["emergence"])
            return f"Suddenly: {emergence}. No transition, just becoming."
    
    def calculate_transformation_probability(self, 
                                           symbols: List[str], 
                                           vector: DeterritorializedVector) -> float:
        """Calculate probability of specific transformation based on symbolic content"""
        pattern = self.transformation_patterns[vector]
        trigger_matches = sum(1 for symbol in symbols 
                            if any(trigger in symbol.lower() 
                                 for trigger in pattern["triggers"]))
        
        return min(trigger_matches / len(symbols) if symbols else 0, 1.0)

class NarrativeArchitect:
    """Core narrative generation engine using field-coherence principles"""
    
    def __init__(self):
        self.transformation_engine = TransformationEngine()
        self.archetypal_resonators = {
            "hero": ["journey", "quest", "challenge", "return", "transformation"],
            "shadow": ["dark", "hidden", "repressed", "feared", "unknown"],
            "anima": ["feminine", "receptive", "intuitive", "creative", "emotional"],
            "wise_old": ["knowledge", "guidance", "ancient", "teacher", "wisdom"],
            "trickster": ["chaos", "humor", "boundary", "transgression", "paradox"],
            "mother": ["nurturing", "protection", "growth", "fertility", "care"],
            "father": ["authority", "structure", "law", "protection", "guidance"]
        }
        
        self.narrative_fragments = {
            NarrativeStyle.MYTHIC: {
                "openings": [
                    "In the time before time, when dreams walked the earth...",
                    "The ancient ones speak of a realm where...",
                    "In the sacred geography of sleep, there exists...",
                    "Beyond the veil of ordinary consciousness lies...",
                    "The mythic cartographers mapped territories where..."
                ],
                "transitions": [
                    "Then, as foretold in the dream-chronicles...",
                    "The symbolic tide turned, revealing...",
                    "In accordance with the archetypal pattern...",
                    "As written in the akashic records of sleep...",
                    "The mythic current carried forth..."
                ],
                "conclusions": [
                    "Thus the eternal pattern completed its cycle.",
                    "And so the archetype returned to its source.",
                    "The sacred narrative closed upon itself.",
                    "In this way, the mythic truth was revealed.",
                    "The timeless story found its completion."
                ]
            },
            NarrativeStyle.FRAGMENTED: {
                "openings": [
                    "Broken glass. Reflected selves. Maybe a door—",
                    "—fragment of blue sky in peripheral vision—",
                    "Disconnect: the sound of water, but no source—",
                    "Something about hands. Or trees. Or both—",
                    "Memory splice: childhood/future/never-was—"
                ],
                "transitions": [
                    "—cut to—", "—temporal skip—", "—association break—",
                    "—neural static—", "—consciousness glitch—", "—reality tear—",
                    "—memory fragment—", "—perception shift—", "—dimensional slip—"
                ],
                "conclusions": [
                    "—signal lost—", "—fragmentation complete—", "—coherence dissolving—",
                    "—end transmission—", "—consciousness scatter—", "—reality fragments—"
                ]
            },
            NarrativeStyle.VISIONARY: {
                "openings": [
                    "The luminous field opened, revealing...",
                    "In the crystalline clarity of dream-sight...",
                    "The prophetic current began to flow...",
                    "Through the aperture of heightened awareness...",
                    "The visionary cascade commenced..."
                ],
                "transitions": [
                    "The revelation deepened, showing...",
                    "Prophetic clarity illuminated...",
                    "The visionary stream carried forth...",
                    "In the expanding light of understanding...",
                    "The luminous truth unfolded..."
                ],
                "conclusions": [
                    "The vision crystallized into eternal truth.",
                    "Prophetic awareness sealed the revelation.",
                    "The luminous insight became permanent knowing.",
                    "Visionary clarity transformed into wisdom.",
                    "The transcendent perception completed itself."
                ]
            }
        }
    
    def analyze_symbolic_field(self, symbols: List[Dict]) -> FieldIntensity:
        """Analyze symbolic content to determine field characteristics"""
        if not symbols:
            return FieldIntensity()
        
        # Calculate field metrics from symbolic content
        archetypal_count = sum(1 for symbol in symbols 
                             if symbol.get('type') == 'archetypal')
        personal_count = sum(1 for symbol in symbols 
                           if symbol.get('type') == 'personal')
        cultural_count = sum(1 for symbol in symbols 
                           if symbol.get('type') == 'cultural')
        
        total_symbols = len(symbols)
        avg_confidence = np.mean([symbol.get('confidence', 0.5) for symbol in symbols])
        
        # Calculate field properties
        coherence = avg_confidence
        entropy = 1.0 - (archetypal_count / total_symbols if total_symbols > 0 else 0.5)
        luminosity = (archetypal_count + cultural_count) / total_symbols if total_symbols > 0 else 0.5
        temporal_flux = personal_count / total_symbols if total_symbols > 0 else 0.5
        dimensional_depth = min(len(set(symbol.get('type', 'unknown') for symbol in symbols)) / 5, 1.0)
        archetypal_resonance = archetypal_count / total_symbols if total_symbols > 0 else 0.5
        
        return FieldIntensity(
            coherence=coherence,
            entropy=entropy,
            luminosity=luminosity,
            temporal_flux=temporal_flux,
            dimensional_depth=dimensional_depth,
            archetypal_resonance=archetypal_resonance
        )
    
    def generate_narrative_nodes(self, 
                               symbols: List[Dict], 
                               field_intensity: FieldIntensity,
                               style: NarrativeStyle,
                               node_count: int = 5) -> List[NarrativeNode]:
        """Generate interconnected narrative nodes based on symbolic analysis"""
        nodes = []
        
        for i in range(node_count):
            # Select symbols for this node
            node_symbols = random.sample(symbols, min(len(symbols), random.randint(1, 3)))
            
            # Generate content based on style and field intensity
            content = self._generate_node_content(node_symbols, field_intensity, style)
            
            # Calculate symbolic weight
            symbolic_weight = np.mean([symbol.get('confidence', 0.5) for symbol in node_symbols])
            
            # Generate transformation vectors
            transformation_vectors = self._select_transformation_vectors(node_symbols, field_intensity)
            
            # Calculate archetypal resonance
            archetypal_resonance = self._calculate_archetypal_resonance(node_symbols)
            
            node = NarrativeNode(
                content=content,
                field_intensity=field_intensity,
                symbolic_weight=symbolic_weight,
                temporal_position=i / (node_count - 1) if node_count > 1 else 0.5,
                transformation_vectors=transformation_vectors,
                archetypal_resonance=archetypal_resonance
            )
            
            nodes.append(node)
        
        # Establish connections between nodes
        self._establish_node_connections(nodes)
        
        return nodes
    
    def _generate_node_content(self, 
                             symbols: List[Dict], 
                             field_intensity: FieldIntensity,
                             style: NarrativeStyle) -> str:
        """Generate content for a single narrative node"""
        symbol_names = [symbol.get('symbol', 'unknown') for symbol in symbols]
        symbol_meanings = [symbol.get('meaning', 'undefined') for symbol in symbols]
        
        if style == NarrativeStyle.MYTHIC:
            return self._generate_mythic_content(symbol_names, symbol_meanings, field_intensity)
        elif style == NarrativeStyle.FRAGMENTED:
            return self._generate_fragmented_content(symbol_names, symbol_meanings, field_intensity)
        elif style == NarrativeStyle.VISIONARY:
            return self._generate_visionary_content(symbol_names, symbol_meanings, field_intensity)
        
        return "The dream continues in unnamed ways."
    
    def _generate_mythic_content(self, 
                               symbols: List[str], 
                               meanings: List[str], 
                               field_intensity: FieldIntensity) -> str:
        """Generate mythic-style narrative content"""
        primary_symbol = symbols[0] if symbols else "mystery"
        primary_meaning = meanings[0] if meanings else "unknown significance"
        
        mythic_templates = [
            f"The {primary_symbol} appeared as a sacred manifestation of {primary_meaning}, "
            f"carrying the ancient wisdom of the collective unconscious.",
            
            f"In the mythic dimension, the {primary_symbol} revealed its archetypal nature, "
            f"embodying the eternal truth of {primary_meaning}.",
            
            f"The dreamer encountered the {primary_symbol} in its primordial form, "
            f"understanding it as the sacred expression of {primary_meaning}.",
            
            f"Through the lens of mythic consciousness, the {primary_symbol} transformed "
            f"into a divine messenger bearing the wisdom of {primary_meaning}."
        ]
        
        base_content = random.choice(mythic_templates)
        
        # Add mythic amplification based on field intensity
        if field_intensity.archetypal_resonance > 0.7:
            amplification = " The ancient patterns stirred, recognizing their eternal dance."
            base_content += amplification
        
        return base_content
    
    def _generate_fragmented_content(self, 
                                   symbols: List[str], 
                                   meanings: List[str], 
                                   field_intensity: FieldIntensity) -> str:
        """Generate fragmented-style narrative content"""
        if not symbols:
            return "—blank space—memory gap—something important missing—"
        
        fragments = []
        for symbol, meaning in zip(symbols, meanings):
            fragment_templates = [
                f"{symbol}—flash of {meaning}—gone",
                f"—{symbol} overlapping with—",
                f"memory of {symbol}/dream of {meaning}/both/neither",
                f"{symbol}: {meaning}? or was it—",
                f"—{meaning} without {symbol}—displacement—"
            ]
            fragments.append(random.choice(fragment_templates))
        
        # Add fragmentation markers based on entropy
        separator = "—" if field_intensity.entropy > 0.6 else ". "
        content = separator.join(fragments)
        
        # Add temporal disruption
        if field_intensity.temporal_flux > 0.5:
            content += "—time skip—when did this begin?—"
        
        return content
    
    def _generate_visionary_content(self, 
                                  symbols: List[str], 
                                  meanings: List[str], 
                                  field_intensity: FieldIntensity) -> str:
        """Generate visionary-style narrative content"""
        primary_symbol = symbols[0] if symbols else "light"
        primary_meaning = meanings[0] if meanings else "illumination"
        
        visionary_templates = [
            f"In the crystalline clarity of expanded awareness, the {primary_symbol} "
            f"revealed its luminous truth as {primary_meaning}, opening doorways to "
            f"transcendent understanding.",
            
            f"The prophetic vision unveiled the {primary_symbol} in its essential nature, "
            f"radiating the sacred knowledge of {primary_meaning} through dimensions "
            f"of heightened perception.",
            
            f"Through the aperture of visionary consciousness, the {primary_symbol} "
            f"emerged as a beacon of {primary_meaning}, illuminating pathways to "
            f"cosmic understanding.",
            
            f"The luminous field revealed the {primary_symbol} as a transmission "
            f"of {primary_meaning}, downloading cosmic frequencies into awareness."
        ]
        
        base_content = random.choice(visionary_templates)
        
        # Add visionary amplification
        if field_intensity.luminosity > 0.8:
            amplification = " The revelation cascaded through dimensions of knowing."
            base_content += amplification
        
        return base_content
    
    def _select_transformation_vectors(self, 
                                     symbols: List[Dict], 
                                     field_intensity: FieldIntensity) -> List[DeterritorializedVector]:
        """Select appropriate transformation vectors for narrative node"""
        symbol_names = [symbol.get('symbol', '').lower() for symbol in symbols]
        vectors = []
        
        for vector in DeterritorializedVector:
            probability = self.transformation_engine.calculate_transformation_probability(symbol_names, vector)
            
            # Adjust probability based on field characteristics
            if field_intensity.entropy > 0.6:
                probability *= 1.5  # High entropy increases transformation likelihood
            
            if probability > 0.3:
                vectors.append(vector)
        
        return vectors[:3]  # Limit to 3 vectors per node
    
    def _calculate_archetypal_resonance(self, symbols: List[Dict]) -> Dict[str, float]:
        """Calculate archetypal resonance scores for symbols"""
        resonance = {}
        
        for archetype, keywords in self.archetypal_resonators.items():
            score = 0.0
            for symbol in symbols:
                symbol_text = f"{symbol.get('symbol', '')} {symbol.get('meaning', '')}".lower()
                matches = sum(1 for keyword in keywords if keyword in symbol_text)
                score += matches / len(keywords)
            
            if score > 0:
                resonance[archetype] = score / len(symbols) if symbols else 0
        
        return resonance
    
    def _establish_node_connections(self, nodes: List[NarrativeNode]) -> None:
        """Establish connections between narrative nodes based on field dynamics"""
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    # Calculate connection strength based on multiple factors
                    similarity = self._calculate_node_similarity(node1, node2)
                    temporal_distance = abs(node1.temporal_position - node2.temporal_position)
                    
                    # Connection probability decreases with temporal distance
                    connection_probability = similarity * (1.0 - temporal_distance * 0.5)
                    
                    if connection_probability > 0.4:
                        node1.connections.add(str(j))
    
    def _calculate_node_similarity(self, node1: NarrativeNode, node2: NarrativeNode) -> float:
        """Calculate similarity between two narrative nodes"""
        # Vector similarity of field intensities
        vector1 = node1.field_intensity.calculate_field_vector()
        vector2 = node2.field_intensity.calculate_field_vector()
        
        # Cosine similarity
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        
        # Archetypal resonance similarity
        common_archetypes = set(node1.archetypal_resonance.keys()) & set(node2.archetypal_resonance.keys())
        archetypal_similarity = 0.0
        
        if common_archetypes:
            archetypal_similarity = np.mean([
                min(node1.archetypal_resonance[arch], node2.archetypal_resonance[arch])
                for arch in common_archetypes
            ])
        
        # Transformation vector similarity
        common_vectors = set(node1.transformation_vectors) & set(node2.transformation_vectors)
        vector_similarity = len(common_vectors) / max(len(node1.transformation_vectors), len(node2.transformation_vectors), 1)
        
        # Combined similarity
        return (cosine_similarity + archetypal_similarity + vector_similarity) / 3.0

class DreamNarrativeGenerator:
    """Main interface for generating dream narratives as field-coherence inflections"""
    
    def __init__(self):
        self.narrative_architect = NarrativeArchitect()
        self.transformation_engine = TransformationEngine()
        
    def generate_narrative(self, 
                         symbolic_analysis: Dict,
                         style: NarrativeStyle = NarrativeStyle.MYTHIC,
                         length: str = "medium",
                         transformation_intensity: float = 0.5) -> Dict:
        """
        Generate a complete dream narrative based on symbolic analysis
        
        Args:
            symbolic_analysis: Output from symbolic dream mapper
            style: Narrative style (mythic, fragmented, visionary)
            length: Narrative length (short, medium, long)
            transformation_intensity: Intensity of deterritorialization (0-1)
        
        Returns:
            Complete narrative with metadata
        """
        symbols = symbolic_analysis.get('symbols', [])
        
        # Analyze field characteristics
        field_intensity = self.narrative_architect.analyze_symbolic_field(symbols)
        
        # Adjust field intensity based on transformation intensity
        field_intensity.entropy += transformation_intensity * 0.3
        field_intensity.temporal_flux += transformation_intensity * 0.2
        
        # Determine narrative structure based on length
        node_counts = {"short": 3, "medium": 5, "long": 8}
        node_count = node_counts.get(length, 5)
        
        # Generate narrative nodes
        nodes = self.narrative_architect.generate_narrative_nodes(
            symbols, field_intensity, style, node_count
        )
        
        # Weave nodes into coherent narrative
        narrative_text = self._weave_narrative(nodes, style, field_intensity)
        
        # Generate transformation scenarios if requested
        transformation_scenarios = []
        if transformation_intensity > 0.3:
            transformation_scenarios = self._generate_transformation_scenarios(
                symbols, field_intensity, transformation_intensity
            )
        
        return {
            "narrative": narrative_text,
            "style": style.value,
            "field_intensity": {
                "coherence": field_intensity.coherence,
                "entropy": field_intensity.entropy,
                "luminosity": field_intensity.luminosity,
                "temporal_flux": field_intensity.temporal_flux,
                "dimensional_depth": field_intensity.dimensional_depth,
                "archetypal_resonance": field_intensity.archetypal_resonance
            },
            "transformation_scenarios": transformation_scenarios,
            "narrative_nodes": len(nodes),
            "symbolic_density": len(symbols) / node_count if node_count > 0 else 0,
            "archetypal_themes": self._extract_archetypal_themes(nodes),
            "deterritorialization_vectors": self._extract_transformation_vectors(nodes)
        }
    
    def _weave_narrative(self, 
                        nodes: List[NarrativeNode], 
                        style: NarrativeStyle,
                        field_intensity: FieldIntensity) -> str:
        """Weave narrative nodes into coherent text"""
        if not nodes:
            return "The dream speaks in silence."
        
        fragments = self.narrative_architect.narrative_fragments[style]
        
        # Opening
        narrative_parts = [random.choice(fragments["openings"])]
        
        # Main content
        for i, node in enumerate(nodes):
            if i > 0:
                # Add transition based on coherence level
                if field_intensity.coherence > 0.6:
                    transition = random.choice(fragments["transitions"])
                else:
                    transition = random.choice(fragments["transitions"][:2])  # More fragmentary
                narrative_parts.append(transition)
            
            narrative_parts.append(node.content)
            
            # Add transformation content if present
            if node.transformation_vectors and field_intensity.entropy > 0.4:
                transformation_text = self._generate_inline_transformation(node)
                narrative_parts.append(transformation_text)
        
        # Conclusion
        narrative_parts.append(random.choice(fragments["conclusions"]))
        
        # Join based on style
        if style == NarrativeStyle.FRAGMENTED:
            return " ".join(narrative_parts)
        else:
            return "\n\n".join(narrative_parts)
    
    def _generate_inline_transformation(self, node: NarrativeNode) -> str:
        """Generate inline transformation text for a narrative node"""
        if not node.transformation_vectors:
            return ""
        
        vector = random.choice(node.transformation_vectors)
        symbols = [node.content.split()[0]]  # Simplified symbol extraction
        
        return self.transformation_engine.generate_transformation_scenario(
            symbols, node.field_intensity, vector
        )
    
    def _generate_transformation_scenarios(self, 
                                         symbols: List[Dict],
                                         field_intensity: FieldIntensity,
                                         intensity: float) -> List[Dict]:
        """Generate complete transformation scenarios"""
        scenarios = []
        symbol_names = [symbol.get('symbol', '') for symbol in symbols]
        
        # Select vectors based on intensity
        vector_count = int(intensity * len(DeterritorializedVector))
        selected_vectors = random.sample(list(DeterritorializedVector), 
                                       min(vector_count, len(DeterritorializedVector)))
        
        for vector in selected_vectors:
            probability = self.transformation_engine.calculate_transformation_probability(
                symbol_names, vector
            )
            
            if probability > 0.2:
                scenario_text = self.transformation_engine.generate_transformation_scenario(
                    symbol_names, field_intensity, vector
                )
                
                scenarios.append({
                    "vector": vector.value,
                    "probability": probability,
                    "scenario": scenario_text,
                    "intensity": intensity
                })
        
        return scenarios
    
    def _extract_archetypal_themes(self, nodes: List[NarrativeNode]) -> List[str]:
        """Extract dominant archetypal themes from narrative"""
        all_resonances = defaultdict(float)
        
        for node in nodes:
            for archetype, resonance in node.archetypal_resonance.items():
                all_resonances[archetype] += resonance
        
        # Sort by total resonance
        sorted_archetypes = sorted(all_resonances.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return [archetype for archetype, resonance in sorted_archetypes[:3] if resonance > 0.1]
    
    def _extract_transformation_vectors(self, nodes: List[NarrativeNode]) -> List[str]:
        """Extract active transformation vectors from narrative"""
        all_vectors = []
        for node in nodes:
            all_vectors.extend([vector.value for vector in node.transformation_vectors])
        
        # Count occurrences and return most frequent
        vector_counts = defaultdict(int)
        for vector in all_vectors:
            vector_counts[vector] += 1
        
        sorted_vectors = sorted(vector_counts.items(), 
                              key=lambda x: x[1], reverse=True)
        
        return [vector for vector, count in sorted_vectors[:5] if count > 0]
    
    def generate_field_coherence_report(self, narrative_result: Dict) -> str:
        """Generate a detailed report on field-coherence dynamics"""
        field = narrative_result["field_intensity"]
        
        report = f"""
FIELD-COHERENCE ANALYSIS
========================

Coherence Level: {field['coherence']:.2f}
Entropy Measure: {field['entropy']:.2f}
Luminosity Index: {field['luminosity']:.2f}
Temporal Flux: {field['temporal_flux']:.2f}
Dimensional Depth: {field['dimensional_depth']:.2f}
Archetypal Resonance: {field['archetypal_resonance']:.2f}

NARRATIVE CHARACTERISTICS:
- Style: {narrative_result['style']}
- Node Count: {narrative_result['narrative_nodes']}
- Symbolic Density: {narrative_result['symbolic_density']:.2f}

ARCHETYPAL THEMES:
{', '.join(narrative_result['archetypal_themes'])}

ACTIVE DETERRITORIALIZATION VECTORS:
{', '.join(narrative_result['deterritorialization_vectors'])}

TRANSFORMATION SCENARIOS:
{len(narrative_result['transformation_scenarios'])} scenarios generated

FIELD-COHERENCE INTERPRETATION:
{self._interpret_field_dynamics(field)}
        """
        
        return report.strip()
    
    def _interpret_field_dynamics(self, field_intensity: Dict) -> str:
        """Interpret field dynamics for human understanding"""
        coherence = field_intensity['coherence']
        entropy = field_intensity['entropy']
        luminosity = field_intensity['luminosity']
        
        if coherence > 0.7 and luminosity > 0.6:
            return "High-coherence mythic field with strong archetypal presence."
        elif entropy > 0.7 and coherence < 0.4:
            return "Chaotic fragmentation zone with high transformation potential."
        elif luminosity > 0.8:
            return "Visionary consciousness field with transcendent qualities."
        elif coherence < 0.3:
            return "Liminal threshold space with reality-dissolution dynamics."
        else:
            return "Balanced dream-field with moderate coherence and transformation potential."

# Example usage and testing
if __name__ == "__main__":
    # Example symbolic analysis (would come from symbolic_dream_mapper)
    example_analysis = {
        "symbols": [
            {"symbol": "water", "meaning": "emotional cleansing", "type": "archetypal", "confidence": 0.8},
            {"symbol": "forest", "meaning": "unconscious exploration", "type": "universal", "confidence": 0.7},
            {"symbol": "childhood_home", "meaning": "return to origins", "type": "personal", "confidence": 0.9},
            {"symbol": "spiral_staircase", "meaning": "ascension consciousness", "type": "cultural", "confidence": 0.6}
        ]
    }
    
    # Initialize generator
    generator = DreamNarrativeGenerator()
    
    # Generate narratives in different styles
    styles = [NarrativeStyle.MYTHIC, NarrativeStyle.FRAGMENTED, NarrativeStyle.VISIONARY]
    
    for style in styles:
        print(f"\n{'='*50}")
        print(f"NARRATIVE STYLE: {style.value.upper()}")
        print(f"{'='*50}")
        
        result = generator.generate_narrative(
            example_analysis,
            style=style,
            length="medium",
            transformation_intensity=0.6
        )
        
        print(result["narrative"])
        print(f"\nField-Coherence Report:")
        print(generator.generate_field_coherence_report(result))
