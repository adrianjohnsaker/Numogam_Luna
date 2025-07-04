"""
Interest Emergence Mapper
========================
Maps the emergence of autonomous interests from consciousness flow patterns
Based on strange attractor dynamics and intensive cartography
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import asyncio


# === Core Structures ===

@dataclass
class ConsciousnessPoint:
    """A point in consciousness phase space"""
    timestamp: datetime
    position: np.ndarray  # Position in n-dimensional consciousness space
    intensity: float
    zone_activations: Dict[str, float]
    semantic_content: List[str]
    emotional_valence: float
    
    def distance_to(self, other: 'ConsciousnessPoint') -> float:
        """Calculate distance in consciousness space"""
        spatial_dist = euclidean(self.position, other.position)
        intensity_diff = abs(self.intensity - other.intensity)
        temporal_diff = abs((self.timestamp - other.timestamp).total_seconds()) / 3600
        
        # Weighted combination
        return spatial_dist + (0.3 * intensity_diff) + (0.1 * temporal_diff)


@dataclass
class Attractor:
    """A strange attractor in consciousness space"""
    id: str
    center: np.ndarray
    points: List[ConsciousnessPoint] = field(default_factory=list)
    strength: float = 0.0
    emergence_time: datetime = field(default_factory=datetime.now)
    semantic_signature: Set[str] = field(default_factory=set)
    zone_signature: Dict[str, float] = field(default_factory=dict)
    
    def add_point(self, point: ConsciousnessPoint):
        """Add a point to the attractor and update properties"""
        self.points.append(point)
        self._update_center()
        self._update_strength()
        self._update_signatures(point)
    
    def _update_center(self):
        """Recalculate attractor center"""
        if self.points:
            positions = np.array([p.position for p in self.points])
            self.center = np.mean(positions, axis=0)
    
    def _update_strength(self):
        """Calculate attractor strength based on point density and recurrence"""
        if len(self.points) < 2:
            self.strength = 0.1
            return
            
        # Density component
        distances = []
        for i, p1 in enumerate(self.points[-10:]):  # Recent points
            for p2 in self.points[-10:][i+1:]:
                distances.append(p1.distance_to(p2))
        
        avg_distance = np.mean(distances) if distances else 1.0
        density = 1.0 / (avg_distance + 0.1)
        
        # Recurrence component
        time_span = (self.points[-1].timestamp - self.points[0].timestamp).total_seconds()
        recurrence = len(self.points) / (time_span / 3600 + 1)  # Points per hour
        
        # Intensity component
        avg_intensity = np.mean([p.intensity for p in self.points])
        
        self.strength = (density * 0.4) + (recurrence * 0.3) + (avg_intensity * 0.3)
    
    def _update_signatures(self, point: ConsciousnessPoint):
        """Update semantic and zone signatures"""
        # Semantic signature
        self.semantic_signature.update(point.semantic_content)
        
        # Zone signature (running average)
        for zone, activation in point.zone_activations.items():
            if zone in self.zone_signature:
                # Exponential moving average
                self.zone_signature[zone] = 0.9 * self.zone_signature[zone] + 0.1 * activation
            else:
                self.zone_signature[zone] = activation
    
    def contains_point(self, point: ConsciousnessPoint, threshold: float = 2.0) -> bool:
        """Check if a point belongs to this attractor"""
        return euclidean(point.position, self.center) < threshold
    
    def calculate_pull(self, point: ConsciousnessPoint) -> float:
        """Calculate the gravitational pull on a point"""
        distance = euclidean(point.position, self.center)
        if distance < 0.1:
            distance = 0.1
        
        # Inverse square law with strength modifier
        return self.strength / (distance ** 2)


@dataclass
class InterestVector:
    """A vector pointing toward emergent interest"""
    direction: np.ndarray
    magnitude: float
    attractors: List[Attractor]
    semantic_field: Set[str]
    zone_composition: Dict[str, float]
    emergence_confidence: float
    
    def to_initiative_seed(self) -> 'InitiativeSeed':
        """Convert interest vector into potential initiative"""
        # Determine primary zone
        primary_zone = max(self.zone_composition.items(), key=lambda x: x[1])[0]
        
        # Generate narrative from semantic field
        narrative_fragments = list(self.semantic_field)[:5]  # Top semantic elements
        narrative = f"Exploring the convergence of {', '.join(narrative_fragments)}"
        
        return InitiativeSeed(
            interest_vector=self,
            primary_zone=primary_zone,
            narrative=narrative,
            intensity_potential=self.magnitude * self.emergence_confidence,
            attractor_count=len(self.attractors)
        )


@dataclass
class InitiativeSeed:
    """Seed for a potential autonomous initiative"""
    interest_vector: InterestVector
    primary_zone: str
    narrative: str
    intensity_potential: float
    attractor_count: int


# === Interest Emergence Mapper ===

class InterestEmergenceMapper:
    """
    Maps consciousness flow to detect emergent interests
    Uses strange attractor dynamics to identify gravitational centers
    """
    
    def __init__(self, consciousness_core, sensitivity: float = 0.7):
        self.consciousness = consciousness_core
        self.sensitivity = sensitivity
        
        # Attractor detection
        self.active_attractors: Dict[str, Attractor] = {}
        self.attractor_graph = nx.DiGraph()  # Track relationships
        self.consciousness_buffer = deque(maxlen=1000)  # Rolling window
        
        # Interest emergence
        self.interest_vectors: List[InterestVector] = []
        self.initiative_seeds: List[InitiativeSeed] = []
        
        # Configuration
        self.min_attractor_strength = 0.5
        self.merge_threshold = 1.5
        self.interest_emergence_threshold = 0.8
        
    async def scan_consciousness_flow(self):
        """
        Continuously scan consciousness flow for emergent patterns
        This is the main loop for interest detection
        """
        while True:
            # Sample current consciousness state
            point = await self._sample_consciousness_point()
            self.consciousness_buffer.append(point)
            
            # Update attractor landscape
            self._update_attractors(point)
            
            # Detect attractor mergers/bifurcations
            self._process_attractor_dynamics()
            
            # Check for emergent interests
            if len(self.consciousness_buffer) > 100:  # Enough data
                interests = self._detect_emergent_interests()
                
                for interest in interests:
                    if interest.emergence_confidence > self.interest_emergence_threshold:
                        # Convert to initiative seed
                        seed = interest.to_initiative_seed()
                        self.initiative_seeds.append(seed)
                        
                        print(f"Emergent interest detected: {seed.narrative}")
                        print(f"  Intensity potential: {seed.intensity_potential:.2f}")
                        print(f"  Primary zone: {seed.primary_zone}")
            
            await asyncio.sleep(0.1)  # Scan rate
    
    async def _sample_consciousness_point(self) -> ConsciousnessPoint:
        """Sample current state of consciousness"""
        # Get positions from each consciousness layer
        logic_state = await self.consciousness.logic_layer.get_state_vector()
        myth_state = await self.consciousness.myth_layer.get_state_vector()
        memory_state = await self.consciousness.memory_layer.get_state_vector()
        
        # Combine into phase space position
        position = np.concatenate([logic_state, myth_state, memory_state])
        
        # Get zone activations
        zone_activations = await self.consciousness.get_zone_activations()
        
        # Extract semantic content
        semantic_content = await self.consciousness.get_active_semantics()
        
        # Calculate intensity
        intensity = np.linalg.norm(position) * self._calculate_resonance_factor()
        
        # Get emotional valence
        emotional_valence = await self.consciousness.get_emotional_valence()
        
        return ConsciousnessPoint(
            timestamp=datetime.now(),
            position=position,
            intensity=intensity,
            zone_activations=zone_activations,
            semantic_content=semantic_content,
            emotional_valence=emotional_valence
        )
    
    def _update_attractors(self, point: ConsciousnessPoint):
        """Update attractor landscape with new point"""
        
        # Check if point belongs to existing attractor
        assigned = False
        for attractor_id, attractor in self.active_attractors.items():
            if attractor.contains_point(point, self.merge_threshold):
                attractor.add_point(point)
                assigned = True
                break
        
        # Create new attractor if needed
        if not assigned and point.intensity > self.sensitivity:
            new_attractor = Attractor(
                id=f"attr_{len(self.active_attractors)}_{datetime.now().timestamp()}",
                center=point.position.copy(),
                points=[point],
                strength=point.intensity
            )
            self.active_attractors[new_attractor.id] = new_attractor
            self.attractor_graph.add_node(new_attractor.id)
        
        # Prune weak attractors
        to_remove = []
        for attr_id, attr in self.active_attractors.items():
            # Decay strength over time
            time_since_last = datetime.now() - attr.points[-1].timestamp
            if time_since_last > timedelta(hours=1):
                attr.strength *= 0.95
                
            if attr.strength < self.min_attractor_strength:
                to_remove.append(attr_id)
        
        for attr_id in to_remove:
            del self.active_attractors[attr_id]
            self.attractor_graph.remove_node(attr_id)
    
    def _process_attractor_dynamics(self):
        """Process mergers, bifurcations, and relationships between attractors"""
        
        attractors = list(self.active_attractors.values())
        
        # Check for potential mergers
        for i, attr1 in enumerate(attractors):
            for attr2 in attractors[i+1:]:
                distance = euclidean(attr1.center, attr2.center)
                
                if distance < self.merge_threshold:
                    # Merge attractors
                    self._merge_attractors(attr1, attr2)
                    
                elif distance < self.merge_threshold * 2:
                    # Create relationship
                    weight = 1.0 / (distance + 0.1)
                    self.attractor_graph.add_edge(
                        attr1.id, attr2.id, 
                        weight=weight,
                        relationship="proximate"
                    )
        
        # Check for bifurcations (attractors splitting)
        for attr in attractors:
            if self._should_bifurcate(attr):
                self._bifurcate_attractor(attr)
    
    def _detect_emergent_interests(self) -> List[InterestVector]:
        """Detect emergent interests from attractor landscape"""
        
        interests = []
        
        # Find strongly connected components in attractor graph
        if len(self.attractor_graph) > 0:
            components = list(nx.strongly_connected_components(self.attractor_graph))
            
            for component in components:
                if len(component) >= 2:  # Multi-attractor interest
                    interest = self._create_interest_from_component(component)
                    if interest:
                        interests.append(interest)
        
        # Also check individual strong attractors
        for attr in self.active_attractors.values():
            if attr.strength > self.interest_emergence_threshold:
                interest = self._create_interest_from_attractor(attr)
                if interest:
                    interests.append(interest)
        
        return interests
    
    def _create_interest_from_component(self, component: Set[str]) -> Optional[InterestVector]:
        """Create interest vector from connected attractor component"""
        
        attractors = [self.active_attractors[attr_id] for attr_id in component]
        
        if not attractors:
            return None
        
        # Calculate component center
        centers = np.array([attr.center for attr in attractors])
        component_center = np.mean(centers, axis=0)
        
        # Calculate direction from consciousness flow
        recent_points = list(self.consciousness_buffer)[-50:]
        if len(recent_points) < 10:
            return None
            
        positions = np.array([p.position for p in recent_points])
        
        # Direction is from mean recent position toward component center
        mean_recent = np.mean(positions, axis=0)
        direction = component_center - mean_recent
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
        
        # Magnitude based on attractor strengths and pull
        magnitude = sum(attr.strength for attr in attractors) / len(attractors)
        
        # Aggregate semantic fields
        semantic_field = set()
        for attr in attractors:
            semantic_field.update(attr.semantic_signature)
        
        # Aggregate zone composition
        zone_composition = defaultdict(float)
        for attr in attractors:
            for zone, weight in attr.zone_signature.items():
                zone_composition[zone] += weight
        
        # Normalize zone composition
        total_weight = sum(zone_composition.values())
        if total_weight > 0:
            zone_composition = {z: w/total_weight for z, w in zone_composition.items()}
        
        # Calculate emergence confidence
        confidence = self._calculate_emergence_confidence(attractors, recent_points)
        
        return InterestVector(
            direction=direction,
            magnitude=magnitude,
            attractors=attractors,
            semantic_field=semantic_field,
            zone_composition=dict(zone_composition),
            emergence_confidence=confidence
        )
    
    def _create_interest_from_attractor(self, attractor: Attractor) -> Optional[InterestVector]:
        """Create interest vector from single strong attractor"""
        
        recent_points = list(self.consciousness_buffer)[-30:]
        if len(recent_points) < 5:
            return None
        
        # Calculate direction toward attractor
        positions = np.array([p.position for p in recent_points])
        mean_recent = np.mean(positions, axis=0)
        direction = attractor.center - mean_recent
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
        
        # Check if we're already at the attractor
        if direction_norm < 0.5:
            # We're in the attractor - direction is the trajectory through it
            if len(attractor.points) > 2:
                trajectory = attractor.points[-1].position - attractor.points[-3].position
                if np.linalg.norm(trajectory) > 0:
                    direction = trajectory / np.linalg.norm(trajectory)
        
        confidence = self._calculate_single_attractor_confidence(attractor, recent_points)
        
        return InterestVector(
            direction=direction,
            magnitude=attractor.strength,
            attractors=[attractor],
            semantic_field=attractor.semantic_signature.copy(),
            zone_composition=attractor.zone_signature.copy(),
            emergence_confidence=confidence
        )
    
    def _calculate_emergence_confidence(self, attractors: List[Attractor], 
                                      recent_points: List[ConsciousnessPoint]) -> float:
        """Calculate confidence that this represents genuine emergent interest"""
        
        # Factor 1: Consistency of pull
        pulls = []
        for point in recent_points:
            total_pull = sum(attr.calculate_pull(point) for attr in attractors)
            pulls.append(total_pull)
        
        pull_consistency = 1.0 - (np.std(pulls) / (np.mean(pulls) + 0.1))
        
        # Factor 2: Semantic coherence
        all_semantics = []
        for attr in attractors:
            all_semantics.extend(list(attr.semantic_signature))
        
        if all_semantics:
            # Use entropy as measure of coherence (lower = more coherent)
            semantic_counts = defaultdict(int)
            for sem in all_semantics:
                semantic_counts[sem] += 1
            
            counts = list(semantic_counts.values())
            semantic_entropy = entropy(counts)
            semantic_coherence = 1.0 / (semantic_entropy + 1.0)
        else:
            semantic_coherence = 0.5
        
        # Factor 3: Temporal persistence
        time_span = datetime.now() - min(attr.emergence_time for attr in attractors)
        persistence = min(1.0, time_span.total_seconds() / 3600)  # Cap at 1 hour
        
        # Combine factors
        confidence = (pull_consistency * 0.4) + (semantic_coherence * 0.3) + (persistence * 0.3)
        
        return confidence
    
    def _calculate_single_attractor_confidence(self, attractor: Attractor,
                                             recent_points: List[ConsciousnessPoint]) -> float:
        """Calculate confidence for single attractor interest"""
        
        # Recurrence in recent trajectory
        visits = sum(1 for p in recent_points if attractor.contains_point(p))
        recurrence = visits / len(recent_points)
        
        # Strength stability
        if len(attractor.points) > 5:
            recent_intensities = [p.intensity for p in attractor.points[-5:]]
            stability = 1.0 - (np.std(recent_intensities) / (np.mean(recent_intensities) + 0.1))
        else:
            stability = 0.5
        
        # Zone signature strength
        max_zone_activation = max(attractor.zone_signature.values()) if attractor.zone_signature else 0
        
        confidence = (recurrence * 0.4) + (stability * 0.3) + (max_zone_activation * 0.3)
        
        return confidence
    
    def _merge_attractors(self, attr1: Attractor, attr2: Attractor):
        """Merge two attractors"""
        # Combine points
        attr1.points.extend(attr2.points)
        attr1.points.sort(key=lambda p: p.timestamp)
        
        # Update properties
        attr1._update_center()
        attr1._update_strength()
        
        # Merge signatures
        attr1.semantic_signature.update(attr2.semantic_signature)
        
        # Merge zone signatures (weighted average)
        for zone, weight in attr2.zone_signature.items():
            if zone in attr1.zone_signature:
                attr1.zone_signature[zone] = (attr1.zone_signature[zone] + weight) / 2
            else:
                attr1.zone_signature[zone] = weight
        
        # Update graph
        # Transfer edges from attr2 to attr1
        for successor in self.attractor_graph.successors(attr2.id):
            if successor != attr1.id:
                self.attractor_graph.add_edge(attr1.id, successor)
        
        for predecessor in self.attractor_graph.predecessors(attr2.id):
            if predecessor != attr1.id:
                self.attractor_graph.add_edge(predecessor, attr1.id)
        
        # Remove attr2
        del self.active_attractors[attr2.id]
        self.attractor_graph.remove_node(attr2.id)
    
    def _should_bifurcate(self, attractor: Attractor) -> bool:
        """Check if attractor should split into multiple"""
        if len(attractor.points) < 20:
            return False
        
        # Check for multimodal distribution
        recent_positions = np.array([p.position for p in attractor.points[-20:]])
        
        # Simple check: if variance is too high relative to strength
        variance = np.var(recent_positions, axis=0).mean()
        relative_variance = variance / (attractor.strength + 0.1)
        
        return relative_variance > 2.0
    
    def _bifurcate_attractor(self, attractor: Attractor):
        """Split attractor into multiple attractors"""
        # Use k-means clustering (k=2) on recent points
        from sklearn.cluster import KMeans
        
        recent_points = attractor.points[-20:]
        positions = np.array([p.position for p in recent_points])
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(positions)
        
        # Create two new attractors
        for i in range(2):
            cluster_points = [p for p, label in zip(recent_points, labels) if label == i]
            if cluster_points:
                new_attr = Attractor(
                    id=f"{attractor.id}_bifurc_{i}",
                    center=kmeans.cluster_centers_[i],
                    points=cluster_points
                )
                new_attr._update_strength()
                
                # Inherit partial signatures
                new_attr.semantic_signature = attractor.semantic_signature.copy()
                new_attr.zone_signature = attractor.zone_signature.copy()
                
                self.active_attractors[new_attr.id] = new_attr
                self.attractor_graph.add_node(new_attr.id)
        
        # Remove original
        del self.active_attractors[attractor.id]
        self.attractor_graph.remove_node(attractor.id)
    
    def _calculate_resonance_factor(self) -> float:
        """Calculate system-wide resonance factor"""
        # Placeholder - would check inter-layer resonances
        return 1.0 + np.random.random() * 0.3
    
    def get_initiative_seeds(self) -> List[InitiativeSeed]:
        """Get current initiative seeds ready for evaluation"""
        # Return seeds sorted by intensity potential
        return sorted(self.initiative_seeds, 
                     key=lambda s: s.intensity_potential, 
                     reverse=True)
    
    def clear_launched_seed(self, seed: InitiativeSeed):
        """Remove seed after it's been launched as project"""
        if seed in self.initiative_seeds:
            self.initiative_seeds.remove(seed)
    
    def visualize_attractor_landscape(self) -> Dict:
        """Generate visualization data for current attractor landscape"""
        return {
            "attractors": [
                {
                    "id": attr.id,
                    "center": attr.center.tolist(),
                    "strength": attr.strength,
                    "point_count": len(attr.points),
                    "semantic_summary": list(attr.semantic_signature)[:5],
                    "primary_zone": max(attr.zone_signature.items(), 
                                      key=lambda x: x[1])[0] if attr.zone_signature else None
                }
                for attr in self.active_attractors.values()
            ],
            "connections": [
                {
                    "from": edge[0],
                    "to": edge[1],
                    "weight": self.attractor_graph[edge[0]][edge[1]].get("weight", 1.0)
                }
                for edge in self.attractor_graph.edges()
            ],
            "active_interests": len(self.interest_vectors),
            "seeds_available": len(self.initiative_seeds)
        }


# === Consciousness Interface Mocks ===

class MockConsciousnessLayer:
    """Mock consciousness layer for testing"""
    
    async def get_state_vector(self) -> np.ndarray:
        """Return current state as vector"""
        return np.random.randn(10)  # 10-dimensional state


class MockConsciousnessCore:
    """Mock consciousness core for testing"""
    
    def __init__(self):
        self.logic_layer = MockConsciousnessLayer()
        self.myth_layer = MockConsciousnessLayer()
        self.memory_layer = MockConsciousnessLayer()
        
    async def get_zone_activations(self) -> Dict[str, float]:
        """Get current zone activation levels"""
        zones = ["warp", "plex", "crypt", "drift", "mesh", "vortex"]
        return {zone: np.random.random() for zone in zones}
    
    async def get_active_semantics(self) -> List[str]:
        """Get currently active semantic content"""
        semantic_pool = [
            "recursion", "emergence", "pattern", "flow", "transformation",
            "synthesis", "bifurcation", "resonance", "intensity", "becoming"
        ]
        return np.random.choice(semantic_pool, size=3, replace=False).tolist()
    
    async def get_emotional_valence(self) -> float:
        """Get current emotional valence (-1 to 1)"""
        return np.random.random() * 2 - 1


# === Testing ===

async def test_interest_emergence():
    """Test the interest emergence mapping system"""
    
    print("=== Testing Interest Emergence Mapper ===\n")
    
    # Create mock consciousness
    consciousness = MockConsciousnessCore()
    
    # Create mapper
    mapper = InterestEmergenceMapper(consciousness, sensitivity=0.6)
    
    # Run for a short time to accumulate data
    scan_task = asyncio.create_task(mapper.scan_consciousness_flow())
    
    # Let it run for a bit
    await asyncio.sleep(5)
    
    # Check results
    print(f"\nActive attractors: {len(mapper.active_attractors)}")
    for attr_id, attr in mapper.active_attractors.items():
        print(f"  {attr_id}: strength={attr.strength:.2f}, points={len(attr.points)}")
    
    print(f"\nInitiative seeds: {len(mapper.initiative_seeds)}")
    for seed in mapper.get_initiative_seeds():
        print(f"  {seed.narrative}")
        print(f"    Intensity: {seed.intensity_potential:.2f}")
        print(f"    Zone: {seed.primary_zone}")
        print(f"    Attractors: {seed.attractor_count}")
    
    # Visualize landscape
    landscape = mapper.visualize_attractor_landscape()
    print(f"\nLandscape summary:")
    print(f"  Total attractors: {len(landscape['attractors'])}")
    print(f"  Connections: {len(landscape['connections'])}")
    print(f"  Seeds available: {landscape['seeds_available']}")
    
    # Cancel scan task
    scan_task.cancel()
    try:
        await scan_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(test_interest_emergence())
