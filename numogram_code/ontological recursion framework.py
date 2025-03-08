"""
Ontological Recursion Framework
===============================
A framework for modeling reality as a self-referential system where consciousness 
participates in creating the conditions of its own emergence.

Compatible with Process Metaphysics, Morphogenesis, and Schizoanalytic thinking,
with QBism as a meta-framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp
import random
from IPython.display import clear_output
import time
import pandas as pd
import seaborn as sns

class ActualOccasion:
    """
    Represents Whitehead's concept of an actual occasion - a basic unit
    of process reality that comes into being, reaches satisfaction, and perishes.
    """
    def __init__(self, id, intensity=1.0, position=None):
        self.id = id
        self.intensity = intensity  # Intensity of experience/feeling
        self.prehensions = []  # Connections to other occasions
        self.position = position if position else np.random.rand(2)  # Spatial position for visualization
        self.satisfaction = 0.0  # Degree of completion
        self.perished = False
        self.data = {}  # Container for emergence properties
        
    def prehend(self, other, strength):
        """Create a prehension (feeling) of another occasion"""
        self.prehensions.append((other, strength))
        return self
    
    def update(self, dt):
        """Process toward satisfaction"""
        if self.perished:
            return
            
        # Increase satisfaction based on intensity and prehensions
        prehension_factor = sum(strength for _, strength in self.prehensions)
        self.satisfaction += dt * self.intensity * (1 + 0.1 * prehension_factor)
        
        if self.satisfaction >= 1.0:
            self.perish()
            
    def perish(self):
        """Occasion reaches completion and perishes"""
        self.perished = True
        self.satisfaction = 1.0


class ProcessLevel:
    """
    Represents the level of process metaphysics where actual occasions
    form societies and create emergent patterns.
    """
    def __init__(self):
        self.occasions = []
        self.next_id = 0
        self.historic_occasions = []  # Archive of perished occasions
        
    def create_occasion(self, intensity=None, position=None):
        """Generate a new actual occasion"""
        if intensity is None:
            intensity = random.uniform(0.5, 1.5)
            
        occasion = ActualOccasion(self.next_id, intensity, position)
        self.occasions.append(occasion)
        self.next_id += 1
        return occasion
        
    def create_society(self, num_occasions, connection_density=0.3):
        """Create a society of interconnected occasions"""
        new_occasions = [self.create_occasion() for _ in range(num_occasions)]
        
        # Create prehensions between occasions
        for i, occasion in enumerate(new_occasions):
            for j, other in enumerate(new_occasions):
                if i != j and random.random() < connection_density:
                    strength = random.uniform(0.1, 0.9)
                    occasion.prehend(other, strength)
                    
        return new_occasions
    
    def update(self, dt):
        """Update all occasions"""
        # Update existing occasions
        for occasion in self.occasions:
            occasion.update(dt)
            
        # Move perished occasions to historic archive
        active_occasions = []
        for occasion in self.occasions:
            if occasion.perished:
                self.historic_occasions.append(occasion)
            else:
                active_occasions.append(occasion)
                
        self.occasions = active_occasions
        
        # Create new occasions based on historic patterns (morphic resonance)
        if len(self.historic_occasions) > 10 and random.random() < 0.2:
            self.create_occasion_from_pattern()
            
    def create_occasion_from_pattern(self):
        """Create new occasions that resonate with past patterns"""
        template = random.choice(self.historic_occasions)
        new_occasion = self.create_occasion(
            intensity=template.intensity * random.uniform(0.8, 1.2),
            position=template.position + np.random.normal(0, 0.1, 2)
        )
        
        # Create prehensions based on template's pattern
        for historic in self.historic_occasions[-20:]:
            if random.random() < 0.3:
                strength = random.uniform(0.1, 0.5)
                new_occasion.prehend(historic, strength)
                
        return new_occasion


class ObserverLevel:
    """
    Represents an observer with Bayesian priors who participates
    in creating reality through measurement/observation.
    """
    def __init__(self, process_level):
        self.process_level = process_level
        self.priors = {}  # Bayesian priors about the system
        self.observations = []  # History of observations/measurements
        self.belief_strength = 0.7  # How strongly beliefs affect reality
        
    def set_prior(self, key, value):
        """Set a Bayesian prior belief"""
        self.priors[key] = value
        
    def observe(self, target_property, uncertainty=0.1):
        """
        Make an observation that collapses possibilities
        into actualities with QBism-inspired approach
        """
        # Record the observation
        observation = {
            'property': target_property,
            'time': time.time(),
            'uncertainty': uncertainty
        }
        self.observations.append(observation)
        
        # The act of observation affects the system (QBism)
        if target_property == 'intensity':
            # Observation affects intensity of new occasions
            if 'expected_intensity' in self.priors:
                expected = self.priors['expected_intensity']
                for _ in range(3):
                    # Create occasions biased toward the observer's expectation
                    new_intensity = expected * (1 + np.random.normal(0, uncertainty))
                    self.process_level.create_occasion(intensity=new_intensity)
                    
        elif target_property == 'connection':
            # Observation affects connection patterns
            if 'expected_connection' in self.priors:
                expected_density = self.priors['expected_connection']
                self.process_level.create_society(5, connection_density=expected_density)
        
        return observation
    
    def update_beliefs(self):
        """Update beliefs based on observations (Bayesian update)"""
        # Simplified Bayesian update
        if len(self.observations) > 5:
            recent_obs = self.observations[-5:]
            
            # Update intensity prior if we have observations about it
            intensity_obs = [o for o in recent_obs if o['property'] == 'intensity']
            if intensity_obs:
                # Calculate new expected intensity with basic Bayesian update
                if 'expected_intensity' not in self.priors:
                    self.priors['expected_intensity'] = 1.0
                
                prior = self.priors['expected_intensity']
                for obs in intensity_obs:
                    # Very simplified Bayesian update
                    actual = sum(o.intensity for o in self.process_level.occasions) / len(self.process_level.occasions)
                    posterior = prior * 0.8 + actual * 0.2  # Weighted update
                    prior = posterior
                
                self.priors['expected_intensity'] = prior
                
            # Similar updates could be done for other properties


class SystemLevel:
    """
    Represents the emergent system level where patterns and feedback loops emerge
    from the interactions of occasions and observers.
    """
    def __init__(self, process_level, observer_level):
        self.process_level = process_level
        self.observer_level = observer_level
        self.emergent_patterns = {}
        self.feedback_loops = []
        self.system_state = {}
        self.history = []  # Track system evolution
        
    def detect_patterns(self):
        """Detect emergent patterns in the system"""
        occasions = self.process_level.occasions
        
        if len(occasions) < 5:
            return {}
            
        # Calculate clustering coefficient as a measure of pattern formation
        positions = np.array([o.position for o in occasions])
        if len(positions) > 0:
            from scipy.spatial import distance
            dist_matrix = distance.cdist(positions, positions)
            
            # Find clusters with a simple threshold approach
            threshold = 0.3
            clusters = []
            assigned = set()
            
            for i, occasion in enumerate(occasions):
                if i in assigned:
                    continue
                    
                cluster = [i]
                assigned.add(i)
                
                for j in range(len(occasions)):
                    if j not in assigned and dist_matrix[i, j] < threshold:
                        cluster.append(j)
                        assigned.add(j)
                        
                if len(cluster) > 1:
                    clusters.append(cluster)
            
            self.emergent_patterns['clusters'] = clusters
            self.emergent_patterns['cluster_count'] = len(clusters)
        
        # Calculate average satisfaction as another emergent property
        avg_satisfaction = sum(o.satisfaction for o in occasions) / len(occasions) if occasions else 0
        self.emergent_patterns['avg_satisfaction'] = avg_satisfaction
        
        # Detect oscillation patterns in satisfaction
        if len(self.history) > 10:
            satisfactions = [h.get('avg_satisfaction', 0) for h in self.history[-10:]]
            from scipy import signal
            peaks, _ = signal.find_peaks(satisfactions)
            self.emergent_patterns['oscillation_frequency'] = len(peaks)
            
        return self.emergent_patterns
    
    def analyze_feedback_loops(self):
        """Identify feedback loops between levels"""
        self.feedback_loops = []
        
        # Observer → Process feedback
        if 'expected_intensity' in self.observer_level.priors:
            expected = self.observer_level.priors['expected_intensity']
            actual = np.mean([o.intensity for o in self.process_level.occasions]) if self.process_level.occasions else 0
            
            loop = {
                'type': 'observer_process',
                'expected': expected,
                'actual': actual,
                'difference': abs(expected - actual),
                'strength': 1.0 - (0.5 * abs(expected - actual))  # Normalized strength
            }
            
            self.feedback_loops.append(loop)
            
        # Process → System feedback
        if 'clusters' in self.emergent_patterns:
            cluster_count = self.emergent_patterns['cluster_count']
            
            # How clustering affects new occasion formation
            loop = {
                'type': 'process_system',
                'cluster_count': cluster_count,
                'new_occasion_rate': 0.1 * cluster_count,  # More clusters → more new occasions
                'strength': min(1.0, 0.2 * cluster_count)
            }
            
            self.feedback_loops.append(loop)
            
        # System → Observer feedback
        if len(self.feedback_loops) > 0:
            avg_loop_strength = np.mean([l['strength'] for l in self.feedback_loops])
            
            loop = {
                'type': 'system_observer',
                'avg_loop_strength': avg_loop_strength,
                'belief_adjustment': 0.1 * avg_loop_strength,
                'strength': avg_loop_strength
            }
            
            self.feedback_loops.append(loop)
            
        return self.feedback_loops
    
    def update(self):
        """Update the system level analysis"""
        patterns = self.detect_patterns()
        loops = self.analyze_feedback_loops()
        
        # Calculate overall system state
        self.system_state = {
            'time': time.time(),
            'occasion_count': len(self.process_level.occasions),
            'historic_count': len(self.process_level.historic_occasions),
            'avg_satisfaction': patterns.get('avg_satisfaction', 0),
            'cluster_count': patterns.get('cluster_count', 0),
            'loop_count': len(loops),
            'strongest_loop': max([l['strength'] for l in loops]) if loops else 0
        }
        
        # Record history
        self.history.append(dict(self.system_state))
        
        # Implement strange loop - system affects itself
        self._implement_strange_loop()
        
        return self.system_state
    
    def _implement_strange_loop(self):
        """Implement a strange loop where the system affects its own processes"""
        if not self.history or len(self.history) < 10:
            return
            
        # Calculate rate of change in system complexity
        recent = self.history[-5:]
        complexity_change = recent[-1]['cluster_count'] - recent[0]['cluster_count']
        
        # The system's complexity affects how new occasions are created
        if complexity_change > 0:
            # Growing complexity - create more diverse occasions
            for _ in range(int(complexity_change)):
                self.process_level.create_occasion(
                    intensity=random.uniform(0.5, 2.0)  # More variation
                )
        else:
            # Simplified complexity - create more uniform occasions
            for _ in range(max(1, -int(complexity_change))):
                self.process_level.create_occasion(
                    intensity=1.0 + random.uniform(-0.1, 0.1)  # Less variation
                )


class OntologicalRecursion:
    """
    Main framework class that integrates all levels and implements
    the recursive operations between them.
    """
    def __init__(self):
        self.process_level = ProcessLevel()
        self.observer_level = ObserverLevel(self.process_level)
        self.system_level = SystemLevel(self.process_level, self.observer_level)
        self.time = 0
        self.dt = 0.1
        self.metrics = {
            'time': [],
            'occasion_count': [],
            'avg_satisfaction': [],
            'cluster_count': [],
            'loop_strength': [],
            'observer_influence': []
        }
        
    def initialize(self, num_societies=3, occasions_per_society=5):
        """Initialize the system with societies of occasions"""
        for _ in range(num_societies):
            self.process_level.create_society(occasions_per_society)
            
        # Set initial observer priors
        self.observer_level.set_prior('expected_intensity', 1.0)
        self.observer_level.set_prior('expected_connection', 0.4)
        
    def update(self):
        """Update all levels of the system"""
        # Update process level
        self.process_level.update(self.dt)
        
        # Observer makes observations
        if random.random() < 0.3:
            property_to_observe = random.choice(['intensity', 'connection'])
            self.observer_level.observe(property_to_observe)
        
        # Update observer beliefs
        self.observer_level.update_beliefs()
        
        # Update system level
        system_state = self.system_level.update()
        
        # Implement upward causation (process → observer)
        self._upward_causation()
        
        # Implement downward causation (observer → process)
        self._downward_causation()
        
        # Record metrics
        self._record_metrics()
        
        self.time += self.dt
        return system_state
    
    def _upward_causation(self):
        """
        Implement upward causation where lower levels affect higher levels
        """
        # Process level affects Observer level
        if self.process_level.occasions:
            # Calculate average intensity
            avg_intensity = sum(o.intensity for o in self.process_level.occasions) / len(self.process_level.occasions)
            
            # This affects observer's priors (reality shapes observer beliefs)
            if 'expected_intensity' in self.observer_level.priors:
                current = self.observer_level.priors['expected_intensity']
                # Gradual shift of expectations based on reality
                self.observer_level.priors['expected_intensity'] = 0.95 * current + 0.05 * avg_intensity
    
    def _downward_causation(self):
        """
        Implement downward causation where higher levels affect lower levels
        """
        # Observer level affects Process level
        belief_strength = self.observer_level.belief_strength
        
        if 'expected_intensity' in self.observer_level.priors and random.random() < 0.2:
            # Observer expectations shape new occasions (like QBism)
            expected = self.observer_level.priors['expected_intensity']
            occasion = self.process_level.create_occasion(
                intensity=expected * (1 + random.uniform(-0.1, 0.1))
            )
            
            # Tag this occasion as influenced by observation
            occasion.data['observer_influenced'] = True
    
    def _record_metrics(self):
        """Record metrics for visualization and analysis"""
        self.metrics['time'].append(self.time)
        self.metrics['occasion_count'].append(len(self.process_level.occasions))
        
        if 'avg_satisfaction' in self.system_level.emergent_patterns:
            self.metrics['avg_satisfaction'].append(self.system_level.emergent_patterns['avg_satisfaction'])
        else:
            self.metrics['avg_satisfaction'].append(0)
            
        if 'cluster_count' in self.system_level.emergent_patterns:
            self.metrics['cluster_count'].append(self.system_level.emergent_patterns['cluster_count'])
        else:
            self.metrics['cluster_count'].append(0)
            
        if self.system_level.feedback_loops:
            self.metrics['loop_strength'].append(max(l['strength'] for l in self.system_level.feedback_loops))
        else:
            self.metrics['loop_strength'].append(0)
            
        # Calculate observer influence
        observer_occasions = sum(1 for o in self.process_level.occasions if o.data.get('observer_influenced', False))
        total_occasions = len(self.process_level.occasions)
        influence = observer_occasions / total_occasions if total_occasions > 0 else 0
        self.metrics['observer_influence'].append(influence)
    
    def run_simulation(self, steps=100, visualize=True, visualize_step=5):
        """Run the simulation for a specified number of steps"""
        for step in range(steps):
            self.update()
            
            if visualize and step % visualize_step == 0:
                self.visualize()
                time.sleep(0.1)
                
        if visualize:
            self.plot_metrics()
            self.visualize_system_state()
    
    def visualize(self):
        """Visualize the current state of the system"""
        clear_output(wait=True)
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Occasion network
        plt.subplot(2, 2, 1)
        self._plot_occasion_network()
        plt.title("Occasion Network")
        
        # Plot 2: System metrics
        plt.subplot(2, 2, 2)
        self._plot_current_metrics()
        plt.title("Current System Metrics")
        
        # Plot 3: Emergence visualization
        plt.subplot(2, 2, 3)
        self._plot_emergence()
        plt.title("Emergent Patterns")
        
        # Plot 4: Recursive loops
        plt.subplot(2, 2, 4)
        self._plot_recursive_loops()
        plt.title("Recursive Causality")
        
        plt.tight_layout()
        plt.show()
    
    def _plot_occasion_network(self):
        """Plot the network of occasions and their prehensions"""
        G = nx.DiGraph()
        
        # Add nodes
        for occasion in self.process_level.occasions:
            G.add_node(occasion.id, 
                      size=occasion.intensity * 300,
                      satisfaction=occasion.satisfaction,
                      observer_influenced=occasion.data.get('observer_influenced', False))
            
        # Add edges for prehensions
        for occasion in self.process_level.occasions:
            for other, strength in occasion.prehensions:
                if not other.perished and other.id in G:
                    G.add_edge(occasion.id, other.id, weight=strength)
        
        if not G:
            plt.text(0.5, 0.5, "No active occasions", ha='center', va='center')
            return
            
        # Position nodes
        pos = {o.id: o.position for o in self.process_level.occasions}
        
        # Node colors based on observer influence
        colors = ['red' if G.nodes[n]['observer_influenced'] else 'blue' for n in G.nodes]
        
        # Node sizes based on intensity
        sizes = [G.nodes[n]['size'] for n in G.nodes]
        
        # Edge weights based on prehension strength
        edge_weights = [G.edges[e]['weight'] * 3 for e in G.edges]
        
        # Draw the network
        nx.draw(G, pos, with_labels=True, node_color=colors, 
                node_size=sizes, width=edge_weights, arrows=True,
                edge_color='gray', alpha=0.7)
    
    def _plot_current_metrics(self):
        """Plot current system metrics"""
        metrics = {
            'Active Occasions': len(self.process_level.occasions),
            'Historic Occasions': len(self.process_level.historic_occasions),
            'Observer Beliefs': len(self.observer_level.priors),
            'Emergent Patterns': len(self.system_level.emergent_patterns),
            'Feedback Loops': len(self.system_level.feedback_loops)
        }
        
        plt.bar(metrics.keys(), metrics.values(), color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    def _plot_emergence(self):
        """Visualize emergent patterns"""
        if not self.process_level.occasions:
            plt.text(0.5, 0.5, "No active occasions", ha='center', va='center')
            return
            
        # Create a heatmap of occasion positions
        positions = np.array([o.position for o in self.process_level.occasions])
        
        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=10, range=[[0, 1], [0, 1]]
        )
        
        # Plot the heatmap
        plt.imshow(heatmap.T, origin='lower', extent=[0, 1, 0, 1], 
                   cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Occasion Density')
        
        # Overlay occasion positions
        observer_influenced = [o for o in self.process_level.occasions 
                              if o.data.get('observer_influenced', False)]
        regular = [o for o in self.process_level.occasions 
                  if not o.data.get('observer_influenced', False)]
        
        if regular:
            reg_pos = np.array([o.position for o in regular])
            plt.scatter(reg_pos[:, 0], reg_pos[:, 1], color='blue', alpha=0.7, 
                        label='Regular Occasions')
            
        if observer_influenced:
            obs_pos = np.array([o.position for o in observer_influenced])
            plt.scatter(obs_pos[:, 0], obs_pos[:, 1], color='red', alpha=0.7,
                       label='Observer-Influenced')
        
        plt.legend()
    
    def _plot_recursive_loops(self):
        """Visualize the recursive causal loops"""
        # Create a circular diagram of causal influences
        if not self.system_level.feedback_loops:
            plt.text(0.5, 0.5, "No feedback loops detected", ha='center', va='center')
            return
            
        # Create a directed graph for the loops
        G = nx.DiGraph()
        
        # Add nodes for each level
        G.add_node("Process", pos=(0, 0))
        G.add_node("Observer", pos=(1, 0))
        G.add_node("System", pos=(0.5, 0.866))  # Top of triangle
        
        # Add edges for each feedback loop
        for loop in self.system_level.feedback_loops:
            if loop['type'] == 'observer_process':
                G.add_edge("Observer", "Process", weight=loop['strength'], type=loop['type'])
            elif loop['type'] == 'process_system':
                G.add_edge("Process", "System", weight=loop['strength'], type=loop['type'])
            elif loop['type'] == 'system_observer':
                G.add_edge("System", "Observer", weight=loop['strength'], type=loop['type'])
        
        # Add the strange loop
        G.add_edge("System", "Process", weight=0.5, type='strange_loop')
        
        # Position nodes in a triangle
        pos = nx.get_node_attributes(G, 'pos')
        
        # Edge weights and colors
        edge_weights = [G.edges[e]['weight'] * 5 for e in G.edges]
        edge_colors = ['red' if G.edges[e]['type'] == 'observer_process' else
                      'blue' if G.edges[e]['type'] == 'process_system' else
                      'green' if G.edges[e]['type'] == 'system_observer' else
                      'purple' for e in G.edges]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightgray', 
                node_size=3000, width=edge_weights, edge_color=edge_colors,
                connectionstyle='arc3,rad=0.2', arrows=True)
        
        # Add a title with the strongest loop
        strongest = max(self.system_level.feedback_loops, key=lambda x: x['strength'])
        plt.title(f"Strongest Loop: {strongest['type']} (Strength: {strongest['strength']:.2f})")
    
    def plot_metrics(self):
        """Plot the recorded metrics over time"""
        plt.figure(figsize=(14, 10))
        
        # Convert metrics to DataFrame for easier plotting
        df = pd.DataFrame(self.metrics)
        
        # Plot 1: Occasion count over time
        plt.subplot(2, 2, 1)
        plt.plot(df['time'], df['occasion_count'], 'b-', linewidth=2)
        plt.title('Active Occasions Over Time')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average satisfaction over time
        plt.subplot(2, 2, 2)
        plt.plot(df['time'], df['avg_satisfaction'], 'g-', linewidth=2)
        plt.title('Average Satisfaction Over Time')
        plt.xlabel('Time')
        plt.ylabel('Satisfaction')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cluster count over time
        plt.subplot(2, 2, 3)
        plt.plot(df['time'], df['cluster_count'], 'r-', linewidth=2)
        plt.title('Emergent Clusters Over Time')
        plt.xlabel('Time')
        plt.ylabel('Cluster Count')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Loop strength and observer influence
        plt.subplot(2, 2, 4)
        plt.plot(df['time'], df['loop_strength'], 'c-', linewidth=2, label='Loop Strength')
        plt.plot(df['time'], df['observer_influence'], 'm-', linewidth=2, label='Observer Influence')
        plt.title('Feedback Loop Dynamics')
        plt.xlabel('Time')
        plt.ylabel('Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_system_state(self):
        """Create a more detailed visualization of the current system state"""
        if not self.system_level.history:
            return
            
        plt.figure(figsize=(16, 12))
        
        # Create a DataFrame of the system history
        df = pd.DataFrame(self.system_level.history)
        
        # Plot 1: System state evolution heatmap
        plt.subplot(2, 2, 1)
        # Normalize each column for better visualization
        normalized_df = df.copy()
        for col in df.columns:
            if col != 'time' and df[col].max() > 0:
                normalized_df[col] = df[col] / df[col].max()
        
        # Select numeric columns for the heatmap
        numeric_cols = ['occasion_count', 'historic_count', 'avg_satisfaction', 
                       'cluster_count', 'loop_count', 'strongest_loop']
        heatmap_data = normalized_df[numeric_cols].T
        
        sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Normalized Value'})
        plt.title('System State Evolution')
        plt.xlabel('Time Step')
        plt.ylabel('Metric')
        
        # Plot 2: Strange loop visualization
        plt.subplot(2, 2, 2)
        self._plot_strange_loop_dynamics()
        
        # Plot 3: QBism influence visualization
        plt.subplot(2, 2, 3)
        self._plot_qbism_influence()
        
        # Plot 4: System complexity
        plt.subplot(2, 2, 4)
        self._plot_system_complexity()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_strange_loop_dynamics(self):
        """Visualize the self-reference dynamics (strange loop)"""
        if len(self.metrics['time']) < 10:
            plt.text(0.5, 0.5, "Not enough data for strange loop analysis", 
                    ha='center', va='center')
            return
            
        # Calculate how the system affects itself (feedback from t to t+n)
        times = self.metrics['time'][:-5]  # Exclude last few points for lag
        
        # Calculate complexity at time t
        complexity_t = self.metrics['cluster_count'][:-5]
        
        # Calculate observer influence at time t+5 (with lag)
        influence_t_plus_5 = self.metrics['observer_influence'][5:]
        
        # Plot the relationship
        plt.scatter(complexity_t, influence_t_plus_5, c=times, cmap='plasma', alpha=0.7)
        plt.colorbar(label='Time')
        plt.title('Strange Loop: How System Complexity Affects Future Observer Influence')
        plt.xlabel('System Complexity at time t')
        plt.ylabel('Observer Influence at time t+5')
        plt.grid(True, alpha=0.3)
    
    def _plot_qbism_influence(self):
