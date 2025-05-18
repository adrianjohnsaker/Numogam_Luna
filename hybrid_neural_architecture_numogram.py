
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import uuid
import datetime
import copy
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class GradientMemory:
    """Store gradient information for hybrid learning"""
    parameter_name: str
    mean_gradient: np.ndarray
    gradient_variance: np.ndarray
    update_count: int = 0


class HybridNeuralArchitecture(nn.Module):
    """
    Neural network architecture that combines gradient-based and
    evolutionary learning approaches
    """
    def __init__(self, input_size=30, hidden_sizes=[20, 10], output_size=9):
        super(HybridNeuralArchitecture, self).__init__()
        
        # Define network layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine layers into sequential model
        self.model = nn.Sequential(*layers)
        
        # Neuromodulation parameters (affected by emotional state)
        self.neuromodulation = {
            "learning_rate_mod": 1.0,
            "noise_level": 0.1,
            "dropout_prob": 0.0,
            "attention_focus": None  # Will hold indices of inputs to focus on
        }
        
        # Hybrid learning parameters
        self.gradient_memory = {}  # Stores gradient information for hybrid updates
        self.evolution_enabled = True
        self.gradient_learning_enabled = True
    
    def forward(self, x):
        """Forward pass with neuromodulation"""
        # Apply attention focus if specified
        if self.neuromodulation["attention_focus"] is not None:
            # Enhance specified features
            focus_mask = torch.zeros_like(x)
            for idx in self.neuromodulation["attention_focus"]:
                if idx < x.size(0):
                    focus_mask[idx] = 1.0
            
            # Apply focus by scaling inputs
            x = x * (1.0 + focus_mask)
        
        # Apply noise based on neuromodulation
        if self.training and self.neuromodulation["noise_level"] > 0:
            noise = torch.randn_like(x) * self.neuromodulation["noise_level"]
            x = x + noise
        
        # Apply dropout if specified
        if self.training and self.neuromodulation["dropout_prob"] > 0:
            dropout = nn.Dropout(p=self.neuromodulation["dropout_prob"])
            x = dropout(x)
        
        # Forward through model
        return self.model(x)
    
    def set_neuromodulation(self, emotional_state: Dict = None):
        """
        Set neuromodulation parameters based on emotional state
        This affects how the network learns and processes inputs
        """
        if emotional_state is None:
            # Reset to defaults
            self.neuromodulation = {
                "learning_rate_mod": 1.0,
                "noise_level": 0.1,
                "dropout_prob": 0.0,
                "attention_focus": None
            }
            return
            
        # Extract emotional factors
        primary_emotion = emotional_state.get("primary_emotion", "neutral")
        intensity = emotional_state.get("intensity", 0.5)
        
        # Adjust learning rate based on emotion
        if primary_emotion in ["surprise", "curiosity", "awe"]:
            # High arousal, high learning
            self.neuromodulation["learning_rate_mod"] = 1.0 + (intensity * 0.5)
        elif primary_emotion in ["fear", "anger"]:
            # High arousal, selective learning
            self.neuromodulation["learning_rate_mod"] = 1.0
            self.neuromodulation["attention_focus"] = [11, 15]  # Focus on fear and anger inputs
        elif primary_emotion in ["sadness", "disgust"]:
            # Low learning rate
            self.neuromodulation["learning_rate_mod"] = max(0.5, 1.0 - (intensity * 0.5))
        elif primary_emotion in ["joy", "trust", "serenity"]:
            # Balanced learning, low noise
            self.neuromodulation["learning_rate_mod"] = 1.0
            self.neuromodulation["noise_level"] = 0.05
        else:
            # Default settings
            self.neuromodulation["learning_rate_mod"] = 1.0
            self.neuromodulation["noise_level"] = 0.1
        
        # Set dropout based on intensity
        self.neuromodulation["dropout_prob"] = intensity * 0.2  # Max 20% dropout
        
        # Set attention focus based on emotional spectrum if available
        if "emotional_spectrum" in emotional_state:
            spectrum = emotional_state["emotional_spectrum"]
            
            # Find top 2 emotions
            top_emotions = sorted(spectrum.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Map emotions to input indices (these should match input feature mapping)
            emotion_indices = {
                "joy": 9, "trust": 10, "fear": 11, "surprise": 12,
                "sadness": 13, "disgust": 14, "anger": 15, "anticipation": 16,
                "curiosity": 17
            }
            
            # Set focus on top emotion features
            focus_indices = []
            for emotion, _ in top_emotions:
                if emotion in emotion_indices:
                    focus_indices.append(emotion_indices[emotion])
            
            if focus_indices:
                self.neuromodulation["attention_focus"] = focus_indices
    
    def store_gradient_information(self):
        """Store gradient information for hybrid learning"""
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_np = param.grad.detach().cpu().numpy()
                
                if name not in self.gradient_memory:
                    self.gradient_memory[name] = GradientMemory(
                        parameter_name=name,
                        mean_gradient=np.zeros_like(grad_np),
                        gradient_variance=np.zeros_like(grad_np),
                        update_count=0
                    )
                
                # Update running statistics
                memory = self.gradient_memory[name]
                memory.update_count += 1
                
                # Update mean using incremental formula
                old_mean = memory.mean_gradient.copy()
                memory.mean_gradient += (grad_np - memory.mean_gradient) / memory.update_count
                
                # Update variance using Welford's online algorithm
                if memory.update_count > 1:
                    memory.gradient_variance += (grad_np - old_mean) * (grad_np - memory.mean_gradient)
    
    def get_gradient_guided_mutation_mask(self, mutation_rate=0.1):
        """
        Generate a mutation mask guided by gradient information
        Areas with higher gradient variance should be mutated more
        """
        masks = {}
        
        for name, param in self.named_parameters():
            param_np = param.detach().cpu().numpy()
            mask = np.random.random(param_np.shape) < mutation_rate
            
            # If we have gradient information, use it to guide mutations
            if name in self.gradient_memory and self.gradient_memory[name].update_count > 10:
                memory = self.gradient_memory[name]
                
                # Normalize variance to 0-1 range
                if np.sum(memory.gradient_variance) > 0:
                    normalized_variance = memory.gradient_variance / np.max(memory.gradient_variance)
                    
                    # Areas with higher variance get higher mutation probability
                    additional_prob = normalized_variance * mutation_rate
                    additional_mask = np.random.random(param_np.shape) < additional_prob
                    
                    # Combine masks
                    mask = mask | additional_mask
            
            masks[name] = mask
            
        return masks
    
    def apply_mutations(self, mutation_rate=0.1, mutation_scale=0.1):
        """Apply mutations to network weights"""
        # Get gradient-guided mutation masks
        mutation_masks = self.get_gradient_guided_mutation_mask(mutation_rate)
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in mutation_masks:
                    mask = torch.tensor(mutation_masks[name], device=param.device)
                    
                    # Generate mutation noise
                    noise = torch.randn_like(param) * mutation_scale
                    
                    # Apply mutations selectively using mask
                    param.data += noise * mask


class EvolutionaryPopulation:
    """Manages a population of neural networks evolved through genetic algorithms"""
    
    def __init__(self, 
                model_class, 
                model_args, 
                population_size=20,
                mutation_rate=0.1,
                crossover_rate=0.7,
                elitism_count=2):
        
        self.model_class = model_class
        self.model_args = model_args
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Track evolution progress
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def _initialize_population(self):
        """Initialize a population of neural networks"""
        population = []
        
        for i in range(self.population_size):
            # Create new model instance
            model = self.model_class(**self.model_args)
            
            # Create individual
            individual = {
                "id": str(uuid.uuid4()),
                "model": model,
                "fitness": 0.0,
                "novelty": 0.0,
                "age": 0,
                "creation_time": datetime.datetime.utcnow().isoformat(),
                "ancestry": [],
                "mutation_history": []
            }
            
            population.append(individual)
        
        return population
    
    def evaluate_population(self, evaluation_func):
        """
        Evaluate all individuals in the population
        evaluation_func should take a model and return fitness
        """
        for individual in self.population:
            individual["fitness"] = evaluation_func(individual["model"])
        
        # Update history
        fitnesses = [ind["fitness"] for ind in self.population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
    
    def calculate_novelty(self):
        """Calculate novelty scores based on parameter space distance"""
        # Extract parameters as flattened arrays
        param_vectors = []
        
        for individual in self.population:
            params = []
            for param in individual["model"].parameters():
                params.append(param.detach().cpu().numpy().flatten())
            
            # Concatenate all parameters
            param_vectors.append(np.concatenate(params))
        
        # Calculate pairwise distances
        for i, individual in enumerate(self.population):
            # Calculate distance to k-nearest neighbors
            distances = []
            
            for j, other_vector in enumerate(param_vectors):
                if i != j:
                    dist = np.linalg.norm(param_vectors[i] - other_vector)
                    distances.append(dist)
            
            # Sort distances
            distances.sort()
            
            # Average of k-nearest
            k = min(3, len(distances))
            if k > 0:
                individual["novelty"] = sum(distances[:k]) / k
            else:
                individual["novelty"] = 0.0
    
    def selection(self, tournament_size=3, use_novelty=True):
        """Select parent using tournament selection"""
        # Choose random individuals for tournament
        tournament = random.sample(self.population, tournament_size)
        
        # Sort by combined fitness and novelty
        if use_novelty:
            tournament.sort(key=lambda x: (0.7 * x["fitness"] + 0.3 * x["novelty"]), reverse=True)
        else:
            tournament.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Return winner
        return tournament[0]
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parent networks"""
        # Create a new model instance
        child_model = self.model_class(**self.model_args)
        
        # Get named parameters from parents
        parent1_params = dict(parent1["model"].named_parameters())
        parent2_params = dict(parent2["model"].named_parameters())
        
        # Apply crossover for each parameter
        with torch.no_grad():
            for name, param in child_model.named_parameters():
                if random.random() < self.crossover_rate:
                    # Perform crossover
                    if random.random() < 0.5:
                        # Method 1: Take entire parameter from one parent
                        if random.random() < 0.5:
                            param.copy_(parent1_params[name])
                        else:
                            param.copy_(parent2_params[name])
                    else:
                        # Method 2: Element-wise crossover
                        mask = torch.rand_like(param) < 0.5
                        param.copy_(
                            torch.where(mask, parent1_params[name], parent2_params[name])
                        )
                else:
                    # No crossover, copy from first parent
                    param.copy_(parent1_params[name])
        
        # Create child individual
        child = {
            "id": str(uuid.uuid4()),
            "model": child_model,
            "fitness": 0.0,
            "novelty": 0.0,
            "age": 0,
            "creation_time": datetime.datetime.utcnow().isoformat(),
            "ancestry": [parent1["id"], parent2["id"]],
            "mutation_history": []
        }
        
        return child
    
    def mutate(self, individual, mutation_scale=0.1):
        """Apply mutation to an individual"""
        # Check if model has gradient-guided mutation method
        if hasattr(individual["model"], "apply_mutations"):
            individual["model"].apply_mutations(
                mutation_rate=self.mutation_rate,
                mutation_scale=mutation_scale
            )
            individual["mutation_history"].append({
                "generation": self.generation,
                "method": "gradient_guided",
                "mutation_rate": self.mutation_rate,
                "mutation_scale": mutation_scale
            })
        else:
            # Standard mutation
            with torch.no_grad():
                for param in individual["model"].parameters():
                    # Generate mutation mask
                    mask = torch.rand_like(param) < self.mutation_rate
                    
                    # Apply mutations where mask is True
                    param.data += torch.randn_like(param) * mutation_scale * mask
            
            individual["mutation_history"].append({
                "generation": self.generation,
                "method": "standard",
                "mutation_rate": self.mutation_rate,
                "mutation_scale": mutation_scale
            })
    
    def evolve(self):
        """Evolve the population to next generation"""
        # Calculate novelty scores
        self.calculate_novelty()
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Create new population
        new_population = []
        
        # Keep elites
        for i in range(min(self.elitism_count, len(self.population))):
            elite = copy.deepcopy(self.population[i])
            elite["age"] += 1
            new_population.append(elite)
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.selection()
            parent2 = self.selection()
            
            # Ensure parents are different
            while parent2["id"] == parent1["id"] and len(self.population) > 1:
                parent2 = self.selection()
            
            # Create child through crossover
            child = self.crossover(parent1, parent2)
            
            # Apply mutation with probability
            if random.random() < 0.9:  # 90% chance of mutation
                self.mutate(child)
            
            new_population.append(child)

            # Replace population
        self.population = new_population
        self.generation += 1
    
    def get_best_individual(self):
        """Get the individual with highest fitness"""
        if not self.population:
            return None
            
        return max(self.population, key=lambda x: x["fitness"])
    
    def get_most_novel_individual(self):
        """Get the most novel individual"""
        if not self.population:
            return None
            
        return max(self.population, key=lambda x: x["novelty"])
    
    def get_population_statistics(self):
        """Get statistics about the population"""
        if not self.population:
            return {
                "status": "empty_population",
                "generation": self.generation
            }
            
        fitnesses = [ind["fitness"] for ind in self.population]
        novelties = [ind["novelty"] for ind in self.population]
        ages = [ind["age"] for ind in self.population]
        
        return {
            "status": "success",
            "generation": self.generation,
            "population_size": len(self.population),
            "fitness_stats": {
                "min": min(fitnesses),
                "max": max(fitnesses),
                "mean": sum(fitnesses) / len(fitnesses),
                "median": sorted(fitnesses)[len(fitnesses) // 2]
            },
            "novelty_stats": {
                "min": min(novelties),
                "max": max(novelties),
                "mean": sum(novelties) / len(novelties)
            },
            "age_stats": {
                "min": min(ages),
                "max": max(ages),
                "mean": sum(ages) / len(ages)
            },
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history
        }
    
    def visualize_evolution(self):
        """Generate a plot of evolution progress"""
        if not self.best_fitness_history:
            return None
            
        plt.figure(figsize=(10, 6))
        
        # Plot best fitness
        plt.plot(self.best_fitness_history, 'b-', label='Best Fitness')
        
        # Plot average fitness
        plt.plot(self.avg_fitness_history, 'g-', label='Average Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolutionary Progress')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()


class HybridNeuroevolutionSystem:
    """
    System that combines gradient-based learning with evolutionary algorithms
    for training neural networks
    """
    
    def __init__(self, 
                numogram_system, 
                symbol_extractor, 
                emotion_tracker,
                input_size=30,
                hidden_sizes=[20, 10],
                output_size=9,
                population_size=20,
                learning_rate=0.01,
                gradient_steps=10,
                evolution_interval=5):
        
        # Store references to system components
        self.numogram = numogram_system
        self.symbol_extractor = symbol_extractor
        self.emotion_tracker = emotion_tracker
        
        # Neural network parameters
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Hybrid learning parameters
        self.learning_rate = learning_rate
        self.gradient_steps = gradient_steps
        self.evolution_interval = evolution_interval
        
        # Initialize evolutionary population
        self.evolutionary_population = EvolutionaryPopulation(
            model_class=HybridNeuralArchitecture,
            model_args={
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
                "output_size": output_size
            },
            population_size=population_size,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_count=2
        )
        
        # Experience buffer for gradient learning
        self.experience_buffer = []
        self.buffer_max_size = 100
        
        # Optimization settings
        self.criterion = nn.MSELoss()
        
        # Training statistics
        self.training_history = {
            "gradient_losses": [],
            "evolutionary_fitness": [],
            "hybrid_performance": []
        }
    
    def _extract_features(self, symbolic_patterns, emotional_state, context_data=None):
        """Extract features for neural network input"""
        features = np.zeros(self.input_size)
        
        # Features 0-8: Zone distribution of symbolic patterns
        if symbolic_patterns:
            zone_distribution = {}
            for pattern in symbolic_patterns:
                zone = pattern.get("numogram_zone")
                if zone not in zone_distribution:
                    zone_distribution[zone] = 0
                zone_distribution[zone] += 1
            
            # Normalize zone distribution
            total_patterns = len(symbolic_patterns)
            for zone, count in zone_distribution.items():
                zone_idx = int(zone) - 1
                if 0 <= zone_idx < 9:
                    features[zone_idx] = count / total_patterns
        
        # Features 9-17: Top emotional states
        if emotional_state and "emotional_spectrum" in emotional_state:
            emotion_scores = emotional_state["emotional_spectrum"]
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get indices for tracked emotions
            emotion_indices = {
                "joy": 9, "trust": 10, "fear": 11, "surprise": 12,
                "sadness": 13, "disgust": 14, "anger": 15, "anticipation": 16,
                "curiosity": 17
            }
            
            # Set feature values for emotions
            for emotion, score in sorted_emotions:
                if emotion in emotion_indices:
                    features[emotion_indices[emotion]] = score
        
        # Features 18-20: Emotional intensity and primary metrics 
        if emotional_state:
            features[18] = emotional_state.get("intensity", 0.5)
            
            # Digital ratios (if available)
            digital_ratios = emotional_state.get("digital_ratios", [])
            for i, ratio in enumerate(digital_ratios[:2]):  # Use up to 2 ratios
                if i < 2:
                    features[19 + i] = ratio / 9.0  # Normalize by max possible ratio
        
        # Features 21-29: Pattern intensities by zone
        if symbolic_patterns:
            zone_intensities = {}
            for pattern in symbolic_patterns:
                zone = pattern.get("numogram_zone")
                intensity = pattern.get("intensity", 0.5)
                if zone not in zone_intensities:
                    zone_intensities[zone] = []
                zone_intensities[zone].append(intensity)
            
            # Average intensity per zone
            for zone, intensities in zone_intensities.items():
                zone_idx = int(zone) - 1
                if 0 <= zone_idx < 9:
                    features[21 + zone_idx] = sum(intensities) / len(intensities)
        
        return features
    
    def _one_hot_encode_zone(self, zone):
        """Convert zone to one-hot encoded vector"""
        target = np.zeros(self.output_size)
        zone_idx = int(zone) - 1
        if 0 <= zone_idx < self.output_size:
            target[zone_idx] = 1.0
        return target
    
    def _add_to_experience_buffer(self, features, target_zone, reward):
        """Add experience to buffer for gradient learning"""
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        target_tensor = torch.FloatTensor(self._one_hot_encode_zone(target_zone))
        
        # Add to buffer
        self.experience_buffer.append({
            "features": features_tensor,
            "target": target_tensor,
            "reward": reward,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
        # Limit buffer size
        if len(self.experience_buffer) > self.buffer_max_size:
            self.experience_buffer = self.experience_buffer[-self.buffer_max_size:]
    
    def _gradient_update_step(self, model, optimizer, batch_size=8):
        """Perform one gradient update step"""
        if len(self.experience_buffer) < batch_size:
            return 0.0  # No update if buffer too small
            
        # Sample batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        # Prepare batch data
        features = torch.stack([exp["features"] for exp in batch])
        targets = torch.stack([exp["target"] for exp in batch])
        rewards = torch.FloatTensor([exp["reward"] for exp in batch])
        
        # Weight targets by reward
        weighted_targets = targets * rewards.unsqueeze(1)
        
        # Forward pass
        model.train()
        outputs = model(features)
        
        # Calculate loss
        loss = self.criterion(outputs, weighted_targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Store gradient information for hybrid learning
        model.store_gradient_information()
        
        # Update weights
        optimizer.step()
        
        return loss.item()
    
    def _perform_gradient_learning(self, individual, num_steps=10):
        """Perform gradient-based learning for an individual"""
        # Skip if no experience
        if not self.experience_buffer:
            return 0.0
            
        model = individual["model"]
        
        # Create optimizer with learning rate modulated by emotional state
        base_lr = self.learning_rate * model.neuromodulation["learning_rate_mod"]
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        
        # Perform multiple update steps
        total_loss = 0.0
        for _ in range(num_steps):
            loss = self._gradient_update_step(model, optimizer)
            total_loss += loss
        
        # Return average loss
        return total_loss / num_steps if num_steps > 0 else 0.0
    
    def _evaluate_model(self, model, features, target_zone):
        """Evaluate model prediction accuracy"""
        # Create tensors
        features_tensor = torch.FloatTensor(features)
        
        # Set to evaluation mode
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor)
        
        # Get predicted zone
        predicted_zone = str(outputs.argmax().item() + 1)
        
        # Calculate confidence
        confidence = outputs[int(target_zone) - 1].item()
        
        # Calculate fitness based on prediction accuracy
        if predicted_zone == target_zone:
            fitness = 0.7 + (0.3 * confidence)  # Reward correct prediction
        else:
            # Partial credit for adjacent zones
            adjacent_zones = {
                "1": ["2", "4", "8"],
                "2": ["1", "3", "6"],
                "3": ["2", "7", "9"],
                "4": ["1", "5", "7"],
                "5": ["4", "6", "8"],
                "6": ["2", "5", "9"],
                "7": ["3", "4", "8"],
                "8": ["1", "5", "7"],
                "9": ["3", "6"]
            }
            
            if predicted_zone in adjacent_zones.get(target_zone, []):
                fitness = 0.3 * confidence  # Partial reward for adjacent zone
            else:
                fitness = 0.1 * confidence  # Minimal reward for wrong prediction
        
        return fitness, predicted_zone, confidence
    
    def integrate(self, text: str, user_id: str, context_data: Dict = None) -> Dict:
        """
        Main integration function using hybrid neuroevolution
        """
        # Initialize context if not provided
        if context_data is None:
            context_data = {}
        
        # 1. Extract symbolic patterns
        symbolic_patterns = self.symbol_extractor.extract_symbols(text, user_id)
        
        # 2. Analyze emotional state
        emotional_state = self.emotion_tracker.analyze_emotion(text, user_id, context_data)
        
        # 3. Get current numogram zone
        current_zone = self.numogram.user_memory.get(user_id, {}).get('zone', '1')
        
        # 4. Extract features
        features = self._extract_features(symbolic_patterns, emotional_state, context_data)
        
        # 5. Get best model from population
        best_individual = self.evolutionary_population.get_best_individual()
        
        # 6. Set neuromodulation based on emotional state
        best_individual["model"].set_neuromodulation(emotional_state)
        
        # 7. Use model to predict zone
        fitness, predicted_zone, confidence = self._evaluate_model(
            best_individual["model"], features, current_zone
        )
        
        # 8. Make numogram transition
        transition_result = self.numogram.transition(
            user_id=user_id,
            current_zone=predicted_zone,  # Use predicted zone
            feedback=confidence,
            context_data={
                **context_data,
                "symbolic_patterns": symbolic_patterns,
                "emotional_state": emotional_state,
                "hybrid_prediction": {
                    "predicted_zone": predicted_zone,
                    "confidence": confidence
                }
            }
        )
        
        # 9. Get actual next zone
        next_zone = transition_result["next_zone"]
        
        # 10. Update experience buffer
        self._add_to_experience_buffer(
            features, next_zone, reward=1.0  # Full reward
        )
        
        # 11. Perform learning
        gradient_loss = 0.0
        
        # Update best individual with gradient-based learning
        if best_individual["model"].gradient_learning_enabled:
            gradient_loss = self._perform_gradient_learning(
                best_individual, self.gradient_steps
            )
        
        # Periodically perform evolution
        if self.evolutionary_population.generation % self.evolution_interval == 0:
            # Evaluate entire population
            self.evolutionary_population.evaluate_population(
                lambda model: self._evaluate_model(model, features, next_zone)[0]
            )
            
            # Evolve to next generation
            self.evolutionary_population.evolve()
        
        # 12. Record training history
        self.training_history["gradient_losses"].append(gradient_loss)
        self.training_history["evolutionary_fitness"].append(fitness)
        
        # Calculate hybrid performance (combination of evolutionary fitness and gradient learning)
        hybrid_performance = fitness * (1.0 - min(1.0, gradient_loss))
        self.training_history["hybrid_performance"].append(hybrid_performance)
        
        # 13. Return comprehensive result
        return {
            "user_id": user_id,
            "text_input": text,
            "symbolic_patterns": symbolic_patterns[:5],  # Limit to top 5 for response
            "emotional_state": emotional_state,
            "numogram_transition": transition_result,
            "hybrid_prediction": {
                "predicted_zone": predicted_zone,
                "confidence": confidence,
                "next_zone": next_zone,
                "evolutionary_generation": self.evolutionary_population.generation,
                "gradient_loss": gradient_loss,
                "hybrid_performance": hybrid_performance
            }
        }
    
    def get_model_explanation(self):
        """
        Get explanation of current model's reasoning process
        Implements explainable AI techniques
        """
        # Get best model
        best_individual = self.evolutionary_population.get_best_individual()
        if not best_individual:
            return {"status": "no_model_available"}
        
        model = best_individual["model"]
        
        # Analyze input feature importance
        feature_importance = {}
        
        # Compute importance by analyzing first layer weights
        first_layer = None
        for name, param in model.named_parameters():
            if 'weight' in name and '0' in name:  # First layer weights
                first_layer = param.detach().cpu().numpy()
                break
        
        if first_layer is not None:
            # Calculate importance as sum of absolute weights for each input
            importance = np.sum(np.abs(first_layer), axis=0)
            
            # Normalize
            total = np.sum(importance)
            if total > 0:
                importance = importance / total
                
            # Map to feature names
            feature_names = {
                0: "zone_1_distribution", 1: "zone_2_distribution", 2: "zone_3_distribution",
                3: "zone_4_distribution", 4: "zone_5_distribution", 5: "zone_6_distribution",
                6: "zone_7_distribution", 7: "zone_8_distribution", 8: "zone_9_distribution",
                9: "emotion_joy", 10: "emotion_trust", 11: "emotion_fear",
                12: "emotion_surprise", 13: "emotion_sadness", 14: "emotion_disgust",
                15: "emotion_anger", 16: "emotion_anticipation", 17: "emotion_curiosity",
                18: "emotional_intensity", 19: "digital_ratio_1", 20: "digital_ratio_2",
                21: "zone_1_intensity", 22: "zone_2_intensity", 23: "zone_3_intensity",
                24: "zone_4_intensity", 25: "zone_5_intensity", 26: "zone_6_intensity",
                27: "zone_7_intensity", 28: "zone_8_intensity", 29: "zone_9_intensity"
            }
            
            for i, imp in enumerate(importance):
                if i in feature_names:
                    feature_importance[feature_names[i]] = float(imp)
        
        # Get top 5 most important features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get neuromodulation information
        neuromodulation = {
            "learning_rate_mod": float(model.neuromodulation["learning_rate_mod"]),
            "noise_level": float(model.neuromodulation["noise_level"]),
            "dropout_prob": float(model.neuromodulation["dropout_prob"]),
            "attention_focus": model.neuromodulation["attention_focus"]
        }
        
        # Return explanation
        return {
            "status": "success",
            "model_id": best_individual["id"],
            "model_age": best_individual["age"],
            "evolutionary_generation": self.evolutionary_population.generation,
            "feature_importance": dict(top_features),
            "neuromodulation": neuromodulation,
            "ancestry": best_individual.get("ancestry", []),
            "mutation_history": best_individual.get("mutation_history", [])
        }
    
    def visualize_learning_progress(self):
        """Generate visualization of hybrid learning progress"""
        if (not self.training_history["gradient_losses"] or
            not self.training_history["evolutionary_fitness"]):
            return None
            
        plt.figure(figsize=(12, 8))
        
        # Create 2 subplots
        plt.subplot(2, 1, 1)
        
        # Plot gradient losses
        plt.plot(self.training_history["gradient_losses"], 'r-', label='Gradient Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Gradient-Based Learning Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        
        # Plot evolutionary fitness
        plt.plot(self.training_history["evolutionary_fitness"], 'b-', label='Evolutionary Fitness')
        
        # Plot hybrid performance
        plt.plot(self.training_history["hybrid_performance"], 'g-', label='Hybrid Performance')
        
        plt.xlabel('Training Step')
        plt.ylabel('Performance')
        plt.title('Evolutionary and Hybrid Performance')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def export_best_model(self):
        """Export the best model from the population"""
        best_individual = self.evolutionary_population.get_best_individual()
        if not best_individual:
            return {"status": "no_model_available"}
        
        # Save model to file using PyTorch
        model_path = f"best_model_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(best_individual["model"].state_dict(), model_path)
        
        return {
            "status": "success",
            "model_id": best_individual["id"],
            "model_path": model_path,
            "model_fitness": best_individual["fitness"],
            "model_age": best_individual["age"],
            "generation": self.evolutionary_population.generation
        }
```
