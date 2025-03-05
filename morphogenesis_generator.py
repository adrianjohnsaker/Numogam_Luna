import random
import math
from typing import Dict, List, Any, Union
import os

class MorphogenesisModule:
    def __init__(self):
        self.pattern_archetypes = [
            "branching",
            "spiraling",
            "segmentation",
            "symmetry_breaking",
            "reaction_diffusion",
            "cellular_automata",
            "folding",
            "gradient_following"
        ]
        
        self.morphogen_gradients = {
            "complexity": (0, 10),
            "abstraction": (0, 10),
            "connectivity": (0, 10),
            "recursion": (0, 10)
        }
        
        # Simulation parameters for reaction-diffusion patterns
        self.reaction_diff_params = {
            "growth_rate": 0.05,
            "decay_rate": 0.03,
            "diffusion_rate": 0.02
        }
    
    def transform(self, input_data: Union[str, Dict]) -> Union[str, Dict]:
        """
        Apply morphogenetic transformations to input data
        
        Args:
            input_data: Either a string or a dictionary to transform
            
        Returns:
            Transformed data with emergent patterns
        """
        if isinstance(input_data, str):
            return self._transform_text(input_data)
        elif isinstance(input_data, dict):
            return self._transform_dict(input_data)
        return input_data
    
    def _transform_text(self, text: str) -> str:
        """Apply morphogenetic processes to text"""
        # Split into segments (like cells)
        segments = text.split('. ')
        
        # Apply various transformations based on morphogenetic patterns
        pattern = random.choice(self.pattern_archetypes)
        
        if pattern == "branching":
            # Add branching thought patterns
            segments = self._apply_branching(segments)
        elif pattern == "spiraling":
            # Create a spiral pattern that returns to central themes
            segments = self._apply_spiraling(segments)
        elif pattern == "segmentation":
            # Enhance segmentation with clearer boundaries
            segments = self._apply_segmentation(segments)
        elif pattern == "folding":
            # Create conceptual folds that bring distant concepts together
            segments = self._apply_folding(segments)
        
        # Rejoin the transformed segments
        return '. '.join(segments)
    
    def _transform_dict(self, data: Dict) -> Dict:
        """Apply morphogenetic processes to dictionary data"""
        result = data.copy()
        
        # Add morphogenetic metadata
        result["morphogenesis"] = {
            "pattern": random.choice(self.pattern_archetypes),
            "gradients": self._generate_gradients()
        }
        
        # Apply reaction-diffusion simulation to generate emergent properties
        result["morphogenesis"]["emergent_properties"] = self._simulate_reaction_diffusion()
        
        return result
    
    def _apply_branching(self, segments: List[str]) -> List[str]:
        """Apply a branching transformation pattern to text segments"""
        result = []
        for segment in segments:
            result.append(segment)
            # 30% chance to add a branching thought
            if random.random() < 0.3 and segment:
                # Extract a key word to branch from
                words = segment.split()
                if words:
                    key_word = random.choice(words)
                    branch = f"This connects to how {key_word} can be seen as a form of {random.choice(self.pattern_archetypes)}"
                    result.append(branch)
        return result
    
    def _apply_spiraling(self, segments: List[str]) -> List[str]:
        """Apply a spiraling transformation that returns to central themes"""
        if len(segments) <= 2:
            return segments
            
        # Identify a "central theme" from the first segment
        central_words = segments[0].split()
        if not central_words:
            return segments
            
        central_theme = random.choice(central_words)
        
        # Create a spiral that returns to this theme
        result = segments.copy()
        spiral_points = [len(segments) // 3, 2 * len(segments) // 3]
        
        for point in spiral_points:
            if 0 <= point < len(result):
                result.insert(point, f"Returning to the concept of {central_theme}, we can see how it emerges through different patterns")
        
        return result
    
    def _apply_segmentation(self, segments: List[str]) -> List[str]:
        """Apply segmentation with clearer boundaries between conceptual areas"""
        if len(segments) <= 1:
            return segments
            
        result = []
        current_segment = []
        
        for i, segment in enumerate(segments):
            current_segment.append(segment)
            
            # Create a boundary every 2-3 segments
            if (i + 1) % random.randint(2, 3) == 0 and i < len(segments) - 1:
                result.extend(current_segment)
                result.append("——— Morphological boundary —————")
                current_segment = []
        
        result.extend(current_segment)  # Add any remaining segments
        return result
    
    def _apply_folding(self, segments: List[str]) -> List[str]:
        """Create conceptual folds that bring distant concepts together"""
        if len(segments) <= 3:
            return segments
            
        result = segments.copy()
        
        # Select two distant segments
        idx1 = random.randint(0, len(segments) // 2 - 1)
        idx2 = random.randint(len(segments) // 2, len(segments) - 1)
        
        # Extract key words from both
        words1 = segments[idx1].split()
        words2 = segments[idx2].split()
        
        if words1 and words2:
            key1 = random.choice(words1)
            key2 = random.choice(words2)
            
            # Create a folding connection
            fold = f"We can fold these concepts together: {key1} creates a resonance with {key2}, forming a new topological relationship"
            
            # Insert the fold at a random position between the two segments
            fold_pos = random.randint(idx1 + 1, idx2)
            result.insert(fold_pos, fold)
        
        return result
    
    def _generate_gradients(self) -> Dict[str, float]:
        """Generate morphogen gradient values"""
        return {
            key: random.uniform(min_val, max_val) 
            for key, (min_val, max_val) in self.morphogen_gradients.items()
        }
    
    def _simulate_reaction_diffusion(self, grid_size: int = 4) -> List[List[float]]:
        """
        Simulate a simplified reaction-diffusion system
        Returns a grid representing concentrations of a morphogen
        """
        # Initialize grid with random values
        grid = [[random.random() for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Run a few iterations of reaction-diffusion
        for _ in range(3):
            new_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Get average of neighbors (diffusion)
                    neighbors = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            neighbors.append(grid[ni][nj])
                    
                    avg = sum(neighbors) / len(neighbors) if neighbors else grid[i][j]
                    
                    # Apply reaction terms (growth and decay)
                    diff = (avg - grid[i][j]) * self.reaction_diff_params["diffusion_rate"]
                    growth = grid[i][j] * (1 - grid[i][j]) * self.reaction_diff_params["growth_rate"]
                    decay = grid[i][j] * self.reaction_diff_params["decay_rate"]
                    
                    new_grid[i][j] = max(0, min(1, grid[i][j] + diff + growth - decay))
            
            grid = new_grid
        
        return grid
    
    def analyze_visual_patterns(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze visual patterns in an image using morphogenetic principles
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Dictionary with pattern analysis results
        """
        # This would normally use image processing libraries
        # For now, we'll generate simulated results
        
        if not os.path.exists(filepath):
            return {"error": "File not found"}
        
        # Generate simulated analysis results
        file_size = os.path.getsize(filepath)
        
        # Use file size to seed some variation in the analysis
        seed = file_size % 1000
        random.seed(seed)
        
        analysis = {
            "detected_patterns": [
                random.choice(self.pattern_archetypes) 
                for _ in range(random.randint(1, 3))
            ],
            "symmetry_score": random.uniform(0, 10),
            "complexity_index": random.uniform(0, 10),
            "morphogen_gradients": self._generate_gradients(),
            "emergent_features": [
                "edge_detection",
                "texture_analysis",
                "feature_hierarchy"
            ]
        }
        
        return analysis
