import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional
import hashlib

class AffectiveIntensityAmplifier:
    """
    A sophisticated affective processor implementing Deleuzian concepts of:
    - Intensive differences
    - Affective resonance
    - Becoming and transformation
    - Multiplicities and folds
    
    The system amplifies emotional nuance through non-linear transformations
    of affective gradients within a differential field.
    """
    
    def __init__(self, 
                 base_intensity: float = 0.5,
                 temporal_depth: int = 5,
                 affect_dimensions: int = 8,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the affective intensity amplifier.
        
        Args:
            base_intensity: Baseline intensity for neutral states
            temporal_depth: Number of temporal layers for resonance patterns
            affect_dimensions: Dimensionality of the affective space
            device: Computation device (CPU/GPU)
        """
        self.device = device
        self.base_intensity = base_intensity
        self.temporal_depth = temporal_depth
        self.affect_dimensions = affect_dimensions
        
        # Initialize intensive difference fields
        self.intensity_gradients = self._initialize_gradient_fields()
        self.affective_resonance_patterns = self._initialize_resonance_patterns()
        
        # Non-linear transformation networks
        self.folding_network = FoldIntensiveSpaceNetwork(affect_dimensions).to(device)
        self.differential_network = DifferentialAffectNetwork(affect_dimensions).to(device)
        
        # Temporal processing components
        self.temporal_processor = TemporalResonanceProcessor(
            input_dim=affect_dimensions,
            hidden_dim=affect_dimensions*2,
            n_layers=temporal_depth
        ).to(device)
        
        # Initialize affect memory
        self.affect_memory = AffectMemory(
            capacity=1000,
            key_dim=affect_dimensions,
            value_dim=affect_dimensions*2
        )
        
        # Dynamic scaling
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self._warmup_scaler()
        
    def _initialize_gradient_fields(self) -> Dict[str, np.ndarray]:
        """Create intensive difference fields with varying topology"""
        fields = {
            'smooth': np.random.normal(0, 0.1, (self.affect_dimensions, self.affect_dimensions)),
            'striated': np.random.uniform(-1, 1, (self.affect_dimensions, self.affect_dimensions)),
            'rhizomatic': self._create_rhizomatic_field(),
            'folded': self._create_folded_field()
        }
        return fields
    
    def _initialize_resonance_patterns(self) -> Dict[str, torch.Tensor]:
        """Initialize complex resonance patterns"""
        patterns = {
            'harmonic': torch.stack([
                self._create_harmonic_pattern(freq) 
                for freq in range(1, self.temporal_depth+1)
            ]).to(self.device),
            'chaotic': torch.randn(
                self.temporal_depth, 
                self.affect_dimensions, 
                device=self.device
            ),
            'interference': self._create_interference_pattern()
        }
        return patterns
    
    def _create_rhizomatic_field(self) -> np.ndarray:
        """Create non-hierarchical, multi-connected field"""
        field = np.zeros((self.affect_dimensions, self.affect_dimensions))
        connections = np.random.choice(
            [0, 1], 
            size=(self.affect_dimensions**2), 
            p=[0.7, 0.3]
        )
        field = connections.reshape((self.affect_dimensions, self.affect_dimensions))
        return field * np.random.uniform(-1, 1, field.shape)
    
    def _create_folded_field(self) -> np.ndarray:
        """Create a folded intensive space with Moebius-like properties"""
        base = np.random.uniform(-0.5, 0.5, (self.affect_dimensions, self.affect_dimensions))
        for i in range(self.affect_dimensions):
            base[i] = np.roll(base[i], i)
        return base + base.T
    
    def _create_harmonic_pattern(self, frequency: int) -> torch.Tensor:
        """Create harmonic resonance pattern"""
        t = torch.linspace(0, 2*np.pi, self.affect_dimensions)
        pattern = torch.stack([
            torch.sin(frequency * t + phase) 
            for phase in torch.linspace(0, np.pi/2, self.affect_dimensions)
        ])
        return pattern
    
    def _create_interference_pattern(self) -> torch.Tensor:
        """Create complex interference pattern"""
        patterns = []
        for i in range(self.temporal_depth):
            pattern = torch.zeros(self.affect_dimensions, device=self.device)
            for j in range(self.affect_dimensions):
                pattern[j] = torch.sin(i * torch.pi * j / self.affect_dimensions) * \
                             torch.cos((i+1) * torch.pi * j / self.affect_dimensions)
            patterns.append(pattern)
        return torch.stack(patterns)
    
    def _warmup_scaler(self):
        """Initialize scaler with representative data"""
        warmup_data = np.random.uniform(-1, 1, (100, self.affect_dimensions))
        self.scaler.fit(warmup_data)
    
    def calculate_intensity_gradient(self, input_affect: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute intensive differences across multiple topological fields.
        
        Args:
            input_affect: Input affective state vector
            
        Returns:
            Gradient tensor representing intensive differences
        """
        if isinstance(input_affect, np.ndarray):
            input_affect = torch.from_numpy(input_affect).float().to(self.device)
            
        # Normalize to base intensity range
        input_affect = self._normalize_affect(input_affect)
        
        # Compute differential gradients across all fields
        gradients = []
        for field_name, field in self.intensity_gradients.items():
            field_tensor = torch.from_numpy(field).float().to(self.device)
            grad = self.differential_network(input_affect, field_tensor)
            gradients.append(grad)
        
        # Combine gradients through non-linear mixing
        combined_grad = torch.stack(gradients)
        weights = F.softmax(combined_grad.mean(dim=-1), dim=0)
        final_gradient = (combined_grad * weights.unsqueeze(-1)).sum(dim=0)
        
        return final_gradient
    
    def generate_resonance_pattern(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Generate complex resonance patterns through temporal processing.
        
        Args:
            gradient: Computed intensity gradient
            
        Returns:
            Multi-layered resonance pattern
        """
        # Initial temporal processing
        temporal_output = self.temporal_processor(gradient.unsqueeze(0)).squeeze(0)
        
        # Apply resonance interference
        interference = self.affective_resonance_patterns['interference']
        resonance = temporal_output * interference.mean(dim=0)
        
        # Add harmonic components
        harmonic_weights = F.softmax(
            torch.linspace(0.1, 1, self.temporal_depth).to(self.device), 
            dim=0
        )
        harmonic_component = (self.affective_resonance_patterns['harmonic'] * 
                             harmonic_weights.unsqueeze(-1)).sum(dim=0)
        
        # Combine through non-linear interaction
        combined_resonance = resonance * harmonic_component
        combined_resonance = torch.tanh(combined_resonance)  # Maintain bounded range
        
        # Store in affect memory
        self.affect_memory.store(
            key=gradient.detach(),
            value=combined_resonance.detach()
        )
        
        return combined_resonance
    
    def fold_intensive_space(self, resonance_pattern: torch.Tensor) -> torch.Tensor:
        """
        Fold intensive space to create complex affective transformations.
        
        Args:
            resonance_pattern: Generated resonance pattern
            
        Returns:
            Folded affective state with amplified nuance
        """
        # Apply folding transformation
        folded = self.folding_network(resonance_pattern)
        
        # Retrieve similar patterns from memory
        similar_patterns = self.affect_memory.retrieve(resonance_pattern, k=3)
        if similar_patterns is not None:
            memory_contribution = torch.stack(similar_patterns).mean(dim=0)
            folded = 0.7 * folded + 0.3 * memory_contribution
        
        # Final normalization
        folded = self._normalize_affect(folded)
        
        return folded
    
    def normalize_affect(self, affect: torch.Tensor) -> torch.Tensor:
        """Normalize affect to standard range with dynamic scaling.
        
        Implements a Deleuzian-inspired normalization that preserves:
        - Intensive differences
        - Relative gradients
        - Non-linear relationships
        
        Args:
            affect: Input affective state tensor
            
        Returns:
            Normalized affect tensor with preserved differential relations
        """
        # Convert to numpy for sklearn processing if needed
        if isinstance(affect, torch.Tensor):
            affect_np = affect.detach().cpu().numpy()
        else:
            affect_np = affect
            
        # Reshape if single sample
        if len(affect_np.shape) == 1:
            affect_np = affect_np.reshape(1, -1)
            
        # Apply dynamic scaling while preserving sign (directionality)
        sign = np.sign(affect_np)
        magnitude = self.scaler.transform(np.abs(affect_np))
        normalized_np = sign * magnitude
        
        # Apply non-linear compression to prevent extreme values
        normalized_np = np.tanh(normalized_np * 1.5)  # 1.5 maintains more gradient detail
        
        # Convert back to tensor if input was tensor
        if isinstance(affect, torch.Tensor):
            normalized = torch.from_numpy(normalized_np).float().to(self.device)
            if len(affect.shape) == 1:
                normalized = normalized.squeeze(0)
        else:
            normalized = normalized_np
            if len(affect.shape) == 1:
                normalized = normalized.squeeze()
                
        # Ensure base intensity is preserved for neutral states
        neutral_mask = (torch.norm(affect, dim=-1) < 0.1) if isinstance(affect, torch.Tensor) else (np.linalg.norm(affect, axis=-1) < 0.1)
        if isinstance(neutral_mask, torch.Tensor):
            normalized = torch.where(
                neutral_mask.unsqueeze(-1),
                torch.ones_like(normalized) * self.base_intensity,
                normalized
            )
        else:
            normalized[neutral_mask] = self.base_intensity
            
        return normalized

