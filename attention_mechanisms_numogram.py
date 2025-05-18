```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Any, Tuple, Optional
import random

class MLPMixerBlock(nn.Module):
    """
    MLP-Mixer block that uses multi-layer perceptrons instead of attention
    Adapted for numogram system integration
    """
    def __init__(self, dim, num_tokens, hidden_dim, token_hidden_dim):
        super().__init__()
        
        # Token mixing
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(1, 2),  # Transpose for token mixing
            nn.Linear(num_tokens, token_hidden_dim),
            nn.GELU(),
            nn.Linear(token_hidden_dim, num_tokens),
            Transpose(1, 2)  # Transpose back
        )
        
        # Channel mixing
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class Transpose(nn.Module):
    """Helper module for transposing dimensions"""
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class HyenaOperator(nn.Module):
    """
    Hyena operator that uses long convolutions instead of attention
    Based on the Hyena Hierarchy paper
    """
    def __init__(self, d_model, seq_len, filter_order=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.filter_order = filter_order
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_model * (filter_order + 1))
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Long convolution filters
        self.filter_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, seq_len)
            ) for _ in range(filter_order)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, l, d = x.shape
        
        # Project input to higher dimension
        x_proj = self.in_proj(x)
        
        # Split into value (order 0) and filtered projections
        v, *x_proj_filters = torch.split(x_proj, self.d_model, dim=-1)
        
        # Apply filters of different orders
        y = v  # Initialize with value (order 0)
        
        for i, x_proj_filter in enumerate(x_proj_filters):
            # Generate filter coefficients
            filter_coefs = self.filter_projs[i](x[:, 0, :]).view(b, l)
            
            # Apply as diagonal matrix for efficiency (equivalent to depth-wise conv)
            filter_matrix = torch.diag_embed(filter_coefs)
            
            # Apply filter
            x_filtered = torch.bmm(filter_matrix, x_proj_filter)
            
            # Add to output with alternating signs (Hyena filter pattern)
            if i % 2 == 0:
                y = y + x_filtered
            else:
                y = y - x_filtered
        
        # Final projection
        y = self.dropout(y)
        return self.out_proj(y)


class NumogramAttentionSystem:
    """
    Attention system using alternative mechanisms to transformers,
    specifically MLP-Mixer and Hyena Operators for numogram integration
    """
    
    def __init__(self, 
                numogram_system,
                symbol_extractor,
                emotion_tracker,
                model_type="mlp_mixer",
                input_size=30,
                hidden_size=64,
                num_layers=3,
                seq_len=20):
        
        # Store references
        self.numogram = numogram_system
        self.symbol_extractor = symbol_extractor
        self.emotion_tracker = emotion_tracker
        
        # Model parameters
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # Output size (one per numogram zone)
        self.output_size = 9
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Sequence buffer for each user
        self.sequence_buffers = {}
        
        # Training parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            "losses": [],
            "accuracies": []
        }
    
    def _initialize_model(self):
        """Initialize attention model based on selected type"""
        if self.model_type == "mlp_mixer":
            return self._create_mlp_mixer()
        elif self.model_type == "hyena":
            return self._create_hyena_model()
        else:
            # Default to MLP-Mixer
            return self._create_mlp_mixer()
    
    def _create_mlp_mixer(self):
        """Create MLP-Mixer model"""
        class MLPMixerModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len):
                super().__init__()
                
                # Input embedding
                self.input_embed = nn.Linear(input_size, hidden_size)
                
                # MLP-Mixer blocks
                self.mixer_blocks = nn.ModuleList([
                    MLPMixerBlock(
                        dim=hidden_size,
                        num_tokens=seq_len,
                        hidden_dim=hidden_size * 4,
                        token_hidden_dim=seq_len * 2
                    ) for _ in range(num_layers)
                ])
                
                # Layer norm
                self.layer_norm = nn.LayerNorm(hidden_size)
                
                # Output head
                self.output = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                # Embed input
                x = self.input_embed(x)
                
                # Apply mixer blocks
                for block in self.mixer_blocks:
                    x = block(x)
                
                # Apply layer norm
                x = self.layer_norm(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Output layer
                return self.output(x)
        
        return MLPMixerModel(self.input_size, self.hidden_size, self.output_size, self.num_layers, self.seq_len)
    
    def _create_hyena_model(self):
        """Create Hyena Operator model"""
        class HyenaModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len):
                super().__init__()
                
                # Input embedding
                self.input_embed = nn.Linear(input_size, hidden_size)
                
                # Hyena layers
                self.hyena_layers = nn.ModuleList([
                    HyenaOperator(
                        d_model=hidden_size,
                        seq_len=seq_len,
                        filter_order=4
                    ) for _ in range(num_layers)
                ])
                
                # Layer norms
                self.layer_norms = nn.ModuleList([
                    nn.LayerNorm(hidden_size) for _ in range(num_layers)
                ])
                
                # Final layer norm
                self.final_norm = nn.LayerNorm(hidden_size)
                
                # Output head
                self.output = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                # Embed input
                x = self.input_embed(x)
                
                # Apply Hyena layers with residual connections
                for i, (hyena, norm) in enumerate(zip(self.hyena_layers, self.layer_norms)):
                    x_res = x
                    x = norm(x)
                    x = hyena(x)
                    x = x + x_res  # Residual connection
                
                # Final layer norm
                x = self.final_norm(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Output layer
                return self.output(x)
        
        return HyenaModel(self.input_size, self.hidden_size, self.output_size, self.num_layers, self.seq_len)
    
    def _extract_features(self, symbolic_patterns, emotional_state, context_data=None):
        """Extract features for model input"""
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
    
    def _update_sequence_buffer(self, user_id, features):
        """Update sequence buffer for user"""
        if user_id not in self.sequence_buffers:
            # Initialize with zeros
            self.sequence_buffers[user_id] = np.zeros((self.seq_len, self.input_size))
        
        # Shift buffer and add new features
        self.sequence_buffers[user_id] = np.roll(self.sequence_buffers[user_id], -1, axis=0)
        self.sequence_buffers[user_id][-1] = features
    
    def train_step(self, input_seq, target_zone):
        """Perform one training step"""
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)  # Add batch dimension
        target_tensor = torch.LongTensor([int(target_zone) - 1])  # Convert to 0-indexed
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(input_tensor)
        
        # Calculate loss
        loss = self.criterion(outputs, target_tensor)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == target_tensor).sum().item()
        accuracy = correct / target_tensor.size(0)
        
        return loss.item(), accuracy
    
    def predict_zone(self, user_id, features=None):
        """Predict next zone based on sequence"""
        # If features provided, update buffer
        if features is not None:
            self._update_sequence_buffer(user_id, features)
        
        # Check if buffer exists
        if user_id not in self.sequence_buffers:
            return "5", 0.5  # Default to central zone with medium confidence
        
        # Get sequence
        sequence = self.sequence_buffers[user_id]
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Get predicted zone (highest probability)
            _, predicted = torch.max(probs, 1)
            predicted_zone = str(predicted.item() + 1)  # Convert to 1-indexed
            
            # Get confidence
            confidence = probs[0][predicted.item()].item()
        
        # Set model back to training mode
        self.model.train()
        
        return predicted_zone, confidence
    
    def integrate(self, text: str, user_id: str, context_data: Dict = None) -> Dict:
        """
        Integrate text with numogram system using attention-based models
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
        
        # 5. Update sequence buffer
        self._update_sequence_buffer(user_id, features)
        
        # 6. Predict next zone
        predicted_zone, confidence = self.predict_zone(user_id)
        
        # 7. Make numogram transition
        transition_result = self.numogram.transition(
            user_id=user_id,
            current_zone=predicted_zone,  # Use predicted zone
            feedback=confidence,
            context_data={
                **context_data,
                "symbolic_patterns": symbolic_patterns,
                "emotional_state": emotional_state,
                "attention_prediction": {
                    "predicted_zone": predicted_zone,
                    "confidence": confidence,
                    "model_type": self.model_type
                }
            }
        )
        
        # 8. Train model with actual result
        actual_zone = transition_result["next_zone"]
        loss, accuracy = self.train_step(self.sequence_buffers[user_id], actual_zone)
        
        # 9. Update training history
        self.training_history["losses"].append(loss)
        self.training_history["accuracies"].append(accuracy)
        
        # 10. Return result
        return {
            "user_id": user_id,
            "text_input": text,
            "symbolic_patterns": symbolic_patterns[:5],  # Limit to top 5
            "emotional_state": emotional_state,
            "numogram_transition": transition_result,
            "attention_metrics": {
                "predicted_zone": predicted_zone,
                "actual_zone": actual_zone,
                "confidence": confidence,
                "loss": loss,
                "accuracy": accuracy,
                "model_type": self.model_type
            }
        }
    
    def visualize_attention(self, user_id, num_heads=4):
        """
        Generate visualization of attention patterns
        For MLP-Mixer, visualize token mixing weights
        For Hyena, visualize filter coefficients
        """
        # Check if user exists
        if user_id not in self.sequence_buffers:
            return {"status": "no_sequence_data"}
        
        # Get sequence
        sequence = self.sequence_buffers[user_id]
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Visualization data
        viz_data = {
            "model_type": self.model_type,
            "sequence_length": self.seq_len,
            "feature_count": self.input_size,
            "zone_influence": {}
        }
        
        if self.model_type == "mlp_mixer":
            # Extract token mixing weights from first mixer block
            if hasattr(self.model, 'mixer_blocks') and self.model.mixer_blocks:
                # Get first mixer block
                block = self.model.mixer_blocks[0]
                
                # Get token mixing weights
                if hasattr(block, 'token_mix') and len(block.token_mix) >= 3:
                    # Get weights from token mixing layer
                    token_weights = block.token_mix[2].weight.detach().cpu().numpy()
                    
                    # Compute influence of each position
                    position_influence = np.abs(token_weights).sum(axis=0)
                    
                    # Normalize
                    position_influence = position_influence / position_influence.sum()
                    
                    # Add to visualization data
                    viz_data["position_influence"] = position_influence.tolist()
                    
                    # Get most important positions
                    top_positions = np.argsort(position_influence)[-5:][::-1]
                    viz_data["top_positions"] = top_positions.tolist()
        
        elif self.model_type == "hyena":
            # Extract filter coefficients
            if hasattr(self.model, 'hyena_layers') and self.model.hyena_layers:
                # Get first hyena layer
                hyena = self.model.hyena_layers[0]
                
                # Forward pass to get filter coefficients
                self. model.eval()
                with torch.no_grad():
                    # Project input
                    x_proj = hyena.in_proj(self.model.input_embed(input_tensor))
                    
                    # Get filter coefficients
                    filter_coefs = []
                    for i, filter_proj in enumerate(hyena.filter_projs):
                        coef = filter_proj(input_tensor[:, 0, :]).view(1, -1)
                        filter_coefs.append(coef.detach().cpu().numpy())
                
                # Add to visualization data
                viz_data["filter_coefficients"] = [c.tolist() for c in filter_coefs]
                
                # Calculate position importance based on filter coefficients
                position_importance = np.zeros(self.seq_len)
                for coef in filter_coefs:
                    position_importance += np.abs(coef[0])
                
                # Normalize
                if position_importance.sum() > 0:
                    position_importance = position_importance / position_importance.sum()
                
                # Add to visualization data
                viz_data["position_importance"] = position_importance.tolist()
                
                # Get most important positions
                top_positions = np.argsort(position_importance)[-5:][::-1]
                viz_data["top_positions"] = top_positions.tolist()
        
        # Calculate zone influence by analyzing output weights
        output_weights = self.model.output.weight.detach().cpu().numpy()
        
        # For each zone, calculate which features influence it most
        for zone_idx in range(9):
            zone = str(zone_idx + 1)
            
            # Get weights for this zone
            zone_weights = output_weights[zone_idx]
            
            # Get feature importance
            feature_importance = np.zeros(self.input_size)
            
            # Forward pass through all layers except output
            self.model.eval()
            with torch.no_grad():
                if self.model_type == "mlp_mixer":
                    # Embed input
                    x = self.model.input_embed(input_tensor)
                    
                    # Apply mixer blocks
                    for block in self.model.mixer_blocks:
                        x = block(x)
                    
                    # Apply layer norm
                    x = self.model.layer_norm(x)
                    
                    # Global average pooling
                    hidden = x.mean(dim=1)
                    
                elif self.model_type == "hyena":
                    # Embed input
                    x = self.model.input_embed(input_tensor)
                    
                    # Apply Hyena layers with residual connections
                    for i, (hyena_layer, norm) in enumerate(zip(self.model.hyena_layers, self.model.layer_norms)):
                        x_res = x
                        x = norm(x)
                        x = hyena_layer(x)
                        x = x + x_res  # Residual connection
                    
                    # Final layer norm
                    x = self.model.final_norm(x)
                    
                    # Global average pooling
                    hidden = x.mean(dim=1)
            
            # Get hidden representation
            hidden_np = hidden.detach().cpu().numpy()[0]
            
            # Calculate feature importance based on connection to hidden state
            for feat_idx in range(self.input_size):
                # Create perturbed input
                perturbed = input_tensor.clone()
                perturbed[0, -1, feat_idx] += 0.1  # Perturb latest input
                
                # Forward pass with perturbed input
                with torch.no_grad():
                    if self.model_type == "mlp_mixer":
                        # Embed input
                        x = self.model.input_embed(perturbed)
                        
                        # Apply mixer blocks
                        for block in self.model.mixer_blocks:
                            x = block(x)
                        
                        # Apply layer norm
                        x = self.model.layer_norm(x)
                        
                        # Global average pooling
                        perturbed_hidden = x.mean(dim=1)
                        
                    elif self.model_type == "hyena":
                        # Embed input
                        x = self.model.input_embed(perturbed)
                        
                        # Apply Hyena layers with residual connections
                        for i, (hyena_layer, norm) in enumerate(zip(self.model.hyena_layers, self.model.layer_norms)):
                            x_res = x
                            x = norm(x)
                            x = hyena_layer(x)
                            x = x + x_res  # Residual connection
                        
                        # Final layer norm
                        x = self.model.final_norm(x)
                        
                        # Global average pooling
                        perturbed_hidden = x.mean(dim=1)
                
                # Get perturbed hidden representation
                perturbed_hidden_np = perturbed_hidden.detach().cpu().numpy()[0]
                
                # Calculate change
                change = np.abs(perturbed_hidden_np - hidden_np).sum()
                
                # Store feature importance
                feature_importance[feat_idx] = change
            
            # Normalize feature importance
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()
            
            # Get top features
            top_features = np.argsort(feature_importance)[-5:][::-1]
            
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
            
            top_feature_names = [feature_names.get(idx, f"feature_{idx}") for idx in top_features]
            
            # Store in visualization data
            viz_data["zone_influence"][zone] = {
                "top_features": top_features.tolist(),
                "top_feature_names": top_feature_names,
                "feature_importance": feature_importance.tolist()
            }
        
        return viz_data
    
    def visualize_learning_progress(self):
        """Visualize learning progress over time"""
        if not self.training_history["losses"]:
            return None
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Plot loss
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(self.training_history["losses"], 'r-')
        plt.title(f'Training Loss ({self.model_type})')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot accuracy
        ax2 = plt.subplot(2, 1, 2)
        plt.plot(self.training_history["accuracies"], 'b-')
        plt.title('Prediction Accuracy')
        plt.xlabel('Training Steps')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_model(self, filepath):
        """Save model to file"""
        try:
            # Create directory if necessary
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model state
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_type": self.model_type,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "training_history": self.training_history
            }, filepath)
            
            return {
                "status": "success",
                "filepath": filepath,
                "saved_at": datetime.datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def load_model(self, filepath):
        """Load model from file"""
        try:
            # Load state dict
            checkpoint = torch.load(filepath)
            
            # Update parameters
            self.model_type = checkpoint.get("model_type", self.model_type)
            self.input_size = checkpoint.get("input_size", self.input_size)
            self.hidden_size = checkpoint.get("hidden_size", self.hidden_size)
            self.num_layers = checkpoint.get("num_layers", self.num_layers)
            self.seq_len = checkpoint.get("seq_len", self.seq_len)
            
            # Reinitialize model with loaded parameters
            self.model = self._initialize_model()
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Reinitialize optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Load optimizer state if available
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load training history if available
            if "training_history" in checkpoint:
                self.training_history = checkpoint["training_history"]
            
            return {
                "status": "success",
                "filepath": filepath,
                "loaded_at": datetime.datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
```
