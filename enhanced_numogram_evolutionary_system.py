
```python
import numpy as np
import torch
from typing import Dict, List, Any, Optional
import uuid
import datetime
import json
import os

# Import from other modules
from numogram_system import NumogramSystem
from symbolic_pattern_extraction import NumogramaticSymbolExtractor
from emotional_memory_system import EmotionalEvolutionSystem, EmotionalMemorySystem
from hybrid_neural_system import HybridNeuroevolutionSystem
from tensor_numogram_system import TensorBasedNumogramSystem
from attention_mechanisms import NumogramAttentionSystem
```
class EnhancedNumogramEvolutionarySystem:
    """
    Enhanced version of NumogramEvolutionarySystem that integrates:
    1. Emotional Memory System
    2. Hybrid Neural System with Evolutionary Algorithms
    3. Dynamic Zone Relationships with Tensor-based representation
    4. Temporal aspects via Circadian Narrative Cycles
    5. Alternative Attention Mechanisms
    """
    
    def __init__(self, config_path=None):
        """Initialize the enhanced system"""
        # Load configuration if provided
        self.config = self._load_config(config_path)

        # Create the base numogram system
        self.base_numogram = NumogramSystem(
            zones_file=self.config.get("zones_file", "numogram_code/zones.json"),
            memory_file=self.config.get("memory_file", "numogram_code/user_memory.json")
        )
        
        # Create the symbolic pattern extractor
        self.symbol_extractor = NumogramaticSymbolExtractor(self.base_numogram)
        
        # Create the emotional evolution system
        self.emotion_tracker = EmotionalEvolutionSystem(
            self.base_numogram, 
            population_size=self.config.get("emotion_population_size", 50),
            mutation_rate=self.config.get("emotion_mutation_rate", 0.1)
        )
        
        # Create the emotional memory system
        self.emotional_memory = EmotionalMemorySystem(
            self.base_numogram,
            self.emotion_tracker,
            self.symbol_extractor,
            memory_file=self.config.get("emotional_memory_file", "numogram_code/emotional_memory.json")
        )
        
        # Create the tensor-based numogram system
        self.tensor_numogram = TensorBasedNumogramSystem(
            self.base_numogram,
            dimension=self.config.get("tensor_dimension", 3),
            learning_rate=self.config.get("tensor_learning_rate", 0.01)
        )
        
        # Create the hybrid neuroevolution system
        self.hybrid_neural = HybridNeuroevolutionSystem(
            self.base_numogram,
            self.symbol_extractor,
            self.emotion_tracker,
            input_size=30,
            hidden_sizes=[20, 10],
            output_size=9,
            population_size=self.config.get("network_population_size", 30),
            learning_rate=self.config.get("learning_rate", 0.01),
            gradient_steps=self.config.get("gradient_steps", 10),
            evolution_interval=self.config.get("evolution_interval", 5)
        )
        
        # Create the attention system
        self.attention_system = NumogramAttentionSystem(
            self.base_numogram,
            self.symbol_extractor,
            self.emotion_tracker,
            model_type=self.config.get("attention_model", "mlp_mixer"),
            input_size=30,
            hidden_size=64,
            num_layers=3,
            seq_len=20
        )
        
        # Try to load tesseract module if available
        self.tesseract_module = None
        try:
            import tesseract_module
            self.tesseract_module = tesseract_module
        except ImportError:
            pass

        # Try to load circadian narrative module if available
        self.circadian_module = None
        try:
            import circadian_narrative_cycles
            self.circadian_module = circadian_narrative_cycles
        except ImportError:
            pass
        
        # System metadata
        self.system_id = str(uuid.uuid4())
        self.created_at = datetime.datetime.utcnow().isoformat()
        self.system_version = "2.0.0"
        self.active_sessions = {}
        
        # Integration mode
        # Controls which subsystems have primary control over transitions
        self.primary_mode = self.config.get("primary_mode", "hybrid")  # Options: hybrid, tensor, attention, emotional_memory
        
        # Secondary modes and their weights
        self.mode_weights = {
            "hybrid": self.config.get("hybrid_weight", 0.3),
            "tensor": self.config.get("tensor_weight", 0.3),
            "attention": self.config.get("attention_weight", 0.2),
            "emotional_memory": self.config.get("emotional_memory_weight", 0.2)
        }
        
        # Normalize weights
        total_weight = sum(self.mode_weights.values())
        if total_weight > 0:
            self.mode_weights = {k: v/total_weight for k, v in self.mode_weights.items()}
    
    def _load_config(self, config_path) -> Dict:
        """Load configuration from JSON file or use defaults"""
        default_config = {
            "zones_file": "numogram_code/zones.json",
            "memory_file": "numogram_code/user_memory.json",
            "emotional_memory_file": "numogram_code/emotional_memory.json",
            "emotion_population_size": 50,
            "emotion_mutation_rate": 0.1,
            "network_population_size": 30,
            "network_mutation_rate": 0.15,
            "network_crossover_rate": 0.7,
            "tensor_dimension": 3,
            "tensor_learning_rate": 0.01,
            "learning_rate": 0.01,
            "gradient_steps": 10,
            "evolution_interval": 5,
            "attention_model": "mlp_mixer",
            "primary_mode": "hybrid",
            "hybrid_weight": 0.3,
            "tensor_weight": 0.3,
            "attention_weight": 0.2,
            "emotional_memory_weight": 0.2,
            "save_directory": "output",
            "auto_save_interval": 10,  # Save every 10 interactions
            "circadian_module_path": None,
            "tesseract_module_path": None
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update defaults with loaded config
                    default_config.update(loaded_config)
            except (json.JSONDecodeError, IOError):
                pass  # Silently use defaults on error
        
        return default_config
    
    def initialize_session(self, user_id: str, session_name: str = None) -> Dict[str, Any]:
        """Initialize a new session"""
        session_id = str(uuid.uuid4())
        session_name = session_name or f"EnhancedNumogramSession-{session_id[:8]}"
        
        session = {
            "id": session_id,
            "name": session_name,
            "user_id": user_id,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "interactions": [],
            "symbolic_patterns": [],
            "emotional_states": [],
            "numogram_transitions": [],
            "active": True,
            "config": {
                "primary_mode": self.primary_mode,
                "mode_weights": self.mode_weights.copy()
            }
        }
        
        self.active_sessions[session_id] = session
        return session
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End an active session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["active"] = False
        session["ended_at"] = datetime.datetime.utcnow().isoformat()
        
        return session
    
    def set_integration_mode(self, session_id: str, primary_mode: str, mode_weights: Dict = None) -> Dict[str, Any]:
        """Set the integration mode for a session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        # Validate primary mode
        valid_modes = ["hybrid", "tensor", "attention", "emotional_memory"]
        if primary_mode not in valid_modes:
            return {"error": f"Invalid mode. Valid modes are: {', '.join(valid_modes)}"}
        
        session = self.active_sessions[session_id]
        
        # Update primary mode
        session["config"]["primary_mode"] = primary_mode
        
        # Update mode weights if provided
        if mode_weights:
            # Validate weights
            for mode in mode_weights:
                if mode not in valid_modes:
                    return {"error": f"Invalid mode in weights: {mode}"}
            
            # Merge with existing weights
            updated_weights = session["config"]["mode_weights"].copy()
            updated_weights.update(mode_weights)
            
            # Normalize weights
            total_weight = sum(updated_weights.values())
            if total_weight > 0:
                updated_weights = {k: v/total_weight for k, v in updated_weights.items()}
            
            # Update session config
            session["config"]["mode_weights"] = updated_weights
        
        return {
            "status": "success",
            "session_id": session_id,
            "primary_mode": session["config"]["primary_mode"],
            "mode_weights": session["config"]["mode_weights"]
        }
    
    def process(self, session_id: str, text: str, context_data: Dict = None) -> Dict[str, Any]:
        """
        Process text input through the integrated enhanced system
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session = self.active_sessions[session_id]
        user_id = session["user_id"]
        
        # Initialize context if not provided
        if context_data is None:
            context_data = {}
        
        # Add session context
        context_data["session_id"] = session_id
        context_data["session_name"] = session["name"]
        context_data["interaction_count"] = len(session["interactions"])
        
        # 1. Extract symbolic patterns
        symbolic_patterns = self.symbol_extractor.extract_symbols(text, user_id)
        
        # 2. Analyze emotional state
        emotional_state = self.emotion_tracker.analyze_emotion(text, user_id, context_data)
        
        # 3. Process with all subsystems to gather predictions
        
        # 3.1 Hybrid neural system
        hybrid_result = self.hybrid_neural.integrate(text, user_id, {
            **context_data,
            "symbolic_patterns": symbolic_patterns,
            "emotional_state": emotional_state
        })
        
        # 3.2 Tensor numogram system
        # First predict zones
        tensor_transition_probs = self.tensor_numogram.predict_transitions(
            hybrid_result["numogram_transition"]["current_zone"],
            {
                **context_data,
                "symbolic_patterns": symbolic_patterns,
                "emotional_state": emotional_state
            }
        )
        
        # 3.3 Attention system
        attention_features = self.attention_system._extract_features(
            symbolic_patterns, emotional_state, context_data
        )
        attention_zone, attention_confidence = self.attention_system.predict_zone(
            user_id, attention_features
        )
        
        # 3.4 Emotional memory system
        emotional_memory_influence = self.emotional_memory.get_memory_influence(
            user_id,
            {
                "current_zone": hybrid_result["numogram_transition"]["current_zone"],
                "current_emotion": emotional_state.get("primary_emotion"),
                "current_symbols": [s.get("core_symbols", [""])[0] for s in symbolic_patterns[:5]] if symbolic_patterns else []
            }
        )
        
        # Get emotional memory predicted zone from trajectory influence
        emotional_memory_zone = None
        emotional_memory_confidence = 0.5
        
        if "influence_factors" in emotional_memory_influence:
            factors = emotional_memory_influence["influence_factors"]
            if "trajectory_influence" in factors and factors["trajectory_influence"]:
                emotional_memory_zone = factors["trajectory_influence"].get("next_zone")
                emotional_memory_confidence = factors["trajectory_influence"].get("confidence", 0.5)
        
        # 4. Determine final zone based on integration mode
        primary_mode = session["config"]["primary_mode"]
        mode_weights = session["config"]["mode_weights"]
        
        # Get zone predictions from each system
        predictions = {
            "hybrid": hybrid_result["hybrid_prediction"]["predicted_zone"],
            "tensor": max(tensor_transition_probs.items(), key=lambda x: x[1])[0],
            "attention": attention_zone
        }
        
        if emotional_memory_zone:
            predictions["emotional_memory"] = emotional_memory_zone
        
        # Get confidences
        confidences = {
            "hybrid": hybrid_result["hybrid_prediction"]["confidence"],
            "tensor": max(tensor_transition_probs.values()),
            "attention": attention_confidence
        }
        
        if emotional_memory_zone:
            confidences["emotional_memory"] = emotional_memory_confidence
        
        # Primary mode has highest influence
        if primary_mode in predictions:
            primary_prediction = predictions[primary_mode]
            primary_confidence = confidences[primary_mode]
        else:
            # Fallback if primary mode not available
            primary_prediction = predictions.get("hybrid", "5")
            primary_confidence = confidences.get("hybrid", 0.5)
        
        # Apply weighted voting for final decision
        zone_votes = {}
        for mode, zone in predictions.items():
            weight = mode_weights.get(mode, 0.0) * confidences.get(mode, 0.5)
            if zone not in zone_votes:
                zone_votes[zone] = 0.0
            zone_votes[zone] += weight
        
        # Add extra weight to primary prediction
        primary_boost = 0.3  # Additional weight for primary mode
        if primary_prediction in zone_votes:
            zone_votes[primary_prediction] += primary_boost * primary_confidence
        
        # Select zone with highest vote
        if zone_votes:
            final_zone = max(zone_votes.items(), key=lambda x: x[1])[0]
            # Scale confidence based on vote distribution
            total_votes = sum(zone_votes.values())
            if total_votes > 0:
                final_confidence = zone_votes[final_zone] / total_votes
            else:
                final_confidence = 0.5
        else:
            # Fallback
            final_zone = primary_prediction
            final_confidence = primary_confidence
        
        # 5. Execute transition with tensor numogram system using final decision
        final_transition = self.tensor_numogram.transition(
            user_id=user_id,
            current_zone=final_zone,
            feedback=final_confidence,
            context_data={
                **context_data,
                "symbolic_patterns": symbolic_patterns,
                "emotional_state": emotional_state,
                "integration_data": {
                    "predictions": predictions,
                    "confidences": confidences,
                    "primary_mode": primary_mode,
                    "mode_weights": mode_weights,
                    "zone_votes": zone_votes
                }
            }
        )
        
        # 6. Update emotional memory
        emotional_memory_result = self.emotional_memory.store_emotional_memory(
            emotional_state, user_id, text, symbolic_patterns
        )
        
        # 7. Integrate with circadian module if available
        circadian_result = None
        if self.circadian_module:
            try:
                circadian_result = self.tensor_numogram.link_to_circadian_narrative(
                    self.circadian_module, user_id
                )
            except Exception as e:
                circadian_result = {"status": "error", "message": str(e)}
        
        # 8. Train attention system with actual result
        actual_zone = final_transition["next_zone"]
        attention_loss, attention_accuracy = self.attention_system.train_step(
            self.attention_system.sequence_buffers.get(user_id, np.zeros((20, 30))), 
            actual_zone
        )
        
        # 9. Store results in session history
        interaction = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "text_input": text,
            "symbolic_patterns": symbolic_patterns,
            "emotional_state": emotional_state,
            "system_predictions": predictions,
            "system_confidences": confidences,
            "integration_result": {
                "final_zone": final_zone,
                "final_confidence": final_confidence,
                "actual_zone": actual_zone,
                "primary_mode": primary_mode,
                "zone_votes": zone_votes
            },
            "numogram_transition": final_transition
        }
        
        session["interactions"].append(interaction)
        
        # Store symbolic patterns
        session["symbolic_patterns"].extend(symbolic_patterns)
        
        # Store emotional state
        session["emotional_states"].append(emotional_state)
        
        # Store numogram transition
        session["numogram_transitions"].append(final_transition)
        
        # Auto-save if needed
        if len(session["interactions"]) % self.config.get("auto_save_interval", 10) == 0:
            self._auto_save_session(session_id)
        
        # 10. Return comprehensive result
        return {
            "session_id": session_id,
            "interaction_id": interaction["id"],
            "text_input": text,
            "symbolic_patterns": symbolic_patterns[:5],  # Limit to top 5
            "emotional_state": emotional_state,
            "system_predictions": predictions,
            "integration_result": {
                "final_zone": final_zone,
                "final_confidence": final_confidence,
                "actual_zone": actual_zone,
                "primary_mode": primary_mode,
                "zone_votes": zone_votes
            },
            "numogram_transition": final_transition,
            "emotional_memory": {
                "memory_id": emotional_memory_result,
                "influence": emotional_memory_influence
            },
            "attention_metrics": {
                "loss": attention_loss,
                "accuracy": attention_accuracy
            },
            "circadian_integration": circadian_result
        }
    
    def _auto_save_session(self, session_id: str) -> None:
        """Automatically save session to disk"""
        if session_id not in self.active_sessions:
            return
            
        session = self.active_sessions[session_id]
        save_dir = self.config.get("save_directory", "output")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Build filename with timestamp
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/enhanced_session_{session_id}_{timestamp}.json"
        
        # Save session data
        with open(filename, 'w') as f:
            json.dump(session, f, indent=2)
    
    def get_zone_info(self, zone: str) -> Dict[str, Any]:
        """Get enhanced information about a numogram zone"""
        # Get base zone data
        zone_data = self.base_numogram.ZONE_DATA.get(zone, {})
        
        # Get tensor representation
        tensor_data = self.tensor_numogram.get_zone_tensor(zone)
        tensor_position = tensor_data["position"].tolist() if tensor_data else None
        
        # Get symbolic associations
        symbolic_associations = []
        for symbol_category, symbols in self.symbol_extractor.zone_symbols.items():
            if zone in symbol_category:
                symbolic_associations.extend(symbols)
        
        # Get emotional associations
        emotional_associations = self.emotion_tracker.zone_emotion_affinities.get(zone, {})
        
        # Get active hyperedges containing this zone
        active_hyperedges = []
        for edge in self.tensor_numogram.hyperedges:
            if edge["active"] and zone in edge["zones"]:
                active_hyperedges.append(edge)
        
        return {
            "zone": zone,
            "zone_data": zone_data,
            "tensor_representation": {
                "position": tensor_position,
                "energy": float(tensor_data["energy"]) if tensor_data else None,
                "stability": float(tensor_data["stability"]) if tensor_data else None,
                "flux": float(tensor_data["flux"]) if tensor_data else None,
                "dimension": self.tensor_numogram.dimension
            },
            "symbolic_associations": symbolic_associations,
            "emotional_associations": emotional_associations,
            "active_hyperedges": [
                {
                    "id": edge["id"],
                    "name": edge["name"],
                    "zones": edge["zones"],
                    "strength": edge["strength"]
                }
                for edge in active_hyperedges
            ]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed status of the enhanced system"""
        # Count active sessions
        active_count = sum(1 for s in self.active_sessions.values() if s.get("active", False))
        
        # Get base numogram stats
        numogram_user_count = len(self.base_numogram.user_memory) if hasattr(self.base_numogram, 'user_memory') else 0
        
        # Get hybrid neural stats
        neural_stats = self.hybrid_neural.get_model_explanation()
        
        # Get tensor space stats
        tensor_viz = self.tensor_numogram.visualize_tensor_space()
        
        # Get attention system stats
        if self.attention_system.training_history["losses"]:
            attention_loss = self.attention_system.training_history["losses"][-1]
            attention_accuracy = self.attention_system.training_history["accuracies"][-1]
        else:
            attention_loss = None
            attention_accuracy = None
        
        # Get emotional memory stats
        memory_count = 0
        for user_memories in self.emotional_memory.emotional_memories.values():
            memory_count += len(user_memories)
        
        # Compile component status
        component_status = {
            "base_numogram": "active",
            "symbol_extractor": "active",
            "emotion_tracker": "active",
            "emotional_memory": "active",
            "tensor_numogram": "active",
            "hybrid_neural": "active",
            "attention_system": "active",
            "circadian_module": "active" if self.circadian_module else "unavailable",
            "tesseract_module": "active" if self.tesseract_module else "unavailable"
        }
        
        return {
            "system_id": self.system_id,
            "version": self.system_version,
            "created_at": self.created_at,
            "current_time": datetime.datetime.utcnow().isoformat(),
            "active_sessions": active_count,
            "total_sessions": len(self.active_sessions),
            "numogram_users": numogram_user_count,
            "emotional_memories": memory_count,
            "components": component_status,
            "primary_mode": self.primary_mode,
            "mode_weights": self.mode_weights,
            "tensor_dimension": self.tensor_numogram.dimension,
            "attention_model_type": self.attention_system.model_type,
            "neural_stats": neural_stats,
            "attention_metrics": {
                "current_loss": attention_loss,
                "current_accuracy": attention_accuracy,
                "model_type": self.attention_system.model_type
            }
        }
    
    def export_session_data(self, session_id: str, format: str = "json") -> str:
        """Export session data in the requested format"""
        if session_id not in self.active_sessions:
            return json.dumps({"error": "Session not found"})
        
        session = self.active_sessions[session_id]
        
        if format.lower() == "json":
            return json.dumps(session, indent=2)
        else:
            return json.dumps({"error": f"Unsupported format: {format}"})
    
    def visualize_system(self) -> Dict[str, Any]:
        """Generate comprehensive visualization data for the system"""
        # Get tensor space visualization
        tensor_viz = self.tensor_numogram.visualize_tensor_space()
        
        # Get emotional landscape visualization
        # Just use first user in base numogram memory
        user_ids = list(self.base_numogram.user_memory.keys())
        emotional_viz = None
        if user_ids:
            emotional_viz = self.emotional_memory.visualize_emotional_landscape(user_ids[0])
        
        # Get neural evolution visualization
        neural_viz = self.hybrid_neural.visualize_learning_progress()
        
        # Get attention visualization
        attention_viz = None
        if user_ids:
            attention_viz = self.attention_system.visualize_attention(user_ids[0])
        
        # Get tesseract visualization if available
        tesseract_viz = None
        if self.tesseract_module:
            tesseract_viz = self.tensor_numogram.tesseract_integration(self.tesseract_module)
        
        return {
            "tensor_space": tensor_viz,
            "emotional_landscape": emotional_viz,
            "neural_evolution": neural_viz is not None,  # Just indicate if available
            "attention_system": attention_viz,
            "tesseract": tesseract_viz,
            "system_config": {
                "primary_mode": self.primary_mode,
                "mode_weights": self.mode_weights,
                "tensor_dimension": self.tensor_numogram.dimension,
                "attention_model_type": self.attention_system.model_type
            }
        }
    
    def save_system_state(self, directory: str) -> Dict[str, Any]:
        """Save complete system state for later resumption"""
        try:
            # Create directory if necessary
            os.makedirs(directory, exist_ok=True)
            
            # Save base numogram state
            base_numogram_path = os.path.join(directory, "base_numogram_state.json")
            self.base_numogram.save_user_memory()
            
            # Save tensor numogram state
            tensor_numogram_path = os.path.join(directory, "tensor_numogram_state.json")
            self.tensor_numogram.save_tensor_state(tensor_numogram_path)
            
            # Save emotional memory state
            emotional_memory_path = os.path.join(directory, "emotional_memory_state.json")
            self.emotional_memory.save_memory()
            
            # Save hybrid neural model
            hybrid_neural_path = os.path.join(directory, "hybrid_neural_model.pt")
            self.hybrid_neural.export_best_model()
            
            # Save attention model
            attention_model_path = os.path.join(directory, "attention_model.pt")
            self.attention_system.save_model(attention_model_path)
            
            # Save active sessions
            sessions_path = os.path.join(directory, "active_sessions.json")
            with open(sessions_path, 'w') as f:
                json.dump(self.active_sessions, f, indent=2)
            
            # Save system configuration
            config_path = os.path.join(directory, "system_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "system_id": self.system_id,
                    "version": self.system_version,
                    "created_at": self.created_at,
                    "saved_at": datetime.datetime.utcnow().isoformat(),
                    "config": self.config,
                    "primary_mode": self.primary_mode,
                    "mode_weights": self.mode_weights
                }, f, indent=2)
            
            return {
                "status": "success",
                "directory": directory,
                "saved_at": datetime.datetime.utcnow().isoformat(),
                "saved_files": [
                    "base_numogram_state.json",
                    "tensor_numogram_state.json",
                    "emotional_memory_state.json",
                    "hybrid_neural_model.pt",
                    "attention_model.pt",
                    "active_sessions.json",
                    "system_config.json"
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def load_system_state(self, directory: str) -> Dict[str, Any]:
        """Load system state from saved files"""
        try:
            # Check if directory exists
            if not os.path.isdir(directory):
                return {
                    "status": "error",
                    "message": f"Directory not found: {directory}"
                }
            
            # Load system configuration
            config_path = os.path.join(directory, "system_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                    # Update system metadata
                    self.system_id = config_data.get("system_id", self.system_id)
                    self.system_version = config_data.get("version", self.system_version)
                    self.created_at = config_data.get("created_at", self.created_at)
                    
                    # Update config
                    if "config" in config_data:
                        self.config.update(config_data["config"])
                    
                    # Update integration mode
                    if "primary_mode" in config_data:
                        self.primary_mode = config_data["primary_mode"]
                    
                    if "mode_weights" in config_data:
                        self.mode_weights = config_data["mode_weights"]
            
            # Load active sessions
            sessions_path = os.path.join(directory, "active_sessions.json")
            if os.path.exists(sessions_path):
                with open(sessions_path, 'r') as f:
                    self.active_sessions = json.load(f)
            
            # Load tensor numogram state
            tensor_numogram_path = os.path.join(directory, "tensor_numogram_state.json")
            if os.path.exists(tensor_numogram_path):
                self.tensor_numogram.load_tensor_state(tensor_numogram_path)
            
            # Load emotional memory state
            # Memory file path is already specified in config
            if os.path.exists(self.config.get("emotional_memory_file", "")):
                self.emotional_memory._load_memory()
            
            # Load hybrid neural model
            hybrid_neural_path = os.path.join(directory, "hybrid_neural_model.pt")
            if os.path.exists(hybrid_neural_path):
                try:
                    model_data = torch.load(hybrid_neural_path)
                    # Load model would be implemented in hybrid neural system
                except:
                    pass
            
            # Load attention model
            attention_model_path = os.path.join(directory, "attention_model.pt")
            if os.path.exists(attention_model_path):
                self.attention_system.load_model(attention_model_path)
            
            return {
                "status": "success",
                "directory": directory,
                "loaded_at": datetime.datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_zone_trajectory(self, user_id: str, lookback: int = 10) -> Dict[str, Any]:
        """
        Get zone trajectory information for a user with temporal patterns
        """
        # Get user's transition history
        user_data = self.base_numogram.user_memory.get(user_id, {})
        history = user_data.get("history", [])
        
        # Get most recent transitions
        recent_transitions = history[-lookback:] if len(history) > lookback else history
        
        # Extract zone sequence
        zone_sequence = []
        timestamps = []
        
        for transition in recent_transitions:
            zone_sequence.append(transition.get("next_zone", "1"))
            timestamps.append(transition.get("timestamp", ""))
        
        # Add current zone
        current_zone = user_data.get("zone", "1")
        if not zone_sequence or zone_sequence[-1] != current_zone:
            zone_sequence.append(current_zone)
            timestamps.append(datetime.datetime.utcnow().isoformat())
        
        # Get emotional signature path
        signature_path = self.emotional_memory.create_signature_path(user_id)
        
        # Get temporal pattern if circadian module available
        temporal_pattern = None
        if self.circadian_module:
            try:
                temporal_pattern = self.circadian_module.detect_temporal_pattern(user_id)
            except:
                pass
        
        # Get tensor hyperedges containing current zone
        active_edges = []
        for edge in self.tensor_numogram.hyperedges:
            if edge["active"] and current_zone in edge["zones"]:
                active_edges.append({
                    "id": edge["id"],
                    "name": edge["name"],
                    "zones": edge["zones"],
                    "strength": edge["strength"]
                })
        
        # Predict future trajectory
        future_zones = []
        
        # Use emotional trajectory prediction if available
        emotional_prediction = self.emotional_memory.predict_emotional_trajectory(user_id, steps=3)
        if emotional_prediction.get("status") != "no_trajectory":
            # Map emotional predictions to zones
            emotion_zone_affinities = {
                "joy": ["6", "9", "3"],
                "trust": ["1", "4", "6"],
                "fear": ["8", "2", "7"],
                "surprise": ["2", "5", "3"],
                "sadness": ["7", "4", "1"],
                "disgust": ["8", "6", "2"],
                "anger": ["8", "7", "5"],
                "anticipation": ["5", "3", "9"],
                "curiosity": ["5", "2", "7"],
                "awe": ["9", "3", "7"],
                "serenity": ["1", "6", "4"],
                "confusion": ["2", "5", "8"]
            }
            
            for prediction in emotional_prediction.get("predictions", []):
                emotion = prediction.get("emotion")
                confidence = prediction.get("confidence", 0.5)
                
                if emotion in emotion_zone_affinities:
                    # Use first affiliated zone
                    predicted_zone = emotion_zone_affinities[emotion][0]
                    future_zones.append({
                        "zone": predicted_zone,
                        "confidence": confidence,
                        "source": "emotional_trajectory",
                        "emotion": emotion
                    })
        
        # Also use tensor prediction for current zone
        tensor_predictions = self.tensor_numogram.predict_transitions(current_zone)
        if tensor_predictions:
            # Get top 3 predicted zones
            top_zones = sorted(tensor_predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for zone, prob in top_zones:
                # Add to future zones if not already predicted
                if not any(fz["zone"] == zone for fz in future_zones):
                    future_zones.append({
                        "zone": zone,
                        "confidence": prob,
                        "source": "tensor_prediction"
                    })
        
        # Combine results
        return {
            "user_id": user_id,
            "current_zone": current_zone,
            "zone_sequence": zone_sequence,
            "timestamps": timestamps,
            "signature_path": signature_path.get("signature_path") if signature_path.get("status") == "success" else None,
            "temporal_pattern": temporal_pattern,
            "active_hyperedges": active_edges,
            "predicted_trajectory": future_zones[:5]  # Limit to 5 predictions
        }
```
