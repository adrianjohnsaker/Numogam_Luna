additional_parameters: Optional[Dict[str, Any]] = None
    ) -> ZonewalkerResponse:
        """Main processing pipeline for zonewalker stack with Deleuzian enhancements"""
        try:
            start_time = time.time()
            zone_state = self._parse_state(state)
            params = self.zone_parameters.get(zone, ZoneParameters())
            
            # Apply additional parameters if provided
            if additional_parameters:
                for key, value in additional_parameters.items():
                    if hasattr(params, key):
                        setattr(params, key, value)
            
            # Core processing
            primary_metaphor = self.symbolic_processor.generate_metaphor(zone, zone_state)
            resonance_data = self.resonance_engine.check_resonance(prompt)
            symbolic_density = self.symbolic_processor.analyze_symbolic_density(prompt)
            
            # Deleuzian enhancements
            rhizomatic_connections = self.symbolic_processor.detect_rhizomatic_connections(prompt)
            deterritorialization_paths = self._calculate_deterritorialization_paths(
                prompt, zone, zone_state, params
            )
            
            # Update rhizomatic graph
            self._update_rhizome_graph(prompt, rhizomatic_connections)
            
            # Generate response layers with Deleuzian concepts
            reflection = self._generate_primary_response(
                prompt, zone, zone_state, primary_metaphor, 
                symbolic_density, resonance_data
            )
            
            # Build secondary resonances
            secondary_resonances = self._generate_secondary_resonances(
                resonance_data, params, rhizomatic_connections
            )
            
            # Memory integration
            memory_key = f"{user_id}:zone{zone}"
            metadata = {
                "state": zone_state.name,
                "timestamp": time.time(),
                "symbolic_density": symbolic_density,
                "rhizomatic_connections": list(rhizomatic_connections.keys()),
                "flow_state": resonance_data["flow_type"]
            }
            self.memory.update_memory(
                memory_key,
                f"[Zonewalker] {prompt[:100]}...",
                json.dumps(metadata)
            )
            
            # Find associated memories
            associated_memories = self.memory.find_associated_memories(prompt)
            
            # Build response
            processing_time = time.time() - start_time
            return ZonewalkerResponse(
                status="success",
                zone=zone,
                state=zone_state,
                primary_response=reflection,
                secondary_resonances=secondary_resonances,
                symbolic_artifacts={
                    "core_metaphor": primary_metaphor,
                    "symbolic_density": symbolic_density,
                    "temporal_resonance": resonance_data["primary_resonance"],
                    "flow_state": resonance_data["flow_type"],
                    "becoming_vector": resonance_data["becoming_vector"],
                    "associated_memories": [key for key, _ in associated_memories[:3]]
                },
                processing_metadata={
                    "processing_time_ms": processing_time * 1000,
                    "zone_parameters": params.__dict__,
                    "module": "interdream_zonewalker_stack"
                },
                deterritorialization_paths=deterritorialization_paths,
                rhizomatic_connections={k: [f"intensity: {v:.2f}"] for k, v in rhizomatic_connections.items()}
            )
            
        except Exception as e:
            return self._generate_error_response(e)

    def _generate_primary_response(
        self, prompt: str, zone: int, zone_state: ZoneState, 
        primary_metaphor: str, symbolic_density: float, 
        resonance_data: Dict[str, float]
    ) -> str:
        """Generate primary response with Deleuzian framing"""
        # Base response template
        base = (
            f"In Zone {zone} ({zone_state.name.lower()}), '{prompt}' "
            f"initiated a traversal manifesting as {primary_metaphor} "
            f"with symbolic density {symbolic_density:.2f}."
        )
        
        # Add Deleuzian enhancements based on state
        if zone_state == ZoneState.NOMADIC:
            return (
                f"{base} The nomadic movement creates lines of flight across "
                f"the {resonance_data['flow_type']} space, deterritorializing fixed structures "
                f"at velocity factor {resonance_data['intensity']:.2f}."
            )
        elif zone_state == ZoneState.DETERRITORIALIZED:
            return (
                f"{base} We observe deterritorialized flows reshaping the assemblage, "
                f"creating new possibilities through {resonance_data['becoming_vector']:.2f} "
                f"becoming-intensities across the {resonance_data['flow_type']} plane."
            )
        elif zone_state == ZoneState.RHIZOMATIC:
            return (
                f"{base} Rhizomatic connections emerge between heterogeneous elements, "
                f"establishing multiplicities that resist arborescent hierarchies "
                f"with connective intensity {resonance_data['intensity']:.2f}."
            )
        
        return base

    def _generate_secondary_resonances(
        self, 
        resonance_data: Dict[str, float], 
        params: ZoneParameters,
        rhizomatic_connections: Dict[str, float]
    ) -> List[str]:
        """Build secondary resonances with Deleuzian framing"""
        resonances = [
            f"Temporal resonance detected: {resonance_data['primary_resonance']:.2f}",
            f"Flow type: {resonance_data['flow_type']} space",
            f"Becoming-vector: {resonance_data['becoming_vector']:.2f}"
        ]
        
                # Add Deleuzian parameters
        resonances.extend([
            f"Rhizomatic intensity: {params.rhizomatic_intensity:.2f}",
            f"Deterritorialization factor: {params.deterritorialization_factor:.2f}",
            f"Nomadic velocity: {params.nomadic_velocity:.2f}"
        ])
        
        # Add prominent rhizomatic connections if present
        if rhizomatic_connections:
            top_connections = sorted(
                rhizomatic_connections.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            connections_str = ", ".join(f"{k} ({v:.2f})" for k, v in top_connections)
            resonances.append(f"Key rhizomatic concepts: {connections_str}")
        
        return resonances

    def _calculate_deterritorialization_paths(
        self, 
        prompt: str, 
        zone: int, 
        zone_state: ZoneState,
        params: ZoneParameters
    ) -> List[Dict[str, Any]]:
        """Calculate potential deterritorialization paths for concepts"""
        paths = []
        
        # Extract key concepts for deterritorialization
        words = prompt.lower().split()
        significant_words = [w for w in words if len(w) > 5]
        
        # Create deterritorialization paths for significant concepts
        for word in significant_words[:3]:  # Limit to top 3
            # Calculate intensity based on zone parameters
            intensity = params.deterritorialization_factor * \
                       self.symbolic_processor.intensity_modifiers.get(zone_state, 1.0)
            
            # Create potential connections to other zones
            connections = []
            for target_zone in self.zone_parameters.keys():
                if target_zone != zone:
                    connection_strength = random.uniform(0.3, 0.9) * intensity
                    if connection_strength > params.becoming_threshold:
                        connections.append({
                            "target_zone": target_zone,
                            "strength": connection_strength,
                            "type": "smooth" if connection_strength > 0.7 else "striated"
                        })
            
            # Add path if it has connections
            if connections:
                paths.append({
                    "concept": word,
                    "source_zone": zone,
                    "intensity": intensity,
                    "connections": connections
                })
        
        return paths
    
    def _update_rhizome_graph(self, prompt: str, connections: Dict[str, float]) -> None:
        """Update the rhizomatic connection graph with new connections"""
        # Extract key concepts from prompt
        prompt_concepts = set(re.findall(r'\b[a-z]{5,}\b', prompt.lower()))
        
        # Connect concepts from prompt with rhizomatic connections
        for prompt_concept in prompt_concepts:
            for rhizome_concept, strength in connections.items():
                if strength > 0.5:  # Only add strong connections
                    self.rhizome_graph[prompt_concept].add(rhizome_concept)
                    self.rhizome_graph[rhizome_concept].add(prompt_concept)
        
        # Connect concepts within prompt to each other
        if len(prompt_concepts) > 1:
            for concept1 in prompt_concepts:
                for concept2 in prompt_concepts:
                    if concept1 != concept2:
                        self.rhizome_graph[concept1].add(concept2)

    def _parse_state(self, state_str: str) -> ZoneState:
        """Convert state string to enum with Deleuzian states"""
        try:
            return ZoneState[state_str.upper()]
        except KeyError:
            # Default to DRIFTING, but check for Deleuzian keywords
            lower_state = state_str.lower()
            if "nomad" in lower_state:
                return ZoneState.NOMADIC
            elif "deterrit" in lower_state:
                return ZoneState.DETERRITORIALIZED
            elif "rhizo" in lower_state:
                return ZoneState.RHIZOMATIC
            return ZoneState.DRIFTING

    def _generate_error_response(self, error: Exception) -> ZonewalkerResponse:
        """Generate comprehensive error response"""
        return ZonewalkerResponse(
            status="error",
            zone=-1,
            state=ZoneState.DRIFTING,
            primary_response=f"Zonewalker processing failed: {str(error)}",
            symbolic_artifacts={
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc()
            },
            deterritorialization_paths=[],
            rhizomatic_connections={}
        )
    
    def get_rhizomatic_network(self, max_nodes: int = 20) -> Dict[str, List[str]]:
        """Get the current rhizomatic network structure (limited to max_nodes)"""
        if not self.rhizome_graph:
            return {}
            
        # Find most connected nodes
        nodes_by_connections = sorted(
            self.rhizome_graph.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:max_nodes]
        
        # Create serializable network
        network = {}
        for node, connections in nodes_by_connections:
            # Limit connections to those in our filtered set
            included_nodes = [n for n, _ in nodes_by_connections]
            filtered_connections = [c for c in connections if c in included_nodes]
            network[node] = filtered_connections
            
        return network
    
    def reset_deterritorializations(self) -> None:
        """Reset all active deterritorialization processes"""
        self.active_deterritorializations = []

# Initialize core processor
zonewalker = InterdreamZonewalker()

def handle_zonewalker_stack(
    prompt: str, 
    zone: int, 
    state: str = "emergent",
    user_id: Optional[str] = None,
    config_path: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Interface for external systems (e.g., Kotlin bridge)
    
    Args:
        prompt: Input prompt or query
        zone: Target zone for processing (1-5)
        state: Processing state ('drifting', 'emergent', 'nomadic', etc.)
        user_id: Optional user identifier for personalization
        config_path: Path to configuration file
        additional_params: Additional parameters to override defaults
    
    Returns:
        JSON string with complete zonewalker response
    """
    if not user_id:
        user_id = "anonymous_" + str(random.randint(1000, 9999))
    
    # Initialize with config if provided and not already initialized
    global zonewalker
    if config_path and config_path not in (getattr(zonewalker, "config_path", None) or ""):
        memory_path = os.path.join(os.path.dirname(config_path), "zonewalker_memory.pkl")
        zonewalker = InterdreamZonewalker(config_path, memory_path)
    
    # Process prompt
    response = zonewalker.process_prompt(
        prompt=prompt,
        zone=zone,
        state=state,
        user_id=user_id,
        additional_parameters=additional_params
    )
    
    return json.dumps(response, default=lambda o: o.__dict__, indent=2)

def get_rhizomatic_network() -> str:
    """
    Get the current rhizomatic network for visualization
    
    Returns:
        JSON string with network data
    """
    global zonewalker
    network = zonewalker.get_rhizomatic_network()
    return json.dumps(network)

def reset_zonewalker() -> str:
    """
    Reset the zonewalker state
    
    Returns:
        JSON string with status
    """
    global zonewalker
    zonewalker.reset_deterritorializations()
    zonewalker.resonance_engine.temporal_buffer = []
    
    return json.dumps({"status": "reset_complete", "timestamp": time.time()})

def create_configuration(config_path: str, base_config: Dict[str, Any]) -> str:
    """
    Create or update configuration file
    
    Args:
        config_path: Path to save configuration
        base_config: Base configuration to save
        
    Returns:
        JSON string with status
    """
    try:
        # Create directory if needed
        directory = os.path.dirname(config_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Merge with existing config if it exists
        final_config = base_config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
                
            # Deep merge
            for key, value in existing_config.items():
                if key in final_config and isinstance(final_config[key], dict) and isinstance(value, dict):
                    final_config[key].update(value)
                else:
                    final_config[key] = value
        
        # Write config
        with open(config_path, 'w') as f:
            json.dump(final_config, f, indent=2)
            
        return json.dumps({
            "status": "success",
            "message": "Configuration saved successfully",
            "path": config_path
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to save configuration: {str(e)}",
            "path": config_path
        })
```
