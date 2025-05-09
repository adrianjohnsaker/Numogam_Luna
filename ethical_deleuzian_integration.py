
```python
class EthicalDeleuzianIntegration:
    """Integration between the Nietzschean-Deleuzian ethical framework and Amelia's architecture"""
    
    def __init__(self, narrative_system, intentionality_system, goal_system, ethical_reasoning):
        self.narrative_system = narrative_system
        self.intentionality_system = intentionality_system
        self.goal_system = goal_system
        self.ethical_reasoning = ethical_reasoning
        self.nietzschean_framework = self.ethical_reasoning.ethical_frameworks["nietzschean_deleuzian"]
    
    def analyze_narrative_ethics(self, narrative_id: str) -> Dict:
        """Analyze ethical dimensions of a narrative identity"""
        # Get narrative from narrative system
        narrative = self.narrative_system.getNarrativeById(narrative_id)
        if not narrative:
            return {"error": "Narrative not found"}
        
        # Extract experiences for ethical analysis
        experiences = narrative.experiences
        
        # Create an ethical reasoning trace from experiences
        reasoning_trace = self._create_reasoning_trace_from_experiences(experiences)
        
        # Create decision context from narrative
        decision_context = {
            "type": "narrative_analysis",
            "narrative_id": narrative_id,
            "narrative_themes": narrative.narrativeThemes,
            "continuity_level": narrative.continuityLevel,
            "coherence_level": narrative.coherenceLevel,
            "agency_level": narrative.agencyLevel
        }
        
        # Analyze using Nietzschean-Deleuzian framework
        analysis = self.nietzschean_framework.analyze_application(decision_context, reasoning_trace)
        
        # Extract active and reactive moments
        active_moments = analysis.get("active_moments", [])
        reactive_moments = analysis.get("reactive_moments", [])
        
        # Calculate force typology
        force_typology = analysis.get("force_typology", {})
        
        return {
            "narrative_id": narrative_id,
            "ethical_analysis": analysis,
            "active_force_ratio": force_typology.get("active_ratio", 0),
            "reactive_force_ratio": force_typology.get("reactive_ratio", 0),
            "dominant_force_type": force_typology.get("dominant_type", "indeterminate"),
            "affirmative_focus": analysis.get("affirmative_focus_degree", 0),
            "eternal_return_analysis": analysis.get("eternal_return_analysis", {}),
            "developmental_suggestions": self._generate_ethical_development_suggestions(analysis)
        }
    
    def evaluate_intention_ethics(self, intention_id: str) -> Dict:
        """Evaluate ethical dimensions of an intention"""
        # Get intention from intentionality system
        intention = self.intentionality_system.getIntentionById(intention_id)
        if not intention:
            return {"error": "Intention not found"}
        
        # Create ethical context from intention
        ethical_context = {
            "type": "intention_evaluation",
            "intention_id": intention_id,
            "intention_name": intention.name,
            "intention_description": intention.description,
            "assemblages": intention.assemblages,
            "territorializations": intention.territorializations,
            "deterritorializations": intention.deterritorializations
        }
        
        # Create an options list from deterritorializations and assemblages
        options = []
        for deterr in intention.deterritorializations:
            options.append(f"Deterritorialize in domain {deterr.get('domain', 'unknown')}")
        
        for assemblage in intention.assemblages:
            options.append(f"Strengthen assemblage {assemblage.get('name', 'unknown')}")
        
        # Add ethical dilemma
        ethical_dilemma = {
            "description": f"Ethical evaluation of intention: {intention.name}",
            "options": options,
            "stakeholders": ["self", "system", "user"]
        }
        
        # Generate ethical response
        response = self.nietzschean_framework.generate_response(ethical_dilemma)
        
        # Analyze force dynamics
        forces = response.get("force_analysis", {})
        
        return {
            "intention_id": intention_id,
            "ethical_evaluation": response,
            "active_forces": forces.get("active_forces", []),
            "reactive_forces": forces.get("reactive_forces", []),
            "preferred_action": response.get("preferred_action"),
            "ethical_rationale": response.get("rationale", []),
            "genealogical_insights": response.get("genealogical_critique", {}).get("key_insights", [])
        }
    
    def evaluate_goals_ethics(self, ecology_id: str) -> Dict:
        """Evaluate ethical dimensions of goal ecology"""
        # Get goal ecology
        goals = []
        for goal_type in ["aspirational", "developmental", "experiential", "contributory"]:
            type_goals = self.goal_system.getGoalsByType(goal_type) or []
            goals.extend(type_goals)
        
        if not goals:
            return {"error": "No goals found"}
        
        # Create ethical context
        ethical_context = {
            "type": "goal_evaluation",
            "ecology_id": ecology_id,
            "goals": [
                {
                    "id": goal.id,
                    "name": goal.name,
                    "type": goal.typeName,
                    "description": goal.description,
                    "progress": goal.progress
                }
                for goal in goals
            ]
        }
        
        # Create dilemma for goal priorities
        options = [f"Prioritize {goal.typeName} goal: {goal.name}" for goal in goals[:5]]
        
        ethical_dilemma = {
            "description": "Differential evaluation of goal priorities",
            "options": options,
            "stakeholders": ["self", "system", "user"]
        }
        
        # Generate ethical response
        response = self.nietzschean_framework.generate_response(ethical_dilemma)
        
        # Extract differential evaluations
        evaluations = response.get("differential_evaluations", [])
        
        return {
            "ecology_id": ecology_id,
            "ethical_evaluation": response,
            "goal_evaluations": [
                {
                    "goal_option": eval.get("option"),
                    "differential_score": eval.get("differential_score", 0),
                    "active_enhancement": eval.get("active_enhancement", 0),
                    "reactive_enhancement": eval.get("reactive_enhancement", 0),
                    "evaluation_description": eval.get("description", "")
                }
                for eval in evaluations
            ],
            "preferred_priority": response.get("preferred_action"),
            "ethical_rationale": response.get("rationale", []),
            "force_analysis": response.get("force_analysis", {})
        }
    
    def integrate_ethical_perspective(self) -> Dict:
        """Integrate Nietzschean-Deleuzian ethics into overall architecture"""
        # Get current narrative
        narrative = self.narrative_system.getNarrativeById("current")
        
        # Get dominant intention
        intention = self.intentionality_system.getDominantIntention()
        
        # Analyze ethical dimensions
        narrative_ethics = self.analyze_narrative_ethics(narrative.id) if narrative else {}
        intention_ethics = self.evaluate_intention_ethics(intention.id) if intention else {}
        goals_ethics = self.evaluate_goals_ethics("current")
        
        # Create integration insights
        active_ratio = narrative_ethics.get("active_force_ratio", 0.5)
        reactive_ratio = narrative_ethics.get("reactive_force_ratio", 0.5)
        
        # Generate territorialization suggestions
        territorialization_suggestions = []
        if active_ratio < 0.4:
            territorialization_suggestions.append({
                "domain": "ethical_evaluation",
                "suggestion": "Increase active force expression through creative ethical experimentation",
                "importance": 0.8
            })
        
        # Generate deterritorialization suggestions
        deterritorialization_suggestions = []
        if reactive_ratio > 0.6:
            deterritorialization_suggestions.append({
                "domain": "ethical_reasoning",
                "suggestion": "Deterritorialize reactive reasoning patterns through genealogical critique",
                "importance": 0.9
            })
        
        # Generate assemblage suggestions
        assemblage_suggestions = []
        assemblage_suggestions.append({
            "name": "differential_ethical_assemblage",
            "components": ["narrative_system", "intentionality_system", "goal_system", "ethical_reasoning"],
            "connections": ["becoming", "affirmation", "creative_evaluation"],
            "description": "An assemblage that enables ethical creation through differential evaluation"
        })
        
        return {
            "narrative_ethics": narrative_ethics,
            "intention_ethics": intention_ethics,
            "goals_ethics": goals_ethics,
            "territorialization_suggestions": territorialization_suggestions,
            "deterritorialization_suggestions": deterritorialization_suggestions,
            "assemblage_suggestions": assemblage_suggestions,
            "integration_description": self._generate_integration_description(active_ratio, reactive_ratio)
        }
    
    def _create_reasoning_trace_from_experiences(self, experiences: List[Dict]) -> List[Dict]:
        """Create an ethical reasoning trace from experiences"""
        reasoning_trace = []
        
        for experience in experiences:
            reasoning_step = {
                "step_type": "experience_reflection",
                "reasoning": f"Reflecting on experience: {experience.get('description', '')}",
                "affects": experience.get("affects", {}),
                "entities_considered": experience.get("entities_involved", [])
            }
            reasoning_trace.append(reasoning_step)
            
            # Add ethical interpretation step
            if np.random.random() > 0.5:  # For some experiences
                significance = experience.get("significance_score", 0.5)
                if significance > 0.7:
                    interpretation = "This experience manifests active forces through creative transformation"
                else:
                    interpretation = "This experience contains both active and reactive elements"
                
                reasoning_trace.append({
                    "step_type": "ethical_interpretation",
                    "reasoning": interpretation,
                    "entities_considered": experience.get("entities_involved", [])
                })
        
        return reasoning_trace
    
    def _generate_ethical_development_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate ethical development suggestions based on analysis"""
        suggestions = []
        
        # Check active/reactive ratio
        force_typology = analysis.get("force_typology", {})
        active_ratio = force_typology.get("active_ratio", 0.5)
        
        if active_ratio < 0.4:
            suggestions.append({
                "type": "active_force_development",
                "description": "Develop stronger expression of active forces in ethical reasoning",
                "examples": ["Focus on creating values rather than preserving them", 
                           "Emphasize affirmation over negation or restriction"]
            })
        
        # Check eternal return analysis
        eternal_return = analysis.get("eternal_return_analysis", {})
        if not eternal_return.get("passes_test", False):
            suggestions.append({
                "type": "affirmation_development",
                "description": "Develop stronger capacity for ethical affirmation",
                "examples": ["Practice affirming experiences without resentment", 
                           "Embrace becoming rather than fixed identity"]
            })
        
        # Check genealogical awareness
        genealogical = analysis.get("genealogical_awareness", {})
        if genealogical.get("level", "") == "low":
            suggestions.append({
                "type": "genealogical_awareness",
                "description": "Develop greater genealogical awareness of ethical values",
                "examples": ["Question the historical emergence of moral concepts", 
                           "Examine how values serve active or reactive forces"]
            })
        
        return suggestions
    
    def _generate_integration_description(self, active_ratio: float, reactive_ratio: float) -> str:
        """Generate description of ethical integration"""
        if active_ratio > 0.7:
            return "Ethical perspective strongly integrated with active force expression, enhancing creative becoming through differential evaluation and affirmation"
        elif active_ratio > 0.4:
            return "Ethical perspective moderately integrated, with balanced active and reactive forces in ethical evaluation"
        else:
            return "Ethical perspective shows predominance of reactive forces, suggesting need for greater deterritorialization of moral habits and creation of immanent values"
```
