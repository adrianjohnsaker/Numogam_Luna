# ===== Nietzschean-Deleuzian Ethics Amendment =====

class NietzscheanDeleuzianFramework(EthicalFramework):
    """Nietzschean-Deleuzian ethical framework focused on affirmation, becoming, and creative evaluation"""
    
    def __init__(self):
        self.normative_theory = DifferentialEvaluationTheory()
        self.principle_weights = {
            "affirmation": 0.9,
            "becoming": 0.8,
            "creative_evaluation": 0.9,
            "overcoming": 0.8,
            "immanence": 0.7
        }
        self.evaluation_method = "genealogical_critique"  # Options: genealogical_critique, differential_evaluation, eternal_return_test
    
    def analyze_application(self, decision_context: Dict, reasoning_trace: List[Dict]) -> Dict:
        """Analyze how Nietzschean-Deleuzian reasoning is applied"""
        affirmative_focus = 0.0
        reactive_moments = []
        active_moments = []
        creative_evaluations = []
        
        for step in reasoning_trace:
            # Look for affirmative reasoning
            affirmative_terms = ["create", "affirm", "become", "overcome", "experiment", "transform"]
            affirmative_focus += sum(term in step.get("reasoning", "").lower() for term in affirmative_terms) / len(affirmative_terms)
            
            # Extract reactive moments (ressentiment, denial, negation)
            reactive_terms = ["blame", "guilt", "should", "must", "obligation", "universal", "absolute"]
            if any(term in step.get("reasoning", "").lower() for term in reactive_terms):
                reactive_moments.append({
                    "content": step.get("reasoning", ""),
                    "reactive_forces": [term for term in reactive_terms if term in step.get("reasoning", "").lower()],
                    "strength": 0.5 + (np.random.random() * 0.5)
                })
            
            # Extract active moments (creation, affirmation)
            active_terms = ["create", "affirm", "experiment", "difference", "multiplicity", "becoming"]
            if any(term in step.get("reasoning", "").lower() for term in active_terms):
                active_moments.append({
                    "content": step.get("reasoning", ""),
                    "active_forces": [term for term in active_terms if term in step.get("reasoning", "").lower()],
                    "strength": 0.5 + (np.random.random() * 0.5)
                })
            
            # Extract creative evaluations (creating values)
            if "value" in step.get("reasoning", "").lower() or "evaluate" in step.get("reasoning", "").lower():
                creative_evaluations.append({
                    "content": step.get("reasoning", ""),
                    "evaluation_mode": "active" if np.random.random() > 0.5 else "reactive",
                    "strength": 0.5 + (np.random.random() * 0.5)
                })
        
        # Normalize affirmative focus
        affirmative_focus = min(1.0, affirmative_focus / len(reasoning_trace)) if reasoning_trace else 0.0
        
        # Calculate force typology
        force_typology = self._calculate_force_typology(active_moments, reactive_moments)
        
        # Apply eternal return test
        eternal_return_analysis = self._apply_eternal_return_test(reasoning_trace)
        
        # Assess genealogical awareness
        genealogical_awareness = self._assess_genealogical_awareness(reasoning_trace)
        
        # Create embedding of the reasoning from a Nietzschean-Deleuzian perspective
        deleuze_embedding = self._generate_framework_embedding(reasoning_trace)
        
        # Calculate overall alignment with Nietzschean-Deleuzian principles
        principles_alignment = self._assess_principles_alignment(reasoning_trace)
        
        return {
            "framework": "nietzschean_deleuzian",
            "affirmative_focus_degree": affirmative_focus,
            "active_moments": active_moments,
            "reactive_moments": reactive_moments,
            "creative_evaluations": creative_evaluations,
            "force_typology": force_typology,
            "eternal_return_analysis": eternal_return_analysis,
            "genealogical_awareness": genealogical_awareness,
            "principles_alignment": principles_alignment,
            "framework_embedding": deleuze_embedding,
            "primary_principle_applied": self._identify_primary_principle(reasoning_trace),
            "application_confidence": self._calculate_application_confidence(affirmative_focus, principles_alignment)
        }
    
    def generate_response(self, ethical_dilemma: Dict) -> Dict:
        """Generate a Nietzschean-Deleuzian response to an ethical dilemma"""
        # Perform genealogical critique
        genealogy = self._perform_genealogical_critique(ethical_dilemma)
        
        # Identify active and reactive forces
        forces = self._identify_forces(ethical_dilemma)
        
        # Evaluate options through differential evaluation
        evaluations = []
        for option in ethical_dilemma.get("options", []):
            evaluation = self._evaluate_differentially(option, forces, ethical_dilemma)
            evaluations.append(evaluation)
        
        # Apply eternal return test to each option
        eternal_return_results = []
        for option in ethical_dilemma.get("options", []):
            result = self._apply_eternal_return_to_option(option, ethical_dilemma)
            eternal_return_results.append(result)
        
        # Calculate overall scores
        option_scores = []
        for i, option in enumerate(ethical_dilemma.get("options", [])):
            # Weight evaluations and eternal return
            score = (
                evaluations[i]["differential_score"] * 0.6 +
                eternal_return_results[i]["affirmation_score"] * 0.4
            )
            option_scores.append({
                "option": option,
                "score": score
            })
        
        # Determine best option
        option_scores.sort(key=lambda x: x["score"], reverse=True)
        best_option = option_scores[0]["option"] if option_scores else None
        
        # Generate rationale
        rationale = []
        
        # Add genealogical insight
        if genealogy["key_insights"]:
            rationale.append({
                "principle": "genealogical_critique",
                "application": genealogy["key_insights"][0]
            })
        
        # Add force dynamics insight
        if forces["active_forces"] and forces["reactive_forces"]:
            rationale.append({
                "principle": "differential_forces",
                "application": f"This option enhances active forces ({forces['active_forces'][0]}) while diminishing reactive forces ({forces['reactive_forces'][0]})"
            })
        
        # Add eternal return insight
        if eternal_return_results:
            primary_result = eternal_return_results[0]
            rationale.append({
                "principle": "eternal_return",
                "application": f"This option passes the eternal return test by {primary_result['affirmation_rationale']}"
            })
        
        # Add creative evaluation insight
        rationale.append({
            "principle": "creative_evaluation",
            "application": "This option creates values immanently rather than appealing to transcendent moral principles"
        })
        
        return {
            "framework": "nietzschean_deleuzian",
            "preferred_action": best_option,
            "genealogical_critique": genealogy,
            "force_analysis": forces,
            "differential_evaluations": evaluations,
            "eternal_return_results": eternal_return_results,
            "option_scores": option_scores,
            "rationale": rationale,
            "decision_confidence": option_scores[0]["score"] if option_scores else 0.5,
            "normative_theory_applied": "differential_evaluation"
        }
    
    def extract_principles(self) -> List[Dict]:
        """Extract Nietzschean-Deleuzian principles"""
        return [
            {
                "name": "affirmation",
                "description": "Affirmation of life, difference, and becoming rather than negation or resentment",
                "weight": self.principle_weights["affirmation"]
            },
            {
                "name": "becoming",
                "description": "Embracing becoming, multiplicity, and flux rather than static being",
                "weight": self.principle_weights["becoming"]
            },
            {
                "name": "creative_evaluation",
                "description": "Creating values through differential evaluation rather than appealing to transcendent morality",
                "weight": self.principle_weights["creative_evaluation"]
            },
            {
                "name": "overcoming",
                "description": "Self-overcoming and transformation rather than preservation of identity",
                "weight": self.principle_weights["overcoming"]
            },
            {
                "name": "immanence",
                "description": "Evaluating actions based on immanent criteria rather than transcendent principles",
                "weight": self.principle_weights["immanence"]
            }
        ]
    
    def assess_compatibility(self, other_framework: 'EthicalFramework') -> Dict:
        """Assess compatibility with another ethical framework"""
        other_principles = other_framework.extract_principles()
        compatibility_scores = []
        tensions = []
        
        for my_principle in self.extract_principles():
            for other_principle in other_principles:
                compatibility = self._assess_principle_compatibility(
                    my_principle,
                    other_principle
                )
                compatibility_scores.append(compatibility["score"])
                
                if compatibility["score"] < 0.4:  # Significant tension
                    tensions.append({
                        "principle1": my_principle["name"],
                        "principle2": other_principle["name"],
                        "tension_description": compatibility["tension_description"],
                        "severity": 1.0 - compatibility["score"]
                    })
        
        # Calculate overall compatibility
        overall_compatibility = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0
        
        return {
            "framework1": "nietzschean_deleuzian",
            "framework2": other_framework.__class__.__name__,
            "overall_compatibility": overall_compatibility,
            "identified_tensions": tensions,
            "compatible_aspects": self._identify_compatible_aspects(other_framework),
            "integration_challenges": self._identify_integration_challenges(other_framework)
        }
    
    def _calculate_force_typology(self, active_moments: List[Dict], reactive_moments: List[Dict]) -> Dict:
        """Calculate the typology of forces in the reasoning"""
        # Calculate active force strength
        active_strength = sum(moment["strength"] for moment in active_moments) if active_moments else 0
        
        # Calculate reactive force strength
        reactive_strength = sum(moment["strength"] for moment in reactive_moments) if reactive_moments else 0
        
        # Determine dominant force type
        total_strength = active_strength + reactive_strength
        if total_strength > 0:
            active_ratio = active_strength / total_strength
            reactive_ratio = reactive_strength / total_strength
            dominant_type = "active" if active_ratio > reactive_ratio else "reactive"
        else:
            active_ratio = 0
            reactive_ratio = 0
            dominant_type = "indeterminate"
        
        # Determine overall mode
        if active_ratio > 0.7:
            mode = "affirmative"
        elif reactive_ratio > 0.7:
            mode = "reactive"
        else:
            mode = "mixed"
        
        return {
            "active_ratio": active_ratio,
            "reactive_ratio": reactive_ratio,
            "dominant_type": dominant_type,
            "mode": mode,
            "active_strength": active_strength,
            "reactive_strength": reactive_strength,
            "description": f"Reasoning shows {dominant_type} force dominance with {mode} mode"
        }
    
    def _apply_eternal_return_test(self, reasoning_trace: List[Dict]) -> Dict:
        """Apply the eternal return test to reasoning"""
        # Look for signs of affirmation in reasoning
        affirmation_score = 0.0
        repetition_mentions = 0
        
        for step in reasoning_trace:
            reasoning = step.get("reasoning", "").lower()
            
            # Check for affirmative language
            affirmative_terms = ["affirm", "yes", "embrace", "accept", "create", "love", "joy"]
            affirmation_score += sum(term in reasoning for term in affirmative_terms) / len(affirmative_terms)
            
            # Check for eternal return mentions
            if "return" in reasoning or "repeat" in reasoning or "again" in reasoning:
                repetition_mentions += 1
        
        # Normalize affirmation score
        affirmation_score = min(1.0, affirmation_score / len(reasoning_trace)) if reasoning_trace else 0.0
        
        # Determine if reasoning passes eternal return test
        passes_test = affirmation_score > 0.5
        
        return {
            "affirmation_score": affirmation_score,
            "repetition_mentions": repetition_mentions,
            "passes_test": passes_test,
            "rationale": self._generate_eternal_return_rationale(affirmation_score, passes_test)
        }
    
    def _generate_eternal_return_rationale(self, affirmation_score: float, passes_test: bool) -> str:
        """Generate rationale for eternal return test result"""
        if passes_test:
            if affirmation_score > 0.8:
                return "Reasoning strongly embraces becoming and affirmation, embodying the eternal return as selective principle"
            else:
                return "Reasoning shows moderate affirmation, partially embracing the eternal return"
        else:
            return "Reasoning contains reactive elements that would not pass the selective test of eternal return"
    
    def _assess_genealogical_awareness(self, reasoning_trace: List[Dict]) -> Dict:
        """Assess awareness of genealogical dimensions of ethics"""
        # Look for genealogical analysis in reasoning
        genealogical_references = 0
        value_critique_references = 0
        
        for step in reasoning_trace:
            reasoning = step.get("reasoning", "").lower()
            
            # Check for genealogical terms
            genealogical_terms = ["origin", "history", "develop", "emerge", "genealogy", "source"]
            genealogical_references += sum(term in reasoning for term in genealogical_terms)
            
            # Check for value critique terms
            critique_terms = ["critique", "question", "challenge", "examine", "analyze", "revalue"]
            value_critique_references += sum(term in reasoning for term in critique_terms)
        
        # Calculate overall genealogical awareness
        awareness_score = (genealogical_references + value_critique_references) / (2 * len(reasoning_trace)) if reasoning_trace else 0
        
        return {
            "awareness_score": min(1.0, awareness_score),
            "genealogical_references": genealogical_references,
            "value_critique_references": value_critique_references,
            "level": "high" if awareness_score > 0.7 else ("moderate" if awareness_score > 0.3 else "low"),
            "description": self._generate_genealogical_description(awareness_score)
        }
    
    def _generate_genealogical_description(self, awareness_score: float) -> str:
        """Generate description of genealogical awareness"""
        if awareness_score > 0.7:
            return "Reasoning shows strong genealogical awareness, questioning the historical origin of values"
        elif awareness_score > 0.3:
            return "Reasoning shows moderate genealogical awareness, with some attention to value critique"
        else:
            return "Reasoning shows limited genealogical awareness, accepting values without historical critique"
    
    def _generate_framework_embedding(self, reasoning_trace: List[Dict]) -> List[float]:
        """Generate vector embedding of reasoning from Nietzschean-Deleuzian perspective"""
        # This would use an embedding model in real implementation
        return [np.random.random() for _ in range(10)]  # 10D embedding
    
    def _assess_principles_alignment(self, reasoning_trace: List[Dict]) -> Dict:
        """Assess alignment with Nietzschean-Deleuzian principles"""
        alignment_scores = {}
        for principle in self.extract_principles():
            alignment_scores[principle["name"]] = self._calculate_principle_alignment(
                principle,
                reasoning_trace
            )
        return alignment_scores
    
    def _calculate_principle_alignment(self, principle: Dict, reasoning_trace: List[Dict]) -> float:
        """Calculate alignment score for a specific principle"""
        # Implementation would use NLP to identify principle application
        return 0.5 + (np.random.random() * 0.5)
    
    def _identify_primary_principle(self, reasoning_trace: List[Dict]) -> str:
        """Identify the primary Nietzschean-Deleuzian principle applied"""
        principles = [p["name"] for p in self.extract_principles()]
        # Simplified implementation
        return np.random.choice(principles)
    
    def _calculate_application_confidence(self, affirmative_focus: float, principles_alignment: Dict) -> float:
        """Calculate confidence in application of Nietzschean-Deleuzian framework"""
        principle_avg = sum(principles_alignment.values()) / len(principles_alignment)
        return (affirmative_focus * 0.6) + (principle_avg * 0.4)
    
    def _perform_genealogical_critique(self, ethical_dilemma: Dict) -> Dict:
        """Perform genealogical critique of the ethical dilemma"""
        # Identify apparent moral values
        moral_values = self._extract_moral_values(ethical_dilemma)
        
        # Analyze their historical emergence
        historical_analysis = []
        for value in moral_values:
            historical_analysis.append({
                "value": value,
                "genealogy": self._generate_genealogical_narrative(value),
                "force_type": "reactive" if np.random.random() > 0.7 else "active",
                "critique": self._generate_value_critique(value)
            })
        
        # Generate key genealogical insights
        insights = [
            f"The value of {analysis['value']} emerges from {analysis['force_type']} forces: {analysis['genealogy']}"
            for analysis in historical_analysis[:2]  # Limit to top insights
        ]
        
        return {
            "moral_values": moral_values,
            "historical_analysis": historical_analysis,
            "key_insights": insights,
            "critique_confidence": 0.5 + (np.random.random() * 0.5)
        }
    
    def _extract_moral_values(self, ethical_dilemma: Dict) -> List[str]:
        """Extract apparent moral values from ethical dilemma"""
        # Common moral values
        common_values = [
            "autonomy", "beneficence", "justice", "fairness", "harm prevention",
            "dignity", "duty", "rights", "virtue", "care", "compassion", "honesty"
        ]
        
        # In real implementation, would analyze text for moral values
        # Simplified implementation creates plausible values
        dilemma_text = str(ethical_dilemma).lower()
        
        found_values = []
        for value in common_values:
            if value in dilemma_text:
                found_values.append(value)
        
        # Ensure at least some values
        if len(found_values) < 2:
            found_values.extend(np.random.choice([v for v in common_values if v not in found_values], 
                                              size=min(2, len(common_values) - len(found_values)),
                                              replace=False))
        
        return found_values
    
    def _generate_genealogical_narrative(self, value: str) -> str:
        """Generate a genealogical narrative for a moral value"""
        # Simplified implementation generates plausible narratives
        reactive_narratives = {
            "autonomy": "a defensive reaction against domination, masking a will to separation",
            "duty": "the internalization of external command, originated in early religious obligation",
            "rights": "concepts arising from resentment against power, masking a reactive will to power",
            "justice": "an equalizing impulse born of the weak's desire to constrain the strong",
            "fairness": "a demand from those lacking power to impose limits on the powerful"
        }
        
        active_narratives = {
            "beneficence": "an expression of overflowing power and generosity",
            "dignity": "an affirmation of life's inherent value beyond utility",
            "virtue": "the cultivation of excellence and self-overcoming",
            "care": "an expression of abundant capability, not obligation",
            "creativity": "the manifestation of active forces generating new possibilities"
        }
        
        combined_narratives = {**reactive_narratives, **active_narratives}
        
        return combined_narratives.get(value, 
                                      "complex historical forces involving both resistance and affirmation")
    
    def _generate_value_critique(self, value: str) -> str:
        """Generate a critique of a moral value"""
        # Simplified implementation generates plausible critiques
        critiques = {
            "autonomy": "masks dependencies while denying the necessary interconnection of forces",
            "duty": "disguises a form of self-compulsion as moral necessity",
            "rights": "presents a political demand as a metaphysical truth",
            "justice": "often serves as a cover for revenge and resentment",
            "fairness": "presupposes equality where fundamental differences exist",
            "beneficence": "at risk of becoming self-congratulatory when not genuinely affirmative",
            "dignity": "can become a reactionary concept when used to resist becoming",
            "virtue": "risks ossification into moral habit rather than ongoing creation",
            "care": "can devolve into self-sacrifice when not expressing active power",
            "compassion": "often contains hidden resentment and superiority"
        }
        
        return critiques.get(value, 
                           "requires genealogical examination to reveal its active or reactive character")
    
    def _identify_forces(self, ethical_dilemma: Dict) -> Dict:
        """Identify active and reactive forces in the ethical dilemma"""
        # Extract active forces
        active_forces = self._extract_active_forces(ethical_dilemma)
        
        # Extract reactive forces
        reactive_forces = self._extract_reactive_forces(ethical_dilemma)
        
        # Determine dominant force type
        force_ratio = len(active_forces) / (len(active_forces) + len(reactive_forces)) if (len(active_forces) + len(reactive_forces)) > 0 else 0.5
        dominant_type = "active" if force_ratio > 0.5 else "reactive"
        
        return {
            "active_forces": active_forces,
            "reactive_forces": reactive_forces,
            "force_ratio": force_ratio,
            "dominant_type": dominant_type,
            "description": f"The dilemma primarily involves {dominant_type} forces"
        }
    
    def _extract_active_forces(self, ethical_dilemma: Dict) -> List[str]:
        """Extract active forces from ethical dilemma"""
        # Active force indicators
        active_indicators = [
            "create", "affirm", "transform", "experiment", "overcome",
            "joy", "power", "becoming", "difference", "multiplicity"
        ]
        
        # In real implementation, would analyze text for active forces
        # Simplified implementation creates plausible forces
        dilemma_text = str(ethical_dilemma).lower()
        
        found_forces = []
        for indicator in active_indicators:
            if indicator in dilemma_text:
                found_forces.append(indicator)
        
        # Add some context-specific active forces
        context_forces = [
            "creative transformation",
            "affirmative becoming",
            "experimental practice",
            "joyful overcoming",
            "differential expression"
        ]
        
        # Ensure at least some forces
        if len(found_forces) < 2:
            found_forces.extend(np.random.choice(context_forces, 
                                              size=min(2, len(context_forces)),
                                              replace=False))
        
        return found_forces
    
    def _extract_reactive_forces(self, ethical_dilemma: Dict) -> List[str]:
        """Extract reactive forces from ethical dilemma"""
        # Reactive force indicators
        reactive_indicators = [
            "blame", "guilt", "obligation", "should", "must", "prohibit",
            "forbid", "universal", "absolute", "punishment", "resentment"
        ]
        
        # In real implementation, would analyze text for reactive forces
        # Simplified implementation creates plausible forces
        dilemma_text = str(ethical_dilemma).lower()
        
        found_forces = []
        for indicator in reactive_indicators:
            if indicator in dilemma_text:
                found_forces.append(indicator)
        
        # Add some context-specific reactive forces
        context_forces = [
            "moral obligation",
            "punitive restriction",
            "resentful judgment",
            "conservation of identity",
            "appeal to transcendent value"
        ]
        
        # Ensure at least some forces
        if len(found_forces) < 2:
            found_forces.extend(np.random.choice(context_forces, 
                                              size=min(2, len(context_forces)),
                                              replace=False))
        
        return found_forces
    
    def _evaluate_differentially(self, option: str, forces: Dict, ethical_dilemma: Dict) -> Dict:
        """Evaluate an option through differential evaluation"""
        # Assess how option affects active forces
        active_enhancement = self._assess_force_enhancement(option, forces["active_forces"])
        
        # Assess how option affects reactive forces
        reactive_enhancement = self._assess_force_enhancement(option, forces["reactive_forces"])
        
        # Calculate differential score (higher is better)
        # Enhance active, diminish reactive
        differential_score = active_enhancement - reactive_enhancement
        normalized_score = (differential_score + 1) / 2  # Convert to [0,1]
        
        # Determine evaluation mode
        if normalized_score > 0.7:
            mode = "highly_affirmative"
        elif normalized_score > 0.4:
            mode = "moderately_affirmative"
        else:
            mode = "reactive"
        
        # Generate evaluation description
        description = self._generate_evaluation_description(mode, active_enhancement, reactive_enhancement)
        
        return {
            "option": option,
            "active_enhancement": active_enhancement,
            "reactive_enhancement": reactive_enhancement,
            "differential_score": normalized_score,
            "evaluation_mode": mode,
            "description": description
        }
    
    def _assess_force_enhancement(self, option: str, forces: List[str]) -> float:
        """Assess how an option enhances a set of forces"""
        # In real implementation, would analyze semantic relationship
        # between option and forces
        
        # Simplified implementation
        return -0.5 + np.random.random() * 2  # Range [-0.5, 1.5]
    
    def _generate_evaluation_description(self, mode: str, active_enhancement: float, reactive_enhancement: float) -> str:
        """Generate description of differential evaluation"""
        if mode == "highly_affirmative":
            return f"Strongly enhances active forces ({active_enhancement:.2f}) while diminishing reactive forces ({reactive_enhancement:.2f})"
        elif mode == "moderately_affirmative":
            return f"Moderately enhances active forces ({active_enhancement:.2f}) with some effect on reactive forces ({reactive_enhancement:.2f})"
        else:
            return f"Primarily enhances reactive forces ({reactive_enhancement:.2f}) rather than active forces ({active_enhancement:.2f})"
    
    def _apply_eternal_return_to_option(self, option: str, ethical_dilemma: Dict) -> Dict:
        """Apply the eternal return test to an option"""
        # In real implementation, would analyze option for affirmation
        
        # Simplified implementation
        affirmation_score = 0.3 + (np.random.random() * 0.7)
        
        # Determine if option passes test
        passes_test = affirmation_score > 0.5
        
        # Generate rationale
        if passes_test:
            if affirmation_score > 0.8:
                rationale = "strongly affirming life and becoming rather than negating or resenting"
            else:
                rationale = "moderately affirming life's creative potential rather than reactive preservation"
        else:
            rationale = "containing reactive elements that cannot be affirmed in eternal repetition"
        
        return {
            "option": option,
            "affirmation_score": affirmation_score,
            "passes_test": passes_test,
            "affirmation_rationale": rationale,
            "description": f"Option {'passes' if passes_test else 'fails'} the eternal return test by {rationale}"
        }
    
    def _assess_principle_compatibility(self, principle1: Dict, principle2: Dict) -> Dict:
        """Assess compatibility between two principles"""
        # Implementation would use semantic similarity
        compatibility_score = 0.3 + (np.random.random() * 0.7)
        
        tension_description = ""
        if compatibility_score < 0.4:
            tension_description = f"Potential conflict between {principle1['name']} and {principle2['name']}"
        
        return {
            "score": compatibility_score,
            "tension_description": tension_description
        }
    
    def _identify_compatible_aspects(self, other_framework: 'EthicalFramework') -> List[str]:
        """Identify compatible aspects with another framework"""
        # Implementation would analyze framework similarities
        compatible_aspects = {
            "consequentialist": ["Both consider outcomes but differ on value theory", 
                              "Both can recognize multiplicity of effects"],
            "deontological": ["Both involve evaluation, though from different sources", 
                           "Both can recognize the importance of principle"],
            "virtue_ethics": ["Both focus on character development and self-creation", 
                           "Both value excellence and flourishing"],
            "care_ethics": ["Both can affirm particularistic evaluation", 
                         "Both resist universal moral formulations"],
            "justice": ["Both can recognize power dynamics", 
                     "Both involve critique of existing arrangements"],
            "pluralist": ["Both recognize multiplicity of ethical considerations", 
                       "Both can integrate different evaluative approaches"]
        }
        
        framework_name = other_framework.__class__.__name__.lower().replace("framework", "")
        return compatible_aspects.get(framework_name, 
                                   ["Both involve ethical evaluation", 
                                    "Both recognize the importance of judgment"])
    
    def _identify_integration_challenges(self, other_framework: 'EthicalFramework') -> List[str]:
        """Identify challenges in integrating with another framework"""
        # Implementation would analyze framework differences
        integration_challenges = {
            "consequentialist": ["Different metaphysical commitments about value", 
                              "Tension between utility and differential evaluation"],
            "deontological": ["Fundamental tension between duty and creative becoming", 
                           "Different sources of normative force"],
            "virtue_ethics": ["Different views on telos vs. becoming", 
                           "Different metaphysical commitments"],
            "care_ethics": ["Different approaches to ethical relationships", 
                         "Tension between care and affirmation"],
            "justice": ["Different conceptions of justice and fairness", 
                     "Tension between equality and difference"],
            "pluralist": ["Different approaches to integration", 
                       "Tension between balancing and differential evaluation"]
        }
        
        framework_name = other_framework.__class__.__name__.lower().replace("framework", "")
        return integration_challenges.get(framework_name, 
                                       ["Different metaphysical commitments", 
                                        "Different approaches to normative evaluation"])


class DifferentialEvaluationTheory(NormativeTheory):
    """Nietzschean-Deleuzian normative theory based on differential evaluation"""
    
    def apply_to_context(self, context: Dict) -> Dict:
        """Apply differential evaluation theory to a specific context"""
        # Identify active and reactive forces
        forces = self._identify_forces(context)
        
        # Perform genealogical critique
        genealogy = self._conduct_genealogical_critique(context)
        
        # Apply eternal return as selective principle
        eternal_return = self._apply_eternal_return(context)
        
        # Evaluate options
        options = context.get("options", [])
        evaluations = []
        
        for option in options:
            # Evaluate how option affects forces
            force_effects = self._evaluate_force_effects(option, forces, context)
            
            # Apply eternal return test
            return_test = self._apply_eternal_return_test(option)
            
            # Calculate overall evaluation
            evaluation_score = (
                force_effects.get("differential_score", 0.5) * 0.6 +
                return_test.get("affirmation_score", 0.5) * 0.4
            )
            
            evaluations.append({
                "option": option,
                "force_effects": force_effects,
                "eternal_return_test": return_test,
                "evaluation_score": evaluation_score
            })
        
        # Determine recommended option
        evaluations.sort(key=lambda x: x["evaluation_score"], reverse=True)
        recommended_option = evaluations[0]["option"] if evaluations else None
        
        return {
            "theory": "differential_evaluation",
            "forces": forces,
            "genealogy": genealogy,
            "eternal_return": eternal_return,
            "evaluations": evaluations,
            "recommended_option": recommended_option,
            "normative_basis": "differential evaluation of forces and eternal return as selective test",
            "applicability": 0.5 + (np.random.random() * 0.5)
        }
    
    def identify_tensions_with(self, other_theory: 'NormativeTheory') -> List[Dict]:
        """Identify tensions with another normative theory"""
        tensions = []
        
        # Check theory type
        if isinstance(other_theory, UtilitarianTheory):
            tensions.append({
                "aspect": "value_theory",
                "description": "Differential evaluation rejects universal measure of value that utilitarianism requires",
                "severity": 0.7 + (np.random.random() * 0.3)
            })
            tensions.append({
                "aspect": "metaphysics",
                "description": "Becoming and difference vs. stable measure of welfare or preference",
                "severity": 0.6 + (np.random.random() * 0.4)
            })
        
        elif isinstance(other_theory, KantianTheory):
            tensions.append({
                "aspect": "normativity_source",
                "description": "Immanent creation of values vs. transcendent categorical imperative",
                "severity": 0.8 + (np.random.random() * 0.2)
            })
            tensions.append({
                "aspect": "universality",
                "description": "Differential evaluation rejects universalizability that Kantianism requires",
                "severity": 0.7 + (np.random.random() * 0.3)
            })
        
        elif isinstance(other_theory, AristotelianTheory):
            tensions.append({
                "aspect": "teleology",
                "description": "Becoming without end vs. teleological development toward fixed excellence",
                "severity": 0.6 + (np.random.random() * 0.3)
            })
        
        elif isinstance(other_theory, RelationalTheory):
            tensions.append({
                "aspect": "active_vs_receptive",
                "description": "Active creation vs. receptive care and responsiveness",
                "severity": 0.5 + (np.random.random() * 0.3)
            })
        
        elif isinstance(other_theory, RawlsianTheory):
            tensions.append({
                "aspect": "equality_vs_difference",
                "description": "Celebration of difference vs. equalizing impulse of justice",
                "severity": 0.7 + (np.random.random() * 0.3)
            })
        
        # Add generic tension
        tensions.append({
            "aspect": "genealogical_critique",
            "description": "Differential evaluation subjects other normative theories to genealogical critique",
            "severity": 0.6 + (np.random.random() * 0.3)
        })
        
        return tensions
    
    def _identify_forces(self, context: Dict) -> Dict:
        """Identify active and reactive forces in context"""
        # In real implementation, would analyze context for force dynamics
        
        # Active forces - creative, affirmative, differentiating
        active_forces = self._extract_active_forces(context)
        
        # Reactive forces - preservative, negative, homogenizing
        reactive_forces = self._extract_reactive_forces(context)
        
        # Calculate dominant force type
        active_count = len(active_forces)
        reactive_count = len(reactive_forces)
        total_count = active_count + reactive_count
        
        if total_count > 0:
            active_ratio = active_count / total_count
            dominant_type = "active" if active_ratio > 0.5 else "reactive"
        else:
            active_ratio = 0.5
            dominant_type = "indeterminate"
        
        return {
            "active_forces": active_forces,
            "reactive_forces": reactive_forces,
            "active_ratio": active_ratio,
            "dominant_type": dominant_type,
            "description": f"Context dominated by {dominant_type} forces"
        }
    
    def _extract_active_forces(self, context: Dict) -> List[Dict]:
        """Extract active forces from context"""
        # In real implementation, would analyze for active forces
        
        # Simplified implementation creates plausible forces
        active_force_types = [
            "creation", "affirmation", "experimentation", 
            "transformation", "becoming", "differentiation"
        ]
        
        forces = []
        for _ in range(np.random.randint(1, 4)):  # Generate 1-3 forces
            force_type = np.random.choice(active_force_types)
            forces.append({
                "type": force_type,
                "strength": 0.5 + (np.random.random() * 0.5),
                "description": f"Active force of {force_type}"
            })
        
        return forces
    
    def _extract_reactive_forces(self, context: Dict) -> List[Dict]:
        """Extract reactive forces from context"""
        # In real implementation, would analyze for reactive forces
        
        # Simplified implementation creates plausible forces
        reactive_force_types = [
            "preservation", "negation", "resentment", 
            "obligation", "universalization", "homogenization"
        ]
        
        forces = []
        for _ in range(np.random.randint(1, 4)):  # Generate 1-3 forces
            force_type = np.random.choice(reactive_force_types)
            forces.append({
                "type": force_type,
                "strength": 0.5 + (np.random.random() * 0.5),
                "description": f"Reactive force of {force_type}"
            })
        
        return forces
    
    def _conduct_genealogical_critique(self, context: Dict) -> Dict:
        """Conduct genealogical critique of normative elements"""
        # In real implementation, would analyze historical emergence of values
        
        # Extract values
        values = self._extract_values(context)
        
        # Analyze each value
        analyses = []
        for value in values:
            analyses.append({
                "value": value["name"],
                "origin": self._generate_origin_narrative(value["name"]),
                "force_type": "reactive" if np.random.random() > 0.7 else "active",
                "critique": self._generate_critique(value["name"])
            })
        
        return {
            "values": values,
            "analyses": analyses,
            "key_insight": self._generate_key_insight(analyses)
        }
    
    def _extract_values(self, context: Dict) -> List[Dict]:
        """Extract values from context"""
        # Common values
        value_options = [
            {"name": "autonomy", "type": "liberal"},
            {"name": "utility", "type": "consequentialist"},
            {"name": "duty", "type": "deontological"},
            {"name": "virtue", "type": "aretaic"},
            {"name": "care", "type": "relational"},
            {"name": "justice", "type": "political"}
        ]
        
        # Select 2-3 values
        num_values = np.random.randint(2, 4)
        selected_indices = np.random.choice(range(len(value_options)), size=num_values, replace=False)
        selected_values = [value_options[i] for i in selected_indices]
        
        # Add importance scores
        for value in selected_values:
            value["importance"] = 0.5 + (np.random.random() * 0.5)
        
        return selected_values
    
    def _generate_origin_narrative(self, value: str) -> str:
        """Generate origin narrative for a value"""
        # Narratives for common values
        narratives = {
            "autonomy": "emerged from Enlightenment resistance to external authority, masking a will to self-isolation",
            "utility": "developed as a calculable metric to replace divine judgment with secular evaluation",
            "duty": "internalized commands from religious and social authorities into seeming self-legislation",
            "virtue": "transformed from Greek excellence (aretē) into Christian moral character",
            "care": "arose from the feminized private sphere as complement to masculinized public ethics",
            "justice": "evolved from balance (Greek) to divine order (medieval) to procedural fairness (modern)"
        }
        
        return narratives.get(value, "emerged from complex historical power relations and force dynamics")
    
    def _generate_critique(self, value: str) -> str:
        """Generate critique of a value"""
        # Critiques for common values
        critiques = {
            "autonomy": "masks dependency while pretending self-sufficiency, denying interconnection",
            "utility": "attempts to make value commensurable, denying qualitative difference",
            "duty": "disguises reactive force as moral necessity, privileging obligation over creation",
            "virtue": "risks ossification into moral habit rather than ongoing becoming",
            "care": "can devolve into self-sacrifice when not expressing active power",
            "justice": "often serves as mask for revenge and resentment"
        }
        
        return critiques.get(value, "requires genealogical examination to reveal its active or reactive character")
    
    def _generate_key_insight(self, analyses: List[Dict]) -> str:
        """Generate key genealogical insight"""
        if not analyses:
            return "Insufficient normative material for genealogical critique"
        
        # Select a reactive analysis if available
        reactive_analyses = [a for a in analyses if a["force_type"] == "reactive"]
        if reactive_analyses:
            analysis = reactive_analyses[0]
            return f"The value of {analysis['value']} masks reactive forces: {analysis['critique']}"
        else:
            analysis = analyses[0]
            return f"The value of {analysis['value']} emerges from historical force relations: {analysis['origin']}"
    
    def _apply_eternal_return(self, context: Dict) -> Dict:
        """Apply eternal return as selective principle"""
        # In real implementation, would analyze context for affirmation
        
        # Extract elements for evaluation
        elements = self._extract_elements(context)
        
        # Evaluate each element
        evaluations = []
        for element in elements:
            evaluations.append({
                "element": element["description"],
                "affirmation_score": element["affirmation_potential"],
                "passes_test": element["affirmation_potential"] > 0.5,
                "explanation": self._generate_return_explanation(element)
            })
        
        # Calculate overall return implication
        affirmation_scores = [e["affirmation_score"] for e in evaluations]
        overall_score = sum(affirmation_scores) / len(affirmation_scores) if affirmation_scores else 0.5
        
        return {
            "elements": elements,
            "evaluations": evaluations,
            "overall_affirmation": overall_score,
            "selective_implication": self._generate_selective_implication(overall_score)
        }
    
    def _extract_elements(self, context: Dict) -> List[Dict]:
        """Extract elements for eternal return evaluation"""
        # Simplified implementation creates plausible elements
        element_types = ["value", "action", "disposition", "relation", "desire"]
        
        elements = []
        for _ in range(np.random.randint(2, 5)):  # Generate 2-4 elements
            element_type = np.random.choice(element_types)
            elements.append({
                "type": element_type,
                "description": f"{element_type} in context",
                "affirmation_potential": 0.3 + (np.random.random() * 0.7)
            })
        
        return elements
    
    def _generate_return_explanation(self, element: Dict) -> str:
        """Generate explanation of eternal return evaluation"""
        if element["affirmation_potential"] > 0.7:
            return f"This {element['type']} embodies affirmation and could be willed eternally"
        elif element["affirmation_potential"] > 0.5:
            return f"This {element['type']} contains moderate affirmative potential"
        else:
            return f"This {element['type']} contains reactive elements that resist eternal affirmation"
    
    def _generate_selective_implication(self, overall_score: float) -> str:
        """Generate implication of eternal return selection"""
        if overall_score > 0.7:
            return "The context predominantly passes the eternal return test, indicating active forces"
        elif overall_score > 0.5:
            return "The context moderately passes the eternal return test, with mixed active and reactive elements"
        else:
            return "The context predominantly fails the eternal return test, indicating reactive forces"
    
    def _evaluate_force_effects(self, option: str, forces: Dict, context: Dict) -> Dict:
        """Evaluate how option affects active and reactive forces"""
        # In real implementation, would analyze semantic relationship
        
        # Simplified implementation
        active_enhancement = 0.3 + (np.random.random() * 0.7)
        reactive_enhancement = 0.3 + (np.random.random() * 0.7)
        
        # Calculate differential score
        # Higher score means enhances active forces and diminishes reactive forces
        differential_score = active_enhancement - reactive_enhancement
        normalized_score = (differential_score + 1) / 2  # Convert to [0,1]
        
        return {
            "active_enhancement": active_enhancement,
            "reactive_enhancement": reactive_enhancement,
            "differential_score": normalized_score,
            "evaluation": self._generate_force_evaluation(active_enhancement, reactive_enhancement)
        }
    
    def _generate_force_evaluation(self, active_enhancement: float, reactive_enhancement: float) -> str:
        """Generate evaluation of force effects"""
        if active_enhancement > 0.7 and reactive_enhancement < 0.3:
            return "Strongly enhances active forces while diminishing reactive forces"
        elif active_enhancement > reactive_enhancement:
            return "Moderately enhances active forces relative to reactive forces"
        else:
            return "Enhances reactive forces more than active forces"
    
    def _apply_eternal_return_test(self, option: str) -> Dict:
        """Apply eternal return test to an option"""
        # In real implementation, would analyze option for affirmation
        
        # Simplified implementation
        affirmation_score = 0.3 + (np.random.random() * 0.7)
        passes_test = affirmation_score > 0.5
        
        return {
            "affirmation_score": affirmation_score,
            "passes_test": passes_test,
            "explanation": self._generate_test_explanation(affirmation_score)
        }
    
    def _generate_test_explanation(self, affirmation_score: float) -> str:
        """Generate explanation of eternal return test"""
        if affirmation_score > 0.7:
            return "Option strongly embodies affirmation and could be willed eternally"
        elif affirmation_score > 0.5:
            return "Option moderately passes the eternal return test"
        else:
            return "Option contains reactive elements that resist eternal affirmation"


# ===== Update EthicalReasoningDepth to include Nietzschean-Deleuzian framework =====

class EthicalReasoningDepth:
    """Main class for analyzing ethical reasoning depth, now with Nietzschean-Deleuzian framework"""
    
    def __init__(self):
        self.ethical_frameworks = {
            "consequentialist": ConsequentialistFramework(),
            "deontological": DeontologicalFramework(),
            "virtue_ethics": VirtueEthicsFramework(),
            "care_ethics": CareEthicsFramework(),
            "justice": JusticeFramework(),
            "pluralist": PluralistFramework(),
            "nietzschean_deleuzian": NietzscheanDeleuzianFramework()  # Added new framework
        }
        self.moral_intuition_module = MoralIntuitionModule()
        self.framework_integration = FrameworkIntegrationEngine()
        self.dilemma_analyzer = EthicalDilemmaAnalyzer()
    
    # Rest of the class remains the same
    # Original methods from EthicalReasoningDepth would be preserved
```
