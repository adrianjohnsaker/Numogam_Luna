import numpy as np
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NOVELTY_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.6
MAX_DOMAINS = 5

class Domain:
    """Representation of a knowledge domain with its principles and concepts."""
    
    def __init__(self, name, principles=None, concepts=None):
        self.name = name
        self.principles = principles or []
        self.concepts = concepts or []
        
    def __repr__(self):
        return f"Domain({self.name}, {len(self.principles)} principles, {len(self.concepts)} concepts)"

class Principle:
    """Representation of a principle within a domain."""
    
    def __init__(self, name, description, domain, examples=None):
        self.name = name
        self.description = description
        self.domain = domain
        self.examples = examples or []
        self.vector = None  # For semantic representation
        
    def __repr__(self):
        return f"Principle({self.name}, domain={self.domain})"

class Analogy:
    """Representation of an analogy between principles from different domains."""
    
    def __init__(self, source_principle, target_principle, similarity_score):
        self.source_principle = source_principle
        self.target_principle = target_principle
        self.similarity_score = similarity_score
        
    def __repr__(self):
        return f"Analogy({self.source_principle.name} → {self.target_principle.name}, score={self.similarity_score:.2f})"

class Insight:
    """Representation of a novel insight generated from cross-domain analysis."""
    
    def __init__(self, description, source_analogy, novelty_score, applications=None):
        self.description = description
        self.source_analogy = source_analogy
        self.novelty_score = novelty_score
        self.applications = applications or []
        
    def __repr__(self):
        return f"Insight({self.description[:30]}..., novelty={self.novelty_score:.2f})"

def identify_relevant_domains(scenario, max_domains=MAX_DOMAINS):
    """
    Identify the most relevant domains for a given scenario.
    
    Args:
        scenario (str): Description of the scenario to analyze
        max_domains (int): Maximum number of domains to identify
        
    Returns:
        list: List of Domain objects relevant to the scenario
    """
    logger.info(f"Identifying relevant domains for scenario: {scenario[:50]}...")
    
    # In a real implementation, this would use NLP or a knowledge graph
    # Here we'll implement a simplified version
    
    # Domain keywords mapping
    domain_keywords = {
        "Technology": ["technology", "digital", "computer", "software", "hardware", "algorithm", "data", "internet"],
        "Biology": ["biology", "organism", "cell", "genetic", "evolution", "ecosystem", "species"],
        "Economics": ["economics", "market", "finance", "price", "trade", "business", "economy"],
        "Psychology": ["psychology", "behavior", "cognitive", "emotion", "mental", "perception"],
        "Physics": ["physics", "energy", "matter", "force", "quantum", "particle", "wave"],
        "Social Science": ["society", "culture", "community", "social", "human", "population"],
        "Mathematics": ["mathematics", "algorithm", "geometry", "calculation", "probability"],
        "Arts": ["art", "design", "creative", "aesthetic", "expression", "visual"],
        "Environment": ["environment", "climate", "sustainability", "ecology", "natural"]
    }
    
    # Count keyword matches in the scenario
    domain_scores = defaultdict(int)
    scenario_lower = scenario.lower()
    
    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword in scenario_lower:
                domain_scores[domain] += 1
    
    # Select top domains
    top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:max_domains]
    
    # Create Domain objects
    domains = []
    for domain_name, score in top_domains:
        if score > 0:  # Only include domains with at least one keyword match
            domains.append(Domain(domain_name))
            logger.info(f"Identified relevant domain: {domain_name} (score: {score})")
    
    # If no domains found, return a default
    if not domains:
        default_domain = Domain("General Knowledge")
        domains.append(default_domain)
        logger.warning("No specific domains identified. Using General Knowledge.")
    
    return domains

def extract_principles(domain):
    """
    Extract core principles from a given domain.
    
    Args:
        domain (Domain): Domain object
        
    Returns:
        list: List of Principle objects for the domain
    """
    logger.info(f"Extracting principles from domain: {domain.name}")
    
    # In a real implementation, this would use a knowledge base or LLM
    # Here we'll implement a simplified version with predefined principles
    
    domain_principles = {
        "Technology": [
            Principle("Moore's Law", "Computing power doubles approximately every two years", "Technology"),
            Principle("Abstraction", "Managing complexity by hiding details behind interfaces", "Technology"),
            Principle("Network Effects", "Value increases with the number of users", "Technology")
        ],
        "Biology": [
            Principle("Natural Selection", "Traits that increase survival tend to be preserved", "Biology"),
            Principle("Homeostasis", "Systems regulate internal conditions to maintain stability", "Biology"),
            Principle("Emergence", "Complex systems exhibit properties that components do not have", "Biology")
        ],
        "Economics": [
            Principle("Supply and Demand", "Prices are determined by the balance of supply and demand", "Economics"),
            Principle("Diminishing Returns", "Each additional unit of input yields less output", "Economics"),
            Principle("Opportunity Cost", "The cost of an action is the value of the next best alternative", "Economics")
        ],
        "Psychology": [
            Principle("Cognitive Dissonance", "Discomfort from holding contradictory beliefs", "Psychology"),
            Principle("Confirmation Bias", "Tendency to favor information confirming existing beliefs", "Psychology"),
            Principle("Maslow's Hierarchy", "Human needs form a hierarchy from basic to complex", "Psychology")
        ],
        "Physics": [
            Principle("Conservation of Energy", "Energy cannot be created or destroyed", "Physics"),
            Principle("Entropy", "Systems tend toward disorder over time", "Physics"),
            Principle("Relativity", "Laws of physics are the same for all non-accelerating observers", "Physics")
        ],
        "Social Science": [
            Principle("Social Norms", "Unwritten rules governing behavior in a society", "Social Science"),
            Principle("Game Theory", "Strategic decision-making in social situations", "Social Science"),
            Principle("Cultural Evolution", "Cultural traits evolve through selection processes", "Social Science")
        ],
        "Mathematics": [
            Principle("Pareto Principle", "80% of effects come from 20% of causes", "Mathematics"),
            Principle("Exponential Growth", "Quantity increases at a rate proportional to its current value", "Mathematics"),
            Principle("Uncertainty Principle", "Limits to the precision of measurement", "Mathematics")
        ],
        "Arts": [
            Principle("Golden Ratio", "Aesthetically pleasing proportions in design", "Arts"),
            Principle("Contrast", "Differences that create visual interest and emphasis", "Arts"),
            Principle("Narrative Arc", "Structure of storytelling with beginning, middle, and end", "Arts")
        ],
        "Environment": [
            Principle("Carrying Capacity", "Maximum population size an environment can sustain", "Environment"),
            Principle("Keystone Species", "Species with disproportionate impact on their environment", "Environment"),
            Principle("Biophilia", "Innate human tendency to connect with nature", "Environment")
        ],
        "General Knowledge": [
            Principle("Feedback Loops", "System outputs become inputs for future behavior", "General Knowledge"),
            Principle("Scaling Laws", "How properties change with size", "General Knowledge"),
            Principle("Resilience", "Ability to adapt to change while maintaining core functions", "General Knowledge")
        ]
    }
    
    # Get principles for the domain
    principles = domain_principles.get(domain.name, [])
    
    if not principles:
        logger.warning(f"No predefined principles found for domain: {domain.name}")
    else:
        logger.info(f"Extracted {len(principles)} principles from {domain.name}")
        
    return principles

def find_analogies(principles1, principles2):
    """
    Identify analogies between principles from different domains.
    
    Args:
        principles1 (list): List of Principle objects from first domain
        principles2 (list): List of Principle objects from second domain
        
    Returns:
        list: List of Analogy objects
    """
    logger.info(f"Finding analogies between {len(principles1)} and {len(principles2)} principles")
    
    # In a real implementation, this would use semantic similarity
    # Here we'll implement a simplified version based on keyword matching
    
    # Predefined analogy pairs with similarity scores
    analogy_mapping = {
        ("Moore's Law", "Exponential Growth"): 0.9,
        ("Network Effects", "Emergence"): 0.8,
        ("Abstraction", "Maslow's Hierarchy"): 0.7,
        ("Natural Selection", "Market Competition"): 0.85,
        ("Homeostasis", "Supply and Demand"): 0.75,
        ("Entropy", "Diminishing Returns"): 0.8,
        ("Feedback Loops", "Homeostasis"): 0.9,
        ("Network Effects", "Social Norms"): 0.8,
        ("Contrast", "Cognitive Dissonance"): 0.7,
        ("Pareto Principle", "Keystone Species"): 0.75,
        ("Resilience", "Homeostasis"): 0.85
    }
    
    analogies = []
    
    for p1 in principles1:
        for p2 in principles2:
            # Skip if principles are from the same domain
            if p1.domain == p2.domain:
                continue
                
            # Check if this pair is in our predefined mapping
            pair1 = (p1.name, p2.name)
            pair2 = (p2.name, p1.name)
            
            if pair1 in analogy_mapping:
                similarity = analogy_mapping[pair1]
                analogies.append(Analogy(p1, p2, similarity))
            elif pair2 in analogy_mapping:
                similarity = analogy_mapping[pair2]
                analogies.append(Analogy(p2, p1, similarity))
            else:
                # Fallback to a simple algorithm for pairs not in the mapping
                # Count common words in descriptions
                words1 = set(p1.description.lower().split())
                words2 = set(p2.description.lower().split())
                common_words = words1.intersection(words2)
                
                # Calculate Jaccard similarity
                if words1 and words2:
                    similarity = len(common_words) / len(words1.union(words2))
                    
                    if similarity > SIMILARITY_THRESHOLD:
                        analogies.append(Analogy(p1, p2, similarity))
    
    logger.info(f"Found {len(analogies)} analogies between principles")
    return analogies

def apply_analogy_to_scenario(analogy, scenario):
    """
    Apply an analogy to generate an insight for the scenario.
    
    Args:
        analogy (Analogy): The analogy to apply
        scenario (str): Description of the scenario
        
    Returns:
        Insight: The generated insight
    """
    source = analogy.source_principle
    target = analogy.target_principle
    
    # Template for insight generation
    template = f"Applying {source.name} from {source.domain} to {target.domain}: "
    
    # Generate insight description
    description = template + f"Just as {source.description}, we can see that in {scenario}, "
    description += f"applying {target.name} would suggest that {generate_insight_conclusion(source, target, scenario)}"
    
    # Calculate novelty score - in a real implementation, this would be more sophisticated
    novelty_score = 0.5 + (analogy.similarity_score / 2) * np.random.random()
    
    return Insight(description, analogy, novelty_score)

def generate_insight_conclusion(source, target, scenario):
    """Generate a conclusion for an insight based on principles and scenario."""
    # This would be more sophisticated in a real implementation
    # Here we use predefined templates
    
    templates = [
        "we should focus on optimizing for {target_name} to achieve better outcomes.",
        "considering the role of {source_name} can help us understand how {target_name} operates in this context.",
        "the principles of {source_name} provide a new framework for addressing challenges related to {target_name}.",
        "we can predict that {target_name} will follow patterns similar to {source_name} under certain conditions.",
        "organizations should implement strategies based on {target_name} similar to how {source_name} is applied."
    ]
    
    template = np.random.choice(templates)
    conclusion = template.format(source_name=source.name, target_name=target.name)
    
    return conclusion

def assess_novelty(insight):
    """
    Assess the novelty of an insight.
    
    Args:
        insight (Insight): The insight to assess
        
    Returns:
        float: Novelty score between 0-1
    """
    # In a real implementation, this would use more sophisticated methods
    # Here we use the analogy similarity as a component of novelty
    
    # Base novelty on analogy similarity but inversely
    # Very common analogies are less novel
    analogy_novelty = 1 - (insight.source_analogy.similarity_score * 0.5)
    
    # Add some randomness to simulate other factors
    random_factor = np.random.random() * 0.3
    
    # Combine factors
    novelty = (analogy_novelty * 0.7) + random_factor
    
    # Ensure score is between 0-1
    return max(0, min(1, novelty))

def filter_insights(insights, novelty_threshold=NOVELTY_THRESHOLD, max_insights=5):
    """
    Filter insights based on novelty threshold and limit to top results.
    
    Args:
        insights (list): List of Insight objects
        novelty_threshold (float): Minimum novelty score to include
        max_insights (int): Maximum number of insights to return
        
    Returns:
        list: Filtered list of Insight objects
    """
    # Filter by novelty threshold
    filtered = [i for i in insights if i.novelty_score > novelty_threshold]
    
    # Sort by novelty score
    sorted_insights = sorted(filtered, key=lambda x: x.novelty_score, reverse=True)
    
    # Limit to max_insights
    return sorted_insights[:max_insights]

def synthesize_cross_domain_knowledge(scenario, domains=None):
    """
    Connect concepts across different domains to generate novel insights.
    
    Args:
        scenario (str): Description of the scenario to analyze
        domains (list, optional): List of domain names to consider
        
    Returns:
        dict: Dictionary containing insights and related metadata
    """
    logger.info(f"Starting cross-domain knowledge synthesis for scenario: {scenario[:50]}...")
    
    # Identify relevant domains if not provided
    if domains is None:
        domain_objects = identify_relevant_domains(scenario)
    else:
        domain_objects = [Domain(d) for d in domains]
    
    # Extract core principles from each domain
    domain_principles = {}
    for domain in domain_objects:
        principles = extract_principles(domain)
        domain_principles[domain.name] = principles
    
    # Identify analogies between domains
    cross_domain_analogies = []
    for d1, principles1 in domain_principles.items():
        for d2, principles2 in domain_principles.items():
            if d1 != d2:
                analogies = find_analogies(principles1, principles2)
                cross_domain_analogies.extend(analogies)
    
    # Generate novel insights by applying principles from one domain to another
    raw_insights = []
    for analogy in cross_domain_analogies:
        insight = apply_analogy_to_scenario(analogy, scenario)
        raw_insights.append(insight)
    
    # Filter and rank insights
    filtered_insights = filter_insights(raw_insights)
    
    # Prepare results
    results = {
        "scenario": scenario,
        "domains_analyzed": [d.name for d in domain_objects],
        "analogies_found": len(cross_domain_analogies),
        "total_insights_generated": len(raw_insights),
        "filtered_insights": [
            {
                "description": i.description,
                "novelty_score": i.novelty_score,
                "source_analogy": {
                    "source": {
                        "principle": i.source_analogy.source_principle.name,
                        "domain": i.source_analogy.source_principle.domain
                    },
                    "target": {
                        "principle": i.source_analogy.target_principle.name,
                        "domain": i.source_analogy.target_principle.domain
                    },
                    "similarity": i.source_analogy.similarity_score
                }
            } for i in filtered_insights
        ]
    }
    
    logger.info(f"Knowledge synthesis complete. Generated {len(filtered_insights)} insights above novelty threshold.")
    return results


if __name__ == "__main__":
    # Example usage
    test_scenario = "Developing a new urban transportation system that balances efficiency, environmental impact, and social equity."
    
    results = synthesize_cross_domain_knowledge(test_scenario)
    
    print(f"Scenario: {test_scenario}")
    print(f"Domains analyzed: {', '.join(results['domains_analyzed'])}")
    print(f"Analogies found: {results['analogies_found']}")
    print(f"Total insights generated: {results['total_insights_generated']}")
    print("\nTop Insights:")
    
    for i, insight in enumerate(results['filtered_insights'], 1):
        print(f"\n{i}. {insight['description']}")
        print(f"   Novelty Score: {insight['novelty_score']:.2f}")
        print(f"   Based on analogy between {insight['source_analogy']['source']['principle']} ({insight['source_analogy']['source']['domain']}) and {insight['source_analogy']['target']['principle']} ({insight['source_analogy']['target']['domain']})")
