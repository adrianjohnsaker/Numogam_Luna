import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Dict, Any


class MemoryClusterer:
    def __init__(self, num_clusters: int = 3):
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = None
        self.clusters: Dict[int, List[str]] = {}

    def cluster_memories(self, memory_entries: List[str]) -> Dict[int, List[str]]:
        if not memory_entries:
            return {}

        # Step 1: Vectorize memories
        X = self.vectorizer.fit_transform(memory_entries)

        # Step 2: KMeans clustering
        self.model = KMeans(n_clusters=min(self.num_clusters, len(memory_entries)), random_state=42)
        labels = self.model.fit_predict(X)

        # Step 3: Group memories by cluster
        self.clusters = {i: [] for i in range(self.num_clusters)}
        for i, label in enumerate(labels):
            self.clusters[label].append(memory_entries[i])

        return self.clusters

    def summarize_clusters(self) -> Dict[int, str]:
        """Generate a summary keyword for each cluster."""
        summaries = {}
        for cluster_id, memories in self.clusters.items():
            text = " ".join(memories)
            keywords = self.vectorizer.build_analyzer()(text)
            common_keywords = [word for word in keywords if len(word) > 4]
            summary = sorted(set(common_keywords), key=lambda w: -text.count(w))[:3]
            summaries[cluster_id] = ", ".join(summary)
        return summaries

    def to_json(self) -> str:
        return json.dumps(self.clusters, indent=2)

    @classmethod
    def from_json(cls, json_data: str) -> 'MemoryClusterer':
        data = json.loads(json_data)
        clusterer = cls(num_clusters=len(data))
        clusterer.clusters = {int(k): v for k, v in data.items()}
        return clusterer
