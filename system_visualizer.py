import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from functools import lru_cache
from typing import Dict, List, Any, Tuple, Optional

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemVisualizer")

class SystemVisualizer:
    def __init__(self, system_level, cache_enabled=True):
        self.system_level = system_level
        self.metrics = self._safe_get_metrics()
        self.cache_enabled = cache_enabled
        self._figure_size = (10, 7)
        plt.style.use('seaborn-v0_8-whitegrid')

    def _safe_get_metrics(self) -> Dict:
        try:
            return getattr(self.system_level, 'metrics', {}) or {}
        except Exception as e:
            logger.warning(f"Failed to get metrics: {e}")
            return {}

    def _safe_get_feedback_loops(self) -> List[Dict]:
        try:
            return getattr(self.system_level, 'feedback_loops', []) or []
        except Exception as e:
            logger.warning(f"Failed to get feedback loops: {e}")
            return []

    @lru_cache(maxsize=8)
    def _calculate_graph_layout(self, nodes: Tuple[str], edges: Tuple[Tuple[str, str]]) -> Dict:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        try:
            return nx.spring_layout(G, seed=42)
        except Exception as e:
            logger.warning(f"Layout failed: {e}")
            return nx.circular_layout(G)

    def draw_feedback_graph(self, title="System Feedback Loops", save_dir="/sdcard/Download") -> str:
        """Draws system feedback loops and returns PNG path."""
        feedback_loops = self._safe_get_feedback_loops()
        if not feedback_loops:
            logger.info("No feedback loops available.")
            plt.figure(figsize=self._figure_size)
            plt.text(0.5, 0.5, "No feedback loops available.", ha='center', va='center', fontsize=14)
            plt.axis('off')
        else:
            G = nx.DiGraph()
            edge_weights, edge_colors, edge_labels = [], [], {}
            for loop in feedback_loops:
                s = loop.get('source', 'unknown')
                t = loop.get('target', 'unknown')
                w = loop.get('strength', 1.0)
                typ = loop.get('type', 'unknown')
                G.add_edge(s, t)
                edge_weights.append(max(0.5, min(w, 5.0)))
                edge_colors.append('firebrick' if typ == 'system_observer' else 'darkorchid')
                edge_labels[(s, t)] = f"{w:.2f}"

            plt.figure(figsize=self._figure_size)
            pos = self._calculate_graph_layout(tuple(G.nodes()), tuple(G.edges()))
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1800, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, arrows=True, alpha=0.7)
            nx.draw_networkx_labels(G, pos, font_size=10)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            plt.title(title)
            plt.axis('off')

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "feedback_graph.png")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_metrics(self, save_dir="/sdcard/Download") -> str:
        """Plots system metrics and returns PNG path."""
        if not self.metrics:
            logger.info("No metrics to plot.")
            plt.figure(figsize=self._figure_size)
            plt.text(0.5, 0.5, "No metrics available.", ha='center', va='center')
            plt.axis('off')
        else:
            df = pd.DataFrame(self.metrics).fillna(0)
            plt.figure(figsize=self._figure_size)
            for col in df.columns:
                if col != 'time':
                    plt.plot(df['time'], df[col], linewidth=2, label=col)
            plt.title("System Metrics Over Time")
            plt.xlabel("Time")
            plt.ylabel("Metric Value")
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "system_metrics.png")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def export_to_json(self, save_dir="/sdcard/Download") -> str:
        """Exports metrics and feedback data as JSON."""
        data = {
            "metrics": self.metrics,
            "feedback_loops": self._safe_get_feedback_loops(),
        }
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "system_data.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path

    def visualize(self, save_dir="/sdcard/Download") -> str:
        """Generates all visual outputs and returns JSON summary."""
        try:
            graph_path = self.draw_feedback_graph(save_dir=save_dir)
            metrics_path = self.plot_metrics(save_dir=save_dir)
            data_path = self.export_to_json(save_dir=save_dir)
            result = {
                "status": "success",
                "paths": {
                    "graph": graph_path,
                    "metrics": metrics_path,
                    "data": data_path
                }
            }
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})

# ---- Example Entry Point for Kotlin ----
def run_visualizer(system_data: Dict[str, Any], save_dir="/sdcard/Download") -> str:
    """Chaquopy-callable entrypoint: takes dict-like data, returns JSON result."""
    class TempSystem:
        def __init__(self, data):
            self.metrics = data.get("metrics", {})
            self.feedback_loops = data.get("feedback_loops", [])
    system = TempSystem(system_data)
    vis = SystemVisualizer(system)
    return vis.visualize(save_dir)
