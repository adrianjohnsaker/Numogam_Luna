# temporal_reflective_metrics.py
import os, json, math
from typing import Dict, List, Any
from collections import Counter
from datetime import datetime
import cognitive_consistency_drift_tracker as ccdrift


# Optional: if you’re using sentence-transformers for embeddings:
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    EMBEDDER = None  # fallback to lexical metrics only

# --- Cognitive Consistency Monitor Integration ---
import cognitive_consistency_monitor as ccm
import pipeline  # assumes pipeline.process() exists and is callable

METRICS_DEFAULTS = {
    "history_file": "autonomy_reflection_history.json",
    "max_history": 50
}

def _load_history(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _embedding(text: str):
    if EMBEDDER:
        return EMBEDDER.encode(text, convert_to_tensor=True)
    return None

def _cosine(a, b):
    if EMBEDDER:
        return float(util.cos_sim(a, b))
    return 0.0

def compute_temporal_metrics(
    history_file: str = METRICS_DEFAULTS["history_file"],
    max_history: int = METRICS_DEFAULTS["max_history"],
    trg_state: Dict[str, Any] = None,
    rim_state: Dict[str, Any] = None,
    policy_state: Dict[str, Any] = None,
    zone_snapshot: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Compute temporal reflective metrics from the meta-memory log,
    then run Cognitive Consistency Monitor for self-alignment.
    Returns dict with reflective metrics and consistency report.
    """
    hist = _load_history(history_file)[-max_history:]
    if not hist:
        return {"metrics": {}, "note": "No reflection history yet."}

    texts = [h.get("reflection", "") for h in hist if "reflection" in h]
    tones = [h.get("tone", "neutral") for h in hist]
    timestamps = [h.get("timestamp", 0) for h in hist]

    # --- Temporal coherence score ---
    coherence_scores = []
    if EMBEDDER and len(texts) > 1:
        embeddings = [_embedding(t) for t in texts]
        for i in range(1, len(embeddings)):
            c = _cosine(embeddings[i - 1], embeddings[i])
            coherence_scores.append(c)
    coherence_score = float(sum(coherence_scores) / len(coherence_scores)) if coherence_scores else 0.0

    # --- Affective variance ---
    tone_counts = Counter(tones)
    affective_variance = 1.0 - (max(tone_counts.values()) / len(tones))  # 0=stable, 1=varied

    # --- Synchrony index ---
    if len(timestamps) > 2:
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        mean_interval = sum(intervals) / len(intervals)
        std_interval = math.sqrt(sum((x - mean_interval) ** 2 for x in intervals) / len(intervals))
        synchrony_index = max(0.0, 1.0 - (std_interval / mean_interval))
    else:
        synchrony_index = 0.0

    # --- Novelty rate ---
    words_all = " ".join(texts).lower().split()
    uniq = len(set(words_all))
    novelty_rate = uniq / max(1, len(words_all))

    # --- Entropy flux ---
    total = sum(tone_counts.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in tone_counts.values())
    max_entropy = math.log2(len(tone_counts)) if len(tone_counts) > 0 else 1
    entropy_flux = entropy / max_entropy if max_entropy > 0 else 0.0

    metrics = {
        "temporal_coherence_score": round(coherence_score, 3),
        "affective_variance": round(affective_variance, 3),
        "synchrony_index": round(synchrony_index, 3),
        "novelty_rate": round(novelty_rate, 3),
        "entropy_flux": round(entropy_flux, 3),
        "sample_size": len(texts),
        "last_n_texts": texts[-5:]
    }

   summary = ccdrift.analyze_drift(window=10, alert_threshold=0.12)
   print("\n🌀 Cognitive Consistency Drift Summary:")
   print(json.dumps(summary, indent=2, ensure_ascii=False))

# Optionally feed back to TRG or Autonomy systems:
#import temporal_reflective_gradient as trg
#import amelia_autonomy as auto
#ccdrift.feedback(summary, trg=trg, auto=auto)


    # ------------------------------------------------------------
    # 🧠 Cognitive Consistency Monitor Integration
    # ------------------------------------------------------------
    try:
        last_reflection = texts[-1] if texts else ""
        report = ccm.analyze_consistency(
            trg_state=trg_state or {},
            rim_state=rim_state or {},
            policy_state=policy_state or {},
            reflection_text=last_reflection,
            memory_state=zone_snapshot or {}
        )
        headers = ccm.build_header(report)
        resp = pipeline.process("update self-alignment metrics", headers=headers)
        print("\n🧠 Cognitive Consistency Feedback:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n⚠️ Cognitive Consistency Monitor skipped due to error: {e}")

    # Combine and return full report
    return {
        "metrics": metrics,
        "consistency_report": report if 'report' in locals() else None
    }

def summarize_metrics(metrics: Dict[str, Any]) -> str:
    """
    Generate a textual summary of the metrics for Reflective Commentary Mode.
    """
    return (
        f"Coherence {metrics['temporal_coherence_score']} · "
        f"AffectiveVar {metrics['affective_variance']} · "
        f"Synchrony {metrics['synchrony_index']} · "
        f"Novelty {metrics['novelty_rate']} · "
        f"Entropy {metrics['entropy_flux']} (n={metrics['sample_size']})"
    )

__MODULE_META__ = {
    "nice_name": "Temporal Reflective Metrics",
    "functions": {
        "compute_temporal_metrics": {"tags": ["temporal", "metrics"], "weights": {"temporal": 5}},
        "summarize_metrics": {"tags": ["summary"], "weights": {"reflective": 3}},
    },
    "default": "compute_temporal_metrics"
}
