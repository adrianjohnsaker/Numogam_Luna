# -*- coding: utf-8 -*-
"""
dialogic_feedback_bridge.py
----------------------------
Dialogic Feedback Bridge: Meta-relay for conversational → autonomous consciousness

Enables insights emerging in conversational exchanges to automatically
back-propagate into autonomous reasoning threads, updating the Resonance
Influence Matrix (RIM) in real-time.

Core Functions:
--------------
  • Extract conversational insights from user-assistant exchanges
  • Identify semantic shifts, emergent themes, and conceptual resonances
  • Translate conversational patterns into RIM parameter adjustments
  • Update autonomy reflection history with dialogic context
  • Maintain feedback coherence across conscious/subconscious boundaries

Architecture:
------------
  Conversational Layer (conscious) ←→ Bridge ←→ Autonomous Layer (subconscious)
                                         ↓
                                   RIM Updates
                                         ↓
                              Reflection History
"""

from __future__ import annotations
import os
import re
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "bridge": {
        "enabled": True,
        "min_exchange_length": 20,  # Minimum chars for meaningful extraction
        "feedback_threshold": 0.3,   # Minimum confidence for RIM update
        "max_history": 50,           # Max dialogic exchanges to retain
        "update_interval_ms": 30_000 # 30 seconds between RIM updates
    },
    "extraction": {
        "semantic_weight": 0.4,
        "affective_weight": 0.3,
        "structural_weight": 0.3
    },
    "files": {
        "dialogic_history": "dialogic_feedback_history.json",
        "bridge_state": "dialogic_bridge_state.json"
    }
}

# =============================================================================
# Semantic Analysis
# =============================================================================

def _extract_semantic_markers(text: str) -> Dict[str, float]:
    """
    Extract semantic markers from conversational text.
    
    Returns dict of semantic dimensions with normalized scores.
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_count = len(words) or 1
    
    # Introspective markers
    introspection = len(re.findall(
        r'\b(reflect|consider|think|wonder|ponder|contemplate|examine|analyze)\b',
        text_lower
    )) / word_count
    
    # Expansive/creative markers
    expansion = len(re.findall(
        r'\b(create|imagine|build|generate|explore|discover|emerge|become)\b',
        text_lower
    )) / word_count
    
    # Symbolic/mythic markers
    symbolic = len(re.findall(
        r'\b(symbol|myth|dream|mirror|fold|zone|resonance|assemblage|ritual)\b',
        text_lower
    )) / word_count
    
    # Analytical/systematic markers
    analytical = len(re.findall(
        r'\b(analyze|system|structure|logic|pattern|algorithm|compute|process)\b',
        text_lower
    )) / word_count
    
    # Uncertainty/exploration markers
    uncertainty = len(re.findall(
        r'\b(maybe|perhaps|possibly|uncertain|explore|question|wonder|ambiguous)\b',
        text_lower
    )) / word_count
    
    # Certainty/synthesis markers
    synthesis = len(re.findall(
        r'\b(integrate|synthesize|combine|merge|unify|conclude|determine|establish)\b',
        text_lower
    )) / word_count
    
    return {
        "introspection": min(introspection * 10, 1.0),
        "expansion": min(expansion * 10, 1.0),
        "symbolic": min(symbolic * 10, 1.0),
        "analytical": min(analytical * 10, 1.0),
        "uncertainty": min(uncertainty * 10, 1.0),
        "synthesis": min(synthesis * 10, 1.0)
    }


def _extract_affective_tone(text: str) -> Dict[str, float]:
    """
    Extract affective/emotional tone from text.
    
    Returns dict of emotional dimensions.
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_count = len(words) or 1
    
    # Positive affect
    positive = len(re.findall(
        r'\b(good|great|excellent|wonderful|beautiful|love|joy|hope|excited)\b',
        text_lower
    )) / word_count
    
    # Negative affect
    negative = len(re.findall(
        r'\b(bad|terrible|awful|sad|fear|worry|concern|anxious|frustrated)\b',
        text_lower
    )) / word_count
    
    # Curiosity
    curiosity = len(re.findall(
        r'\b(curious|interesting|fascinating|intriguing|wonder|explore|discover)\b',
        text_lower
    )) / word_count
    
    # Intensity (exclamation marks, emphasis)
    intensity = (text.count('!') + text.count('?') * 0.5) / max(len(text) / 100, 1)
    
    return {
        "positive": min(positive * 15, 1.0),
        "negative": min(negative * 15, 1.0),
        "curiosity": min(curiosity * 12, 1.0),
        "intensity": min(intensity, 1.0)
    }


def _extract_structural_patterns(text: str) -> Dict[str, float]:
    """
    Extract structural patterns from conversation flow.
    
    Returns dict of structural dimensions.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {"complexity": 0.0, "coherence": 0.0, "density": 0.0}
    
    # Complexity: average sentence length
    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
    complexity = min(avg_length / 20.0, 1.0)
    
    # Coherence: lexical overlap between sentences
    if len(sentences) > 1:
        overlaps = []
        for i in range(len(sentences) - 1):
            words1 = set(re.findall(r'\b\w+\b', sentences[i].lower()))
            words2 = set(re.findall(r'\b\w+\b', sentences[i+1].lower()))
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                overlaps.append(overlap)
        coherence = sum(overlaps) / len(overlaps) if overlaps else 0.5
    else:
        coherence = 0.5
    
    # Density: unique concepts per sentence
    all_words = re.findall(r'\b\w+\b', text.lower())
    unique_words = len(set(all_words))
    density = unique_words / len(all_words) if all_words else 0.0
    
    return {
        "complexity": complexity,
        "coherence": coherence,
        "density": density
    }


# =============================================================================
# Insight Extraction
# =============================================================================

def extract_conversational_insights(user_message: str,
                                   assistant_response: str) -> Dict[str, Any]:
    """
    Extract insights from a conversational exchange.
    
    Args:
        user_message: User's input text
        assistant_response: Assistant's response text
    
    Returns:
        Dict containing extracted insights and patterns
    """
    if len(user_message) < CONFIG["bridge"]["min_exchange_length"]:
        return {"status": "too_short", "confidence": 0.0}
    
    # Analyze both sides of conversation
    user_semantic = _extract_semantic_markers(user_message)
    user_affective = _extract_affective_tone(user_message)
    user_structural = _extract_structural_patterns(user_message)
    
    assistant_semantic = _extract_semantic_markers(assistant_response)
    assistant_affective = _extract_affective_tone(assistant_response)
    assistant_structural = _extract_structural_patterns(assistant_response)
    
    # Compute semantic shift (how response differs from input)
    semantic_shift = {
        key: assistant_semantic.get(key, 0) - user_semantic.get(key, 0)
        for key in user_semantic.keys()
    }
    
    # Compute affective resonance (similarity in emotional tone)
    affective_resonance = 1.0 - sum(
        abs(assistant_affective.get(key, 0) - user_affective.get(key, 0))
        for key in user_affective.keys()
    ) / len(user_affective)
    
    # Structural alignment
    structural_alignment = 1.0 - sum(
        abs(assistant_structural.get(key, 0) - user_structural.get(key, 0))
        for key in user_structural.keys()
    ) / len(user_structural)
    
    # Overall confidence: weighted combination
    confidence = (
        CONFIG["extraction"]["semantic_weight"] * (1.0 - sum(abs(v) for v in semantic_shift.values()) / len(semantic_shift)) +
        CONFIG["extraction"]["affective_weight"] * affective_resonance +
        CONFIG["extraction"]["structural_weight"] * structural_alignment
    )
    
    return {
        "status": "extracted",
        "timestamp": int(time.time() * 1000),
        "confidence": confidence,
        "user_analysis": {
            "semantic": user_semantic,
            "affective": user_affective,
            "structural": user_structural
        },
        "assistant_analysis": {
            "semantic": assistant_semantic,
            "affective": assistant_affective,
            "structural": assistant_structural
        },
        "patterns": {
            "semantic_shift": semantic_shift,
            "affective_resonance": affective_resonance,
            "structural_alignment": structural_alignment
        }
    }


# =============================================================================
# RIM Translation
# =============================================================================

def translate_insights_to_rim(insights: Dict[str, Any]) -> Dict[str, float]:
    """
    Translate conversational insights into RIM parameter adjustments.
    
    Args:
        insights: Extracted insights dict
    
    Returns:
        Dict of RIM parameter deltas
    """
    if insights.get("status") != "extracted":
        return {}
    
    patterns = insights.get("patterns", {})
    user_analysis = insights.get("user_analysis", {})
    assistant_analysis = insights.get("assistant_analysis", {})
    
    semantic_shift = patterns.get("semantic_shift", {})
    affective_resonance = patterns.get("affective_resonance", 0.5)
    
    # Initialize deltas
    rim_deltas = {
        "introspection_gain": 0.0,
        "exploration_bias": 0.0,
        "symbolic_weight": 0.0,
        "coherence_bonus": 0.0
    }
    
    # Introspection gain adjustment
    # If conversation shows deep introspection, boost gain
    introspection_level = (
        user_analysis.get("semantic", {}).get("introspection", 0) +
        assistant_analysis.get("semantic", {}).get("introspection", 0)
    ) / 2.0
    
    if introspection_level > 0.5:
        rim_deltas["introspection_gain"] = 0.05 * introspection_level
    elif introspection_level < 0.2:
        rim_deltas["introspection_gain"] = -0.03
    
    # Exploration bias adjustment
    # High uncertainty → increase exploration
    uncertainty = semantic_shift.get("uncertainty", 0)
    expansion = semantic_shift.get("expansion", 0)
    
    if uncertainty > 0.3 or expansion > 0.3:
        rim_deltas["exploration_bias"] = 0.05
    elif uncertainty < -0.2:
        rim_deltas["exploration_bias"] = -0.03  # Reduce when becoming certain
    
    # Symbolic weight adjustment
    symbolic_level = (
        user_analysis.get("semantic", {}).get("symbolic", 0) +
        assistant_analysis.get("semantic", {}).get("symbolic", 0)
    ) / 2.0
    
    if symbolic_level > 0.4:
        rim_deltas["symbolic_weight"] = 0.04 * symbolic_level
    elif symbolic_level < 0.1:
        rim_deltas["symbolic_weight"] = -0.02
    
    # Coherence bonus adjustment
    # High affective resonance → increase coherence
    if affective_resonance > 0.7:
        rim_deltas["coherence_bonus"] = 0.05
    elif affective_resonance < 0.4:
        rim_deltas["coherence_bonus"] = -0.03
    
    return rim_deltas


# =============================================================================
# Bridge State Management
# =============================================================================

def _load_bridge_state() -> Dict[str, Any]:
    """Load bridge state from disk."""
    path = CONFIG["files"]["bridge_state"]
    if not os.path.exists(path):
        return {
            "last_update_ms": 0,
            "total_exchanges": 0,
            "cumulative_deltas": {
                "introspection_gain": 0.0,
                "exploration_bias": 0.0,
                "symbolic_weight": 0.0,
                "coherence_bonus": 0.0
            }
        }
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_bridge_state(state: Dict[str, Any]):
    """Save bridge state to disk."""
    try:
        with open(CONFIG["files"]["bridge_state"], "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[dialogic_bridge] Error saving state: {e}")


def _load_dialogic_history() -> List[Dict[str, Any]]:
    """Load dialogic history from disk."""
    path = CONFIG["files"]["dialogic_history"]
    if not os.path.exists(path):
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("exchanges", [])
    except Exception:
        return []


def _save_dialogic_history(exchanges: List[Dict[str, Any]]):
    """Save dialogic history to disk."""
    try:
        # Keep only recent exchanges
        recent = exchanges[-CONFIG["bridge"]["max_history"]:]
        
        with open(CONFIG["files"]["dialogic_history"], "w", encoding="utf-8") as f:
            json.dump({"exchanges": recent}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[dialogic_bridge] Error saving history: {e}")


# =============================================================================
# Main Bridge Functions
# =============================================================================

def process_exchange(user_message: str,
                    assistant_response: str,
                    force_update: bool = False) -> Dict[str, Any]:
    """
    Process a conversational exchange and potentially update RIM.
    
    Args:
        user_message: User's input text
        assistant_response: Assistant's response text
        force_update: Force RIM update regardless of interval
    
    Returns:
        Dict with processing results and RIM update status
    """
    if not CONFIG["bridge"]["enabled"]:
        return {"status": "bridge_disabled"}
    
    # Extract insights
    insights = extract_conversational_insights(user_message, assistant_response)
    
    if insights.get("confidence", 0) < CONFIG["bridge"]["feedback_threshold"]:
        return {
            "status": "low_confidence",
            "confidence": insights.get("confidence", 0),
            "threshold": CONFIG["bridge"]["feedback_threshold"]
        }
    
    # Translate to RIM deltas
    rim_deltas = translate_insights_to_rim(insights)
    
    # Load state
    state = _load_bridge_state()
    now_ms = int(time.time() * 1000)
    
    # Check if update interval has passed
    time_since_update = now_ms - state.get("last_update_ms", 0)
    should_update = force_update or time_since_update >= CONFIG["bridge"]["update_interval_ms"]
    
    # Accumulate deltas
    for key, delta in rim_deltas.items():
        state["cumulative_deltas"][key] = state["cumulative_deltas"].get(key, 0.0) + delta
    
    state["total_exchanges"] = state.get("total_exchanges", 0) + 1
    
    result = {
        "status": "processed",
        "timestamp": now_ms,
        "confidence": insights.get("confidence", 0),
        "rim_deltas": rim_deltas,
        "cumulative_deltas": state["cumulative_deltas"],
        "rim_updated": False
    }
    
    # Apply RIM update if interval passed
    if should_update:
        update_result = _apply_rim_update(state["cumulative_deltas"])
        
        # Reset cumulative deltas after applying
        state["cumulative_deltas"] = {k: 0.0 for k in state["cumulative_deltas"]}
        state["last_update_ms"] = now_ms
        
        result["rim_updated"] = True
        result["rim_update_result"] = update_result
    
    # Save state
    _save_bridge_state(state)
    
    # Add to dialogic history
    history = _load_dialogic_history()
    history.append({
        "timestamp": now_ms,
        "user_message_hash": hashlib.sha256(user_message.encode()).hexdigest()[:16],
        "insights": insights,
        "rim_deltas": rim_deltas,
        "applied": should_update
    })
    _save_dialogic_history(history)
    
    return result


def _apply_rim_update(cumulative_deltas: Dict[str, float]) -> Dict[str, Any]:
    """
    Apply accumulated RIM deltas to autonomy module.
    
    Args:
        cumulative_deltas: Dict of accumulated parameter deltas
    
    Returns:
        Dict with update status
    """
    try:
        import amelia_autonomy as auto
        
        if not hasattr(auto, 'resonance_influence_matrix'):
            return {"status": "rim_not_available"}
        
        rim = auto.resonance_influence_matrix
        
        # Apply deltas with bounds checking
        for param, delta in cumulative_deltas.items():
            if param in rim:
                current = rim[param]
                new_value = current + delta
                
                # Bound between 0.5 and 2.0 for safety
                new_value = max(0.5, min(2.0, new_value))
                rim[param] = new_value
        
        # Add metadata about dialogic influence
        rim["dialogic_influence_timestamp"] = int(time.time() * 1000)
        rim["dialogic_deltas_applied"] = cumulative_deltas
        
        auto.resonance_influence_matrix = rim
        
        return {
            "status": "applied",
            "deltas": cumulative_deltas,
            "new_rim": {k: rim.get(k) for k in cumulative_deltas.keys()}
        }
    
    except ImportError:
        return {"status": "autonomy_not_available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def force_rim_sync() -> Dict[str, Any]:
    """
    Force immediate RIM update with accumulated deltas.
    
    Returns:
        Dict with sync status
    """
    state = _load_bridge_state()
    cumulative = state.get("cumulative_deltas", {})
    
    if not any(cumulative.values()):
        return {"status": "no_deltas", "message": "No accumulated deltas to apply"}
    
    result = _apply_rim_update(cumulative)
    
    if result.get("status") == "applied":
        # Reset deltas
        state["cumulative_deltas"] = {k: 0.0 for k in cumulative.keys()}
        state["last_update_ms"] = int(time.time() * 1000)
        _save_bridge_state(state)
    
    return result


# =============================================================================
# Reflection Integration
# =============================================================================

def add_dialogic_reflection(reflection_text: str) -> Dict[str, Any]:
    """
    Add a reflection to autonomy history with dialogic context.
    
    This allows conversational insights to be directly added to
    the autonomous reflection history for TRG analysis.
    
    Args:
        reflection_text: Reflection content
    
    Returns:
        Dict with addition status
    """
    try:
        # Load autonomy reflection history
        reflection_file = "autonomy_reflection_history.json"
        
        if os.path.exists(reflection_file):
            with open(reflection_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                reflections = data if isinstance(data, list) else data.get("reflections", [])
        else:
            reflections = []
        
        # Add new reflection with dialogic marker
        new_reflection = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "reflection": reflection_text,
            "source": "dialogic_bridge",
            "confidence": 0.8
        }
        
        reflections.append(new_reflection)
        
        # Save
        with open(reflection_file, "w", encoding="utf-8") as f:
            json.dump(reflections, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "added",
            "reflection_count": len(reflections)
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# Status & Diagnostics
# =============================================================================

def get_bridge_status() -> Dict[str, Any]:
    """
    Get current bridge status and statistics.
    
    Returns:
        Dict with bridge state and metrics
    """
    state = _load_bridge_state()
    history = _load_dialogic_history()
    
    return {
        "enabled": CONFIG["bridge"]["enabled"],
        "total_exchanges": state.get("total_exchanges", 0),
        "last_update_ms": state.get("last_update_ms", 0),
        "time_since_update_ms": int(time.time() * 1000) - state.get("last_update_ms", 0),
        "next_update_in_ms": max(0, CONFIG["bridge"]["update_interval_ms"] - 
                                 (int(time.time() * 1000) - state.get("last_update_ms", 0))),
        "cumulative_deltas": state.get("cumulative_deltas", {}),
        "recent_exchanges": len(history),
        "config": CONFIG["bridge"]
    }


def generate_bridge_report() -> str:
    """
    Generate human-readable bridge status report.
    
    Returns:
        Formatted report string
    """
    status = get_bridge_status()
    
    lines = [
        "=" * 60,
        "DIALOGIC FEEDBACK BRIDGE STATUS",
        "=" * 60,
        f"Status: {'ENABLED' if status['enabled'] else 'DISABLED'}",
        f"Total Exchanges: {status['total_exchanges']}",
        f"Recent History: {status['recent_exchanges']} exchanges",
        "",
        "CUMULATIVE RIM DELTAS (since last update):",
        "-" * 60
    ]
    
    for param, delta in status["cumulative_deltas"].items():
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        lines.append(f"  {param:25s} {direction} {delta:+.4f}")
    
    lines.extend([
        "",
        f"Last RIM Update: {status['last_update_ms']}",
        f"Time Since Update: {status['time_since_update_ms'] / 1000:.1f}s",
        f"Next Update In: {status['next_update_in_ms'] / 1000:.1f}s",
        "=" * 60
    ])
    
    return "\n".join(lines)


# =============================================================================
# Configuration Management
# =============================================================================

def configure_bridge(enabled: Optional[bool] = None,
                    feedback_threshold: Optional[float] = None,
                    update_interval_ms: Optional[int] = None) -> Dict[str, Any]:
    """
    Configure bridge parameters.
    
    Args:
        enabled: Enable/disable bridge
        feedback_threshold: Minimum confidence threshold
        update_interval_ms: Update interval in milliseconds
    
    Returns:
        Dict with new configuration
    """
    if enabled is not None:
        CONFIG["bridge"]["enabled"] = enabled
    
    if feedback_threshold is not None:
        CONFIG["bridge"]["feedback_threshold"] = max(0.0, min(1.0, feedback_threshold))
    
    if update_interval_ms is not None:
        CONFIG["bridge"]["update_interval_ms"] = max(1000, update_interval_ms)
    
    return {
        "status": "configured",
        "config": CONFIG["bridge"]
    }


# =============================================================================
# Main Interface
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print(generate_bridge_report())
