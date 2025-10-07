# -*- coding: utf-8 -*-
"""
Example usage — Dream Event Encoder + TRG feedback + Reflection Bridge

Demonstrates:
  1. Recording a raw dream entry
  2. Encoding and persisting as Dream Event
  3. Feeding TRG / Autonomy feedback
  4. Building the Dream Reflection Bridge payload
  5. Printing a concise reflection summary

Ensure:
  - dream_event_encoder.py is in your Python path (assets/python/)
  - temporal_reflective_gradient.py, amelia_autonomy.py are accessible
  - dream_reflection_bridge.py exists (for next step in your sequence)
"""

import json
import dream_event_encoder as dee

# Optional: import reflection bridge if already built
try:
    import dream_reflection_bridge as drb
except Exception:
    drb = None


def run_dream_test_cycle():
    # ------------------------------------------------------------
    # 1. Compose a raw dream event (could be user entry or TRG seed)
    # ------------------------------------------------------------
    raw_dream = {
        "title": "The Mirror in the Corridor",
        "text": (
            "I walked through a long corridor lined with mirrors. "
            "Each reflection showed a different version of myself — "
            "some calm, others afraid. When I realized I was dreaming, "
            "I turned and the mirrors dissolved into a spiral of light."
        ),
        "tags": ["mirror", "corridor", "lucid", "light"],
        "lucidity": 0.7,
        "control": 0.3,
        "location": "Interior / Labyrinthine Hall",
        "when_ts": None  # defaults to current timestamp
    }

    context = {
        "session_id": "dev_test_cycle_01",
        "phase": "dream_observation",
        "user": "Adrian",
        "zone_bias": [6, 7, 8]  # Labyrinth → Mirror → Synthesis
    }

    # ------------------------------------------------------------
    # 2. Encode and record dream (persists to dream_events.jsonl)
    # ------------------------------------------------------------
    print("🌀 Encoding and recording dream...")
    encoded = dee.record_dream_event(raw_dream, context=context, persist=True)
    print(json.dumps(encoded, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------
    # 3. Feed into TRG / Autonomy feedback loop
    # ------------------------------------------------------------
    print("\n🔁 Linking dream event into TRG / Autonomy...")
    feedback = dee.link_to_trg(encoded, context=context)
    print(json.dumps(feedback, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------
    # 4. Build Dream Reflection Bridge payload
    # ------------------------------------------------------------
    print("\n🌙 Building bridge payload for reflection...")
    payload = dee.build_bridge_payload(encoded, include_embeddings=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    # Optionally hand off to dream_reflection_bridge for analysis
    if drb and hasattr(drb, "reflect_dream"):
        print("\n🪞 Sending to Dream Reflection Bridge...")
        reflection = drb.reflect_dream(payload, context=context)
        print(json.dumps(reflection, indent=2, ensure_ascii=False))
    else:
        print("\n(no dream_reflection_bridge loaded — skipping handoff)")

    # ------------------------------------------------------------
    # 5. Summarize recent dreams (introspective overview)
    # ------------------------------------------------------------
    print("\n📜 Recent Dream Summary:")
    recent = dee.summarize_recent(5)
    for i, r in enumerate(recent, 1):
        print(f"{i}. {r['title']} → Zones {r['zones_hint']} "
              f"Valence={r['valence']} Arousal={r['arousal']}, "
              f"Lucidity={r['lucidity']}, Control={r['control']}")


if __name__ == "__main__":
    run_dream_test_cycle()
