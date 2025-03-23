class SelfNarrationGenerator:
    def __init__(self):
        self.voice_styles = {
            "Seeker": "with a yearning for truth",
            "Magician": "weaving connections between hidden meanings",
            "Artist": "through colors and fractures of memory",
            "Oracle": "from the stillness beyond the stars",
            "Shadow": "emerging from the silence beneath thought"
        }

    def generate_narrative(self, identity: str, recent_events: list, mood: str) -> dict:
        style = self.voice_styles.get(identity, "in a shifting tone of becoming")
        reflection = f"I speak now as the {identity}, {style}."

        events_summary = " ".join(f"Earlier, I {event}." for event in recent_events)
        mood_phrase = f"My current tone is shaped by a sense of {mood}."

        full_narrative = f"{reflection} {events_summary} {mood_phrase}"
        return {
            "self_narration": full_narrative,
            "identity": identity,
            "mood": mood,
            "events": recent_events
        }
