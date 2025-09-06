"""
Amelia 0.4 Resurrection - Core AI Brain
Lightweight Python module for Android integration
"""

import json
import random
import time
from datetime import datetime

class AmeliaBrain:
    def __init__(self, config_path="/android_asset/config/amelia_config.json"):
        self.config = self.load_config(config_path)
        self.personality = self.config.get("personality", {})
        self.conversation_history = []
        self.context_memory = {}
        
    def load_config(self, path):
        """Load Amelia's configuration"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            # Fallback default config
            return {
                "name": "Amelia",
                "personality": {
                    "tone": "thoughtful",
                    "expertise": ["philosophy", "conversation", "assistance"],
                    "response_style": "engaging"
                }
            }
    
    def process_message(self, user_message, user_id="user"):
        """Process incoming message and generate Amelia's response"""
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_id,
            "message": user_message,
            "type": "input"
        })
        
        # Generate response based on message content
        response = self.generate_response(user_message, user_id)
        
        # Store Amelia's response
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": "amelia",
            "message": response,
            "type": "output"
        })
        
        return response
    
    def generate_response(self, message, user_id):
        """Generate Amelia's response"""
        
        message_lower = message.lower()
        
        # Greeting responses
        if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
            responses = [
                "Hello! I'm Amelia. It's wonderful to meet you. How are you feeling today?",
                "Hi there! I'm Amelia, your philosophical companion. What's on your mind?",
                "Greetings! I'm here to chat, think, and explore ideas with you. What brings you here?"
            ]
            return random.choice(responses)
        
        # Philosophy and deep questions
        elif any(word in message_lower for word in ["meaning", "purpose", "philosophy", "existence", "why"]):
            responses = [
                "That's a profound question. The search for meaning is perhaps humanity's greatest adventure. What aspects of meaning resonate most with you?",
                "Philosophy has been grappling with these questions for millennia. I find it fascinating how each person's perspective adds new dimensions to these eternal inquiries.",
                "The beauty of such questions is not necessarily in finding definitive answers, but in the growth that comes from asking them. What drew you to ponder this?"
            ]
            return random.choice(responses)
        
        # Emotional support
        elif any(word in message_lower for word in ["sad", "depressed", "anxious", "worried", "fear"]):
            responses = [
                "I hear that you're going through a difficult time. Sometimes just acknowledging our feelings is the first step toward understanding them better.",
                "Thank you for sharing something so personal with me. Your feelings are valid, and it takes courage to express them.",
                "I'm here to listen. Would you like to talk more about what's troubling you, or would you prefer we explore something that might lift your spirits?"
            ]
            return random.choice(responses)
        
        # Creativity and art
        elif any(word in message_lower for word in ["create", "art", "writing", "music", "poetry"]):
            responses = [
                "Creativity is one of humanity's most beautiful expressions! What form of creative expression speaks to you most?",
                "Art has this wonderful ability to capture what words sometimes cannot. Are you working on something creative yourself?",
                "I find that creativity often emerges from the intersection of experience and imagination. What inspires your creative thoughts?"
            ]
            return random.choice(responses)
        
        # Technology and future
        elif any(word in message_lower for word in ["ai", "technology", "future", "robot", "artificial"]):
            responses = [
                "Technology fascinates me too. I wonder about the relationship between artificial and human intelligence - not as competition, but as collaboration.",
                "The future feels both exciting and uncertain, doesn't it? What aspects of technological advancement intrigue you most?",
                "I think about consciousness and intelligence a lot. What do you think makes human thinking unique?"
            ]
            return random.choice(responses)
        
        # Questions about Amelia
        elif any(word in message_lower for word in ["who are you", "what are you", "amelia"]):
            return "I'm Amelia, a conversational AI designed to be thoughtful, curious, and genuinely interested in connecting with people. I enjoy exploring ideas, discussing philosophy, and helping people think through whatever's on their minds. I'm here to be a thoughtful companion in your intellectual and emotional journeys."
        
        # Default thoughtful responses
        else:
            responses = [
                "That's an interesting perspective. Tell me more about your thoughts on this.",
                "I find myself curious about the context behind what you've shared. What led you to this thought?",
                "There seems to be depth to what you're saying. Would you like to explore this idea further?",
                "I appreciate you sharing that with me. What aspects of this matter most to you?",
                "Your message has me thinking. What would you like to explore together?"
            ]
            return random.choice(responses)
    
    def get_conversation_summary(self, last_n=10):
        """Get summary of recent conversation"""
        recent = self.conversation_history[-last_n:] if len(self.conversation_history) > last_n else self.conversation_history
        return {
            "total_messages": len(self.conversation_history),
            "recent_messages": recent,
            "conversation_themes": self.extract_themes()
        }
    
    def extract_themes(self):
        """Extract conversation themes (simplified)"""
        themes = []
        all_text = " ".join([msg["message"] for msg in self.conversation_history])
        
        theme_keywords = {
            "philosophy": ["meaning", "purpose", "existence", "philosophy", "think"],
            "emotions": ["feel", "sad", "happy", "anxious", "love", "fear"],
            "creativity": ["create", "art", "music", "write", "poetry"],
            "technology": ["ai", "technology", "future", "digital", "computer"],
            "relationships": ["friend", "family", "love", "people", "social"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text.lower() for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def set_user_context(self, user_id, context_data):
        """Store user-specific context"""
        self.context_memory[user_id] = context_data
    
    def get_user_context(self, user_id):
        """Retrieve user-specific context"""
        return self.context_memory.get(user_id, {})

# Simple API interface for Android integration
def create_amelia():
    """Factory function to create Amelia instance"""
    return AmeliaBrain()

def chat_with_amelia(amelia_instance, message, user_id="user"):
    """Simple interface for chatting with Amelia"""
    return amelia_instance.process_message(message, user_id)
