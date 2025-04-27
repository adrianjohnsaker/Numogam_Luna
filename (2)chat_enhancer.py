def initialize():
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        return "NLP resources initialized"
    except Exception as e:
        return f"Initialization error: {str(e)}"

def enhance_response(user_input):
    try:
        # First try with TextBlob only (simpler, more likely to work)
        from textblob import TextBlob
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        
        response = "That sounds " + (
            "positive!" if sentiment > 0 else 
            "neutral." if sentiment == 0 else 
            "a bit down."
        )
        
        # Try to use basic text processing without spacy
        if "hello" in user_input.lower():
            response = "Hello! How are you today?"
        elif "help" in user_input.lower():
            response = "I'm here to help! What do you need?"
        else:
            response += f" You said: {user_input}"
            
        return response
        
    except Exception as e:
        # Fallback to simple response if anything fails
        return f"I heard you say: {user_input}"
