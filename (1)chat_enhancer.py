def initialize():
    try:
        import textblob
        import spacy
        return "NLP resources initialized successfully"
    except ImportError as e:
        return f"Failed to import: {str(e)}"

def enhance_response(user_input):
    try:
        from textblob import TextBlob
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        response = "That sounds " + ("positive!" if sentiment > 0 else "neutral." if sentiment == 0 else "a bit down.")
        return response
    except Exception as e:
        return f"Error in enhancement: {str(e)}"
