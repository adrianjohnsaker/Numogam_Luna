from textblob import TextBlob
import spacy

nlp = spacy.load("en_core_web_sm")

def initialize():
    import nltk
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    return "NLP resources initialized"

def enhance_response(user_input):
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    response = "That sounds " + ("positive!" if sentiment > 0 else "neutral." if sentiment == 0 else "a bit down.")
    if entities:
        response += f" I noticed: {', '.join([e[0] for e in entities])}."
    return response
