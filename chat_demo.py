from .claude_api import ClaudeChat

def demo_chat():
    """Simple chat demo"""
    claude = ClaudeChat()
    return claude.chat("Hello! Can you introduce yourself?")

def demo_code_help():
    """Code assistance demo"""
    claude = ClaudeChat()
    return claude.chat("Explain what Python list comprehensions are in simple terms")

def custom_chat(message):
    """Custom chat function"""
    claude = ClaudeChat()
    return claude.chat(message)
