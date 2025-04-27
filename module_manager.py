def process_message(message):
    if message.startswith("calc:"):
        expression = message[5:]
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    return message
