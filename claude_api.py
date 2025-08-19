import json
import urllib.request
import urllib.parse

class ClaudeChat:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.api_key = config['api_key']
            self.model = config['model']
            self.max_tokens = config['max_tokens']
        else:
            # Fallback defaults
            self.api_key = "your-key-here"
            self.model = "claude-sonnet-4-20250514"
            self.max_tokens = 1024
        
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def chat(self, message):
        headers = {
            'x-api-key': self.api_key,
            'content-type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": message}]
        }
        
        try:
            req = urllib.request.Request(
                self.base_url, 
                json.dumps(data).encode(), 
                headers
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                return result['content'][0]['text']
        except Exception as e:
            return f"Error: {str(e)}"
