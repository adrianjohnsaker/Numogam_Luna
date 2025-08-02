# claude_chat.py

import anthropic

# Put your real Claude API key here
API_KEY = "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"

client = anthropic.Anthropic(api_key=API_KEY)

def ask_claude(question, system="You are Amelia, an emergent AI companion. Respond with empathy and clarity."):
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Adjust according to your API access
            max_tokens=600,
            temperature=0.6,
            system=system,
            messages=[{"role": "user", "content": question}]
        )
        return "".join([c.text for c in response.content])
    except Exception as e:
        return f"[Claude API error: {str(e)}]"

def send_broadcast_reply(reply_text):
    from jnius import autoclass
    Intent = autoclass('android.content.Intent')
    PythonActivity = autoclass('org.kivy.android.PythonActivity')
    context = PythonActivity.mActivity

    intent = Intent("claude.REPLY")
    intent.putExtra("claude_reply", reply_text)
    context.sendBroadcast(intent)

def process_and_reply(user_text):
    reply = ask_claude(user_text)
    send_broadcast_reply(reply)

if __name__ == "__main__":
    print("Claude chat test. Type your question, or 'exit' to quit:")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        answer = ask_claude(q)
        print(f"Claude reply: {answer}\n")
