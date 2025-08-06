# File: app/src/main/python/app.py

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "")
    # TODO: import and call your enhancement modules here
    # from your_module import generate_response
    # bot_reply = generate_response(user_msg)
    bot_reply = f"You said: {user_msg}"
    return jsonify(reply=bot_reply)

if __name__ == "__main__":
    # When started via Chaquopy, this will run in-process
    app.run(host="0.0.0.0", port=5000)
