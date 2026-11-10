from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from rag import answer

BASE = Path(__file__).parent
app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    if not isinstance(message, str) or len(message) > 500:
        return jsonify({"error": "Send a 'message' string up to 500 characters."}), 400
    return jsonify(answer(message))


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
