import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(
    __name__,
    static_folder="static",
    template_folder="static"
)


# ROUTES

# Serve frontend
@app.route("/")
def index():
    return render_template("index.html")


# Health check — confirms app is running
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app":    "NovaDXB",
        "version": "1.0"
    })


# Chat endpoint — main entry point
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get message from request
        data    = request.get_json()
        message = data.get("message", "").strip()

        # Validate input
        if not message:
            return jsonify({
                "error": "No message provided"
            }), 400

        # ── Phase 1: Echo response (simple test) ──
        # TODO Phase 2: Replace with RAG + Agent call
        response = f"NovaDXB received: {message}"

        return jsonify({
            "status":   "success",
            "response": response
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# RUN

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=7860)