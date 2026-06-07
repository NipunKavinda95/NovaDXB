# ─────────────────────────────────────────
# NovaDXB — app.py
# Flask App — Phase 2 (Agent connected)
# ─────────────────────────────────────────

import os
import threading
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG + Agent
from rag_engine import initialize_rag
from agent import initialize_agent, query_agent

# Initialize Flask
app = Flask(
    __name__,
    static_folder="static",
    template_folder="static"
)

# ─────────────────────────────────────────
# STARTUP — Initialize RAG then Agent
# Background thread — app starts instantly
# ─────────────────────────────────────────

def startup():
    initialize_rag()
    initialize_agent()

threading.Thread(target=startup, daemon=True).start()


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "app":     "NovaDXB",
        "version": "1.0"
    })


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "error": "No message provided"
            }), 400

        # ── Agent response ────────────────
        response = query_agent(message)

        return jsonify({
            "status":   "success",
            "response": response
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 7860))
    print(f"✅ NovaDXB running on port {port}")
    serve(app, host="0.0.0.0", port=port)