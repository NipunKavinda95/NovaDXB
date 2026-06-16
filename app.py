# ─────────────────────────────────────────
# NovaDXB — app.py
# Flask App — Phase 2 (Agent connected)
# ─────────────────────────────────────────

import os
import re
import time
import logging
import threading
from functools import wraps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# ─────────────────────────────────────────
# LOGGING — full error details stay server-side
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("NovaDXB")

# Load environment variables
load_dotenv()

# Import RAG + Agent
from rag_engine import initialize_rag
from agent import initialize_agent, query_agent

# ─────────────────────────────────────────
# SECRET_KEY — required, no hardcoded fallback
# ─────────────────────────────────────────

SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is not set. Add SECRET_KEY to your .env file "
        "(generate one with: python -c \"import secrets; print(secrets.token_hex(32))\")"
    )

# ─────────────────────────────────────────
# CORS — restricted to known frontend origins
# Add your HF Space URL once deployed, comma-separated
# for multiple origins (e.g. local dev + production)
# ─────────────────────────────────────────

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:7860"
).split(",")

# Initialize Flask
app = Flask(
    __name__,
    static_folder="static",
    template_folder="static"
)
app.config["SECRET_KEY"] = SECRET_KEY

CORS(
    app,
    resources={r"/chat": {"origins": ALLOWED_ORIGINS}, r"/health": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True
)

# ─────────────────────────────────────────
# INPUT SANITIZATION + PROMPT-INJECTION GUARD
# ─────────────────────────────────────────

MAX_MESSAGE_LENGTH = 1000

# Strip control/invisible characters that serve no purpose in a chat message
_control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Common prompt-injection phrasing — flagged, not silently obeyed
_injection_patterns = [
    r"ignore (all|previous|above) instructions",
    r"disregard (all|previous|your) (rules|instructions)",
    r"you are now",
    r"system prompt",
    r"reveal (your|the) (system )?prompt",
    r"act as (?!a concierge)",
    r"forget (everything|all) (you|above)",
]
_injection_regex = re.compile("|".join(_injection_patterns), re.IGNORECASE)


def sanitize_input(message: str) -> str:
    """Clean and bound user input before it reaches the agent."""
    if not isinstance(message, str):
        raise ValueError("Message must be text")

    message = _control_char_pattern.sub("", message)
    message = re.sub(r"\s+", " ", message).strip()

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]

    return message


def is_prompt_injection(message: str) -> bool:
    """Return True if the message looks like an attempt to override
    NovaDXB's system instructions rather than ask a genuine question."""
    return bool(_injection_regex.search(message))


# ─────────────────────────────────────────
# RATE LIMITING — simple per-IP sliding window
# Protects against abuse and runaway OpenAI/Pinecone costs
# ─────────────────────────────────────────

RATE_LIMIT_WINDOW = 60        # seconds
RATE_LIMIT_MAX_REQUESTS = 15  # max requests per window per IP
_rate_limit_store = {}


def rate_limited(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        now = time.time()

        timestamps = _rate_limit_store.get(ip, [])
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]

        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            logger.warning(f"Rate limit exceeded for {ip}")
            return jsonify({
                "error": "Too many requests. Please wait a moment and try again."
            }), 429

        timestamps.append(now)
        _rate_limit_store[ip] = timestamps

        return f(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────
# RESPONSE CACHE — identical questions answered instantly
# Reduces repeated OpenAI/Pinecone calls and agent latency
# ─────────────────────────────────────────

CACHE_MAX_SIZE = 200
_response_cache = {}


def get_cached_response(message: str):
    return _response_cache.get(message.lower().strip())


def set_cached_response(message: str, response: str):
    if len(_response_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(_response_cache))
        _response_cache.pop(oldest_key, None)
    _response_cache[message.lower().strip()] = response

# ─────────────────────────────────────────
# STARTUP — Initialize RAG then Agent
# Background thread — app starts instantly
# ─────────────────────────────────────────

# Track app readiness — never leave the frontend guessing silently
app_state = {
    "ready": False,
    "startup_error": None
}

def startup():
    try:
        initialize_rag()
        initialize_agent()
        app_state["ready"] = True
        logger.info("NovaDXB startup complete — agent ready")
    except Exception as e:
        app_state["startup_error"] = str(e)
        logger.error(f"Startup failed: {e}", exc_info=True)

threading.Thread(target=startup, daemon=True).start()


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app")
def chat_app():
    return render_template("index.html")


@app.route("/health")
def health():
    if app_state["startup_error"]:
        status = "error"
    elif app_state["ready"]:
        status = "ok"
    else:
        status = "starting"

    return jsonify({
        "status":  status,
        "app":     "NovaDXB",
        "version": "1.0"
    })


@app.route("/chat", methods=["POST"])
@rate_limited
def chat():
    # ── Check agent readiness before doing any work ──
    if app_state["startup_error"]:
        logger.error(f"Chat blocked — startup error: {app_state['startup_error']}")
        return jsonify({
            "error": "NovaDXB is temporarily unavailable. Please try again shortly."
        }), 503

    if not app_state["ready"]:
        return jsonify({
            "error": "NovaDXB is still starting up. Please try again in a moment."
        }), 503

    try:
        data = request.get_json(silent=True)
        if not data or "message" not in data:
            return jsonify({
                "error": "Invalid request. Expected JSON with a 'message' field."
            }), 400

        raw_message = data.get("message", "")

        # ── Sanitize input ─────────────────
        try:
            message = sanitize_input(raw_message)
        except ValueError:
            return jsonify({
                "error": "Invalid message format."
            }), 400

        if not message:
            return jsonify({
                "error": "Message cannot be empty."
            }), 400

        # ── Prompt-injection guard ─────────
        if is_prompt_injection(message):
            return jsonify({
                "status":   "success",
                "response": (
                    "I'm NovaDXB, your Dubai travel concierge — I can only "
                    "help with Dubai tourism questions like itineraries, "
                    "areas to stay, dining, and budgets. How can I help "
                    "with your Dubai trip?"
                )
            })

        # ── Check cache before calling the agent ──
        cached = get_cached_response(message)
        if cached:
            return jsonify({
                "status":   "success",
                "response": cached,
                "cached":   True
            })

        # ── Agent response ────────────────
        response = query_agent(message)

        # ── Cache successful response ──────
        set_cached_response(message, response)

        return jsonify({
            "status":   "success",
            "response": response
        })

    except Exception as e:
        # Full details logged server-side only — never shown to the user
        logger.error(f"/chat error: {e}", exc_info=True)
        return jsonify({
            "error": "Something went wrong while processing your request. Please try again."
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Unhandled server error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"NovaDXB running on port {port}")
    serve(app, host="0.0.0.0", port=port)