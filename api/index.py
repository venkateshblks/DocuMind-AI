"""
DocuMind AI — Flask application entry point.

Routes:
  GET  /              -> rendered single-page UI
  GET  /api/health    -> service health check
  POST /api/upload    -> upload a PDF, index it, return a session_id
  POST /api/ask       -> stream an answer (SSE) for a question about the doc

This file is the Vercel serverless entry point (exported `app`).
Run locally with:  python api/index.py
"""

import json
import os
import sys
import traceback

# Make the project root importable so `lib.*` resolves on both Vercel
# and when running `python api/index.py` locally.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load .env for local development (no-op on Vercel where env vars are
# configured in the dashboard).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from lib.pdf import MAX_FILE_SIZE, extract_text_from_pdf
from lib.rag import index_document, query_document_stream

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=None,
)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Session-level API key storage (in-memory, lost on server restart)
_session_keys = {}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def _get_api_key(env_var: str, session_key: str) -> str | None:
    """Get API key from environment or session storage."""
    # Environment variable takes precedence
    env_val = os.getenv(env_var)
    if env_val:
        return env_val
    # Fall back to session storage and synchronize to environment
    val = _session_keys.get(session_key)
    if val:
        os.environ[env_var] = val
        return val
    return None


@app.route("/api/health")
def health():
    google_key = _get_api_key("GOOGLE_API_KEY", "google_key")
    groq_key = _get_api_key("GROQ_API_KEY", "groq_key")
    pinecone_key = _get_api_key("PINECONE_API_KEY", "pinecone_key")
    
    has_gemini = bool(google_key)
    has_groq = bool(groq_key)
    has_pinecone = bool(pinecone_key)
    configured = (has_gemini or has_groq) and has_pinecone
    
    return jsonify(
        {
            "status": "ok",
            "service": "DocuMind AI",
            "configured": configured,
            "models": {
                "gemini": has_gemini,
                "groq": has_groq,
            },
            "pinecone": has_pinecone,
        }
    )


@app.route("/api/settings", methods=["POST"])
def save_settings():
    """Save API keys to session storage."""
    try:
        data = request.get_json(silent=True) or {}
        
        if "googleApiKey" in data:
            _session_keys["google_key"] = data["googleApiKey"].strip()
        if "groqApiKey" in data:
            _session_keys["groq_key"] = data["groqApiKey"].strip()
        if "pineconeApiKey" in data:
            _session_keys["pinecone_key"] = data["pineconeApiKey"].strip()
        
        # Update environment for this session
        if _session_keys.get("google_key"):
            os.environ["GOOGLE_API_KEY"] = _session_keys["google_key"]
        if _session_keys.get("groq_key"):
            os.environ["GROQ_API_KEY"] = _session_keys["groq_key"]
        if _session_keys.get("pinecone_key"):
            os.environ["PINECONE_API_KEY"] = _session_keys["pinecone_key"]
        
        return jsonify({"status": "ok", "message": "Settings saved"})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        google_key = _get_api_key("GOOGLE_API_KEY", "google_key")
        pinecone_key = _get_api_key("PINECONE_API_KEY", "pinecone_key")
        if not google_key or not pinecone_key:
            return jsonify({
                "error": "Google API key and Pinecone API key are required to embed and index documents. "
                         "Please configure them in the Settings panel."
            }), 400

        if "file" not in request.files:
            return jsonify({"error": "No file provided. Use form field 'file'."}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400

        file_bytes = file.read()

        # Extract text
        text, page_count = extract_text_from_pdf(file_bytes)

        # Index into Pinecone
        session_id = index_document(text)

        return jsonify(
            {
                "session_id": session_id,
                "filename": file.filename,
                "page_count": page_count,
                "char_count": len(text),
                "chunk_count": (len(text) // 800) + 1,
            }
        )

    except ValueError as exc:
        # User-facing errors (bad PDF, empty, encrypted, etc.)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        question = (data.get("question") or "").strip()
        model = (data.get("model") or "gemini").strip().lower()

        pinecone_key = _get_api_key("PINECONE_API_KEY", "pinecone_key")
        if not pinecone_key:
            return jsonify({"error": "Pinecone API key is required. Please set it in Settings."}), 400

        if model == "groq":
            groq_key = _get_api_key("GROQ_API_KEY", "groq_key")
            if not groq_key:
                return jsonify({"error": "Groq API key is required to query with Groq. Please set it in Settings."}), 400
        else:
            google_key = _get_api_key("GOOGLE_API_KEY", "google_key")
            if not google_key:
                return jsonify({"error": "Google API key is required to query with Gemini. Please set it in Settings."}), 400

        if not session_id:
            return jsonify({"error": "Missing session_id."}), 400
        if not question:
            return jsonify({"error": "Missing question."}), 400
        if len(question) > 1000:
            return jsonify({"error": "Question is too long (max 1000 chars)."}), 400
        if model not in ["gemini", "groq"]:
            return jsonify({"error": "Invalid model. Use 'gemini' or 'groq'."}), 400

        def event_stream():
            try:
                for event in query_document_stream(session_id, question, model):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # disable proxy buffering
            },
        )

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
