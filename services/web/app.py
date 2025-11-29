# services/web/app.py
import os
import argparse
from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS

# ---- Flask app (must exist before using @app.route) ----
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ✅ NEW BACKEND IMPORT (rag_system10.py)
from rag_system10 import search_database, generate_answer

# ---- Initialize RAG system safely (optional, left for future if you add history hooks) ----
try:
    rag_system_ready = True
except Exception as e:
    rag_system_ready = False
    INIT_ERROR = f"Failed to initialize RAG system: {e}"

# ---- Routes ----

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("index.html")

    # Accept JSON or HTML form and guard against empty input
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or request.form.get("question") or "").strip()
    if not question:
        return render_template("index.html", error="Please enter a question."), 400

    if not rag_system_ready:
        # Surface init error cleanly in the UI rather than crashing
        return render_template("index.html", error=globals().get("INIT_ERROR", "RAG system not initialized.")), 500

    # ✅ Call new RAG system
    intent, rows, parsed = search_database(question)
    answer = generate_answer(question, intent, rows, parsed)

    # Return to frontend UI
    return render_template("index.html", user_question=question, answer=answer)

@app.get("/clear_conversation")
def clear_conversation():
    # No history stored? That's fine, no-op
    return redirect(url_for("chat"))

@app.get("/health")
def health():
    status = "ok" if rag_system_ready else "degraded"
    return jsonify(status=status)

@app.get("/clear")
def clear():
    # Just reload UI
    return redirect(url_for("chat"))

# ---- Entrypoint (supports: python app.py --host 0.0.0.0 --port 5029) ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5029")))
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)

