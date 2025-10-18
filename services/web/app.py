# services/web/app.py
import os
import argparse
from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS

# ---- Flask app (must exist before using @app.route) ----
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# services/web/app.py
from rag_system8 import ThesisDataManager, ThesisRAGSystem

try:
    data_manager = ThesisDataManager()              # uses that class’s defaults
    rag_system = ThesisRAGSystem(data_manager)      # <-- pass it in
except Exception as e:
    rag_system = None
    INIT_ERROR = f"Failed to initialize ThesisRAGSystem: {e}"
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

    if rag_system is None:
        # Surface init error cleanly in the UI rather than crashing
        return render_template("index.html", error=globals().get("INIT_ERROR", "RAG system not initialized.")), 500

    answer = rag_system.answer_question(question)
    return render_template("index.html", user_question=question, answer=answer)

@app.get("/clear_conversation")
def clear_conversation():
    # If you track history in rag_system, clear it here (no-op is fine)
    try:
        if rag_system and hasattr(rag_system, "clear_history"):
            rag_system.clear_history()
    except Exception:
        pass
    return redirect(url_for("chat"))

@app.get("/health")
def health():
    status = "ok" if rag_system is not None else "degraded"
    return jsonify(status=status)

@app.get("/clear")
def clear():
    # if you maintain history in rag_system, clear it here
    try:
        if rag_system and hasattr(rag_system, "clear_history"):
            rag_system.clear_history()
    except Exception:
        pass
    return redirect(url_for("chat"))

# ---- Entrypoint (supports: python app.py --host 0.0.0.0 --port 5029) ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5029")))
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)

