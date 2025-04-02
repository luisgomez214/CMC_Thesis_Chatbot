from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_session import Session
import logging
from datetime import datetime
import json

from rag_system8 import ThesisDataManager, ThesisRAGSystem, run_llm

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with strong secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

data_manager = ThesisDataManager()
rag_system = ThesisRAGSystem(data_manager)

@app.route('/')
def index():
    session.pop('conversation', None)  # clear conversation on homepage
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    question = request.form['question']

    if 'conversation' not in session:
        session['conversation'] = []

    session['conversation'].append({"role": "user", "content": question})
    conversation_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in session['conversation']]
    )

    answer = rag_system.answer_question(question, conversation_history=conversation_history)

    session['conversation'].append({"role": "assistant", "content": answer})

    return render_template('results.html', question=question, answer=answer, conversation=session['conversation'])


if __name__ == "__main__":
    if data_manager.load_data():
        app.logger.info("Data preloaded successfully!")
        dummy = run_llm("You are a helpful assistant.", "Hello")
        app.logger.info("LLM warmed up.")
    else:
        app.logger.error("Failed to preload data.")
    app.run(debug=True, port=5015, host='0.0.0.0')
    

