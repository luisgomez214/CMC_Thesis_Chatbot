from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_session import Session
import logging
from datetime import datetime
import json

from rag_system8 import ThesisDataManager, ThesisRAGSystem, run_llm

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

data_manager = ThesisDataManager()
rag_system = ThesisRAGSystem(data_manager)

@app.route('/', methods=['GET', 'POST'])
def chat():
    # Initialize conversation in session if it doesn't exist
    if 'conversation' not in session:
        session['conversation'] = []

    if request.method == 'POST':
        question = request.form.get('question')
        conversation = session.get('conversation', [])
        
        # Process the question with your RAG system (ensure it returns an answer string)
        answer = rag_system.answer_question(question)
        
        conversation.append({'role': 'user', 'content': question})
        conversation.append({'role': 'assistant', 'content': answer})
        session['conversation'] = conversation
        
        # Append the anchor so that after a new question it scrolls to the bottom
        return redirect(url_for('chat') + "#conversation-end")
    else:
        conversation = session.get('conversation', [])
        return render_template('chat.html', conversation=conversation)

@app.route('/clear_conversation')
def clear_conversation():
    session['conversation'] = []  # Clear the conversation history
    return redirect(url_for('chat'))  # Redirect to the chat (home) route

if __name__ == "__main__":
    if data_manager.load_data():
        app.logger.info("Data preloaded successfully!")
        dummy = run_llm("You are a helpful assistant.", "Hello")
        app.logger.info("LLM warmed up.")
    else:
        app.logger.error("Failed to preload data.")
    app.run(debug=True, port=5029, host='0.0.0.0')

