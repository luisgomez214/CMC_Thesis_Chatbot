from flask import Flask, render_template, request, redirect, url_for, flash, session
import logging
from datetime import datetime
from rag_system8 import ThesisDataManager, ThesisRAGSystem, run_llm

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Configure logging
logging.basicConfig(level=logging.INFO)

data_manager = ThesisDataManager()
rag_system = ThesisRAGSystem(data_manager)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    question = request.form.get('question')
    if not question:
        flash("Please enter a question.", "error")
        return redirect(url_for('index'))
    
    # Retrieve conversation history from the session; initialize if not present.
    conversation = session.get('conversation', '')
    
    start_time = datetime.now()
    # Pass conversation history into the answer function.
    answer = rag_system.answer_question(question, conversation_history=conversation)
    end_time = datetime.now()
    
    # Update conversation history (simple concatenation example)
    conversation += f"User: {question}\nAssistant: {answer}\n"
    session['conversation'] = conversation
    
    response_time = (end_time - start_time).total_seconds()
    return render_template('results.html', answer=answer, response_time=response_time, question=question)

if __name__ == "__main__":
    # Preload the data before starting the server
    if data_manager.load_data():
        app.logger.info("Data preloaded successfully!")
        # Optional: Warm up the LLM with a dummy query
        dummy = run_llm("You are a helpful assistant.", "Hello")
        app.logger.info("LLM warmed up.")
    else:
        app.logger.error("Failed to preload data.")
    app.run(debug=True, port=5028)

