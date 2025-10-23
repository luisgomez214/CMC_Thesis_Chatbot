import sqlite3
import re
import logging
from groq import Groq
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Fetch actual columns from the database
def get_table_columns():
    conn = sqlite3.connect("theses2.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(theses_fts);")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()
    return columns

# Extract clean keywords from user input
def extract_keywords(question):
    prompt = f"""
Extract the main concepts from this question as a Python list of lowercase strings.
Question: "{question}"
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You extract minimal search keywords for academic databases."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    raw_output = resp.choices[0].message.content.strip()
    keywords = re.findall(r"'(.*?)'|\"(.*?)\"", raw_output)
    return [k[0] or k[1] for k in keywords] if keywords else [question]

# Expand keywords using synonyms
def expand_keywords_with_synonyms(keywords):
    prompt = f"""
Generate 3-5 synonyms or variations for each keyword for academic search.
Return only a flat Python list.
Keywords: {keywords}
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Generate synonyms relevant to academic research and theses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    raw_output = resp.choices[0].message.content.strip()
    found = re.findall(r"'(.*?)'|\"(.*?)\"", raw_output)
    expanded = list(set([f[0] or f[1] for f in found]))
    return expanded if expanded else keywords

# Choose which columns to search: validate LLM suggestion
def choose_columns(question, all_columns):
    prompt = f"""
Given these columns: {all_columns}
and the user question: "{question}",
select the columns most relevant to search in the database.
Return only a Python list of valid column names.
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Select relevant database columns for academic search."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    raw_output = resp.choices[0].message.content.strip()
    cols = re.findall(r"'(.*?)'|\"(.*?)\"", raw_output)
    suggested = [c[0] or c[1] for c in cols]
    # Keep only real columns
    valid_columns = [col for col in suggested if col in all_columns]
    return valid_columns if valid_columns else all_columns

# Search the database using FTS5
def search_theses(question):
    conn = sqlite3.connect("theses2.db")
    cursor = conn.cursor()

    keywords = extract_keywords(question)
    expanded_keywords = expand_keywords_with_synonyms(keywords)
    logging.info(f"Expanded keywords: {expanded_keywords}")

    all_columns = get_table_columns()
    columns = choose_columns(question, all_columns)
    logging.info(f"Columns chosen for search: {columns}")

    safe_columns = [f'"{col}"' for col in columns]
    where_clause = " OR ".join([f"{col} MATCH ?" for col in safe_columns])
    params = [" OR ".join(expanded_keywords)] * len(safe_columns)

    try:
        query = f"SELECT * FROM theses_fts WHERE {where_clause} LIMIT 5"
        cursor.execute(query, params)
        results = cursor.fetchall()
    except Exception as e:
        logging.error(f"Search failed: {e}")
        results = []

    conn.close()
    return columns, results

# Generate a human-readable summary using only retrieved data
def generate_response(columns, results):
    if not results:
        return "No results found."

    formatted = []
    for i, row in enumerate(results, 1):
        # Convert row to dict for LLM summarization
        row_dict = {col: row[idx] if idx < len(row) else "" for idx, col in enumerate(columns)}

        prompt = f"""
You are an assistant that summarizes a thesis record.
Given this dictionary: {row_dict}
Return a short, factual summary including:
- Title
- Authors
- Advisors
- URL
Do NOT hallucinate. Use only the provided data.
"""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Generate a concise, factual thesis summary from database data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        summary = resp.choices[0].message.content.strip()
        formatted.append(f"{i}. {summary}")

    return "\n\n".join(formatted)

def main():
    print("🎓 Welcome to the CMC Thesis Chatbot!\n")
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        columns, results = search_theses(question)
        answer = generate_response(columns, results)
        print("\n--- Answer ---")
        print(answer)
        print()

if __name__ == "__main__":
    main()

