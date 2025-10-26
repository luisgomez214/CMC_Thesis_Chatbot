import sqlite3
import logging
from groq import Groq
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DB_PATH = "theses2.db"

def get_database_schema():
    """Dynamically retrieve the database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get table structure
    cursor.execute("PRAGMA table_info(theses_fts)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Get sample data to understand content
    cursor.execute(f"SELECT * FROM theses_fts LIMIT 3")
    sample_rows = cursor.fetchall()
    
    conn.close()
    
    schema_info = {
        "columns": columns,
        "sample_data": [dict(zip(columns, row)) for row in sample_rows]
    }
    
    logger.info(f"Database schema: {columns}")
    return schema_info

def extract_search_terms(question, schema_info):
    """Use LLM to extract optimal search terms."""
    prompt = f"""Extract search terms from the user's question for querying an academic thesis database.

User question: "{question}"

Rules:
- If the question is just a name or names, extract ONLY the name components
- DO NOT add generic words like "author", "advisor", "thesis"
- Extract specific keywords for topics
- Maximum 3-4 terms

Examples:
"ethan choi" -> ["Ethan", "Choi"]
"machine learning theses" -> ["machine", "learning"]
"economics department" -> ["economics", "department"]

Respond with ONLY a JSON array of terms:
["term1", "term2"]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        # Try to parse as JSON array
        if content.startswith('['):
            search_terms = json.loads(content)
        # Try to extract JSON array from response
        elif '[' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            search_terms = json.loads(content[start:end])
        else:
            raise json.JSONDecodeError("No array found", content, 0)
    except (json.JSONDecodeError, TypeError):
        # Fallback: extract meaningful words directly from question
        import re
        # Remove common question words
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were',
                     'the', 'a', 'an', 'about', 'find', 'show', 'me', 'tell', 'did', 'do',
                     'thesis', 'theses', 'wrote', 'written', 'author', 'advisor', 'by'}
        words = re.findall(r'\b[A-Za-z]+\b', question)
        search_terms = [w for w in words if w.lower() not in stop_words and len(w) > 1][:4]
    
    # Clean up terms
    search_terms = [term.strip() for term in search_terms if term and len(term.strip()) > 1]
    
    logger.info(f"Search terms extracted: {search_terms}")
    return search_terms

def determine_search_strategy(question, schema_info):
    """Let LLM decide which columns to search and how to prioritize results."""
    prompt = f"""Analyze this question and select the best database columns to display.

Question: "{question}"

Available columns: {schema_info['columns']}

Select 4-6 columns that would best answer this question.
- If asking about a person (name), include: author, advisor1, advisor2, advisor3, Title
- If asking about a topic, include: Title, keywords, abstract, disciplines
- If asking about a department, include: department, second_department, Title, author

Respond with ONLY valid JSON:
{{"primary_columns": ["col1", "col2", "col3"]}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        # Extract JSON
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            result = json.loads(json_str)
            columns = result.get("primary_columns", [])
        else:
            raise json.JSONDecodeError("No JSON found", content, 0)
    except json.JSONDecodeError:
        # Intelligent fallback based on question analysis
        question_lower = question.lower()
        if any(indicator in question_lower for indicator in ['advisor', 'advised', 'professor', 'taught']):
            columns = ['Title', 'author', 'advisor1', 'advisor2', 'advisor3', 'disciplines']
        elif any(indicator in question_lower for indicator in ['author', 'wrote', 'written by']):
            columns = ['Title', 'author', 'abstract', 'keywords', 'advisor1']
        elif any(indicator in question_lower for indicator in ['department', 'major', 'field']):
            columns = ['department', 'second_department', 'Title', 'author', 'disciplines']
        else:
            columns = ['Title', 'author', 'abstract', 'keywords', 'disciplines']
    
    # Validate columns exist in schema
    valid_columns = [col for col in columns if col in schema_info['columns']]
    if not valid_columns:
        valid_columns = ['Title', 'author', 'abstract', 'keywords']
    
    logger.info(f"Selected columns: {valid_columns}")
    return valid_columns

def search_database(search_terms, selected_columns, schema_info):
    """Search database with intelligent fallback strategies."""
    if not search_terms:
        return []
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ensure we're selecting valid columns
    valid_columns = [col for col in selected_columns if col in schema_info['columns']]
    if not valid_columns:
        valid_columns = schema_info['columns']
    
    col_str = ', '.join(valid_columns)
    
    # Build query strategies with increasing generality
    queries_to_try = []
    
    # Strategy 1: Exact phrase match (best for names)
    if len(search_terms) >= 2:
        full_phrase = " ".join(search_terms)
        queries_to_try.append(('phrase', f'"{full_phrase}"'))
    
    # Strategy 2: All terms must be present (AND logic using multiple MATCH)
    if len(search_terms) >= 2:
        and_query = " AND ".join(search_terms)
        queries_to_try.append(('and', and_query))
    
    # Strategy 3: Any term can match (OR logic)
    if len(search_terms) >= 1:
        or_query = " OR ".join(search_terms)
        queries_to_try.append(('or', or_query))
    
    # Strategy 4: Individual terms (one at a time)
    for term in search_terms:
        queries_to_try.append(('single', term))
    
    # Strategy 5: Prefix matching with wildcards
    for term in search_terms:
        if len(term) > 3:
            queries_to_try.append(('wildcard', f"{term}*"))
    
    rows = []
    successful_query = None
    strategy_used = None
    
    for strategy, fts_query in queries_to_try:
        try:
            # Clean query to avoid FTS syntax errors
            fts_query_clean = fts_query.replace("'", "")
            
            sql = f"SELECT {col_str} FROM theses_fts WHERE full_text MATCH ? LIMIT 20"
            logger.info(f"[{strategy}] Trying: {fts_query_clean}")
            cursor.execute(sql, (fts_query_clean,))
            current_rows = cursor.fetchall()
            
            if current_rows:
                # For phrase and AND queries, take results immediately (most precise)
                if strategy in ['phrase', 'and']:
                    rows = current_rows
                    successful_query = fts_query_clean
                    strategy_used = strategy
                    logger.info(f"✓ [{strategy}] Found {len(rows)} results - using these (precise match)")
                    break
                # For OR and other queries, keep searching if we haven't found phrase/AND results
                elif not rows:
                    rows = current_rows
                    successful_query = fts_query_clean
                    strategy_used = strategy
                    logger.info(f"✓ [{strategy}] Found {len(rows)} results")
                    # Continue to see if we can find more precise results
                    if strategy == 'or' and len(rows) > 10:
                        # Too many results with OR, keep trying for more precise matches
                        continue
                    elif strategy in ['single', 'wildcard']:
                        # Good enough, stop here
                        break
        except sqlite3.OperationalError as e:
            logger.warning(f"✗ [{strategy}] Query failed '{fts_query_clean}': {e}")
            continue
    
    conn.close()
    
    if not rows:
        logger.warning("No results found with any query strategy")
    else:
        logger.info(f"Final result: {len(rows)} rows using '{strategy_used}' strategy")
    
    return rows, valid_columns

def generate_answer(question, rows, columns):
    """Use LLM to generate a natural, helpful answer."""
    if not rows:
        return f"""I couldn't find any theses matching "{question}" in the database.

Try:
• Using a different spelling or format
• Searching for a research topic or keyword
• Asking about a department or field"""
    
    # Format results for LLM - be smart about what to include
    results_formatted = []
    for row in rows:
        entry = {}
        for i, col in enumerate(columns):
            if i < len(row) and row[i]:
                value = str(row[i]).strip()
                # Include full text for important fields
                if col in ['author', 'Title', 'advisor1', 'advisor2', 'advisor3', 'department']:
                    entry[col] = value
                # Truncate long fields
                elif col == 'abstract':
                    entry[col] = value[:400] + "..." if len(value) > 400 else value
                else:
                    entry[col] = value[:200] + "..." if len(value) > 200 else value
        
        # Only include entries that have meaningful data
        if entry:
            results_formatted.append(entry)
    
    # Limit to most relevant results to avoid token overflow
    if len(results_formatted) > 8:
        results_formatted = results_formatted[:8]
    
    prompt = f"""You are answering questions about academic theses.

Question: "{question}"

Database results ({len(results_formatted)} theses found):
{json.dumps(results_formatted, indent=2)}

Instructions:
- Answer the specific question directly
- If asked about a person, show what they wrote OR what they advised
- List relevant thesis titles and authors
- Be specific - use actual names and titles from the data
- If multiple results, summarize the key findings
- Be concise and well-formatted

Answer:"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.3
    )
    
    answer = response.choices[0].message.content.strip()
    
    # Post-process: if answer says "no thesis" but we have results, regenerate
    if ('no thesis' in answer.lower() or 'not found' in answer.lower() or "couldn't find" in answer.lower()) and len(results_formatted) > 0:
        logger.warning("LLM incorrectly said no results, forcing regeneration")
        
        # Create a simpler, more direct prompt
        simple_results = "\n\n".join([
            f"Title: {r.get('Title', 'N/A')}\nAuthor: {r.get('author', 'N/A')}\nAdvisor: {r.get('advisor1', 'N/A')}"
            for r in results_formatted[:5]
        ])
        
        prompt2 = f"""List the theses found for: "{question}"

{simple_results}

Format as a clear list with thesis titles and authors."""
        
        response2 = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt2}],
            max_tokens=500,
            temperature=0.1
        )
        answer = response2.choices[0].message.content.strip()
    
    return answer

def main():
    print("🎓 Welcome to the CMC Thesis Chatbot!")
    print("Ask me anything about theses, authors, topics, or departments.\n")
    
    # Get database schema once at startup
    schema_info = get_database_schema()
    
    while True:
        try:
            question = input("\n💬 Your question (or 'exit' to quit): ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thank you for using the CMC Thesis Chatbot!")
                break
            
            print("\n🔍 Searching...")
            
            # Let LLM decide everything
            search_terms = extract_search_terms(question, schema_info)
            selected_columns = determine_search_strategy(question, schema_info)
            rows, actual_columns = search_database(search_terms, selected_columns, schema_info)
            
            logger.info(f"Found {len(rows)} results")
            
            print("\n📚 Generating answer...\n")
            answer = generate_answer(question, rows, actual_columns)
            
            print("="*70)
            print(answer)
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using the CMC Thesis Chatbot!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print("\n❌ Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    main()
