import sqlite3
import logging
from groq import Groq
import json
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DB_PATH = "theses2.db"

def detect_query_type(question):
    """Detect what the user is asking for using simple patterns."""
    q = question.lower()
    
    # Check for content questions first
    if any(word in q for word in ['what is', 'about', 'explain', 'summarize', 'tell me about']):
        return 'content'
    
    # Check for advisor queries
    if 'advis' in q or 'supervised' in q:
        match = re.search(r'(?:advis(?:ed|or)|supervised?)\s+(?:by\s+)?(.+?)$', q)
        if match:
            return ('advisor', match.group(1).strip())
        return 'advisor'
    
    # Check for author queries
    if 'thesis by' in q or 'paper by' in q or 'author' in q:
        match = re.search(r'(?:by|author)\s+(.+?)$', q)
        if match:
            return ('author', match.group(1).strip())
        return 'author'
    
    # Check for award queries
    if any(word in q for word in ['best', 'award', 'won', 'prize']):
        return 'award'
    
    # Check for year
    year_match = re.search(r'\b(19|20)\d{2}\b', question)
    if year_match:
        return ('year', year_match.group(0))
    
    # Check if it's just a person name (2-3 words)
    words = question.strip().split()
    if 1 <= len(words) <= 3 and all(w[0].isupper() or w.islower() for w in words):
        return ('person', question.strip())
    
    # Default: topic search
    return 'topic'

def build_sql_query(question, query_type):
    """Build SQL query based on detected type."""
    
    # Handle tuple types (with extracted values)
    if isinstance(query_type, tuple):
        qtype, value = query_type
    else:
        qtype = query_type
        value = None
    
    if qtype == 'advisor' and value:
        # Search advisor columns
        sql = """
            SELECT * FROM theses 
            WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?
            LIMIT 50
        """
        params = [f'%{value}%'] * 3
        return sql, params
    
    elif qtype == 'author' and value:
        # Search author column
        sql = "SELECT * FROM theses WHERE author LIKE ? LIMIT 50"
        params = [f'%{value}%']
        return sql, params
    
    elif qtype == 'person' and value:
        # Could be author or advisor - search both
        sql = """
            SELECT * FROM theses 
            WHERE author LIKE ? OR advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?
            LIMIT 50
        """
        params = [f'%{value}%'] * 4
        return sql, params
    
    elif qtype == 'content':
        # Extract key words from title
        words = [w for w in question.lower().split() if len(w) > 3 and w not in 
                 ['what', 'about', 'thesis', 'tell', 'explain']]
        if words:
            search_term = ' '.join(words[:5])
            sql = "SELECT * FROM theses WHERE Title LIKE ? LIMIT 10"
            params = [f'%{search_term}%']
            return sql, params
    
    elif qtype == 'award':
        sql = "SELECT * FROM theses WHERE award IS NOT NULL AND award != '' LIMIT 50"
        params = []
        return sql, params
    
    elif qtype == 'year' and value:
        sql = 'SELECT * FROM theses WHERE "First published" LIKE ? LIMIT 50'
        params = [f'%{value}%']
        return sql, params
    
    # Default: topic/keyword search
    words = [w for w in question.split() if len(w) > 2]
    if words:
        conditions = []
        params = []
        for word in words[:3]:  # Use first 3 keywords
            conditions.append(
                "(Title LIKE ? OR keywords LIKE ? OR abstract LIKE ? OR disciplines LIKE ?)"
            )
            params.extend([f'%{word}%'] * 4)
        
        sql = f"SELECT * FROM theses WHERE {' OR '.join(conditions)} LIMIT 50"
        return sql, params
    
    return None, None

def search_database(question):
    """Search the database based on the question."""
    logger.info(f"\n[SEARCH] {question}")
    
    # Step 1: Detect what user wants
    query_type = detect_query_type(question)
    logger.info(f"[TYPE] {query_type}")
    
    # Step 2: Build SQL query
    sql, params = build_sql_query(question, query_type)
    
    if not sql:
        logger.warning("Could not build query")
        return []
    
    # Step 3: Execute query
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA case_sensitive_like = OFF")
        cursor = conn.cursor()
        
        logger.info(f"[SQL] {sql[:100]}...")
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        logger.info(f"[RESULTS] Found {len(rows)} theses")
        return rows
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []

def format_results(rows):
    """Convert database rows to dictionaries."""
    results = []
    for row in rows:
        if row:
            result = {
                'Title': row[0],
                'Author': row[13] if len(row) > 13 else None,
                'Advisor': row[6] if len(row) > 6 else None,
                'Department': row[4] if len(row) > 4 else None,
                'Abstract': row[12] if len(row) > 12 else None,
                'Keywords': row[10] if len(row) > 10 else None,
                'Year': row[17] if len(row) > 17 else None,
                'Award': row[3] if len(row) > 3 else None,
            }
            # Remove empty values
            result = {k: v for k, v in result.items() if v and str(v).strip()}
            if result:
                results.append(result)
    return results

def generate_answer(question, results):
    """Use LLM to generate natural language answer."""
    
    if not results:
        return "I couldn't find any theses matching your query. Try different keywords or names."
    
    # Detect if it's a content question (asking what a thesis is about)
    is_content_question = any(word in question.lower() for word in 
                              ['what is', 'about', 'explain', 'summarize', 'tell me about'])
    
    # Detect if it's an award question
    is_award_question = any(word in question.lower() for word in ['best', 'award', 'won', 'prize'])
    
    if is_content_question and len(results) == 1:
        # Single thesis - provide detailed summary
        prompt = f"""Answer this question about a thesis:

Question: {question}

Thesis Information:
{json.dumps(results[0], indent=2)}

Provide a clear answer focusing on:
- What the research is about
- Main findings (if available in abstract)
- Research approach

Be concise and factual."""
        
    elif any(word in question.lower() for word in ['how many', 'count', 'list all', 'which']):
        # Aggregation question
        prompt = f"""Answer this question using the thesis data:

Question: {question}

Data ({len(results)} theses):
{json.dumps(results[:20], indent=2)}

Provide statistics and answer the question directly."""
    
    elif is_award_question:
        # Award question - emphasize awards and dates
        prompt = f"""List these award-winning theses:

Question: {question}

Theses ({len(results)} found):
{json.dumps(results[:10], indent=2)}

Format as a numbered list. For EACH thesis include:
- Title
- Author
- Advisor
- Award (ONLY if the "Award" field exists and is not empty - otherwise skip this line completely)
- Year published (if present)

IMPORTANT: Do not show "Award: None" or "Award: N/A". Only show the Award line if there is an actual award name.

Example format:
1. **Title**
   - Author: Name
   - Advisor: Name
   - Award: Actual Award Name (only if exists)
   - Published: Year"""
        
    else:
        # List theses
        prompt = f"""List these theses clearly:

Question: {question}

Theses:
{json.dumps(results[:8], indent=2)}

Format as a numbered list with titles, authors, advisors, and publication year if available."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Fallback: simple list
        output = f"Found {len(results)} theses:\n\n"
        for i, r in enumerate(results[:8], 1):
            output += f"{i}. {r['Title']}\n"
            if 'Author' in r:
                output += f"   Author: {r['Author']}\n"
            if 'Advisor' in r:
                output += f"   Advisor: {r['Advisor']}\n"
            # Only show Award if it exists and is not empty
            if 'Award' in r and r['Award'] and str(r['Award']).strip():
                output += f"   Award: {r['Award']}\n"
            if 'Year' in r:
                output += f"   Published: {r['Year']}\n"
            output += "\n"
        return output

def main():
    print("🎓 Thesis Search System")
    print("Ask questions about theses by author, advisor, topic, or year\n")
    
    while True:
        try:
            question = input("💬 Question (or 'exit'): ").strip()
            
            if not question or question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            print("\n🔍 Searching...")
            
            # Search database
            rows = search_database(question)
            
            # Format results
            results = format_results(rows)
            
            # Generate answer
            print("\n" + "="*70)
            answer = generate_answer(question, results)
            print(answer)
            print("="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print("❌ Something went wrong. Try again.\n")

if __name__ == "__main__":
    main()
