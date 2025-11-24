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

# Available columns in the database
AVAILABLE_COLUMNS = [
    'Title', 'publication_date', 'season', 'award', 'department', 'second_department',
    'advisor1', 'advisor2', 'advisor3', 'embargo_date', 'keywords', 'disciplines',
    'abstract', 'author', 'author1_institution', 'author 2', 'URL', 
    'First published', 'State', 'Total'
]

def extract_key_terms(question):
    """Extract key terms from question, keeping titles and names together."""
    prompt = f"""Extract key terms from this question about academic theses.

Question: "{question}"

Rules:
- If the question asks "what is [TITLE] about", extract the ENTIRE title as ONE term
- Keep person names together as ONE term (e.g., "Mike Izbicki" not "Mike" and "Izbicki")
- Keep thesis titles together as ONE complete term, including colons and subtitles
- Keep multi-word topics together (e.g., "machine learning", "computer science")
- Extract years as separate terms
- Remove filler words (show, find, tell, about, from, by, what, is, the)
- NEVER include generic academic terms like: thesis, theses, paper, research, study
- NEVER invent or add names/terms that are not in the question
- NEVER split titles into multiple parts

Examples:
"theses by Mike Izbicki about machine learning from 2020"
→ ["Mike Izbicki", "machine learning", "2020"]

"what is The Value of Youth in Major League Baseball about"
→ ["The Value of Youth in Major League Baseball"]

"what is Case Study: Josh Hamilton - Finding a Long-Term Match at the Right Price about"
→ ["Case Study: Josh Hamilton - Finding a Long-Term Match at the Right Price"]

"thesis about texas"
→ ["texas"]

CRITICAL: When you see "what is [something] about", the [something] is the COMPLETE thesis title. Extract it as ONE single term.

Respond with ONLY a JSON array:
["term1", "term2"]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        if '[' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            key_terms = json.loads(content[start:end])
        else:
            raise json.JSONDecodeError("No array found", content, 0)
            
    except Exception as e:
        logger.warning(f"Key term extraction failed: {e}, using fallback")
        words = re.findall(r'\b[A-Za-z0-9]+\b', question)
        stop_words = {'show', 'tell', 'find', 'me', 'please', 'by', 'about', 'from', 'in', 'the', 'a', 'an', 'thesis', 'theses'}
        key_terms = [w for w in words if w.lower() not in stop_words][:5]
    
    # Filter out generic terms even from LLM results
    generic_terms = {'thesis', 'theses', 'paper', 'papers', 'research', 'study', 'studies'}
    key_terms = [term for term in key_terms if term.lower() not in generic_terms]
    
    logger.info(f"Extracted key terms: {key_terms}")
    return key_terms

def map_terms_to_columns(question, key_terms):
    """Map key terms to specific database columns."""
    prompt = f"""Given a question and extracted key terms, map each term to the appropriate database columns.

Question: "{question}"
Key terms: {key_terms}

Available columns:
- Title: thesis title (use for long multi-word phrases that look like titles)
- author: author name
- author 2: second author name
- advisor1, advisor2, advisor3: thesis advisors
- keywords: research keywords
- abstract: thesis abstract/description
- disciplines: academic disciplines/fields
- department, second_department: academic departments
- publication_date: publication date (format: "1/1/25 0:00")
- First published: first publication date (format: "11/24/10")
- season: academic season (Fall, Spring)
- award: awards received

Instructions:
1. If the question asks "what is [something] about", map [something] to Title column only
2. If a term is a long phrase with colons, dashes, or looks like a complete title, map it to Title
3. Person names (2-3 words, capitalized, common first/last names) → author, advisor columns
4. Single words or short topics → keywords, abstract, disciplines
5. Years → First published
6. NEVER map thesis titles to person columns (author, advisor)
7. NEVER map person names to Title

Examples:
"Case Study: Josh Hamilton - Finding a Long-Term Match at the Right Price" → Title (it's a complete thesis title)
"Mike Izbicki" → author, advisor1, advisor2, advisor3 (it's a person name)
"machine learning" → keywords, abstract, disciplines (it's a topic)

Return ONLY valid JSON:
{{
  "column_mappings": {{
    "Title": ["full title here if applicable"],
    "author": ["person name if applicable"],
    "keywords": ["topic if applicable"]
  }},
  "primary_focus": "title or person or topic"
}}

Only include columns that are relevant."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            mapping = json.loads(content[json_start:json_end])
        else:
            raise json.JSONDecodeError("No JSON found", content, 0)
            
    except Exception as e:
        logger.warning(f"Column mapping failed: {e}, using basic fallback")
        # Basic fallback: detect years and names
        mapping = {"column_mappings": {}, "primary_focus": "topic"}
        
        for term in key_terms:
            # Detect years
            if term.isdigit() and len(term) == 4:
                mapping["column_mappings"]["First published"] = [term]
            # Detect likely names (2+ capitalized words)
            elif len(term.split()) >= 2 and term[0].isupper():
                mapping["column_mappings"]["author"] = [term]
                mapping["column_mappings"]["advisor1"] = [term]
                mapping["column_mappings"]["advisor2"] = [term]
                mapping["primary_focus"] = "person"
            # Everything else as topic
            else:
                if "keywords" not in mapping["column_mappings"]:
                    mapping["column_mappings"]["keywords"] = []
                if "abstract" not in mapping["column_mappings"]:
                    mapping["column_mappings"]["abstract"] = []
                mapping["column_mappings"]["keywords"].append(term)
                mapping["column_mappings"]["abstract"].append(term)
    
    logger.info(f"Column mappings: {mapping}")
    return mapping

def analyze_column_types():
    """Dynamically analyze what type of data each column contains."""
    prompt = f"""Analyze these database columns and categorize them by data type.

Columns: {AVAILABLE_COLUMNS}

Categorize each column into ONE of these types:
- "person": Contains people's names (authors, advisors)
- "topic": Contains research content (keywords, abstract, disciplines, titles)
- "date": Contains dates or time information
- "metadata": Contains other information (awards, departments, URLs, etc.)

Return ONLY valid JSON:
{{
  "person": ["column1", "column2"],
  "topic": ["column3", "column4"],
  "date": ["column5"],
  "metadata": ["column6", "column7"]
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            column_types = json.loads(content[json_start:json_end])
        else:
            raise json.JSONDecodeError("No JSON found", content, 0)
            
    except Exception as e:
        logger.warning(f"Column type analysis failed: {e}, using minimal fallback")
        # Minimal fallback - only detect obvious patterns
        column_types = {"person": [], "topic": [], "date": [], "metadata": []}
        for col in AVAILABLE_COLUMNS:
            col_lower = col.lower()
            if 'author' in col_lower or 'advisor' in col_lower:
                column_types["person"].append(col)
            elif 'date' in col_lower or 'published' in col_lower or 'season' in col_lower:
                column_types["date"].append(col)
            elif 'keyword' in col_lower or 'abstract' in col_lower or 'discipline' in col_lower or 'title' in col_lower:
                column_types["topic"].append(col)
            else:
                column_types["metadata"].append(col)
    
    logger.info(f"Column types: {column_types}")
    return column_types

def build_sql_queries(column_mappings):
    """Build targeted SQL queries dynamically based on column mappings."""
    queries = []
    mappings = column_mappings.get("column_mappings", {})
    
    if not mappings:
        return queries
    
    # Dynamically categorize the mapped columns
    column_types = analyze_column_types()
    
    # Group the mappings by type
    person_mappings = []
    topic_mappings = []
    date_mappings = []
    title_mappings = []
    other_mappings = []
    
    for col, values in mappings.items():
        if not values:
            continue
        
        # Check for Title column first
        if col == "Title":
            title_mappings.append((col, values))
        # Determine which category this column belongs to
        elif col in column_types.get("person", []):
            person_mappings.append((col, values))
        elif col in column_types.get("topic", []):
            topic_mappings.append((col, values))
        elif col in column_types.get("date", []):
            date_mappings.append((col, values))
        else:
            other_mappings.append((col, values))
    
    # Dynamically build query strategies based on what we have
    prompt = f"""Given these mapped search criteria, determine the best SQL query strategies.

Title columns with values: {title_mappings if title_mappings else "None"}
Person columns with values: {person_mappings if person_mappings else "None"}
Topic columns with values: {topic_mappings if topic_mappings else "None"}  
Date columns with values: {date_mappings if date_mappings else "None"}
Other columns with values: {other_mappings if other_mappings else "None"}

Generate search strategies:
1. If we have title: Search by Title first (highest priority)
2. If we have person columns: Try each person column separately with topics and dates
3. If we have topic columns but no person: Combine topics with OR, add dates
4. If we have only dates: Query by date alone

Return ONLY valid JSON array of strategies:
[
  {{
    "description": "Search by title",
    "columns_to_use": ["Title"],
    "logic": "Title match"
  }}
]

Each strategy should specify which columns to include in the WHERE clause."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        if '[' in content:
            start = content.find('[')
            end = content.rfind(']') + 1
            strategies = json.loads(content[start:end])
        else:
            raise json.JSONDecodeError("No array found", content, 0)
            
        # Build SQL from strategies
        for strategy in strategies:
            cols_to_use = strategy.get("columns_to_use", [])
            where_parts = []
            params = []
            
            # Build WHERE clause for each column type
            for col in cols_to_use:
                # Find values for this column
                col_values = None
                for mapped_col, values in mappings.items():
                    if mapped_col == col:
                        col_values = values
                        break
                
                if not col_values:
                    continue
                
                # Quote column name if it contains spaces
                col_quoted = f'"{col}"' if ' ' in col else col
                
                # Handle date columns specially
                if col in column_types.get("date", []):
                    for val in col_values:
                        # If it's a 4-digit year, match last 2 digits in date
                        if val.isdigit() and len(val) == 4:
                            year_2digit = val[-2:]
                            where_parts.append(f"{col_quoted} LIKE ?")
                            params.append(f"%/{year_2digit}")
                        else:
                            where_parts.append(f"{col_quoted} LIKE ?")
                            params.append(f"%{val}%")
                # Handle topic columns - use OR for multiple values
                elif col in column_types.get("topic", []):
                    topic_parts = []
                    for val in col_values:
                        topic_parts.append(f"{col_quoted} LIKE ?")
                        params.append(f"%{val}%")
                    if len(topic_parts) > 1:
                        where_parts.append(f"({' OR '.join(topic_parts)})")
                    elif topic_parts:
                        where_parts.append(topic_parts[0])
                # Handle person and other columns normally
                else:
                    for val in col_values:
                        where_parts.append(f"{col_quoted} LIKE ?")
                        params.append(f"%{val}%")
            
            if where_parts:
                sql = f"SELECT * FROM theses WHERE {' AND '.join(where_parts)} LIMIT 20"
                queries.append((sql, params, strategy.get("description", "strategy")))
                
    except Exception as e:
        logger.warning(f"Dynamic query building failed: {e}, using basic fallback")
        
        # Basic fallback: build simple queries from available mappings
        # Strategy 0: Title search (highest priority)
        for col, values in title_mappings:
            col_quoted = f'"{col}"' if ' ' in col else col
            where_parts = [f"{col_quoted} LIKE ?"]
            params = [f"%{values[0]}%"]
            sql = f"SELECT * FROM theses WHERE {' AND '.join(where_parts)} LIMIT 20"
            queries.append((sql, params, "title_search"))
        
        # Strategy 1: If we have person mappings, try person-only first, then person+topics
        for col, values in person_mappings:
            col_quoted = f'"{col}"' if ' ' in col else col
            
            # First try: Person only (most specific)
            where_parts = [f"{col_quoted} LIKE ?"]
            params = [f"%{values[0]}%"]
            sql = f"SELECT * FROM theses WHERE {' AND '.join(where_parts)} LIMIT 20"
            queries.append((sql, params, f"person_only_{col}"))
            
            # Second try: Person AND topics (if topics exist)
            if topic_mappings or date_mappings:
                where_parts = [f"{col_quoted} LIKE ?"]
                params = [f"%{values[0]}%"]
                
                # Add topic conditions with OR
                if topic_mappings:
                    topic_parts = []
                    for topic_col, topic_vals in topic_mappings:
                        topic_col_quoted = f'"{topic_col}"' if ' ' in topic_col else topic_col
                        for val in topic_vals:
                            topic_parts.append(f"{topic_col_quoted} LIKE ?")
                            params.append(f"%{val}%")
                    if topic_parts:
                        where_parts.append(f"({' OR '.join(topic_parts)})")
                
                # Add date conditions
                for date_col, date_vals in date_mappings:
                    date_col_quoted = f'"{date_col}"' if ' ' in date_col else date_col
                    for val in date_vals:
                        if val.isdigit() and len(val) == 4:
                            year_2digit = val[-2:]
                            where_parts.append(f"{date_col_quoted} LIKE ?")
                            params.append(f"%/{year_2digit}")
                
                sql = f"SELECT * FROM theses WHERE {' AND '.join(where_parts)} LIMIT 20"
                queries.append((sql, params, f"person_with_filters_{col}"))
        
        # Strategy 2: Topic-only if no person mappings
        if not person_mappings and topic_mappings:
            where_parts = []
            params = []
            
            topic_parts = []
            for topic_col, topic_vals in topic_mappings:
                topic_col_quoted = f'"{topic_col}"' if ' ' in topic_col else topic_col
                for val in topic_vals:
                    topic_parts.append(f"{topic_col_quoted} LIKE ?")
                    params.append(f"%{val}%")
            
            if topic_parts:
                where_parts.append(f"({' OR '.join(topic_parts)})")
            
            for date_col, date_vals in date_mappings:
                date_col_quoted = f'"{date_col}"' if ' ' in date_col else date_col
                for val in date_vals:
                    if val.isdigit() and len(val) == 4:
                        year_2digit = val[-2:]
                        where_parts.append(f"{date_col_quoted} LIKE ?")
                        params.append(f"%/{year_2digit}")
            
            if where_parts:
                sql = f"SELECT * FROM theses WHERE {' AND '.join(where_parts)} LIMIT 20"
                queries.append((sql, params, "fallback_topics"))
    
    logger.info(f"Built {len(queries)} SQL queries")
    return queries

def execute_queries(queries):
    """Execute SQL queries and return results."""
    conn = sqlite3.connect(DB_PATH)
    # Enable case-insensitive LIKE for better name matching
    conn.execute("PRAGMA case_sensitive_like = OFF")
    cursor = conn.cursor()
    
    all_results = []
    successful_strategy = None
    
    for sql, params, strategy in queries:
        try:
            logger.info(f"[{strategy}] Executing SQL:")
            logger.info(f"  Query: {sql}")
            logger.info(f"  Params: {params}")
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            if rows:
                logger.info(f"✓ [{strategy}] Found {len(rows)} results")
                all_results.extend(rows)
                if not successful_strategy:
                    successful_strategy = strategy
                
                # If we found results with a precise query, stop
                if 'person' in strategy or 'advisor' in strategy:
                    break
            else:
                logger.info(f"✗ [{strategy}] No results found")
                    
        except sqlite3.OperationalError as e:
            logger.error(f"✗ [{strategy}] Query failed with error: {e}")
            logger.error(f"  SQL was: {sql}")
            logger.error(f"  Params were: {params}")
            continue
    

    
    conn.close()
    
    # Remove duplicates (same Title)
    seen_titles = set()
    unique_results = []
    for row in all_results:
        title = row[0] if row else None
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(row)
    
    if successful_strategy:
        logger.info(f"FINAL: {len(unique_results)} unique results using '{successful_strategy}' strategy")
    else:
        logger.warning(f"FINAL: No results found with any strategy")
    
    return unique_results

def format_results(rows):
    """Format database rows into readable output."""
    if not rows:
        return []
    
    formatted = []
    for row in rows:
        result = {
            'Title': row[0] if len(row) > 0 else None,
            'publication_date': row[1] if len(row) > 1 else None,
            'award': row[3] if len(row) > 3 else None,
            'department': row[4] if len(row) > 4 else None,
            'advisor1': row[6] if len(row) > 6 else None,
            'keywords': row[10] if len(row) > 10 else None,
            'disciplines': row[11] if len(row) > 11 else None,
            'abstract': row[12] if len(row) > 12 else None,
            'author': row[13] if len(row) > 13 else None,
            'First published': row[17] if len(row) > 17 else None,
        }
        
        # Remove None values
        result = {k: v for k, v in result.items() if v is not None and str(v).strip()}
        
        if result:
            formatted.append(result)
    
    return formatted

def generate_answer(question, results):
    """Generate natural language answer from results."""
    if not results:
        return f"""I couldn't find any theses matching "{question}" in the database.

Try:
• Using different keywords or names
• Checking the spelling of names
• Searching for broader topics"""
    
    # Limit for readability
    results_to_show = results[:8]
    
    prompt = f"""Answer this question using ONLY the database results provided.

Question: "{question}"

Database results ({len(results_to_show)} theses found):
{json.dumps(results_to_show, indent=2)}

Instructions:
- ONLY use information explicitly present in the results
- List thesis titles and authors clearly
- Include advisors, departments, and dates when available
- DO NOT make up or infer any information
- Format as a clear, organized list
- Be concise and factual

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        # Fallback: simple list
        output = f"Found {len(results)} theses:\n\n"
        for i, r in enumerate(results_to_show, 1):
            output += f"{i}. {r.get('Title', 'N/A')}\n"
            if r.get('author'):
                output += f"   Author: {r['author']}\n"
            if r.get('advisor1'):
                output += f"   Advisor: {r['advisor1']}\n"
            if r.get('First published'):
                output += f"   Published: {r['First published']}\n"
            output += "\n"
        return output

def main():
    print("🎓 Welcome to the CMC Thesis Chatbot!")
    print("Ask me anything about theses, authors, topics, or departments.\n")
    
    while True:
        try:
            question = input("\n💬 Your question (or 'exit' to quit): ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thank you for using the CMC Thesis Chatbot!")
                break
            
            if len(question) > 500:
                print("❌ Question is too long. Please ask a shorter question.\n")
                continue
            
            # Code detection - check for programming keywords
            code_keywords = ['import', 'def ', 'class ', 'function', 'var ', 'let ', 'const ', '#!/']
            if any(keyword in question.lower() for keyword in code_keywords):
                print("❌ It looks like you pasted code. Please ask a question about theses instead.\n")
                continue
            
            print("\n🔍 Searching...")
            
            key_terms = extract_key_terms(question)
            if not key_terms:
                print("I couldn't extract any keywords. Please try rephrasing your question.\n")
                continue
            
            column_mappings = map_terms_to_columns(question, key_terms)
            queries = build_sql_queries(column_mappings)
            if not queries:
                print("I couldn't build a search query. Please try rephrasing your question.\n")
                continue
            
            rows = execute_queries(queries)
            results = format_results(rows)
            logger.info(f"Found {len(results)} results")
            
            print("\n📚 Generating answer...\n")
            answer = generate_answer(question, results)
            
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
