import sqlite3
import logging
from groq import Groq
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DB_PATH = "theses2.db"

def extract_entities(question):
    """Extract key entities from question using simple patterns."""
    entities = {
        'names': [],
        'years': [],
        'keywords': []
    }
    
    # Extract capitalized names (2-3 words)
    names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', question)
    entities['names'] = list(set(names))
    
    # If no capitalized names, try lowercase (for queries like "mike izbicki")
    if not entities['names']:
        # Look for potential names in lowercase (2-3 words together)
        words = question.lower().split()
        # Check if there are consecutive words that could be a name
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                # Skip common words
                if words[i] not in ['the', 'and', 'for', 'from', 'with', 'about', 'theses', 'thesis']:
                    potential_name = f"{words[i].title()} {words[i+1].title()}"
                    # Check if this could be a name (not common English words)
                    if not any(w in ['About', 'From', 'With', 'Theses', 'Thesis'] for w in potential_name.split()):
                        entities['names'].append(potential_name)
    
    # Extract 4-digit years only
    years = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', question)
    entities['years'] = years
    
    # Extract meaningful keywords (3+ chars, not common words)
    stop_words = {'the', 'and', 'for', 'are', 'was', 'were', 'has', 'have', 'this', 
                  'that', 'with', 'from', 'what', 'who', 'how', 'when', 'where', 'which'}
    words = re.findall(r'\b[a-z]{3,}\b', question.lower())
    entities['keywords'] = [w for w in words if w not in stop_words]
    
    return entities

def detect_intent(question):
    """Detect query intent using simple keyword matching."""
    q = question.lower()
    
    # Intent keywords - minimal hardcoding
    intent_map = {
        'thesis_ideas': ['thesis idea', 'thesis topic', 'brainstorm', 'suggest thesis', 'thesis outline'],
        'advisor_rec': ['who should i ask', 'recommend advisor', 'suggest advisor', 'best advisor'],
        'statistics': ['how many', 'count', 'number of', 'total'],
        'award': ['best', 'award', 'won', 'prize'],
        'advisor': ['advis', 'supervised'],
        'author': ['author', 'thesis by', 'wrote'],
        'who_is': ['who is', 'tell me about']
    }
    
    for intent, keywords in intent_map.items():
        if any(kw in q for kw in keywords):
            return intent
    
    return 'search'  # Default: general search

def build_flexible_query(question, intent, entities):
    """Build SQL query dynamically based on intent and extracted entities."""
    
    # Base query parts
    select = "SELECT * FROM theses"
    where_clauses = []
    params = []
    order_by = 'ORDER BY "First published" DESC'
    limit = "LIMIT 50"
    
    q_lower = question.lower()
    
    # Build WHERE clauses based on entities and intent
    if intent == 'award':
        where_clauses.append("(award IS NOT NULL AND award != '')")
    
    # Check for explicit "by" indicating author
    is_author_query = 'thesis by' in q_lower or 'theses by' in q_lower or 'authored by' in q_lower
    
    # Check for advisor indicators
    is_advisor_query = 'advis' in q_lower or 'supervised' in q_lower
    
    if is_author_query and entities['names']:
        # Explicit author query
        name = entities['names'][0]
        where_clauses.append("author LIKE ?")
        params.append(f'%{name}%')
    elif is_advisor_query and entities['names']:
        # Explicit advisor query
        name = entities['names'][0]
        where_clauses.append("(advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?)")
        params.extend([f'%{name}%'] * 3)
    elif intent in ['advisor', 'statistics'] or is_advisor_query:
        # Advisor-related query
        if entities['names']:
            name = entities['names'][0]
            where_clauses.append("(advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?)")
            params.extend([f'%{name}%'] * 3)
    elif entities['names'] and not where_clauses:
        # Just a name - search both author and advisor
        name = entities['names'][0]
        where_clauses.append(
            "(author LIKE ? OR advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?)"
        )
        params.extend([f'%{name}%'] * 4)
    
    if entities['years']:
        year = entities['years'][0]
        where_clauses.append('"First published" LIKE ?')
        params.append(f'%{year}%')
    
    # If no specific entity match, search by keywords
    if not where_clauses and entities['keywords']:
        keyword_conditions = []
        for kw in entities['keywords'][:3]:
            keyword_conditions.append(
                "(Title LIKE ? OR keywords LIKE ? OR abstract LIKE ? OR disciplines LIKE ? OR department LIKE ?)"
            )
            params.extend([f'%{kw}%'] * 5)
        if keyword_conditions:
            where_clauses.append(f"({' OR '.join(keyword_conditions)})")
    
    # Assemble query
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    query = f"{select} {where} {order_by} {limit}"
    
    return query, params

def search_database(question):
    """Execute database search."""
    intent = detect_intent(question)
    entities = extract_entities(question)
    
    logger.info(f"[INTENT] {intent}")
    logger.info(f"[ENTITIES] Names: {entities['names']}, Years: {entities['years']}, Keywords: {entities['keywords'][:3]}")
    
    # Handle special intents that don't need DB query
    if intent in ['thesis_ideas', 'advisor_rec']:
        return intent, [], entities
    
    # Build and execute query
    query, params = build_flexible_query(question, intent, entities)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA case_sensitive_like = OFF")
        cursor = conn.cursor()
        
        logger.info(f"[SQL] {query[:100]}...")
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        logger.info(f"[RESULTS] {len(rows)} theses")
        return intent, rows, entities
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return intent, [], entities

def format_results(rows):
    """Convert rows to dicts."""
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
            result = {k: v for k, v in result.items() if v and str(v).strip()}
            if result:
                results.append(result)
    return results

def generate_thesis_ideas(question, entities):
    """Generate thesis ideas using LLM."""
    # Search for related work
    related = []
    for kw in entities['keywords'][:3]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Title, advisor1, department, keywords 
            FROM theses 
            WHERE Title LIKE ? OR keywords LIKE ? 
            ORDER BY "First published" DESC 
            LIMIT 5
        """, (f'%{kw}%', f'%{kw}%'))
        related.extend(cursor.fetchall())
        conn.close()
    
    context = f"User request: {question}\n\n"
    if related:
        context += "Related theses in database:\n"
        for r in related[:5]:
            context += f"- \"{r[0]}\" (Advisor: {r[1]}, Dept: {r[2]})\n"
    
    prompt = f"""{context}

Generate 5-7 specific, feasible thesis ideas. For each:
1. **Title**: Specific and compelling
2. **Overview**: What it explores (2-3 sentences)
3. **Research Questions**: 2-3 specific questions
4. **Methodology**: Brief approach
5. **Significance**: Why it matters
6. **Potential Advisor**: Based on related work above (if available)
7. **Timeline**: Realistic estimate

Be specific and academically rigorous."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return "I'm having trouble generating ideas. Please try rephrasing."

def recommend_advisors(question, entities):
    """Recommend advisors based on topic."""
    # Search for relevant advisors - use ALL keywords, not just first 3
    advisor_stats = defaultdict(lambda: {'count': 0, 'latest_year': 0, 'theses': [], 'depts': set()})
    
    # Get ALL matching keywords (machine, learning, etc.)
    search_terms = entities['keywords'][:5]  # Use up to 5 keywords
    
    if not search_terms:
        return "Please provide more specific topics or keywords for advisor recommendations."
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Build comprehensive search
    for kw in search_terms:
        cursor.execute("""
            SELECT advisor1, advisor2, advisor3, "First published", Title, department
            FROM theses
            WHERE Title LIKE ? OR keywords LIKE ? OR abstract LIKE ? OR disciplines LIKE ?
            ORDER BY "First published" DESC
            LIMIT 50
        """, (f'%{kw}%', f'%{kw}%', f'%{kw}%', f'%{kw}%'))
        
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} theses for keyword '{kw}'")
        
        for row in rows:
            year_match = re.search(r'\b(19|20)\d{2}\b', row[3] or '')
            year = int(year_match.group(0)) if year_match else 0
            
            for adv in [row[0], row[1], row[2]]:
                if adv and str(adv).strip():
                    advisor_stats[adv]['count'] += 1
                    advisor_stats[adv]['latest_year'] = max(advisor_stats[adv]['latest_year'], year)
                    if row[4] not in advisor_stats[adv]['theses']:
                        advisor_stats[adv]['theses'].append(row[4])
                    if row[5]:
                        advisor_stats[adv]['depts'].add(row[5])
    
    conn.close()
    
    if not advisor_stats:
        return f"I couldn't find advisors matching those topics ({', '.join(search_terms)}). Try different or broader keywords."
    
    # Score advisors
    current_year = datetime.now().year
    scored = []
    for adv, stats in advisor_stats.items():
        years_since = current_year - stats['latest_year'] if stats['latest_year'] > 0 else 100
        # Weight recent activity heavily
        recency_score = 1.0 if years_since <= 3 else max(0, 1.0 - (years_since - 3) * 0.1)
        score = stats['count'] * recency_score
        
        scored.append({
            'advisor': adv,
            'count': stats['count'],
            'latest_year': stats['latest_year'],
            'is_active': years_since <= 3,
            'theses': stats['theses'][:3],
            'departments': list(stats['depts']),
            'score': score
        })
    
    # Sort by score (relevance + recency)
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Found {len(scored)} advisors, top 3: {[s['advisor'] for s in scored[:3]]}")
    
    # Use LLM to format nicely
    prompt = f"""Question: {question}

Top advisors by expertise in {', '.join(search_terms)}:
{json.dumps(scored[:10], indent=2)}

Provide conversational recommendations. Start with a 2-sentence summary, then list top 5-8 advisors with:
- Name
- Activity status: ✅ if latest_year >= 2022, ⚠️ if older
- Expertise: count of relevant theses
- Departments
- 1-2 sample thesis titles

Be helpful and specific. Focus on the most active and relevant advisors."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        output = f"**Advisor Recommendations for {', '.join(search_terms)}:**\n\n"
        for i, adv in enumerate(scored[:8], 1):
            status = "✅ Active" if adv['is_active'] else f"⚠️ Last {adv['latest_year']}"
            output += f"{i}. **{adv['advisor']}** {status}\n"
            output += f"   {adv['count']} relevant theses | Depts: {', '.join(adv['departments'][:2])}\n"
            if adv['theses']:
                output += f"   Sample: {adv['theses'][0]}\n"
            output += "\n"
        return output

def handle_statistics(question, intent, rows, entities):
    """Handle counting questions - direct answer, no LLM needed."""
    count = len(rows)
    
    # Build description
    parts = []
    if entities['names']:
        if intent == 'advisor' or 'advis' in question.lower():
            parts.append(f"advised by {entities['names'][0]}")
        else:
            parts.append(f"by {entities['names'][0]}")
    
    for kw in entities['keywords']:
        if kw in ['economics', 'computer', 'science', 'math', 'physics', 'biology', 'psychology', 'philosophy']:
            parts.append(f"in {kw}")
            break
    
    if entities['years']:
        parts.append(f"from {entities['years'][0]}")
    
    if 'award' in question.lower():
        parts.append("that won awards")
    
    desc = " ".join(parts)
    
    # Simple, direct answer
    answer = f"**{count} theses** {desc}.\n\n"
    
    if count > 0:
        results = format_results(rows[:5])
        answer += "Here are some examples:\n\n"
        for i, r in enumerate(results, 1):
            answer += f"{i}. {r['Title']} ({r.get('Year', 'N/A')})\n"
            if r.get('Author'):
                answer += f"   Author: {r['Author']}\n"
            if r.get('Advisor'):
                answer += f"   Advisor: {r['Advisor']}\n"
            if r.get('Award'):
                answer += f"   Award: {r['Award']}\n"
            answer += "\n"
    
    return answer

def handle_who_is(entities):
    """Handle 'who is X' questions - direct data retrieval."""
    if not entities['names']:
        return "Please specify a person's name."
    
    name = entities['names'][0]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check as advisor
    cursor.execute("""
        SELECT * FROM theses 
        WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?
        ORDER BY "First published" DESC
    """, (f'%{name}%', f'%{name}%', f'%{name}%'))
    advised = format_results(cursor.fetchall())
    
    # Check as author
    cursor.execute("""
        SELECT * FROM theses 
        WHERE author LIKE ?
        ORDER BY "First published" DESC
    """, (f'%{name}%',))
    authored = format_results(cursor.fetchall())
    
    conn.close()
    
    if not advised and not authored:
        return f"I couldn't find {name} in the thesis database."
    
    info = f"**{name}**\n\n"
    
    if advised:
        years = [t.get('Year', '') for t in advised if t.get('Year')]
        depts = set([t.get('Department') for t in advised if t.get('Department')])
        
        info += f"**As Advisor:** {len(advised)} theses"
        if years:
            info += f" ({min(years)} - {max(years)})"
        info += "\n"
        
        if depts:
            info += f"Departments: {', '.join(list(depts)[:3])}\n"
        
        info += "\nRecent work:\n"
        for i, t in enumerate(advised[:5], 1):
            info += f"{i}. {t['Title']} ({t.get('Year', 'N/A')})\n"
            if t.get('Award'):
                info += f"   🏆 {t['Award']}\n"
    
    if authored:
        info += f"\n**As Author:**\n"
        for t in authored[:3]:
            info += f"- {t['Title']} ({t.get('Year', 'N/A')})\n"
    
    return info

def generate_answer(question, intent, rows, entities):
    """Generate answer - use LLM only when beneficial for natural language."""
    
    # Special intents - handle directly or with targeted LLM use
    if intent == 'thesis_ideas':
        return generate_thesis_ideas(question, entities)
    
    if intent == 'advisor_rec':
        return recommend_advisors(question, entities)
    
    if intent == 'statistics':
        return handle_statistics(question, intent, rows, entities)
    
    if intent == 'who_is':
        return handle_who_is(entities)
    
    results = format_results(rows)
    
    if not results:
        return "I couldn't find any theses matching your query. Try different keywords or names."
    
    # For listing theses, use LLM to create conversational response
    prompt = f"""Question: {question}

Found {len(results)} theses:
{json.dumps(results[:8], indent=2)}

Provide a conversational answer like ChatGPT:
1. Start with a brief summary (1-2 sentences)
2. List theses with title, author, advisor, year
3. Show awards if they exist (don't show "Award: None")
4. Be friendly and helpful"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        # Fallback: simple formatting
        output = f"Found {len(results)} theses:\n\n"
        for i, r in enumerate(results[:8], 1):
            output += f"{i}. **{r['Title']}**\n"
            if r.get('Author'):
                output += f"   Author: {r['Author']}\n"
            if r.get('Advisor'):
                output += f"   Advisor: {r['Advisor']}\n"
            if r.get('Award'):
                output += f"   🏆 {r['Award']}\n"
            if r.get('Year'):
                output += f"   Year: {r['Year']}\n"
            output += "\n"
        return output

def main():
    print("🎓 CMC Thesis Chatbot")
    print("="*50)
    print("Ask me anything about CMC theses!")
    print("="*50)
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("💬 Your question: ").strip()
            
            if not question or question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            print("\n🔍 Searching...")
            
            intent, rows, entities = search_database(question)
            answer = generate_answer(question, intent, rows, entities)
            
            print("\n" + "="*70)
            print(answer)
            print("="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print("❌ Something went wrong. Please try again.\n")

if __name__ == "__main__":
    main()
