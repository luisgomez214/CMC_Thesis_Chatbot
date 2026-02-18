import sqlite3
import logging
from groq import Groq
import json
import os
import re
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DB_PATH = "theses2.db"

def parse_query_with_llm(question):
    """Use LLM to understand the query and extract structured info."""
    prompt = f"""Analyze this thesis database query and extract structured information.

Query: "{question}"

Return ONLY valid JSON (no markdown, no explanation) with this structure:
{{
    "intent": "search|advisor|author|statistics|award|thesis_ideas|advisor_rec|who_is",
    "search_type": "person_name|topic|department|year|award|general",
    "entities": {{
        "person_names": ["list of actual person names, NOT topics like 'Machine Learning'"],
        "topics": ["subject keywords like 'machine learning', 'economics', 'climate change'"],
        "years": ["4-digit years"],
        "departments": ["department names if mentioned"]
    }},
    "role": "advisor|author|both|unknown",
    "needs_recommendation": true|false,
    "needs_generation": true|false
}}

Examples:
- "Find theses about machine learning" → search_type: "topic", topics: ["machine learning"]
- "Who did Mike Izbicki advise?" → search_type: "person_name", person_names: ["Mike Izbicki"], role: "advisor"
- "Theses by John Smith" → search_type: "person_name", person_names: ["John Smith"], role: "author"
- "How many economics theses?" → intent: "statistics", topics: ["economics"]
- "Who should I ask about AI?" → intent: "advisor_rec", topics: ["AI"]

Be smart about distinguishing person names from topics."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        result = re.sub(r'^```json\s*|\s*```$', '', result, flags=re.MULTILINE).strip()
        parsed = json.loads(result)
        return parsed
        
    except Exception as e:
        logger.error(f"LLM parsing failed: {e}")
        return {
            "intent": "search",
            "search_type": "general",
            "entities": {
                "person_names": [],
                "topics": re.findall(r'\b[a-z]{4,}\b', question.lower()),
                "years": re.findall(r'\b(19\d{2}|20[0-2]\d)\b', question),
                "departments": []
            },
            "role": "unknown",
            "needs_recommendation": False,
            "needs_generation": False
        }

def build_sql_query(parsed_query):
    """Build SQL query from LLM-parsed structure."""
    select = "SELECT * FROM theses"
    where_clauses = []
    params = []
    order_by = 'ORDER BY "First published" DESC'
    limit = "LIMIT 50"
    
    entities = parsed_query['entities']
    search_type = parsed_query['search_type']
    role = parsed_query['role']
    intent = parsed_query['intent']
    
    if intent == 'award' or search_type == 'award':
        where_clauses.append("(award IS NOT NULL AND award != '')")
    
    if entities.get('person_names'):
        name = entities['person_names'][0]
        
        if role == 'advisor':
            where_clauses.append("(advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?)")
            params.extend([f'%{name}%'] * 3)
        elif role == 'author':
            where_clauses.append("author LIKE ?")
            params.append(f'%{name}%')
        else:
            where_clauses.append(
                "(author LIKE ? OR advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?)"
            )
            params.extend([f'%{name}%'] * 4)
    
    if entities.get('years'):
        year = entities['years'][0]
        year_2digit = year[-2:]
        where_clauses.append('"First published" LIKE ?')
        params.append(f'%/{year_2digit}')
    
    if entities.get('departments'):
        dept = entities['departments'][0]
        where_clauses.append("department LIKE ?")
        params.append(f'%{dept}%')
    
    if entities.get('topics'):
        topic_conditions = []
        for topic in entities['topics'][:5]:
            topic_conditions.append(
                "(Title LIKE ? OR keywords LIKE ? OR abstract LIKE ? OR disciplines LIKE ? OR department LIKE ?)"
            )
            params.extend([f'%{topic}%'] * 5)
        
        if topic_conditions:
            where_clauses.append(f"({' OR '.join(topic_conditions)})")
    
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    query = f"{select} {where} {order_by} {limit}"
    
    return query, params

def search_database(question):
    """Execute database search using LLM-enhanced parsing."""
    parsed = parse_query_with_llm(question)
    
    if parsed['needs_generation']:
        return parsed['intent'], [], parsed
    
    if parsed['needs_recommendation']:
        return parsed['intent'], [], parsed
    
    query, params = build_sql_query(parsed)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA case_sensitive_like = OFF")
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return parsed['intent'], rows, parsed
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return parsed['intent'], [], parsed

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

def generate_thesis_ideas(question, parsed):
    """Generate thesis ideas using LLM."""
    topics = parsed['entities'].get('topics', [])
    
    related = []
    advisors_found = []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for topic in topics[:3]:
        cursor.execute("""
            SELECT Title, advisor1, advisor2, advisor3, department, keywords 
            FROM theses 
            WHERE Title LIKE ? OR keywords LIKE ? OR abstract LIKE ? OR disciplines LIKE ?
            ORDER BY "First published" DESC 
            LIMIT 10
        """, (f'%{topic}%', f'%{topic}%', f'%{topic}%', f'%{topic}%'))
        results = cursor.fetchall()
        related.extend(results)
        for r in results:
            for adv in [r[1], r[2], r[3]]:
                if adv and str(adv).strip():
                    advisors_found.append((adv, r[4]))
    
    if not advisors_found:
        cursor.execute("""
            SELECT DISTINCT advisor1, department 
            FROM theses 
            WHERE advisor1 IS NOT NULL AND advisor1 != ''
            ORDER BY "First published" DESC 
            LIMIT 20
        """)
        general_advisors = cursor.fetchall()
        advisors_found = [(adv[0], adv[1]) for adv in general_advisors if adv[0]]
    
    conn.close()
    
    advisor_list = {}
    for adv, dept in advisors_found[:15]:
        if adv not in advisor_list:
            advisor_list[adv] = dept
    
    context = f"User request: {question}\n\n"
    
    if related:
        context += "Related theses in database:\n"
        for r in related[:8]:
            context += f"- \"{r[0]}\" (Advisor: {r[1]}, Dept: {r[4]})\n"
        context += "\n"
    else:
        context += "No directly related theses found, but generating ideas for this topic.\n\n"
    
    context += "Available advisors from CMC thesis database:\n"
    for adv, dept in list(advisor_list.items())[:10]:
        context += f"- {adv} ({dept})\n"
    
    if not advisor_list:
        context += "- No advisors found in database for this topic\n"
    
    prompt = f"""You are helping a CMC (Claremont McKenna College) undergraduate student plan a ONE SEMESTER senior thesis.

{context}

YOUR TASK: Generate 3-5 undergraduate thesis ideas.

MANDATORY FORMAT FOR EACH IDEA:

**Title**: [compelling title]

**Overview**: [2-3 sentences]

**Research Questions**:
- [question 1]
- [question 2]  
- [question 3]

**Methodology**: [doable in ONE SEMESTER by an undergraduate]

**Significance**: [why it matters]

**Potential Advisor**: [MUST copy exact name from advisor list above, e.g. "Mike Izbicki" or "Janet Smith". If none fit, write "Consult department"]

**Timeline**: One semester (3-4 months)

=================
NON-NEGOTIABLE RULES:
=================
1. Timeline MUST be "One semester (3-4 months)" - NO exceptions
2. Advisor MUST be copy-pasted from the list above - NEVER invent names like "Dr. Rachel Cummings" or "Dr. Been Kim"
3. Scope MUST fit in 3-4 months of undergraduate work
4. If you cannot find a good advisor match in the list, write exactly: "Consult department for advisor recommendation"
5. DO NOT use advisors from Stanford, Harvard, Princeton, or any other university
6. ONLY use advisors explicitly listed above from CMC

Double-check each thesis idea follows ALL rules before including it."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return "I'm having trouble generating ideas. Please try rephrasing."
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return "I'm having trouble generating ideas. Please try rephrasing."

def recommend_advisors(question, parsed):
    """Recommend advisors based on topic."""
    topics = parsed['entities'].get('topics', [])
    
    if not topics:
        return "Please provide more specific topics or keywords for advisor recommendations."
    
    advisor_stats = defaultdict(lambda: {'count': 0, 'latest_year': 0, 'theses': [], 'depts': set()})
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for topic in topics[:5]:
        cursor.execute("""
            SELECT advisor1, advisor2, advisor3, "First published", Title, department
            FROM theses
            WHERE Title LIKE ? OR keywords LIKE ? OR abstract LIKE ? OR disciplines LIKE ?
            ORDER BY "First published" DESC
            LIMIT 50
        """, (f'%{topic}%', f'%{topic}%', f'%{topic}%', f'%{topic}%'))
        
        rows = cursor.fetchall()
        
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
        return f"I couldn't find advisors matching those topics ({', '.join(topics)}). Try different or broader keywords."
    
    current_year = datetime.now().year
    scored = []
    for adv, stats in advisor_stats.items():
        years_since = current_year - stats['latest_year'] if stats['latest_year'] > 0 else 100
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
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    prompt = f"""Question: {question}

Top advisors by expertise in {', '.join(topics)}:
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
        output = f"**Advisor Recommendations for {', '.join(topics)}:**\n\n"
        for i, adv in enumerate(scored[:8], 1):
            status = "✅ Active" if adv['is_active'] else f"⚠️ Last {adv['latest_year']}"
            output += f"{i}. **{adv['advisor']}** {status}\n"
            output += f"   {adv['count']} relevant theses | Depts: {', '.join(adv['departments'][:2])}\n"
            if adv['theses']:
                output += f"   Sample: {adv['theses'][0]}\n"
            output += "\n"
        return output

def handle_who_is(parsed):
    """Handle 'who is X' questions."""
    person_names = parsed['entities'].get('person_names', [])
    if not person_names:
        return "Please specify a person's name."
    
    name = person_names[0]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM theses 
        WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?
        ORDER BY "First published" DESC
    """, (f'%{name}%', f'%{name}%', f'%{name}%'))
    advised = format_results(cursor.fetchall())
    
    cursor.execute("""
        SELECT * FROM theses 
        WHERE author LIKE ?
        ORDER BY "First published" DESC
    """, (f'%{name}%',))
    authored = format_results(cursor.fetchall())
    
    conn.close()
    
    if not advised and not authored:
        return f"I couldn't find {name} in the thesis database."
    
    data = {
        'name': name,
        'as_advisor': {
            'count': len(advised),
            'theses': advised[:5]
        },
        'as_author': {
            'count': len(authored),
            'theses': authored[:3]
        }
    }
    
    prompt = f"""Create a helpful profile for {name} based on this thesis database info:

{json.dumps(data, indent=2)}

Format conversationally with:
1. Opening summary of who they are (advisor/author/both)
2. If advisor: number of theses, date range, departments, recent work
3. If author: list their thesis/theses
4. Highlight any awards

Be concise and helpful."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        info = f"**{name}**\n\n"
        if advised:
            info += f"**As Advisor:** {len(advised)} theses\n"
            for i, t in enumerate(advised[:5], 1):
                info += f"{i}. {t['Title']} ({t.get('Year', 'N/A')})\n"
        if authored:
            info += f"\n**As Author:**\n"
            for t in authored[:3]:
                info += f"- {t['Title']} ({t.get('Year', 'N/A')})\n"
        return info

def generate_answer(question, intent, rows, parsed):
    """Generate answer using LLM for natural language responses."""
    
    if intent == 'thesis_ideas' or parsed.get('needs_generation'):
        return generate_thesis_ideas(question, parsed)
    
    if intent == 'advisor_rec' or parsed.get('needs_recommendation'):
        return recommend_advisors(question, parsed)
    
    if intent == 'who_is':
        return handle_who_is(parsed)
    
    results = format_results(rows)
    
    if not results:
        prompt = f"""The user asked: "{question}"

No matching theses were found in the database. Generate a helpful response that:
1. Acknowledges no results were found
2. Suggests alternative search strategies (try broader terms, different keywords, check spelling)
3. Offers to help with related queries
4. Keeps it brief (2-3 sentences)"""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content
        except:
            return "I couldn't find any theses matching your query. Try different keywords or broader search terms."
    
    prompt = f"""Question: {question}

Found {len(results)} theses in the database:
{json.dumps(results[:10], indent=2)}

Provide a natural, conversational answer like ChatGPT would:
1. Start with a brief summary (1-2 sentences about what was found)
2. List theses with: title, author, advisor (if present), year
3. Show awards prominently if they exist (use 🏆 emoji)
4. Don't show "Award: None" or empty fields
5. Keep it organized but friendly
6. If more than 8 results, mention the count and show top ones

Be helpful and conversational."""
    
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
        output = f"Found {len(results)} theses:\n\n"
        for i, r in enumerate(results[:10], 1):
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

def about():
    ##create a function to handle "what is this thesis about" questions

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
            
            intent, rows, parsed = search_database(question)
            answer = generate_answer(question, intent, rows, parsed)
            
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
