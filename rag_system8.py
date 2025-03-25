from urllib.parse import urlparse
import datetime
import logging
import re
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import groq
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM Configuration
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
DEFAULT_MODEL = 'llama3-8b-8192'

def run_llm(system: str, user: str, model: str = DEFAULT_MODEL, seed: Optional[int] = None) -> str:
    """Helper function for all LLM uses"""
    try:
        completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ],
            model=model,
            seed=seed,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"Error: {str(e)}"

def analyze_query_type(question: str) -> Dict[str, Any]:
    """Analyze query to determine search strategy"""
    # Add award-related keywords to detect
    award_keywords = ["award", "prize", "recognition", "honor", "win", "won", "winning", "awarded"]
    
    if any(keyword in question.lower() for keyword in award_keywords):
        return {
            "query_type": "award",
            "entities": [e.strip() for e in question.split() if len(e) > 3],
            "columns_needed": ["Title", "department", "publication_date", "author1_fname", "author1_lname", "award", "abstract"]
        }
    
    system_prompt = """You are a query analyzer for a thesis database. 
Determine what kind of information the user is looking for.

Output a JSON object with:
{
    "query_type": "advisor", "author", "department", "topic", "year", or "general",
    "entities": [list of relevant names, terms, or years mentioned],
    "columns_needed": [list of likely needed columns, keep this minimal],
    "is_exact_title_search": true/false,
    "exact_title": "full title if user is looking for a specific thesis" 
}

Include only the JSON in your response."""
    
    response = run_llm(system_prompt, question)
    try:
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except Exception as e:
        logger.error(f"Query analysis parsing error: {e}")
    
    if "advisor" in question.lower():
        return {
            "query_type": "advisor",
            "entities": [e.strip() for e in question.split() if len(e) > 3],
            "columns_needed": ["Title", "department", "publication_date", "advisor1", "advisor2", "disciplines"]
        }
    elif any(term in question.lower() for term in ["author", "student", "wrote"]):
        return {
            "query_type": "author",
            "entities": [e.strip() for e in question.split() if len(e) > 3],
            "columns_needed": ["Title", "publication_date", "author1_fname", "author1_lname", "department", "advisor1", "advisor2", "disciplines"]
        }
    else:
        return {
            "query_type": "general",
            "entities": [e.strip() for e in question.split() if len(e) > 3],
            "columns_needed": ["Title", "abstract", "department", "publication_date", "advisor1", "advisor2"]
        }

def clean_name(name: str) -> str:
    """Remove common prefixes (e.g. 'Name:') and trim whitespace"""
    return re.sub(r'^(Name:\s*)', '', name, flags=re.IGNORECASE).strip()

class ThesisDataManager:
    """Manages thesis data from CSV to SQLite"""
    
    def __init__(self, csv_path: str = "merged_theses.csv", db_path: str = "theses.db"):
        self.csv_path = csv_path
        self.db_path = db_path
        self.conn = None
        self.loaded = False
        self._columns_cache = None
        
    def csv_to_sqlite(self) -> bool:
        """Convert CSV data to SQLite database"""
        try:
            chunk_size = 1000
            df_iter = pd.read_csv(self.csv_path, chunksize=chunk_size)
            # IMPORTANT: Allow connection use in different threads
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            for i, chunk in enumerate(df_iter):
                chunk.to_sql('theses', self.conn, if_exists='replace' if i == 0 else 'append', index=False)
            cursor = self.conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON theses (Title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_author1_lname ON theses (author1_lname)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_author1_fname ON theses (author1_fname)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_publication_date ON theses (publication_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_department ON theses (department)")
            cursor.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines, content='theses', content_rowid='rowid')"
            )
            cursor.execute(
                "INSERT INTO theses_fts(rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines) "
                "SELECT rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines FROM theses"
            )
            self.conn.commit()
            logger.info(f"Successfully converted CSV to SQLite at {self.db_path}")
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"CSV to SQLite conversion failed: {e}")
            return False
    
    def load_data(self) -> bool:
        """Load data into SQLite if not already done"""
        try:
            if not os.path.exists(self.db_path):
                return self.csv_to_sqlite()
            # IMPORTANT: Allow connection use in different threads
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses'")
            if not cursor.fetchone():
                return self.csv_to_sqlite()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses_fts'")
            if not cursor.fetchone():
                try:
                    cursor.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines, content='theses', content_rowid='rowid')"
                    )
                    cursor.execute(
                        "INSERT INTO theses_fts(rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines) "
                        "SELECT rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines FROM theses"
                    )
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Error adding FTS: {e}")
            logger.info(f"Connected to existing database at {self.db_path}")
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False
    
    def get_columns(self) -> List[str]:
        """Get column names in the dataset"""
        if not self.loaded:
            self.load_data()
        if self._columns_cache:
            return self._columns_cache
        try:
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(theses)")
            columns = [row[1] for row in cursor.fetchall()]
            self._columns_cache = columns
            return columns
        except Exception as e:
            logger.error(f"Error getting columns: {e}")
            return []
    
    def execute_query(self, query: str, params: tuple = (), column_subset: List[str] = None) -> List[Dict[str, Any]]:
        """Execute SQL query with option to return specific columns"""
        if not self.loaded:
            self.load_data()
        try:
            cursor = self.conn.cursor()
            logger.debug(f"Executing: {query} with params: {params}")
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                if column_subset:
                    result_dict = {}
                    row_dict = dict(zip(columns, row))
                    for col in column_subset:
                        if col in row_dict:
                            result_dict[col] = row_dict[col]
                    results.append(result_dict)
                else:
                    results.append(dict(zip(columns, row)))
            logger.info(f"Query returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return []
    
    def search_by_year(self, year: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by publication year"""
        try:
            query = "SELECT * FROM theses WHERE publication_date LIKE ? ORDER BY publication_date DESC LIMIT ?"
            return self.execute_query(query, (f"%{year}%", limit))
        except Exception as e:
            logger.error(f"Year search error: {e}")
            return []
    
    def search_by_year_and_department(self, year: str, dept: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by publication year and department"""
        try:
            query = "SELECT * FROM theses WHERE publication_date LIKE ? AND department LIKE ? ORDER BY publication_date DESC LIMIT ?"
            return self.execute_query(query, (f"%{year}%", f"%{dept}%", limit))
        except Exception as e:
            logger.error(f"Year and department search error: {e}")
            return []
    
    def get_publication_year_range(self) -> Tuple[str, str]:
        """Return the minimum and maximum publication_date from the theses table"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT MIN(publication_date), MAX(publication_date) FROM theses")
            row = cursor.fetchone()
            if row:
                return row[0], row[1]
            else:
                return "", ""
        except Exception as e:
            logger.error(f"Error getting publication year range: {e}")
            return "", ""
    
    def get_departments(self) -> List[str]:
        """Return a list of distinct department names from the database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT department FROM theses")
            return [row[0] for row in cursor.fetchall() if row[0]]
        except Exception as e:
            logger.error(f"Error getting departments: {e}")
            return []

    def search_by_advisor(self, advisor_name: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by advisor name"""
        results = []
        try:
            query1 = """
            SELECT Title, department, publication_date, advisor1, advisor2 
            FROM theses 
            WHERE advisor1 = ? OR advisor2 = ? OR advisor3 = ?
            ORDER BY publication_date DESC
            LIMIT ?
            """
            results.extend(self.execute_query(query1, (advisor_name, advisor_name, advisor_name, limit)))
            if len(results) < limit:
                query2 = """
                SELECT Title, department, publication_date, advisor1, advisor2 
                FROM theses 
                WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ? 
                ORDER BY publication_date DESC
                LIMIT ?
                """
                results.extend(self.execute_query(query2, (f"%{advisor_name}%", f"%{advisor_name}%", f"%{advisor_name}%", limit)))
            if len(results) < limit and ' ' in advisor_name:
                for part in advisor_name.split():
                    if len(part) > 2:
                        query3 = """
                        SELECT Title, department, publication_date, advisor1, advisor2 
                        FROM theses 
                        WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ? 
                        ORDER BY publication_date DESC
                        LIMIT ?
                        """
                        results.extend(self.execute_query(query3, (f"%{part}%", f"%{part}%", f"%{part}%", limit - len(results))))
                        if len(results) >= limit:
                            break
            return results[:limit]
        except Exception as e:
            logger.error(f"Advisor search error: {e}")
            return []
    
    def search_by_author(self, first_name: str = None, last_name: str = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by author name"""
        try:
            if first_name and last_name:
                query = """
                SELECT Title, author1_fname, author1_lname, department, publication_date
                FROM theses 
                WHERE author1_fname = ? AND author1_lname = ?
                ORDER BY publication_date DESC
                LIMIT ?
                """
                results = self.execute_query(query, (first_name, last_name, limit))
                if not results:
                    query = """
                    SELECT Title, author1_fname, author1_lname, department, publication_date
                    FROM theses 
                    WHERE author1_fname LIKE ? AND author1_lname LIKE ?
                    ORDER BY publication_date DESC
                    LIMIT ?
                    """
                    results = self.execute_query(query, (f"%{first_name}%", f"%{last_name}%", limit))
                return results
            elif last_name:
                return self.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date FROM theses WHERE author1_lname LIKE ? ORDER BY publication_date DESC LIMIT ?", 
                    (f"%{last_name}%", limit)
                )
            elif first_name:
                return self.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date FROM theses WHERE author1_fname LIKE ? ORDER BY publication_date DESC LIMIT ?", 
                    (f"%{first_name}%", limit)
                )
            return []
        except Exception as e:
            logger.error(f"Author search error: {e}")
            return []
    
    def search_by_department(self, department: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by department"""
        try:
            return self.execute_query(
                "SELECT Title, department, publication_date, author1_fname, author1_lname FROM theses WHERE department LIKE ? ORDER BY publication_date DESC LIMIT ?", 
                (f"%{department}%", limit)
            )
        except Exception as e:
            logger.error(f"Department search error: {e}")
            return []
    
    def search_by_keyword(self, keyword: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by keyword in title or abstract (including award)"""
        try:
            try:
                fts_query = """
                SELECT t.Title, t.department, t.publication_date, t.author1_fname, t.author1_lname
                FROM theses t
                JOIN theses_fts fts ON t.rowid = fts.rowid
                WHERE theses_fts MATCH ?
                ORDER BY t.publication_date DESC
                LIMIT ?
                """
                results = self.execute_query(fts_query, (keyword, limit))
                if results:
                    return results
            except Exception:
                logger.warning("FTS failed, falling back to LIKE")
            return self.execute_query(
                "SELECT Title, department, publication_date, author1_fname, author1_lname FROM theses WHERE Title LIKE ? OR abstract LIKE ? OR keywords LIKE ? OR award LIKE ? ORDER BY publication_date DESC LIMIT ?",
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit)
            )
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def search_by_exact_title(self, title: str) -> List[Dict[str, Any]]:
        """Search for thesis by exact title"""
        try:
            results = self.execute_query("SELECT * FROM theses WHERE Title = ? ORDER BY publication_date DESC LIMIT 5", (title,))
            if not results:
                results = self.execute_query("SELECT * FROM theses WHERE Title LIKE ? ORDER BY publication_date DESC LIMIT 5", (title,))
            if not results:
                partial_title = title[:min(50, len(title))]
                results = self.execute_query("SELECT * FROM theses WHERE Title LIKE ? ORDER BY publication_date DESC LIMIT 10", (f"%{partial_title}%",))
            return results
        except Exception as e:
            logger.error(f"Exact title search error: {e}")
            return []
    
    def search_all_columns(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search across all text columns, including award"""
        if not self.loaded:
            self.load_data()
        columns = self.get_columns()
        text_columns = [col for col in columns if col not in ['rowid', 'publication_date']]
        try:
            try:
                fts_query = """
                SELECT t.*
                FROM theses t
                JOIN theses_fts fts ON t.rowid = fts.rowid
                WHERE theses_fts MATCH ?
                ORDER BY t.publication_date DESC
                LIMIT ?
                """
                results = self.execute_query(fts_query, (search_term, limit))
                if results:
                    return results
            except Exception:
                logger.warning("FTS failed, falling back to LIKE")
            where_clauses = [f'"{col}" LIKE ?' for col in text_columns]
            params = [f"%{search_term}%" for _ in text_columns]
            query = f"""
            SELECT * FROM theses 
            WHERE {" OR ".join(where_clauses)}
            ORDER BY publication_date DESC
            LIMIT {limit}
            """
            return self.execute_query(query, tuple(params))
        except Exception as e:
            logger.error(f"All-columns search error: {e}")
            return []

    def search_by_award(self, award_term: str = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses that have received awards"""
        try:
            if award_term:
                query = """
                SELECT * FROM theses 
                WHERE award IS NOT NULL AND award != '' AND award LIKE ? 
                ORDER BY publication_date DESC 
                LIMIT ?
                """
                return self.execute_query(query, (f"%{award_term}%", limit))
            else:
                query = """
                SELECT * FROM theses 
                WHERE award IS NOT NULL AND award != '' 
                ORDER BY publication_date DESC 
                LIMIT ?
                """
                return self.execute_query(query, (limit,))
        except Exception as e:
            logger.error(f"Award search error: {e}")
            return []

class ThesisRAGSystem:
    """RAG system for answering questions about thesis data"""
    
    def __init__(self, data_manager: ThesisDataManager, model: str = DEFAULT_MODEL):
        self.data_manager = data_manager
        self.model = model
            
    def determine_query_strategy(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Determine best query strategy based on question and execute it"""
        title_analysis_prompt = """Analyze this user query and determine if it appears to be:
1. A direct title of an academic work
2. A question about a specific title
3. A general question not directly referencing a title

Return a JSON with:
{
    "query_type": "direct_title", "title_question", or "general_question",
    "extracted_title": "full title if detected or null",
    "confidence": 0-100 (how confident you are in this assessment)
}

Include only the JSON in your response."""
        response = run_llm(title_analysis_prompt, question)
        try:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                title_info = json.loads(json_match.group(1))
                if (title_info.get("query_type") in ["direct_title", "title_question"] and 
                    title_info.get("confidence", 0) > 70 and
                    title_info.get("extracted_title")):
                    extracted_title = title_info.get("extracted_title")
                    logger.info(f"LLM detected potential title: {extracted_title}")
                    title_results = self.data_manager.search_by_exact_title(extracted_title)
                    if title_results:
                        return "exact_title", title_results
                    if title_info.get("query_type") == "direct_title" and extracted_title != question:
                        title_results = self.data_manager.search_by_exact_title(question)
                        if title_results:
                            return "exact_title", title_results
        except Exception as e:
            logger.error(f"Error parsing title analysis: {e}")
        
        m = re.search(r'what is (.*?) about', question, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                title_results = self.data_manager.search_by_exact_title(candidate)
                if title_results:
                    return "exact_title", title_results

        year_matches = re.findall(r'\b(20\d{2})\b', question)
        if year_matches:
            year = year_matches[0]
            departments = self.data_manager.get_departments()
            dept_keyword = None
            for dept in departments:
                if dept.lower() in question.lower():
                    dept_keyword = dept
                    break
            if dept_keyword:
                combined_results = self.data_manager.search_by_year_and_department(year, dept_keyword)
                if combined_results:
                    return "year_and_department", combined_results
            year_results = self.data_manager.search_by_year(year)
            if year_results:
                return "year", year_results
            else:
                return f"year:{year}", []
        
        query_analysis_prompt = """You are a database query analyzer for a thesis database.
Analyze this question and determine which query types would be most appropriate.
Return a JSON object with:
{
    "possible_query_types": ["advisor", "author", "department", "topic", "award", "general"],
    "query_priority": ["primary_type", "secondary_type", ...],
    "entities": {
        "person_names": ["full name 1", "full name 2"],
        "keywords": ["keyword1", "keyword2"],
        "award_terms": ["prize", "award", "recognition", "honor"]
    }
}
Include only the JSON in your response."""
        response = run_llm(query_analysis_prompt, question)
        try:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                query_info = json.loads(json_match.group(1))
            else:
                query_info = {"query_priority": ["general"], "entities": {"person_names": [], "keywords": []}}
        except Exception:
            query_info = {"query_priority": ["general"], "entities": {"person_names": [], "keywords": []}}
        
        query_priority = query_info.get("query_priority", ["general"])
        person_names = [clean_name(n) for n in query_info.get("entities", {}).get("person_names", [])]
        keywords = query_info.get("entities", {}).get("keywords", [])
        
        if not person_names:
            person_names.extend(re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', question))
            person_names = [clean_name(n) for n in person_names]
        keywords.extend(re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+){2,})', question))
        if len(keywords) < 3:
            keywords.extend([word for word in question.split() if len(word) > 4 and word[0].isupper()])
        
        logger.info(f"Query analysis: priorities={query_priority}, names={person_names}, keywords={keywords}")
        
        all_results = []
        successful_query_type = "general"
        for query_type in query_priority:
            results = []
            if query_type == "advisor":
                for name in person_names:
                    advisor_results = self.data_manager.search_by_advisor(name)
                    if advisor_results:
                        results.extend(advisor_results)
                        successful_query_type = "advisor"
            elif query_type == "author":
                for name in person_names:
                    parts = name.split()
                    if len(parts) >= 2:
                        author_results = self.data_manager.search_by_author(parts[0], parts[1])
                        if author_results:
                            results.extend(author_results)
                            successful_query_type = "author"
            elif query_type == "department":
                for keyword in keywords:
                    dept_results = self.data_manager.search_by_department(keyword)
                    if dept_results:
                        results.extend(dept_results)
                        successful_query_type = "department"
            elif query_type == "topic" or query_type == "general":
                for keyword in keywords:
                    keyword_results = self.data_manager.search_by_keyword(keyword)
                    if keyword_results:
                        results.extend(keyword_results)
                        successful_query_type = query_type
            elif query_type == "award":
                award_terms = query_info.get("entities", {}).get("award_terms", [])
                if not award_terms:
                    award_results = self.data_manager.search_by_award()
                else:
                    for term in award_terms:
                        award_specific_results = self.data_manager.search_by_award(term)
                        if award_specific_results:
                            results.extend(award_specific_results)
                            successful_query_type = "award"

            if results:
                all_results.extend(results)
                if len(all_results) > 20:
                    break
        
        if len(all_results) < 10:
            all_search_terms = person_names + keywords
            for search_term in all_search_terms:
                if len(search_term) > 3:
                    comprehensive_results = self.data_manager.search_all_columns(search_term)
                    if comprehensive_results:
                        all_results.extend(comprehensive_results)
                        successful_query_type = "comprehensive"
                        if len(all_results) > 30:
                            break
        
        if not all_results:
            all_results = self._fallback_retrieval(question)
            successful_query_type = "fallback"
        
        unique_results = []
        seen_titles = set()
        for result in all_results:
            title = result.get('Title', '')
            if title and title not in seen_titles:
                unique_results.append(result)
                seen_titles.add(title)
    
        return successful_query_type, unique_results

    def _check_for_exact_title_search(self, question: str) -> List[Dict[str, Any]]:
        """Check if question is asking about a specific thesis title"""
        title_patterns = [
            r'thesis (titled|called|named) ["\'](.*?)["\']',
            r'information (on|about) ["\'](.*?)["\']',
            r'find ["\'](.*?)["\']',
            r'looking for ["\'](.*?)["\']',
            r'what is (.*?) about'
        ]
        for pattern in title_patterns:
            matches = re.search(pattern, question, re.IGNORECASE)
            if matches and len(matches.groups()) >= 1:
                title = matches.groups()[-1].strip()
                logger.info(f"Detected possible title search: {title}")
                results = self.data_manager.search_by_exact_title(title)
                if results:
                    return results
        return []

    def _fallback_retrieval(self, question: str) -> List[Dict[str, Any]]:
        """Fallback retrieval strategy when main strategies fail"""
        stop_words = {"the", "a", "an", "in", "of", "for", "about", "with", "by", "to", 
                      "what", "who", "how", "when", "where", "which", "thesis", "theses", "dissertation"}
        words = [word.strip(',.?!:;()[]{}') for word in question.lower().split()]
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        results = []
        for keyword in keywords:
            keyword_results = self.data_manager.search_all_columns(keyword, limit=10)
            if keyword_results:
                results.extend(keyword_results)
                if len(results) >= 30:
                    break
        if not results:
            try:
                results = self.data_manager.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date, abstract FROM theses ORDER BY publication_date DESC LIMIT 20"
                )
            except Exception as e:
                logger.error(f"Fallback retrieval error: {e}")
        return results
    
    def _format_theses_for_context(self, theses: List[Dict[str, Any]], query_type: str) -> str:
        """Format retrieved theses in a more conversational Markdown manner."""
        lines = []
        if not theses:
            return "No theses found."
        
        lines.append("## Relevant Theses Found")
        lines.append("---")
    
        for idx, thesis in enumerate(theses):
            title = thesis.get('Title', 'Unknown title')
            department = thesis.get('department', 'Unknown department')
            author_fname = thesis.get('author1_fname', '')
            author_lname = thesis.get('author1_lname', '')
            author = f"{author_fname} {author_lname}".strip() or "Unknown author"
            pub_date = thesis.get('publication_date', 'Unknown date')
            advisor1 = thesis.get('advisor1', '')
            advisor2 = thesis.get('advisor2', '')
            abstract = thesis.get('abstract', '')
    
            if abstract and len(abstract) > 500:
                abstract = abstract[:497] + "..."
    
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"- **Author:** {author}")
            lines.append(f"- **Department:** {department}")
            lines.append(f"- **Year:** {pub_date}")
    
            if advisor1:
                lines.append(f"- **Primary Advisor:** {advisor1}")
            if advisor2:
                lines.append(f"- **Secondary Advisor:** {advisor2}")
    
            if abstract:
                lines.append("")
                lines.append("**Abstract:**")
                lines.append(abstract)
            
            # Optionally include award information if available
            award = thesis.get('award', '')
            if award:
                lines.append("")
                lines.append(f"**Award:** {award}")
    
            if idx < len(theses) - 1:
                lines.append("\n---\n")
    
        return "\n".join(lines)
    


    def answer_question(self, question: str, conversation_history: str = "") -> str:
        """Answer a question about theses, taking previous conversation into account."""
        if not self.data_manager.loaded:
            if not self.data_manager.load_data():
                return "I apologize, but I was unable to load the thesis database. Please try again later."
        query_type, relevant_theses = self.determine_query_strategy(question)
        if query_type.startswith("year:") and not relevant_theses:
            searched_year = query_type.split(":")[1]
            min_year, max_year = self.data_manager.get_publication_year_range()
            return (f"I'm happy to help you with your request! Unfortunately, it appears that the theses in our database are from {min_year}-{max_year}, "
                    f"but there are no theses from {searched_year}. The most recent theses in our database are from {max_year}. "
                    "If you're interested, I can provide you with information about those theses.")
        if not relevant_theses:
            return "I couldn't find any theses matching your query. Could you please rephrase your question or provide more details?"
        max_results = 20
        if len(relevant_theses) > max_results:
            relevant_theses = relevant_theses[:max_results]
        context = self._format_theses_for_context(relevant_theses, query_type)
        system_prompt = """You are a helpful academic assistant answering questions about university theses.
I will provide thesis information from a database based on the user's question.

Your response should:
1. Directly answer the user's question based on the data provided.
2. Structure your answer clearly and informatively.
3. Mention specific thesis titles, authors, and relevant details.
4. Be honest if the data doesn't contain enough information.
5. Don't make up information that isn't in the data.

You are speaking directly to the user who asked the question.
"""
        if conversation_history:
            system_prompt += "\nConversation History:\n" + conversation_history + "\n"
        user_prompt = f"""
Question: {question}

Here is the thesis data from our database:
{context}
"""
        return run_llm(system_prompt, user_prompt, model=self.model)

def main():
    """Main function to initialize and run the system from the terminal."""
    data_manager = ThesisDataManager()
    if not data_manager.load_data():
        logger.error("Failed to load data. Exiting.")
        return
    rag_system = ThesisRAGSystem(data_manager)
    print("Welcome to the Thesis Search System!")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        start_time = datetime.datetime.now()
        answer = rag_system.answer_question(query)
        end_time = datetime.datetime.now()
        print("\nAnswer:")
        print(answer)
        print(f"\nResponse time: {(end_time - start_time).total_seconds():.2f} seconds")

if __name__ == "__main__":
    main()

