#!/usr/bin/env python3
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
MAX_TOKENS = 6000

# Global prompt definitions
TITLE_ANALYSIS_PROMPT = (
    "You are analyzing questions about academic theses.\n"
    "Determine if this question is asking about a specific thesis by title.\n"
    "Return a JSON object with:\n"
    "{\n"
    '    "query_type": "direct_title" or "title_question" or "not_title",\n'
    '    "confidence": percentage from 0-100,\n'
    '    "extracted_title": "the exact thesis title if present"\n'
    "}\n"
    "Include only the JSON in your response."
)

QUERY_ANALYSIS_PROMPT = (
    "You are a database query analyzer for a thesis database.\n"
    "Analyze this question and determine which query types would be most appropriate.\n"
    "Return a JSON object with:\n"
    "{\n"
    '    "possible_query_types": ["advisor", "author", "department", "topic", "award", "general"],\n'
    '    "query_priority": ["primary_type", "secondary_type", ...],\n'
    '    "entities": {\n'
    '        "person_names": ["full name 1", "full name 2"],\n'
    '        "keywords": ["keyword1", "keyword2"],\n'
    '        "award_terms": ["prize", "award", "recognition", "honor"]\n'
    "    }\n"
    "}\n"
    "Include only the JSON in your response."
)

# Update the ADVISOR_SUGGESTION_PROMPT constant to be more comprehensive
ADVISOR_SUGGESTION_PROMPT = (
    "You are a helpful academic assistant specializing in thesis topic development.\n"
    "Given the user's request, generate: \n"
    "1. 5–7 specific and creative thesis ideas with detailed titles and topic descriptions\n"
    "2. For EACH thesis idea, provide a comprehensive outline with these sections:\n"
    "   - Introduction & Problem Statement\n"
    "   - Literature Review\n"
    "   - Methodology (with specific research methods)\n"
    "   - Data Collection & Analysis Approach\n"
    "   - Expected Results\n" 
    "   - Implications & Significance\n"
    "   - Timeline & Feasibility\n"
    "3. Relevant disciplines and interdisciplinary connections\n"
    "4. Advisor profile suggestions based on the topic (area of expertise)\n"
    "Format each thesis idea as a structured, comprehensive outline suitable for academic planning."
)

TOPIC_ANALYSIS_PROMPT = (
    "You are analyzing academic thesis topic requests.\n"
    "Extract key information from this request to help generate relevant thesis ideas.\n"
    "Return a JSON object with:\n"
    "{\n"
    '    "primary_domain": "the main academic field",\n'
    '    "related_domains": ["other relevant fields"],\n'
    '    "key_themes": ["important themes or concepts"],\n'
    '    "research_approach": "potential research approach if mentioned"\n'
    "}\n"
    "Include only the JSON in your response."
)

REQUEST_TYPE_PROMPT = (
    "You are analyzing user queries about academic theses.\n"
    "Determine if this query is a request for thesis ideas, topics, or outlines.\n"
    "Return a JSON object with:\n"
    "{\n"
    '    "is_thesis_idea_request": true/false,\n'
    '    "confidence": percentage from 0-100\n'
    "}\n"
    "Include only the JSON in your response."
)

def run_llm(system: str, user: str, model: str = DEFAULT_MODEL, seed: Optional[int] = None) -> str:
    combined = system + user
    if len(combined) > MAX_TOKENS:
        allowed = max(0, MAX_TOKENS - len(system))
        user = user[:allowed]
        logger.warning("Prompt truncated to meet token limit.")
    try:
        completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ],
            model=model
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "There was an error generating the response."


def sanitize_query(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)

def clean_name(name: str) -> str:
    """Remove common prefixes and trim whitespace"""
    return re.sub(r'^(Name:\s*)', '', name, flags=re.IGNORECASE).strip()

class ThesisDataManager:
    """Manages thesis data with enhanced error handling"""
    
    def __init__(self, csv_path: str = "merged_theses.csv", db_path: str = "theses.db"):
        self.csv_path = csv_path
        self.db_path = db_path
        self.conn = None
        self.loaded = False
        self._columns_cache = None
        
    def _safe_execute(self, cursor, query, params=(), error_message="Query execution error"):
        """Safe query execution with error handling"""
        try:
            cursor.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"{error_message}: {e}")
            raise

    def csv_to_sqlite(self) -> bool:
        """Convert CSV data to SQLite with robust error management"""
        try:
            chunk_size = 1000
            df_iter = pd.read_csv(self.csv_path, chunksize=chunk_size)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            total_rows = 0
            for i, chunk in enumerate(df_iter):
                try:
                    chunk.to_sql('theses', self.conn, 
                                 if_exists='replace' if i == 0 else 'append', 
                                 index=False)
                    total_rows += len(chunk)
                except Exception as chunk_error:
                    logger.error(f"Chunk processing error {i}: {chunk_error}")
            
            cursor = self.conn.cursor()
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_title ON theses (Title)",
                "CREATE INDEX IF NOT EXISTS idx_author1_lname ON theses (author1_lname)",
                "CREATE INDEX IF NOT EXISTS idx_author1_fname ON theses (author1_fname)",
                "CREATE INDEX IF NOT EXISTS idx_publication_date ON theses (publication_date)",
                "CREATE INDEX IF NOT EXISTS idx_department ON theses (department)"
            ]
            
            for index_sql in indexes:
                self._safe_execute(cursor, index_sql)
            
            # Create FTS table with comprehensive error handling
            try:
                self._safe_execute(cursor, """
                    CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(
                        Title, abstract, keywords, advisor1, advisor2, 
                        author1_fname, author1_lname, department, disciplines,
                        content='theses', content_rowid='rowid'
                    )
                """)
                self._safe_execute(cursor, """
                    INSERT INTO theses_fts(
                        rowid, Title, abstract, keywords, 
                        advisor1, advisor2, author1_fname, 
                        author1_lname, department, disciplines
                    ) 
                    SELECT 
                        rowid, Title, abstract, keywords, 
                        advisor1, advisor2, author1_fname, 
                        author1_lname, department, disciplines 
                    FROM theses
                """)
            except sqlite3.OperationalError as fts_error:
                logger.warning(f"FTS table creation warning: {fts_error}")
            
            self.conn.commit()
            logger.info(f"CSV converted to SQLite: {self.db_path}, Rows: {total_rows}")
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"CSV to SQLite conversion failed: {e}")
            return False

    def load_data(self) -> bool:
        """Load data with enhanced error checking"""
        try:
            if not os.path.exists(self.db_path):
                return self.csv_to_sqlite()
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Verify table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses'")
            if not cursor.fetchone():
                return self.csv_to_sqlite()
            
            # Verify FTS table exists; if not, try to create it
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses_fts'")
            if not cursor.fetchone():
                try:
                    self._safe_execute(cursor, """
                        CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(
                            Title, abstract, keywords, advisor1, advisor2, 
                            author1_fname, author1_lname, department, disciplines,
                            content='theses', content_rowid='rowid'
                        )
                    """)
                except sqlite3.OperationalError as fts_error:
                    logger.error(f"FTS table creation error: {fts_error}")
            
            self.conn.commit()
            logger.info(f"Connected to database: {self.db_path}")
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
   
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute SQL query with robust error handling"""
        if not self.loaded:
            self.load_data()
        
        try:
            cursor = self.conn.cursor()
            logger.debug(f"Executing: {query} with params: {params}")
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                results.append(row_dict)
            
            logger.info(f"Query returned {len(results)} results")
            return results
        
        except sqlite3.Error as e:
            logger.error(f"Query execution error: {e}")
            return []   

    def search_by_year(self, year: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by publication year with error handling"""
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
        """Return the minimum and maximum publication dates"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT MIN(publication_date), MAX(publication_date) FROM theses")
            row = cursor.fetchone()
            return (row[0] or "", row[1] or "") if row else ("", "")
        except Exception as e:
            logger.error(f"Error getting publication year range: {e}")
            return "", ""

    def get_departments(self) -> List[str]:
        """Return distinct department names"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT department FROM theses WHERE department IS NOT NULL AND department != ''")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting departments: {e}")
            return []

    def search_by_advisor(self, advisor_name: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by advisor name with comprehensive matching"""
        results = []
        try:
            query1 = """
            SELECT * 
            FROM theses 
            WHERE advisor1 = ? OR advisor2 = ? OR advisor3 = ?
            ORDER BY publication_date DESC
            LIMIT ?
            """
            results.extend(self.execute_query(query1, (advisor_name, advisor_name, advisor_name, limit)))
            
            if len(results) < limit:
                query2 = """
                SELECT * 
                FROM theses 
                WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?
                ORDER BY publication_date DESC
                LIMIT ?
                """
                results.extend(self.execute_query(query2, (
                    f"%{advisor_name}%", 
                    f"%{advisor_name}%", 
                    f"%{advisor_name}%", 
                    limit - len(results)
                )))
            
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
            if last_name:
                return self.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date FROM theses WHERE author1_lname LIKE ? ORDER BY publication_date DESC LIMIT ?", 
                    (f"%{last_name}%", limit)
                )
            if first_name:
                return self.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date FROM theses WHERE author1_fname LIKE ? ORDER BY publication_date DESC LIMIT ?", 
                    (f"%{first_name}%", limit)
                )
            return []
        except Exception as e:
            logger.error(f"Author search error: {e}")
            return []

    def find_co_advisors(self, advisor_name: str) -> Dict[str, Any]:
        """Find advisors who have co-advised theses with the specified advisor"""
        try:
            query = """
            SELECT * FROM theses
            WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ?
            ORDER BY publication_date DESC
            """
            results = self.execute_query(query, (f"%{advisor_name}%", f"%{advisor_name}%", f"%{advisor_name}%"))
            co_advisors = {}
            theses_details = []

            for thesis in results:
                advisors = [thesis.get('advisor1', ''), thesis.get('advisor2', ''), thesis.get('advisor3', '')]
                # Ensure case-insensitive matching for the primary advisor
                if not any(advisor_name.lower() in a.lower() for a in advisors):
                    continue

                other_advisors = [a for a in advisors if a and advisor_name.lower() not in a.lower()]
                for other in other_advisors:
                    co_advisors[other] = co_advisors.get(other, 0) + 1

                theses_details.append(thesis)

            sorted_co_advisors = sorted(co_advisors.items(), key=lambda x: x[1], reverse=True)
            return {"co_advisors": sorted_co_advisors, "theses": theses_details}
        except Exception as e:
            logger.error(f"Co-advisor search error: {e}")
            return {"co_advisors": [], "theses": []}

    def search_by_award(self, award_term: str = None, department: str = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses that have received awards"""
        try:
            base_query = """
            SELECT * FROM theses 
            WHERE award IS NOT NULL AND award != '' 
            """
            conditions = []
            params = []
            
            if award_term:
                conditions.append("award LIKE ?")
                params.append(f"%{award_term}%")
            if department:
                conditions.append("department LIKE ?")
                params.append(f"%{department}%")
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            base_query += " ORDER BY publication_date DESC LIMIT ?"
            params.append(limit)
            
            return self.execute_query(base_query, tuple(params))
        except Exception as e:
            logger.error(f"Award search error: {e}")
            return []
    
    def search_by_department(self, department: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by department and return advisor names too"""
        try:
            return self.execute_query(
                """
                SELECT Title,
                       department,
                       publication_date,
                       advisor1,
                       advisor2,
                       advisor3,
                       author1_fname,
                       author1_lname
                FROM theses
                WHERE department LIKE ?
                ORDER BY publication_date DESC
                LIMIT ?
                """,
                (f"%{department}%", limit)
            )
        except Exception as e:
            logger.error(f"Department search error: {e}")
            return []

    def search_by_keyword(self, keyword: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search the theses database for a keyword, returning up to *limit* rows.

        First tries the FTS5 virtual table for fast full‑text search; if that fails
        (e.g., FTS not available) we fall back to a standard LIKE query.
        The result rows include advisor1/2/3 so downstream code can surface real
        advisor names for thesis‑idea generation.
        """
        # Allow only alphanumerics + whitespace inside the MATCH query
        escaped_keyword = re.sub(r"[^\w\s]", "", keyword)

        # ---------- Fast FTS branch ----------
        try:
            fts_query = f"""
                SELECT  t.Title,
                        t.department,
                        t.publication_date,
                        t.advisor1,
                        t.advisor2,
                        t.advisor3,
                        t.author1_fname,
                        t.author1_lname
                FROM    theses            AS t
                JOIN    theses_fts        AS fts  ON t.rowid = fts.rowid
                WHERE   theses_fts MATCH '{escaped_keyword}'
                ORDER BY t.publication_date DESC
                LIMIT   {limit}
            """
            results = self.execute_query(fts_query)
            if results:
                return results
        except Exception:
            logger.warning("FTS search failed in search_by_keyword – falling back to LIKE")

        # ---------- Fallback LIKE branch ----------
        fallback_query = """
            SELECT  Title,
                    department,
                    publication_date,
                    advisor1,
                    advisor2,
                    advisor3,
                    author1_fname,
                    author1_lname
            FROM    theses
            WHERE   Title    LIKE ?
                OR  abstract LIKE ?
                OR  keywords LIKE ?
            ORDER BY publication_date DESC
            LIMIT   ?
        """
        return self.execute_query(
            fallback_query,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit)
        )
        
   

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
        # Exclude non–text columns (you may adjust this list as needed)
        text_columns = [col for col in columns if col not in ['rowid', 'publication_date']]
        try:
            # Use FTS if available (escape punctuation)
            escaped_search_term = re.sub(r'[^\w\s]', '', search_term)
            fts_query = (
                f"SELECT t.* FROM theses t "
                f"JOIN theses_fts fts ON t.rowid = fts.rowid "
                f"WHERE theses_fts MATCH '{escaped_search_term}' "
                f"ORDER BY t.publication_date DESC LIMIT {limit}"
            )
            results = self.execute_query(fts_query)
            if results:
                return results
        except Exception:
            logger.warning("FTS failed in search_all_columns, falling back to LIKE")
            pass
        where_clauses = [f'"{col}" LIKE ?' for col in text_columns]
        params = [f"%{search_term}%" for _ in text_columns]
        query = (
            f"SELECT * FROM theses WHERE {' OR '.join(where_clauses)} "
            f"ORDER BY publication_date DESC LIMIT {limit}"
        )
        return self.execute_query(query, tuple(params))

    def count_theses_by_criteria(self, criteria: Dict[str, str]) -> int:
        """Count theses based on given criteria"""
        try:
            valid_columns = self.get_columns()
            where_clauses = []
            params = []
            for column, value in criteria.items():
                if value and column in valid_columns:
                    where_clauses.append(f"{column} LIKE ?")
                    params.append(f"%{value}%")
                else:
                    logger.warning(f"Ignoring invalid column: {column}")
            query = "SELECT COUNT(*) FROM theses"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                cursor = self.conn.cursor()
                cursor.execute(query, tuple(params))
            else:
                cursor = self.conn.cursor()
                cursor.execute(query)
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Count query error: {e}")
            return 0

    def count_exact_criteria(self, column: str, value: str) -> int:
        """Count theses with an exact match for a specific column value"""
        try:
            query = f"SELECT COUNT(*) FROM theses WHERE {column} = ?"
            cursor = self.conn.cursor()
            cursor.execute(query, (value,))
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Exact count query error: {e}")
            return 0

class ThesisRAGSystem:
    """RAG system for answering questions about thesis data"""
    
    def __init__(self, data_manager: ThesisDataManager, model: str = DEFAULT_MODEL):
        self.data_manager = data_manager
        self.model = model
        self.latest_thesis_list = []

    # ------------------------------------------------------------------
    # Helper ‑‑ thesis ideas + detailed outline + real advisors
    # ------------------------------------------------------------------
    def generate_thesis_help(self, topic_request: str) -> str:
        """
        Produce 5‑7 thesis ideas on the requested topic using LLM analysis.
        For every idea include:
          • Title
          • Comprehensive section-by-section outline
          • Advisor: <name picked from DB>   — or expertise description if none found
        """
        # --- 1. Analyze the topic request with LLM to extract domains and themes ---
        topic_analysis = run_llm(TOPIC_ANALYSIS_PROMPT, topic_request, model=self.model)
        
        try:
            # Extract the JSON from the response
            json_match = re.search(r'({.*})', topic_analysis, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(1))
            else:
                # Fallback if JSON extraction fails
                analysis = {
                    "primary_domain": "",
                    "related_domains": [],
                    "key_themes": [],
                    "research_approach": ""
                }
        except Exception as e:
            logger.error(f"Error parsing topic analysis: {e}")
            analysis = {
                "primary_domain": "",
                "related_domains": [],
                "key_themes": [],
                "research_approach": ""
            }
        
        # Extract the primary domain and key themes
        primary_domain = analysis.get("primary_domain", "")
        key_themes = analysis.get("key_themes", [])
        
        # --- 2. Search for related advisors based on domain and themes ---
        advisor_pool = []
        
        # First search by domain if available
        if primary_domain:
            domain_results = self.data_manager.search_by_department(primary_domain, limit=30)
            for r in domain_results:
                advisor_pool += [
                    r.get("advisor1", ""), r.get("advisor2", ""), r.get("advisor3", "")
                ]
        
        # Then search by key themes
        for theme in key_themes:
            if theme and len(theme) > 3:
                rows = self.data_manager.search_by_keyword(theme, limit=20)
                for r in rows:
                    advisor_pool += [
                        r.get("advisor1", ""), r.get("advisor2", ""), r.get("advisor3", "")
                    ]
        
        # Also search for advisors using the entire request as context
        general_results = self.data_manager.search_all_columns(topic_request, limit=30)
        for r in general_results:
            advisor_pool += [
                r.get("advisor1", ""), r.get("advisor2", ""), r.get("advisor3", "")
            ]

        # Clean and deduplicate advisors
        advisor_pool = [a.strip() for a in advisor_pool if a and a.strip()]
        from collections import Counter
        advisor_counter = Counter(advisor_pool)
        top_advisors = [name for name, _ in advisor_counter.most_common(12) if name]


        # --- 3. Get relevant thesis examples ---
        relevant_examples = []
        
        # First try to get examples from the primary domain
        if primary_domain:
            domain_examples = self.data_manager.search_by_department(primary_domain, limit=5)
            if domain_examples:
                relevant_examples.extend(domain_examples)
        
        # Then add examples based on key themes
        for theme in key_themes:
            if theme and len(theme) > 3 and len(relevant_examples) < 8:
                theme_examples = self.data_manager.search_by_keyword(theme, limit=3)
                if theme_examples:
                    relevant_examples.extend(theme_examples)
        
        # Format examples for context
        examples_text = ""
        if relevant_examples:
            examples_text = "\n\nRelevant thesis examples from the database:\n"
            seen_titles = set()
            count = 0
            
            for thesis in relevant_examples:
                title = thesis.get("Title", "")
                if title and title not in seen_titles and count < 5:
                    seen_titles.add(title)
                    author = f"{thesis.get('author1_fname', '')} {thesis.get('author1_lname', '')}".strip()
                    department = thesis.get("department", "")
                    examples_text += f"- \"{title}\" by {author} ({department})\n"
                    count += 1
        
        # --- 4. Build advisor context ---
        if top_advisors:
            advisor_block = (
                "### Relevant advisors found in the thesis database:\n"
                + "\n".join(f"- {name}" for name in top_advisors)
            )
            advisor_rule = (
                "For **every** thesis idea below, finish with an **Advisor:** line "
                "choosing a single name *only* from the list above. Match advisor expertise to the thesis topic."
            )
        else:
            advisor_block = "*No specific advisors found in the database for this topic.*"
            advisor_rule = (
                "Since no specific advisors were found, add an "
                "**Advisor Expertise Profile:** section describing the ideal expertise for each thesis idea."
            )
        
        # --- 5. Compose final prompts ---
        system_prompt = (
            ADVISOR_SUGGESTION_PROMPT
            + "\n\n"
            + advisor_rule
            + "\n\nFormat each thesis idea comprehensively with all required sections."
        )
        
        user_prompt = (
            f"User's thesis request: {topic_request}\n\n"
            f"Topic analysis:\n"
            f"- Primary domain: {primary_domain}\n"
            f"- Key themes: {', '.join(key_themes)}\n\n"
            f"{advisor_block}{examples_text}"
        )

        # Generate the ideas with detailed structure
        ideas_text = run_llm(system_prompt, user_prompt, model=self.model)
        
        # --- 6. Add header context ---
        header = (
            f"**Domain:** {primary_domain}\n"
            f"**Key Themes:** {', '.join(key_themes)}\n\n"
            f"{advisor_block}\n\n"
            f"---\n\n"
        )
        
        return header + ideas_text   


    def determine_query_strategy(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Determine best query strategy based on question and execute it"""
        response = run_llm(TITLE_ANALYSIS_PROMPT, question, model=self.model)
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

        response = run_llm(QUERY_ANALYSIS_PROMPT, question, model=self.model)
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
            elif query_type in ["topic", "general"]:
                for keyword in keywords:
                    keyword_results = self.data_manager.search_by_keyword(keyword)
                    if keyword_results:
                        results.extend(keyword_results)
                        successful_query_type = query_type
            elif query_type == "award":
                award_terms = query_info.get("entities", {}).get("award_terms", [])
                if not award_terms:
                    award_results = self.data_manager.search_by_award()
                    results.extend(award_results)
                    successful_query_type = "award"
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
        """Format retrieved theses in Markdown"""
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
            award = thesis.get('award', '')
            if award:
                lines.append("")
                lines.append(f"**Award:** {award}")
            if idx < len(theses) - 1:
                lines.append("\n---\n")
    
        return "\n".join(lines)

    def answer_question(self, question: str, conversation_history: str = "") -> str:
        """Answer a question about theses, incorporating conversation history."""
           
        quick_check_terms = ["thesis idea", "thesis topic", "thesis outline", "brainstorm"]
        if any(term in question.lower() for term in quick_check_terms):
            return self.generate_thesis_help(question)
        
        # For less obvious cases, use LLM to analyze the request type
        # We'll only do this if there's any mention of "thesis" to save API calls
        if "thesis" in question.lower() or "dissertation" in question.lower():
            request_analysis = run_llm(REQUEST_TYPE_PROMPT, question, model=self.model)
            try:
                json_match = re.search(r'({.*})', request_analysis, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                    if analysis.get("is_thesis_idea_request", False) and analysis.get("confidence", 0) > 70:
                        return self.generate_thesis_help(question)
            except Exception as e:
                logger.error(f"Error parsing request analysis: {e}")
        
        # Continue with the existing code for other types of questions...
        # Handle co-advisor queries
        co_advisor_patterns = [
            r"who has co.?advised .* with ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"who co.?advises with ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+)'s co.?advisors"
        ]

        # ---------- Thesis brainstorming shortcut ----------
        brainstorm_triggers = [
            "thesis ideas", "thesis help", "brainstorm", "outline",
            "topic suggestions", "what should i write my thesis on",
            "give me ideas for my thesis", "suggest a thesis topic"
        ]
        if any(trigger in question.lower() for trigger in brainstorm_triggers):
            return self.generate_thesis_help(question)

        for pattern in co_advisor_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                advisor_name = match.group(1)
                co_advisor_data = self.data_manager.find_co_advisors(advisor_name)
                if co_advisor_data["co_advisors"]:
                    context = f"## Co-advisors with {advisor_name}\n\n"
                    for co_advisor, count in co_advisor_data["co_advisors"]:
                        context += f"- {co_advisor}: co-advised {count} theses\n"
                    
                    context += "\n## Co-advised Theses\n\n"
                    for i, thesis in enumerate(co_advisor_data["theses"][:10], 1):
                        context += f"{i}. **Title:** {thesis.get('Title', 'Unknown')}\n"
                        context += f"   **Authors:** {thesis.get('author1_fname', '')} {thesis.get('author1_lname', '')}\n" 
                        context += f"   **Department:** {thesis.get('department', 'Unknown')}\n"
                        context += f"   **Year:** {thesis.get('publication_date', 'Unknown')}\n"
                        context += f"   **Primary Advisor:** {thesis.get('advisor1', '')}\n"
                        if thesis.get('advisor2'):
                            context += f"   **Secondary Advisor:** {thesis.get('advisor2', '')}\n"
                        context += "\n"
                    
                    system_prompt = (
                        "You are a helpful academic assistant answering questions about co-advisorship in university theses.\n"
                        "Analyze the provided data and answer the user's question, focusing on advisor relationships."
                    )
                    return run_llm(system_prompt, f"Question: {question}\n\nData: {context}", model=self.model)
        
        # Handle award-related queries
        if any(term in question.lower() for term in ["award", "prize", "recognition", "honor", "awarded"]):
            dept_match = re.search(r"in ([a-zA-Z ]+)(?:\?|$|\.)", question)
            department = dept_match.group(1) if dept_match else None
            award_results = self.data_manager.search_by_award(department=department)
            if award_results:
                context = self._format_theses_for_context(award_results, "award")
                system_prompt = (
                    "You are a helpful academic assistant answering questions about university theses awards.\n"
                    "Focus on identifying trends and details in awarded theses."
                )
                return run_llm(system_prompt, f"Question: {question}\n\nData: {context}", model=self.model)
        
        # Handling positional queries for the thesis list from the previous response.
        if re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b thesis", question.lower()):
            position_words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            try:
                index = position_words.index([w for w in position_words if w in question.lower()][0])
            except IndexError:
                index = 0
            thesis_list = self.latest_thesis_list
            if 0 <= index < len(thesis_list):
                selected = thesis_list[index]
                return f"Here's what the **{position_words[index]} thesis** is about:\n\n" + self._format_theses_for_context([selected], "follow-up")
            else:
                return "I couldn't find that thesis position from the last response. Try rephrasing or asking again."

        # Handle counting queries
        if any(phrase in question.lower() for phrase in ["how many", "count", "number of"]):
            return self.count_matching_theses(question)

        # Handle thesis topic recommendations
        recommendation_indicators = ["recommend", "suggest", "ideas", "topics", "thesis ideas", "thesis topic", "thesis suggestion"]
        if any(indicator in question.lower() for indicator in recommendation_indicators):
            return self.generate_thesis_recommendations(question)

        if not self.data_manager.loaded:
            if not self.data_manager.load_data():
                return "I apologize, but I was unable to load the thesis database. Please try again later."

        query_type, relevant_theses = self.determine_query_strategy(question)
        if query_type.startswith("year:") and not relevant_theses:
            searched_year = query_type.split(":")[1]
            min_year, max_year = self.data_manager.get_publication_year_range()
            return (
                f"I'm happy to help! It appears the theses in our database cover {min_year} to {max_year}, "
                f"but there are no theses from {searched_year}. The most recent theses are from {max_year}."
            )

        if not relevant_theses:
            return "I couldn't find any theses matching your query. Could you please rephrase your question or provide more details?"

        max_results = 20
        if len(relevant_theses) > max_results:
            relevant_theses = relevant_theses[:max_results]

        context = self._format_theses_for_context(relevant_theses, query_type)
        system_prompt = (
            "You are a helpful academic assistant answering questions about university theses.\n"
            "Please provide a clear and detailed answer based on the data provided below.\n"
            "Mention specific thesis titles, authors, and any award details if available."
        )
        if conversation_history:
            system_prompt += "\nConversation History:\n" + conversation_history + "\n"

        user_prompt = f"Question: {question}\n\nThesis Data:\n{context}"
        self.latest_thesis_list = relevant_theses
        return run_llm(system_prompt, user_prompt, model=self.model)

    def count_matching_theses(self, question: str) -> str:
        """Handle counting queries about theses"""
        # Extract department if mentioned
        department_match = re.search(r'(biology|economics|computer science|comp sci|cs|engineering|physics|chemistry|mathematics|math|english|history|psychology|sociology|political science|art|music)', question.lower())
        department = department_match.group(1) if department_match else None
        department_mapping = {
            'comp sci': 'computer science',
            'cs': 'computer science',
            'math': 'mathematics'
        }
        if department in department_mapping:
            department = department_mapping[department]
        year_match = re.search(r'\b(19|20)\d{2}\b', question)
        year = year_match.group(0) if year_match else None
        
        query_criteria = {}
        if department:
            query_criteria["department"] = department
        if year:
            query_criteria["publication_date"] = year
        
        count = self.data_manager.count_theses_by_criteria(query_criteria)
        if department and year:
            response = (f"There are {count} {department} theses published in {year} according to our records.")
        elif department:
            response = (f"There are {count} {department} theses in our database.")
        elif year:
            response = (f"There are {count} theses published in {year}.")
        else:
            response = f"There are {count} theses in our database."
        return response

    def generate_thesis_recommendations(self, topic_request: str) -> str:
        """Generate thesis topic recommendations based on user request"""
        prompt = f"Extract key themes from the following thesis topic request as a JSON array: {topic_request}"
        themes_response = run_llm("You extract key themes from thesis topic requests.", prompt, model=self.model)
        try:
            themes = json.loads(re.search(r'\[.*\]', themes_response).group())
        except Exception:
            themes = [word for word in topic_request.lower().split() if len(word) > 3 
                      and word not in {"thesis", "idea", "topic", "about", "essay", "give", "recommend"}]
        related_theses = []
        for theme in themes:
            results = self.data_manager.search_by_keyword(theme, limit=5)
            if results:
                related_theses.extend(results)
        context = ""
        if related_theses:
            context = "Here are some existing theses on related topics:\n\n"
            for thesis in related_theses[:10]:
                title = thesis.get('Title', '')
                abstract = thesis.get('abstract', '')
                if abstract and len(abstract) > 200:
                    abstract = abstract[:197] + "..."
                context += f"- {title}\n  {abstract}\n\n"
        system_prompt = (
            "You are an academic advisor specializing in thesis topic development.\n"
            "Generate 5-7 creative and academically rigorous thesis topic recommendations. For each recommendation, provide:\n"
            "1. A specific thesis title\n"
            "2. A brief description of the topic and its significance\n"
            "3. Suggested research approaches\n"
            "4. Relevant disciplines and any cross-disciplinary connections\n"
        )
        if "outline" in topic_request.lower():
            system_prompt += "Also provide a detailed outline including an introduction, literature review, methodology, expected findings, and conclusion.\n"
        user_prompt = f"Request: {topic_request}\n\n{context}"
        return run_llm(system_prompt, user_prompt, model=self.model)

# Main interactive loop
def main():
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
        # Use the public answer_question method
        answer = rag_system.answer_question(query)
        end_time = datetime.datetime.now()
        print("\nAnswer:")
        print(answer)
        print(f"\nResponse time: {(end_time - start_time).total_seconds():.2f} seconds")

if __name__ == "__main__":
    main()

