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
    '''Helper function for all LLM uses'''
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
    
    # Fallback to simpler analysis
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
            "columns_needed": ["Title", "abstract", "department", "publication_date, "advisor1", "advisor2"]
        }

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
            # Read CSV in chunks
            chunk_size = 1000
            df_iter = pd.read_csv(self.csv_path, chunksize=chunk_size)
            
            self.conn = sqlite3.connect(self.db_path)
            
            for i, chunk in enumerate(df_iter):
                chunk.to_sql('theses', self.conn, if_exists='replace' if i==0 else 'append', index=False)
            
            # Create indices
            cursor = self.conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON theses (Title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_author1_lname ON theses (author1_lname)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_author1_fname ON theses (author1_fname)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_publication_date ON theses (publication_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_department ON theses (department)")
            
            # Add full-text search
            cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines, content='theses', content_rowid='rowid')")
            cursor.execute("INSERT INTO theses_fts(rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines) SELECT rowid, Title, disciplines, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department FROM theses")
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
            
            # Connect to existing database
            self.conn = sqlite3.connect(self.db_path)
            
            # Verify tables exist
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses'")
            if not cursor.fetchone():
                return self.csv_to_sqlite()
            
            # Check for FTS table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses_fts'")
            if not cursor.fetchone():
                try:
                    cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines, content='theses', content_rowid='rowid')")
                    cursor.execute("INSERT INTO theses_fts(rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department) SELECT rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, disciplines, FROM theses")
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
    
    # Specialized search functions
    def search_by_advisor(self, advisor_name: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by advisor name"""
        results = []
        
        try:
            # Exact match
            query1 = """
            SELECT Title, department, publication_date, advisor1, advisor2 
            FROM theses 
            WHERE advisor1 = ? OR advisor2 = ? OR advisor3 = ?
            LIMIT ?
            """
            results.extend(self.execute_query(query1, (advisor_name, advisor_name, advisor_name, limit)))
            
            # Partial match
            if len(results) < limit:
                query2 = """
                SELECT Title, department, publication_date, advisor1, advisor2 
                FROM theses 
                WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ? 
                LIMIT ?
                """
                results.extend(self.execute_query(query2, (f"%{advisor_name}%", f"%{advisor_name}%", f"%{advisor_name}%", limit)))
            
            # Try with name parts
            if len(results) < limit and ' ' in advisor_name:
                for part in advisor_name.split():
                    if len(part) > 2:
                        query3 = """
                        SELECT Title, department, publication_date, advisor1, advisor2 
                        FROM theses 
                        WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ? 
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
                LIMIT ?
                """
                results = self.execute_query(query, (first_name, last_name, limit))
                
                if not results:
                    query = """
                    SELECT Title, author1_fname, author1_lname, department, publication_date
                    FROM theses 
                    WHERE author1_fname LIKE ? AND author1_lname LIKE ? 
                    LIMIT ?
                    """
                    results = self.execute_query(query, (f"%{first_name}%", f"%{last_name}%", limit))
                
                return results
            elif last_name:
                return self.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date FROM theses WHERE author1_lname LIKE ? LIMIT ?", 
                    (f"%{last_name}%", limit)
                )
            elif first_name:
                return self.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date FROM theses WHERE author1_fname LIKE ? LIMIT ?", 
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
                "SELECT Title, department, publication_date, author1_fname, author1_lname FROM theses WHERE department LIKE ? LIMIT ?", 
                (f"%{department}%", limit)
            )
        except Exception as e:
            logger.error(f"Department search error: {e}")
            return []
    
    def search_by_keyword(self, keyword: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by keyword in title or abstract"""
        try:
            # Try full-text search first
            try:
                fts_query = """
                SELECT t.Title, t.department, t.publication_date, t.author1_fname, t.author1_lname
                FROM theses t
                JOIN theses_fts fts ON t.rowid = fts.rowid
                WHERE theses_fts MATCH ?
                LIMIT ?
                """
                results = self.execute_query(fts_query, (keyword, limit))
                if results:
                    return results
            except Exception:
                logger.warning("FTS failed, falling back to LIKE")
            
            # Fall back to regular search
            return self.execute_query(
                "SELECT Title, department, publication_date, author1_fname, author1_lname FROM theses WHERE Title LIKE ? OR abstract LIKE ? OR keywords LIKE ? LIMIT ?", 
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit)
            )
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def search_by_exact_title(self, title: str) -> List[Dict[str, Any]]:
        """Search for thesis by exact title"""
        try:
            # Exact match first
            results = self.execute_query("SELECT * FROM theses WHERE Title = ? LIMIT 5", (title,))
            
            # Try case insensitive
            if not results:
                results = self.execute_query("SELECT * FROM theses WHERE Title LIKE ? LIMIT 5", (title,))
                
            # Try partial match
            if not results:
                partial_title = title[:min(50, len(title))]
                results = self.execute_query("SELECT * FROM theses WHERE Title LIKE ? LIMIT 10", (f"%{partial_title}%",))
                
            return results
        except Exception as e:
            logger.error(f"Exact title search error: {e}")
            return []
    
    def search_all_columns(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search across all text columns"""
        if not self.loaded:
            self.load_data()
        
        columns = self.get_columns()
        text_columns = [col for col in columns if col not in ['rowid', 'publication_date']]
            
        try:
            # Try full-text search first
            try:
                fts_query = """
                SELECT t.*
                FROM theses t
                JOIN theses_fts fts ON t.rowid = fts.rowid
                WHERE theses_fts MATCH ?
                LIMIT ?
                """
                results = self.execute_query(fts_query, (search_term, limit))
                if results:
                    return results
            except Exception:
                logger.warning("FTS failed, falling back to LIKE")
                
            # Build query across all text columns
            where_clauses = [f'"{col}" LIKE ?' for col in text_columns]
            params = [f"%{search_term}%" for _ in text_columns]
                
            query = f"""
            SELECT * FROM theses 
            WHERE {" OR ".join(where_clauses)}
            LIMIT {limit}
            """
            
            return self.execute_query(query, tuple(params))
        except Exception as e:
            logger.error(f"All-columns search error: {e}")
            return []

class ThesisRAGSystem:
    """RAG system for answering questions about thesis data"""
    
    def __init__(self, data_manager: ThesisDataManager, model: str = DEFAULT_MODEL):
        self.data_manager = data_manager
        self.model = model
            
    def determine_query_strategy(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Determine best query strategy based on question and execute it"""
        # First use LLM to analyze if the query might be a direct title
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
        
        Include only the JSON in your response.
        """
        
        response = run_llm(title_analysis_prompt, question)
        
        # Parse the response
        try:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                title_info = json.loads(json_match.group(1))
                
                # If LLM thinks this is a direct title or title question with high confidence
                if (title_info.get("query_type") in ["direct_title", "title_question"] and 
                    title_info.get("confidence", 0) > 70 and
                    title_info.get("extracted_title")):
                    
                    extracted_title = title_info.get("extracted_title")
                    logger.info(f"LLM detected potential title: {extracted_title}")
                    title_results = self.data_manager.search_by_exact_title(extracted_title)
                    
                    if title_results:
                        return "exact_title", title_results
                        
                    # If nothing found with extracted title but query type is direct_title,
                    # try the original query as a direct title search
                    if title_info.get("query_type") == "direct_title" and extracted_title != question:
                        title_results = self.data_manager.search_by_exact_title(question)
                        if title_results:
                            return "exact_title", title_results
        except Exception as e:
            logger.error(f"Error parsing title analysis: {e}")
        
        # Fall back to original strategies
        # Check for exact title search using pattern matching
        exact_title_results = self._check_for_exact_title_search(question)
        if exact_title_results:
            return "exact_title", exact_title_results
        
        # Continue with the rest of the original method unchanged...
        # Use LLM to analyze the question
        query_analysis_prompt = """You are a database query analyzer for a thesis database.
        Analyze this question and determine which query types would be most appropriate.
        Return a JSON object with:
        {
            "possible_query_types": ["advisor", "author", "department", "topic", "general"],
            "query_priority": ["primary_type", "secondary_type", ...],
            "entities": {
                "person_names": ["full name 1", "full name 2"],
                "keywords": ["keyword1", "keyword2"]
            }
        }
        Include only the JSON in your response.
        """
        
        response = run_llm(query_analysis_prompt, question)
        
        # Extract JSON from response
        try:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                query_info = json.loads(json_match.group(1))
            else:
                query_info = {
                    "query_priority": ["general"],
                    "entities": {"person_names": [], "keywords": []}
                }
        except Exception:
            query_info = {
                "query_priority": ["general"],
                "entities": {"person_names": [], "keywords": []}
            }       
        # Get query priorities and entities
        query_priority = query_info.get("query_priority", ["general"])
        person_names = query_info.get("entities", {}).get("person_names", [])
        keywords = query_info.get("entities", {}).get("keywords", [])
        
        # Extract names if LLM didn't find any
        if not person_names:
            person_names.extend(re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', question))
        
        # Extract title fragments
        keywords.extend(re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+){2,})', question))
        
        # Extract other potential keywords
        if len(keywords) < 3:
            keywords.extend([word for word in question.split() if len(word) > 4 and word[0].isupper()])
        
        logger.info(f"Query analysis: priorities={query_priority}, names={person_names}, keywords={keywords}")
        
        # Try each query type in priority order
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
            
            # If we found results, add them
            if results:
                all_results.extend(results)
                if len(all_results) > 20:
                    break
        
        # Try comprehensive search if needed
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
        
        # Fallback if no results
        if not all_results:
            all_results = self._fallback_retrieval(question)
            successful_query_type = "fallback"
        
        # Remove duplicates
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
            r'looking for ["\'](.*?)["\']'
        ]
        
        for pattern in title_patterns:
            matches = re.search(pattern, question, re.IGNORECASE)
            if matches and len(matches.groups()) > 1:
                title = matches.group(2)
                logger.info(f"Detected possible title search: {title}")
                results = self.data_manager.search_by_exact_title(title)
                if results:
                    return results
        
        return []
        
    def _fallback_retrieval(self, question: str) -> List[Dict[str, Any]]:
        """Fallback retrieval strategy when main strategies fail"""
        # Extract important words
        stop_words = {"the", "a", "an", "in", "of", "for", "about", "with", "by", "to", 
                      "what", "who", "how", "when", "where", "which", "thesis", "theses", "dissertation"}
        words = [word.strip(',.?!:;()[]{}') for word in question.lower().split()]
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        results = []
        # Try each keyword
        for keyword in keywords:
            keyword_results = self.data_manager.search_all_columns(keyword, limit=10)
            if keyword_results:
                results.extend(keyword_results)
                if len(results) >= 30:
                    break
        
        # Last resort: recent theses
        if not results:
            try:
                results = self.data_manager.execute_query(
                    "SELECT Title, author1_fname, author1_lname, department, publication_date, abstract FROM theses ORDER BY publication_date DESC LIMIT 20"
                )
            except Exception as e:
                logger.error(f"Fallback retrieval error: {e}")
        
        return results
    
    def answer_question(self, question: str) -> str:
        """Answer a question about theses"""
        if not self.data_manager.loaded:
            if not self.data_manager.load_data():
                return "I apologize, but I was unable to load the thesis database. Please try again later."
        
        # Get relevant theses
        query_type, relevant_theses = self.determine_query_strategy(question)
        
        if not relevant_theses:
            return "I couldn't find any theses matching your query. Could you please rephrase your question or provide more details?"
        
        # Limit results to avoid context length issues
        max_results = 20
        if len(relevant_theses) > max_results:
            relevant_theses = relevant_theses[:max_results]
        
        # Format theses for LLM
        context = self._format_theses_for_context(relevant_theses, query_type)
        
        system_prompt = """You are a helpful academic assistant answering questions about university theses.
        I will provide thesis information from a database based on the user's question.
        
        Your response should:
        1. Directly answer the user's question based on the data provided
        2. Structure your answer clearly and informatively
        3. Mention specific thesis titles, authors, and relevant details
        4. Be honest if the data doesn't contain enough information
        5. Don't make up information that isn't in the data
        
        You are speaking directly to the user who asked the question.
        """
        
        user_prompt = f"""
        Question: {question}
        
        Here is the thesis data from our database:
        {context}
        """
        
        return run_llm(system_prompt, user_prompt, model=self.model)
    
    def _format_theses_for_context(self, theses: List[Dict[str, Any]], query_type: str) -> str:
        """Format retrieved theses for use in LLM context"""
        result = f"[Found {len(theses)} relevant theses]\n\n"
        
        for i, thesis in enumerate(theses, 1):
            result += f"Thesis {i}:\n"
            
            # Core fields
            title = thesis.get('Title', 'Unknown title')
            department = thesis.get('department', 'Unknown department')
            
            result += f"Title: {title}\n"
            result += f"Department: {department}\n"
            
            # Author info
            first_name = thesis.get('author1_fname', '')
            last_name = thesis.get('author1_lname', '')
            if first_name or last_name:
                result += f"Author: {first_name} {last_name}\n"
            
            # Publication date
            pub_date = thesis.get('publication_date', '')
            if pub_date:
                result += f"Publication date: {pub_date}\n"
            
            # Advisor info
            advisor1 = thesis.get('advisor1', '')
            advisor2 = thesis.get('advisor2', '')
            if advisor1 and (query_type == 'advisor' or i <= 5):
                result += f"Primary advisor: {advisor1}\n"
            if advisor2 and (query_type == 'advisor' or i <= 5):
                result += f"Secondary advisor: {advisor2}\n"
            
            # Abstract for first few results
            abstract = thesis.get('abstract', '')
            if abstract and (i <= 3 or query_type in ['exact_title', 'comprehensive']):
                if len(abstract) > 500:
                    abstract = abstract[:497] + "..."
                result += f"Abstract: {abstract}\n"
            
            # Additional fields
            degree = thesis.get('degree', '')
            if degree:
                result += f"Degree: {degree}\n"
            
            uri = thesis.get('uri', '')
            if uri:
                result += f"URI: {uri}\n"
            
            result += "\n"
        
        return result

def main():
    """Main function to initialize and run the system"""
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
