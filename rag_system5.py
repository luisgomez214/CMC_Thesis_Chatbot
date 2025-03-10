from urllib.parse import urlparse
import datetime
import logging
import re
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

import groq
from groq import Groq
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

################################################################################
# LLM Configuration
################################################################################

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Default model to use
DEFAULT_MODEL = 'llama3-8b-8192'

################################################################################
# LLM functions
################################################################################

def run_llm(system: str, user: str, model: str = DEFAULT_MODEL, seed: Optional[int] = None) -> str:
    '''
    Helper function for all the uses of LLMs in this file.
    
    Args:
        system: System prompt for the LLM
        user: User query for the LLM
        model: Model to use for completion
        seed: Optional seed for reproducibility
        
    Returns:
        Generated text from the LLM
    '''
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': system,
                },
                {
                    "role": "user",
                    "content": user,
                }
            ],
            model=model,
            seed=seed,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in LLM call: {e}")
        return f"Error: {str(e)}"

################################################################################
# Query Analysis Functions
################################################################################

def analyze_query_type(question: str) -> Dict[str, Any]:
    """
    Analyze the user query to determine the appropriate search strategy.
    
    Args:
        question: User's question
        
    Returns:
        Dictionary with query type and relevant entities
    """
    system_prompt = """You are a query analyzer for a thesis database. 
    Your task is to determine what kind of information the user is looking for.
    
    Output a JSON object with the following structure:
    {
        "query_type": "advisor", "author", "department", "topic", "year", or "general",
        "entities": [list of relevant names, terms, or years mentioned],
        "columns_needed": [list of likely needed columns, keep this minimal],
        "is_exact_title_search": true/false,
        "exact_title": "full title if user is looking for a specific thesis" 
    }
    
    Include only the JSON in your response, no other text.
    """
    
    response = run_llm(system_prompt, question)
    
    # Try to extract JSON from the response
    try:
        # Find JSON pattern in the response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            import json
            query_info = json.loads(json_match.group(1))
            return query_info
    except Exception as e:
        logger.error(f"Error parsing query analysis: {e}")
    
    # Fallback to a simpler analysis if JSON parsing fails
    if "advisor" in question.lower():
        return {
            "query_type": "advisor",
            "entities": [e.strip() for e in question.split() if len(e) > 3],
            "columns_needed": ["Title", "department", "publication_date", "advisor1", "advisor2"]
        }
    elif any(term in question.lower() for term in ["author", "student", "wrote"]):
        return {
            "query_type": "author",
            "entities": [e.strip() for e in question.split() if len(e) > 3],
            "columns_needed": ["Title", "publication_date", "author1_fname", "author1_lname", "department"]
        }
    else:
        return {
            "query_type": "general",
            "entities": [e.strip() for e in question.split() if len(e) > 3],
            "columns_needed": ["Title", "abstract", "department", "publication_date"]
        }

################################################################################
# Data Loading and Management
################################################################################

class ThesisDataManager:
    """Manages loading and accessing thesis data from CSV to SQLite."""
    
    def __init__(self, csv_path: str = "merged_theses.csv", db_path: str = "theses.db"):
        """
        Initialize the data manager with the path to the CSV file and SQLite database.
        
        Args:
            csv_path: Path to the CSV file containing thesis data
            db_path: Path to the SQLite database to create or use
        """
        self.csv_path = csv_path
        self.db_path = db_path
        self.conn = None
        self.loaded = False
        self._columns_cache = None
        
    def csv_to_sqlite(self) -> bool:
        """
        Convert CSV data to SQLite database.
        
        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            # Read CSV in chunks to avoid memory issues
            chunk_size = 1000
            df_iter = pd.read_csv(self.csv_path, chunksize=chunk_size)
            
            # Connect to SQLite database
            self.conn = sqlite3.connect(self.db_path)
            
            for i, chunk in enumerate(df_iter):
                # If first chunk, create tables
                if i == 0:
                    # Create main table
                    chunk.to_sql('theses', self.conn, if_exists='replace', index=False)
                else:
                    # Append to existing table
                    chunk.to_sql('theses', self.conn, if_exists='append', index=False)
            
            # Create indices for common search columns
            cursor = self.conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON theses (Title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_author1_lname ON theses (author1_lname)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_author1_fname ON theses (author1_fname)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_publication_date ON theses (publication_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_department ON theses (department)")
            # Add full-text search capability
            cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, content='theses', content_rowid='rowid')")
            cursor.execute("INSERT INTO theses_fts(rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department) SELECT rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department FROM theses")
            self.conn.commit()
            
            logger.info(f"Successfully converted CSV to SQLite database at {self.db_path}")
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to convert CSV to SQLite: {e}")
            return False
    
    def load_data(self) -> bool:
        """
        Load data into SQLite database if not already done.
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Check if database file exists
            if not os.path.exists(self.db_path):
                logger.info(f"Database file not found. Creating new database from CSV.")
                return self.csv_to_sqlite()
            else:
                # Connect to existing database
                self.conn = sqlite3.connect(self.db_path)
                
                # Verify the database has our tables
                cursor = self.conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses'")
                if not cursor.fetchone():
                    logger.info(f"Database exists but missing tables. Recreating from CSV.")
                    return self.csv_to_sqlite()
                
                # Check for full-text search table, add if missing
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theses_fts'")
                if not cursor.fetchone():
                    try:
                        logger.info("Adding full-text search capability to existing database")
                        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS theses_fts USING fts5(Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department, content='theses', content_rowid='rowid')")
                        cursor.execute("INSERT INTO theses_fts(rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department) SELECT rowid, Title, abstract, keywords, advisor1, advisor2, author1_fname, author1_lname, department FROM theses")
                        self.conn.commit()
                    except Exception as e:
                        logger.error(f"Error adding full-text search: {e}")
                
                logger.info(f"Successfully connected to existing database at {self.db_path}")
                self.loaded = True
                return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def get_columns(self) -> List[str]:
        """
        Get list of column names in the dataset.
        
        Returns:
            List of column names
        """
        if not self.loaded:
            self.load_data()
        
        # Use cached columns if available
        if self._columns_cache is not None:
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
    
    def get_column_sample(self, column_name: str, limit: int = 50) -> List[Any]:
        """
        Get a sample of values for a specific column.
        
        Args:
            column_name: Name of the column to retrieve
            limit: Maximum number of values to retrieve
            
        Returns:
            List of sample values in the column
        """
        if not self.loaded:
            self.load_data()
            
        try:
            cursor = self.conn.cursor()
            # Use double quotes for column names in case they contain spaces or special characters
            query = f'SELECT DISTINCT "{column_name}" FROM theses WHERE "{column_name}" IS NOT NULL LIMIT {limit}'
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting column sample: {e}")
            return []
    
    def search_column(self, column_name: str, search_term: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for a term in a specific column.
        
        Args:
            column_name: Name of the column to search
            search_term: Term to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching records
        """
        if not self.loaded:
            self.load_data()
            
        try:
            cursor = self.conn.cursor()
            
            # Get all column names for the result set
            columns = self.get_columns()
            
            # Construct query with proper escaping
            query = f"""
            SELECT * FROM theses 
            WHERE "{column_name}" LIKE ? 
            LIMIT {limit}
            """
            
            cursor.execute(query, (f"%{search_term}%",))
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            logger.error(f"Error searching column {column_name}: {e}")
            return []
    
    def execute_query(self, query: str, params: tuple = (), column_subset: List[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query with option to return only specific columns.
    
        Args:
            query: SQL query to execute
            params: Parameters for the query
            column_subset: Optional list of columns to include in results (to reduce data size)
    
        Returns:
            List of records matching the query
        """
        if not self.loaded:
            self.load_data()
    
        try:
            cursor = self.conn.cursor()
            # Log the query to check for correctness
            logger.debug(f"Executing SQL query: {query} with params: {params}")
            cursor.execute(query, params)
    
            # Get column names
            columns = [description[0] for description in cursor.description]
    
            results = []
            for row in cursor.fetchall():
                # If column_subset is specified, only include those columns
                if column_subset:
                    result_dict = {}
                    row_dict = dict(zip(columns, row))
                    for col in column_subset:
                        if col in row_dict:
                            result_dict[col] = row_dict[col]
                    results.append(result_dict)
                else:
                    results.append(dict(zip(columns, row)))
    
            logger.info(f"Query successfully executed and returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def get_table_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the theses table (keeping response size small).
        
        Returns:
            Dictionary with table statistics
        """
        if not self.loaded:
            self.load_data()
            
        try:
            cursor = self.conn.cursor()
            
            # Get row count
            cursor.execute("SELECT COUNT(*) FROM theses")
            row_count = cursor.fetchone()[0]
            
            # Get column count
            columns = self.get_columns()
            column_count = len(columns)
            
            return {
                "row_count": row_count,
                "column_count": column_count,
                "columns": columns[:10],  # Just return a subset of columns to keep response size down
            }
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {
                "row_count": "unknown",
                "column_count": "unknown",
                "columns": [],
            }

    # Specialized query functions to reduce data size
    
    def search_by_advisor(self, advisor_name: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by advisor name with multiple strategies."""
        results = []
        
        try:
            # Strategy 1: Exact match on full name
            query1 = """
            SELECT Title, department, publication_date, advisor1, advisor2 
            FROM theses 
            WHERE advisor1 = ? OR advisor2 = ? OR advisor3 = ?
            LIMIT ?
            """
            results.extend(self.execute_query(query1, (advisor_name, advisor_name, advisor_name, limit)))
            
            # Strategy 2: Partial match
            if len(results) < limit:
                query2 = """
                SELECT Title, department, publication_date, advisor1, advisor2 
                FROM theses 
                WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ? 
                LIMIT ?
                """
                results.extend(self.execute_query(query2, (f"%{advisor_name}%", f"%{advisor_name}%", f"%{advisor_name}%", limit)))
            
            # Strategy 3: Try with parts of the name (for first/last name searches)
            if len(results) < limit and ' ' in advisor_name:
                name_parts = advisor_name.split()
                for part in name_parts:
                    if len(part) > 2:  # Skip very short name parts
                        query3 = """
                        SELECT Title, department, publication_date, advisor1, advisor2 
                        FROM theses 
                        WHERE advisor1 LIKE ? OR advisor2 LIKE ? OR advisor3 LIKE ? 
                        LIMIT ?
                        """
                        part_results = self.execute_query(query3, (f"%{part}%", f"%{part}%", f"%{part}%", limit - len(results)))
                        results.extend(part_results)
                        
                        if len(results) >= limit:
                            break
            
            return results[:limit]  # Ensure we don't exceed the limit
        except Exception as e:
            logger.error(f"Error in advisor search: {e}")
            return []
    
    def search_by_author(self, first_name: str = None, last_name: str = None, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by author name."""
        try:
            if first_name and last_name:
                query = """
                SELECT Title, author1_fname, author1_lname, department, publication_date
                FROM theses 
                WHERE author1_fname = ? AND author1_lname = ? 
                LIMIT ?
                """
                results = self.execute_query(query, (first_name, last_name, limit))
                
                # If exact match fails, try LIKE operators
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
                query = """
                SELECT Title, author1_fname, author1_lname, department, publication_date
                FROM theses 
                WHERE author1_lname LIKE ? 
                LIMIT ?
                """
                return self.execute_query(query, (f"%{last_name}%", limit))
            elif first_name:
                query = """
                SELECT Title, author1_fname, author1_lname, department, publication_date
                FROM theses 
                WHERE author1_fname LIKE ? 
                LIMIT ?
                """
                return self.execute_query(query, (f"%{first_name}%", limit))
            return []
        except Exception as e:
            logger.error(f"Error in author search: {e}")
            return []
    
    def search_by_department(self, department: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by department."""
        try:
            query = """
            SELECT Title, department, publication_date, author1_fname, author1_lname
            FROM theses 
            WHERE department LIKE ? 
            LIMIT ?
            """
            return self.execute_query(query, (f"%{department}%", limit))
        except Exception as e:
            logger.error(f"Error in department search: {e}")
            return []
    
    def search_by_keyword(self, keyword: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Search for theses by keyword in title or abstract."""
        try:
            # First try full-text search if available
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
            except Exception as fts_error:
                logger.warning(f"Full-text search failed, falling back to LIKE: {fts_error}")
            
            # Fall back to regular search
            query = """
            SELECT Title, department, publication_date, author1_fname, author1_lname
            FROM theses 
            WHERE Title LIKE ? OR abstract LIKE ? OR keywords LIKE ?
            LIMIT ?
            """
            return self.execute_query(query, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit))
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def search_by_exact_title(self, title: str) -> List[Dict[str, Any]]:
        """
        Search for thesis by exact title.
        
        Args:
            title: The exact title to search for
            
        Returns:
            List of matching records (usually just one)
        """
        try:
            # Try exact match first
            query = """
            SELECT * FROM theses 
            WHERE Title = ?
            LIMIT 5
            """
            results = self.execute_query(query, (title,))
            
            # If no exact match, try with LIKE for case insensitivity
            if not results:
                query = """
                SELECT * FROM theses 
                WHERE Title LIKE ?
                LIMIT 5
                """
                results = self.execute_query(query, (title,))
                
            # If still no results, try with partial match
            if not results:
                # Try partial match with the first 50 chars of the title
                partial_title = title[:min(50, len(title))]
                query = """
                SELECT * FROM theses 
                WHERE Title LIKE ?
                LIMIT 10
                """
                results = self.execute_query(query, (f"%{partial_title}%",))
                
            return results
        except Exception as e:
            logger.error(f"Error in exact title search: {e}")
            return []
    
    def search_all_columns(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search across all text columns for a term (comprehensive search).
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching records
        """
        if not self.loaded:
            self.load_data()
            
        # Get all columns
        columns = self.get_columns()
        text_columns = [col for col in columns if col not in ['rowid', 'publication_date']]
            
        try:
            # Try full-text search first if available
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
            except Exception as fts_error:
                logger.warning(f"Full-text search failed, falling back to LIKE: {fts_error}")
                
            # Build a query that searches across all text columns
            where_clauses = []
            params = []
            
            for col in text_columns:
                where_clauses.append(f'"{col}" LIKE ?')
                params.append(f"%{search_term}%")
                
            query = f"""
            SELECT * FROM theses 
            WHERE {" OR ".join(where_clauses)}
            LIMIT {limit}
            """
            
            return self.execute_query(query, tuple(params))
        except Exception as e:
            logger.error(f"Error in all-columns search: {e}")
            return []

################################################################################
# RAG System
################################################################################

class ThesisRAGSystem:
    """RAG system for answering questions about thesis data using SQLite."""
    
    def __init__(self, data_manager: ThesisDataManager, model: str = DEFAULT_MODEL):
        """
        Initialize the RAG system.
        
        Args:
            data_manager: Data manager instance for accessing thesis data
            model: LLM model to use for answering questions
        """
        self.data_manager = data_manager
        self.model = model
        
    def determine_query_strategy(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Determine the best query strategy based on the question and execute it.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (query_type, query_results)
        """
        # Step 1: Check if this is an exact title search
        exact_title_results = self._check_for_exact_title_search(question)
        if exact_title_results:
            return "exact_title", exact_title_results
        
        # Step 2: Use LLM to analyze the question and determine query types to try
        query_analysis_prompt = """You are a database query analyzer for a thesis database.
        
        Analyze this question and determine which query types would be most appropriate to try.
        The database contains information about theses including authors, advisors, departments, titles, etc.
        
        Return a JSON object with the following structure:
        {
            "possible_query_types": ["advisor", "author", "department", "topic", "general"],
            "query_priority": ["primary_type", "secondary_type", ...],
            "entities": {
                "person_names": ["full name 1", "full name 2"],
                "keywords": ["keyword1", "keyword2"]
            }
        }
        
        Include only the JSON in your response, no other text.
        """
        
        response = run_llm(query_analysis_prompt, question)
        
        # Extract JSON from the response
        try:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                import json
                query_info = json.loads(json_match.group(1))
            else:
                # Default if no JSON found
                query_info = {
                    "possible_query_types": ["advisor", "author", "topic", "general"],
                    "query_priority": ["general"],
                    "entities": {
                        "person_names": [],
                        "keywords": []
                    }
                }
        except Exception as e:
            logger.error(f"Error parsing query analysis: {e}")
            # Provide default values on error
            query_info = {
                "possible_query_types": ["advisor", "author", "topic", "general"],
                "query_priority": ["general"],
                "entities": {
                    "person_names": [],
                    "keywords": []
                }
            }
        
        # Get query priorities
        query_priority = query_info.get("query_priority", ["general"])
        person_names = query_info.get("entities", {}).get("person_names", [])
        keywords = query_info.get("entities", {}).get("keywords", [])
        
        # Extract names from the question if LLM didn't find any
        if not person_names:
            name_matches = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', question)
            person_names.extend(name_matches)
        
        # Extract possible title fragments (sequences of capitalized words)
        title_fragments = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+){2,})', question)
        keywords.extend(title_fragments)
        
        # Extract other potential keywords from the question if LLM didn't find enough
        if len(keywords) < 3:
            potential_keywords = [word for word in question.split() if len(word) > 4 and word[0].isupper()]
            keywords.extend(potential_keywords)
        
        logger.info(f"Query analysis: priorities={query_priority}, names={person_names}, keywords={keywords}")
        
        # Try each query type in priority order
        all_results = []
        successful_query_type = "general"
        
        for query_type in query_priority:
            results = []
            
            if query_type == "advisor":
                # Try each name as an advisor
                for name in person_names:
                    advisor_results = self.data_manager.search_by_advisor(name)
                    if advisor_results:
                        results.extend(advisor_results)
                        successful_query_type = "advisor"
                        logger.info(f"Found {len(advisor_results)} results for advisor: {name}")
            
            elif query_type == "author":
                # Try each name as an author
                for name in person_names:
                    parts = name.split()
                    if len(parts) >= 2:
                        author_results = self.data_manager.search_by_author(parts[0], parts[1])
                        if author_results:
                            results.extend(author_results)
                            successful_query_type = "author"
                            logger.info(f"Found {len(author_results)} results for author: {name}")
            
            elif query_type == "department":
                # Try department search with keywords
                for keyword in keywords:
                    dept_results = self.data_manager.search_by_department(keyword)
                    if dept_results:
                        results.extend(dept_results)
                        successful_query_type = "department"
                        logger.info(f"Found {len(dept_results)} results for department: {keyword}")
            
            elif query_type == "topic" or query_type == "general":
                # Try keyword search
                for keyword in keywords:
                    keyword_results = self.data_manager.search_by_keyword(keyword)
                    if keyword_results:
                        results.extend(keyword_results)
                        successful_query_type = query_type
                        logger.info(f"Found {len(keyword_results)} results for keyword: {keyword}")
            
            # If we found reasonable results, add them but continue searching
            if results:
                all_results.extend(results)
                # Only break if we found a lot of results
                if len(all_results) > 20:
                    break
        
        # If we still don't have many results, try the comprehensive all-columns search
        if len(all_results) < 10:
            # Combine all potential search terms
            all_search_terms = person_names + keywords
            
            for search_term in all_search_terms:
                if len(search_term) > 3:  # Skip very short terms
                    # Try comprehensive search
                    comprehensive_results = self.data_manager.search_all_columns(search_term)
                    if comprehensive_results:
                        all_results.extend(comprehensive_results)
                        successful_query_type = "comprehensive"
                        logger.info(f"All-columns search found {len(comprehensive_results)} results for: {search_term}")
                        
                        # If we have enough results, stop
                        if len(all_results) > 30:
                            break
        
        # If still no results, use a fallback approach
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
            """
            Check if the question is asking about a specific thesis title.
            
            Args:
                question: User's question
                
            Returns:
                List of matching records if it's asking about a specific title, empty list otherwise
            """
            # Use patterns to detect if user is asking about a specific thesis
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
        """
        Fallback retrieval strategy when main strategies fail.
        
        Args:
            question: User's question
            
        Returns:
            List of potentially relevant theses
        """
        # Extract the most important words from the question
        stop_words = {"the", "a", "an", "in", "of", "for", "about", "with", "by", "to", "what", "who", "how", "when", "where", "which", "thesis", "theses", "dissertation"}
        words = [word.strip(',.?!:;()[]{}') for word in question.lower().split()]
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        results = []
        # Try each keyword individually
        for keyword in keywords:
            keyword_results = self.data_manager.search_all_columns(keyword, limit=10)
            if keyword_results:
                results.extend(keyword_results)
                if len(results) >= 30:
                    break
        
        # If that didn't work, get some recent theses as a last resort
        if not results:
            try:
                query = """
                SELECT Title, author1_fname, author1_lname, department, publication_date, abstract
                FROM theses 
                ORDER BY publication_date DESC
                LIMIT 20
                """
                results = self.data_manager.execute_query(query)
            except Exception as e:
                logger.error(f"Error in fallback retrieval: {e}")
        
        return results
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question about theses.
        
        Args:
            question: User's question
            
        Returns:
            Generated answer
        """
        if not self.data_manager.loaded:
            success = self.data_manager.load_data()
            if not success:
                return "I apologize, but I was unable to load the thesis database. Please try again later."
        
        # Get relevant theses based on the question
        query_type, relevant_theses = self.determine_query_strategy(question)
        
        # If we couldn't find any relevant theses
        if not relevant_theses:
            return "I couldn't find any theses matching your query. Could you please rephrase your question or provide more details?"
        
        # Truncate number of results to avoid context length issues
        max_results = 20
        if len(relevant_theses) > max_results:
            logger.info(f"Truncating results from {len(relevant_theses)} to {max_results}")
            relevant_theses = relevant_theses[:max_results]
        
        # Format retrieved theses for the LLM
        context = self._format_theses_for_context(relevant_theses, query_type)
        
        # Generate an answer using the LLM
        system_prompt = """You are a helpful academic assistant answering questions about university theses.
        
        I will provide you with relevant thesis information from a database based on the user's question.
        Use this information to give a thorough and accurate answer.
        
        Your response should:
        1. Directly answer the user's question based on the data provided
        2. Structure your answer in a clear, informative way
        3. Mention specific thesis titles, authors, and other relevant details
        4. Be honest if the data doesn't contain enough information to answer
        5. Don't make up information that isn't in the data
        
        You are speaking directly to the user who asked the question.
        """
        
        user_prompt = f"""
        Question: {question}
        
        Here is the thesis data from our database:
        {context}
        """
        
        response = run_llm(system_prompt, user_prompt, model=self.model)
        return response
    
    def _format_theses_for_context(self, theses: List[Dict[str, Any]], query_type: str) -> str:
        """
        Format retrieved theses for use in LLM context.
        
        Args:
            theses: List of thesis records
            query_type: Type of query that retrieved these theses
            
        Returns:
            Formatted thesis information
        """
        result = f"[Found {len(theses)} relevant theses]\n\n"
        
        # Include different fields based on query type for better answers
        for i, thesis in enumerate(theses, 1):
            result += f"Thesis {i}:\n"
            
            # Always include core fields
            title = thesis.get('Title', 'Unknown title')
            department = thesis.get('department', 'Unknown department')
            
            result += f"Title: {title}\n"
            result += f"Department: {department}\n"
            
            # Include author information if available
            first_name = thesis.get('author1_fname', '')
            last_name = thesis.get('author1_lname', '')
            if first_name or last_name:
                result += f"Author: {first_name} {last_name}\n"
            
            # Include publication date if available
            pub_date = thesis.get('publication_date', '')
            if pub_date:
                result += f"Publication date: {pub_date}\n"
            
            # Include advisor information if available and relevant
            advisor1 = thesis.get('advisor1', '')
            advisor2 = thesis.get('advisor2', '')
            if advisor1 and (query_type == 'advisor' or i <= 5):
                result += f"Primary advisor: {advisor1}\n"
            if advisor2 and (query_type == 'advisor' or i <= 5):
                result += f"Secondary advisor: {advisor2}\n"
            
            # Include abstract for the first few results or if it seems particularly relevant
            abstract = thesis.get('abstract', '')
            if abstract and (i <= 3 or query_type in ['exact_title', 'comprehensive']):
                # Truncate very long abstracts
                if len(abstract) > 500:
                    abstract = abstract[:497] + "..."
                result += f"Abstract: {abstract}\n"
            
            # Include degree and URI if available
            degree = thesis.get('degree', '')
            if degree:
                result += f"Degree: {degree}\n"
            
            uri = thesis.get('uri', '')
            if uri:
                result += f"URI: {uri}\n"
            
            result += "\n"
        
        return result

################################################################################
# Main function
################################################################################

def main():
    """Main function to initialize and run the system."""
    # Initialize data manager
    data_manager = ThesisDataManager()
    success = data_manager.load_data()
    
    if not success:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Initialize RAG system
    rag_system = ThesisRAGSystem(data_manager)
    
    # Simple CLI for testing
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
