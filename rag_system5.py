from urllib.parse import urlparse
import datetime
import logging
import re
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional

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
    
    def search_column(self, column_name: str, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
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
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query to execute
            params: Parameters for the query
            
        Returns:
            List of records matching the query
        """
        if not self.loaded:
            self.load_data()
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def get_table_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the theses table.
        
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
            
            # Get sample of publication dates
            cursor.execute("SELECT DISTINCT publication_date FROM theses WHERE publication_date IS NOT NULL ORDER BY publication_date LIMIT 10")
            date_sample = [row[0] for row in cursor.fetchall()]
            
            # Get common departments
            cursor.execute("""
                SELECT department, COUNT(*) as count 
                FROM theses 
                WHERE department IS NOT NULL 
                GROUP BY department 
                ORDER BY count DESC 
                LIMIT 10
            """)
            departments = [(row[0], row[1]) for row in cursor.fetchall()]
            
            return {
                "row_count": row_count,
                "column_count": column_count,
                "columns": columns,
                "date_sample": date_sample,
                "top_departments": departments
            }
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            
            # Fallback for basic stats
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM theses")
                row_count = cursor.fetchone()[0]
                
                columns = self.get_columns()
                column_count = len(columns)
                
                return {
                    "row_count": row_count,
                    "column_count": column_count,
                    "columns": columns,
                }
            except Exception as e2:
                logger.error(f"Error getting basic table stats: {e2}")
                return {
                    "row_count": "unknown",
                    "column_count": len(self.get_columns()),
                    "columns": self.get_columns(),
                }

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
        
    def generate_retrieval_prompt(self, question: str) -> str:
        """
        Generate a system prompt for the retrieval step.
        
        Args:
            question: User's question about the thesis data
            
        Returns:
            System prompt for retrieval
        """
        columns = self.data_manager.get_columns()
        table_stats = self.data_manager.get_table_stats()
        
        system_prompt = f"""You are a data retrieval assistant tasked with generating a SQL query to answer a user's question about thesis data.

Available columns in the 'theses' table:
{', '.join(columns)}

Dataset statistics:
- Total theses: {table_stats.get('row_count', 'unknown')}
- Column count: {table_stats.get('column_count', 'unknown')}

Based on the user's question, generate a SQL query that will retrieve the relevant information. The query should:
1. Select only columns necessary to answer the question
2. Include appropriate WHERE clauses to filter results
3. Keep the result set small (use LIMIT 10-20 if appropriate)
4. Format as simple raw SQL without any explanation
5. Use double quotes around column names, not square brackets

Return ONLY the SQL query, nothing else.
"""
        return system_prompt
        
    def generate_answer_prompt(self, question: str, query_results: List[Dict[str, Any]]) -> str:
        """
        Generate a system prompt for the answer generation step.
        
        Args:
            question: User's question about the thesis data
            query_results: Results from the SQL query
            
        Returns:
            System prompt for answer generation
        """
        # Format query results for the prompt
        result_str = ""
        if query_results:
            columns = list(query_results[0].keys())
            result_str += f"Columns: {', '.join(columns)}\n\n"
            
            # Format each row
            for i, row in enumerate(query_results[:20]):  # Limit to first 20 results to keep prompt size manageable
                result_str += f"Row {i+1}:\n"
                for col, val in row.items():
                    # Truncate long text values
                    if isinstance(val, str) and len(val) > 200:
                        val = val[:200] + "..."
                    result_str += f"  {col}: {val}\n"
                result_str += "\n"
            
            # Note if results were truncated
            if len(query_results) > 20:
                result_str += f"[Showing 20 of {len(query_results)} results]\n"
        else:
            result_str = "No results found for this query."
        
        system_prompt = f"""You are a knowledgeable research assistant tasked with answering questions about thesis data from a university repository.

The user asked: "{question}"

Here are the query results from the database:

{result_str}

Using the information provided above, please answer the user's question. If the data doesn't contain enough information to give a complete answer, acknowledge the limitations and provide the best answer possible based on the available data. Do not make up information not present in the provided results.
"""
        return system_prompt
        
    def answer_question(self, question: str) -> str:
        """
        Answer a question about the thesis data using the RAG approach with SQL.
        
        Args:
            question: User's question about the thesis data
            
        Returns:
            Answer to the question
        """
        # Step 1: Generate a SQL query for the question
        retrieval_prompt = self.generate_retrieval_prompt(question)
        sql_query = run_llm(retrieval_prompt, question, self.model)
        
        # Clean up the SQL query (remove markdown formatting if present)
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Step 2: Execute the SQL query
        try:
            query_results = self.data_manager.execute_query(sql_query)
            logger.info(f"Query returned {len(query_results)} results")
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            
            # Fallback to simpler approach based on keywords
            logger.info("Falling back to keyword-based approach")
            fallback_results = self._fallback_retrieval(question)
            query_results = fallback_results
        
        # Step 3: Generate an answer using the query results
        if not query_results:
            # Try the fallback method if no results from SQL
            if 'fallback' not in locals():
                fallback_results = self._fallback_retrieval(question)
                query_results = fallback_results
            
            if not query_results:
                return "I couldn't find relevant data to answer your question. Please check if your question relates to the thesis dataset or try reformulating it."
        
        answer_prompt = self.generate_answer_prompt(question, query_results)
        answer = run_llm(answer_prompt, question, self.model)
        return answer
    
    def _fallback_retrieval(self, question: str) -> List[Dict[str, Any]]:
        """
        Fallback method for retrieving data when SQL query fails.
        
        Args:
            question: User's question about the thesis data
            
        Returns:
            List of relevant records
        """
        # Extract potential keywords from the question
        keywords_prompt = """Extract the 3-5 most important search keywords from this question. Return only the keywords separated by commas, no explanation:"""
        keywords_response = run_llm(keywords_prompt, question, self.model)
        keywords = [k.strip() for k in keywords_response.split(',')]
        
        logger.info(f"Extracted keywords: {keywords}")
        
        # Determine potential columns to search based on common patterns
        search_columns = ['Title', 'author1_lname', 'author1_fname', 'department', 'abstract']
        
        # Search for each keyword in relevant columns
        all_results = []
        for keyword in keywords:
            if not keyword:
                continue
                
            for column in search_columns:
                # Skip columns that don't exist
                if column not in self.data_manager.get_columns():
                    continue
                    
                results = self.data_manager.search_column(column, keyword, limit=5)
                all_results.extend(results)
        
        # Remove duplicates (based on same Title if available, otherwise full record)
        unique_results = []
        seen_titles = set()
        
        for result in all_results:
            result_title = result.get('Title', str(result))
            if result_title not in seen_titles:
                unique_results.append(result)
                seen_titles.add(result_title)
        
        return unique_results[:20]  # Limit to top 20 results

################################################################################
# Interactive CLI
################################################################################

def interactive_cli():
    """Run an interactive command-line interface for the RAG system."""
    print("Thesis RAG System - Interactive CLI")
    print("-----------------------------------")
    
    data_manager = ThesisDataManager()
    success = data_manager.load_data()
    
    if not success:
        print("Failed to load data. Please check the file path and try again.")
        return
    
    table_stats = data_manager.get_table_stats()
    print(f"Loaded dataset with {table_stats.get('row_count', 'unknown')} rows and {table_stats.get('column_count', 'unknown')} columns.")
    
    rag_system = ThesisRAGSystem(data_manager)
    
    print("\nYou can now ask questions about the thesis data. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nQuestion: ").strip()
        
        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("Thinking...")
        answer = rag_system.answer_question(user_input)
        print("\nAnswer:")
        print(answer)

################################################################################
# Main function
################################################################################

if __name__ == "__main__":
    interactive_cli()
