#!/bin/python3

'''
Run an interactive QA session with academic theses data using the OpenAI API and retrieval augmented generation (RAG).

New theses can be added to the database with the --add_csv parameter,
and the path to the database can be changed with the --db parameter.
'''

import datetime
import logging
import re
import sqlite3
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
import os
import groq

################################################################################
# LLM functions
################################################################################

client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))


def run_llm(system, user, model='llama3-8b-8192', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
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


def summarize_text(text, seed=None):
    system = 'Summarize the input academic thesis abstract below. Limit the summary to 1 paragraph. Use an advanced reading level similar to the input text, and ensure that all key topics, methodologies, findings, and implications are included in the summary. The summary should capture the essence of the research while maintaining academic rigor.'
    return run_llm(system, text, seed=seed)


def extract_keywords(text, seed=None):
    # System prompt that instructs the AI assistant to extract keywords from the provided text
    system_prompt = '''You are an AI assistant specializing in academic research. Your task is to extract keywords from a given thesis abstract or description. The goal is to generate a comprehensive list of relevant terms that represent the main ideas, topics, research methodologies, and theoretical frameworks of the thesis. Along with key ideas, include words that provide additional context and connections to these core academic concepts. Your output should be a detailed list of keywords, capturing both central and contextually related terms. Include all relevant nouns, verbs, adjectives, and proper nouns, especially those that enhance the understanding of the primary research content. There is no need for punctuation or formatting—just a space-separated list of words. Exclude common filler words like "the," "and," "of," or similar non-essential words. Focus on words that convey academic meaning, ensuring that the list reflects both the primary research subjects and related ideas. Include compound concepts like "machine learning" as two separate words.
Only provide a space-separated list of relevant keywords. Avoid adding explanations, comments, punctuation, or any additional text.'''

    # Define the user input prompt based on the text provided
    user_prompt = f"Extract keywords from the following thesis information: {text}"

    # Use the run_llm function to get the extracted keywords
    keywords = run_llm(system_prompt, user_prompt, seed=seed)

    # Return the keywords as a space-separated string
    return keywords

################################################################################
# helper functions
################################################################################

def _logsql(sql):
    rex = re.compile(r'\W+')
    sql_dewhite = rex.sub(' ', sql)
    logging.debug(f'SQL: {sql_dewhite}')


def _catch_errors(func):
    '''
    This function is intended to be used as a decorator.
    It traps whatever errors the input function raises and logs the errors.
    '''
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(str(e))
    return inner_function


################################################################################
# rag
################################################################################

def rag(text, db):
    """
    This function uses Retrieval Augmented Generation (RAG) to create an LLM response for the input text.
    The `db` argument should be an instance of the `ThesesDB` class containing relevant documents.

    Steps:
    1. Extract keywords from the input text.
    2. Retrieve related theses based on those keywords.
    3. Construct a new prompt using the query and the thesis information.
    4. Pass the prompt to the LLM and return the response.
    """

    # Step 1: Extract keywords from the input text
    keywords = extract_keywords(text)

    # Step 2: Use the extracted keywords to find relevant theses in the database
    related_theses = db.find_theses(query=keywords)

    # Step 3: Compile information from the relevant theses
    thesis_info = f"{text}\n\nTheses:\n\n"
    
    for thesis in related_theses:
        thesis_authors = f"{thesis['author1_fname']} {thesis['author1_lname']}"
        if thesis['author2_fname']:
            thesis_authors += f", {thesis['author2_fname']} {thesis['author2_lname']}"
        
        thesis_info += (
            f"Title: {thesis['title']}\n"
            f"Author(s): {thesis_authors}\n"
            f"Department: {thesis['department']}\n"
            f"Publication Date: {thesis['publication_date']}\n"
            f"Degree: {thesis['degree_name']}\n"
            f"Keywords: {thesis['keywords']}\n"
            f"Abstract: {thesis['abstract']}\n"
            f"Disciplines: {thesis['disciplines']}\n\n"
        )

    # Create the LLM prompt by incorporating the user query and the relevant thesis information
    prompt = (
        f"You are a research assistant skilled at answering questions about academic theses. "
        f"Below is the user's query followed by information from relevant theses in our database. "
        f"Please answer based only on the information provided. Cite specific theses when appropriate. "
        f"If the information isn't available in the provided theses, acknowledge this limitation. "
        f"User Query: \"{text}\"\n\n"
    )

    # Step 4: Pass the constructed prompt to the LLM and return the generated response
    response = run_llm(prompt, thesis_info)

    return response


class ThesesDB:
    '''
    This class represents a database of academic theses.
    It is backed by sqlite3 and designed to have no external dependencies and be easy to understand.

    The following example shows how to add theses from a CSV file to the database.

    >>> db = ThesesDB()
    >>> len(db)
    0
    >>> db.add_csv("merged_theses.csv")
    >>> len(db)
    # number of theses added

    Once theses have been added,
    we can search through those theses to find ones about certain topics.

    >>> theses = db.find_theses('machine learning')

    The output is a list of theses that match the search query.
    Each thesis is represented by a dictionary with fields from the CSV.
    '''

    def __init__(self, filename=':memory:'):
        self.db = sqlite3.connect(filename)
        self.db.row_factory = sqlite3.Row
        self.logger = logging
        self._create_schema()

    def _create_schema(self):
        '''
        Create the DB schema if it doesn't already exist.
        '''
        try:
            sql = '''
            CREATE VIRTUAL TABLE theses
            USING FTS5 (
                title,
                orcid,
                publication_date,
                season,
                custom_date,
                document_type,
                award,
                degree_name,
                department,
                second_department,
                advisor1,
                advisor2,
                advisor3,
                embargo_date,
                distribution_license,
                keywords,
                rights,
                oclc_record_number,
                disciplines,
                abstract,
                comments,
                data_link,
                multimedia_url,
                multimedia_format,
                multimedia_url_2,
                multimedia_format_2,
                fulltext_url,
                author1_fname,
                author1_mname,
                author1_lname,
                author1_suffix,
                author1_institution,
                author2_fname,
                author2_mname,
                author2_lname,
                author2_suffix,
                author2_institution,
                author3_fname,
                author3_mname,
                author3_lname,
                author3_suffix,
                author3_institution,
                author4_fname,
                author4_mname,
                author4_lname,
                author4_suffix,
                author4_institution,
                author5_fname,
                author5_mname,
                author5_lname,
                author5_suffix,
                author5_email,
                author5_institution,
                calc_url,
                context_key,
                issue,
                ctmtime,
                URL,
                First_published,
                State,
                Total,
                summary
                );
            '''
            self.db.execute(sql)
            self.db.commit()

        # if the database already exists,
        # then do nothing
        except sqlite3.OperationalError:
            self.logger.debug('CREATE TABLE failed, table likely already exists')
    
    def find_theses(self, query, limit=10):
        '''
        Return a list of theses in the database that match the specified query.
        '''
    
        cursor = self.db.cursor()
    
        # Modified query to fix the MATCH clause
        # When using FTS5, the table name should not be repeated in the WHERE clause
        sql = """
        SELECT title, abstract, publication_date, degree_name, department, keywords, 
               disciplines, author1_fname, author1_lname, author2_fname, author2_lname,
               advisor1, advisor2, fulltext_url
        FROM theses 
        WHERE theses MATCH ? 
        ORDER BY rank
        LIMIT ?;
        """
    
        # Alternative query if the above still fails:
        # sql = """
        # SELECT title, abstract, publication_date, degree_name, department, keywords, 
        #        disciplines, author1_fname, author1_lname, author2_fname, author2_lname,
        #        advisor1, advisor2, fulltext_url
        # FROM theses 
        # WHERE theses.theses MATCH ? 
        # LIMIT ?;
        # """
    
        cursor.execute(sql, (query, limit))
        rows = cursor.fetchall()
    
        # Get column names from cursor description
        columns = [column[0] for column in cursor.description]
    
        # Convert rows to list of dictionaries
        output = [dict(zip(columns, row)) for row in rows]
        return output   
        @_catch_errors
        def add_csv(self, csv_file, batch_size=100):
            '''
            Add theses from a CSV file to the database.
            
            Args:
                csv_file (str): Path to the CSV file
                batch_size (int): Number of records to process in each batch
            '''
            logging.info(f'Reading CSV file: {csv_file}')
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file, low_memory=False)
                # Clean column names (remove spaces, etc.)
                df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]
                
                # Process in batches to avoid memory issues with large CSV files
                total_rows = len(df)
                batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
                
                for batch_num in range(batches):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, total_rows)
                    
                    logging.info(f'Processing batch {batch_num + 1} of {batches} (rows {start_idx} to {end_idx})')
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    # Process each row in the batch
                    for _, row in batch_df.iterrows():
                        self._process_thesis_row(row)
                    
                    # Commit after each batch
                    self.db.commit()
                    
                logging.info(f'Successfully imported {total_rows} theses from {csv_file}')
                
            except Exception as e:
                logging.error(f'Error importing CSV: {str(e)}')
                raise
        
    def _process_thesis_row(self, row):
        '''
        Process a single thesis row and add it to the database.
        
        Args:
            row (pandas.Series): A row from the thesis DataFrame
        '''
        # Convert row to dictionary and fill missing values
        thesis_data = row.to_dict()
        
        # Fill NaN values with None for proper SQL handling
        for key, value in thesis_data.items():
            if pd.isna(value):
                thesis_data[key] = None
        
        # Generate a summary if abstract exists
        if thesis_data.get('abstract') and isinstance(thesis_data['abstract'], str) and len(thesis_data['abstract']) > 50:
            try:
                summary = summarize_text(thesis_data['abstract'])
                thesis_data['summary'] = summary
            except Exception as e:
                logging.error(f"Error summarizing abstract: {str(e)}")
                thesis_data['summary'] = None
        else:
            thesis_data['summary'] = None
        
        # Prepare SQL statement with all columns
        columns = thesis_data.keys()
        placeholders = ', '.join(['?' for _ in columns])
        columns_str = ', '.join(columns)
        
        sql = f'''
        INSERT INTO theses({columns_str})
        VALUES ({placeholders});
        '''
        
        # Execute SQL statement
        cursor = self.db.cursor()
        cursor.execute(sql, list(thesis_data.values()))
        
    def __len__(self):
        sql = '''
        SELECT count(*)
        FROM theses;
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return row[0]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Interactive QA with academic theses.')
    parser.add_argument('--loglevel', default='warning')
    parser.add_argument('--db', default='theses.db')
    parser.add_argument('--add_csv', help='Add theses from a CSV file to the database')
    parser.add_argument('--test_find_theses', action='store_true', help='Test finding theses')
    parser.add_argument('--query', help='Query for interactive QA')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for CSV import')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel.upper(),
    )

    db = ThesesDB(args.db)

    if args.add_csv:
        db.add_csv(args.add_csv, batch_size=args.batch_size)
    elif args.test_find_theses:
        results = db.find_theses("machine learning")
        for result in results:
            print(f"Title: {result['title']}")
            print(f"Author: {result['author1_fname']} {result['author1_lname']}")
            print(f"Abstract: {result['abstract'][:200]}...")
            print("-" * 80)
    elif args.query:
        output = rag(args.query, db)
        print(output)
    else:
        import readline
        print("Welcome to the Academic Theses QA System!")
        print("Type your questions about academic theses or 'exit' to quit.")
        while True:
            text = input('thesesrag> ')
            if text.strip().lower() in ['exit', 'quit']:
                break
            if len(text.strip()) > 0:
                output = rag(text, db)
                print(output)
