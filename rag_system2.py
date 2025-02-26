import os
import pandas as pd
import re
import html
import groq
import json
import hashlib
import logging
import time
import asyncio
import aiohttp
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(
    filename='rag_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download required NLTK data (uncomment if needed)
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initialize the Groq client
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Simple cache for LLM responses
llm_cache = {}
cache_file = "llm_cache.json"

# Load cache from file if exists
if os.path.exists(cache_file):
    try:
        with open(cache_file, 'r') as f:
            llm_cache = json.load(f)
    except Exception as e:
        logging.error(f"Error loading cache: {str(e)}")
        llm_cache = {}

def run_llm(system_prompt, user_query="", model='llama3-8b-8192', max_tokens=1000):
    """
    Sends a prompt to the LLM and returns the response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling LLM API: {str(e)}")
        return "Unable to generate recommendation at this time."

def run_llm_with_cache(system_prompt, user_query="", model='llama3-8b-8192', max_tokens=1000):
    """
    Sends a prompt to the LLM with caching to improve performance and reduce API costs.
    """
    # Create a cache key from the input parameters
    key_string = f"{system_prompt}|{user_query}|{model}|{max_tokens}"
    cache_key = hashlib.md5(key_string.encode()).hexdigest()
    
    # Check if we have a cached response
    if cache_key in llm_cache:
        logging.info("Using cached LLM response")
        return llm_cache[cache_key]
    
    # Get response from LLM
    logging.info("Making new LLM API call")
    response = run_llm(system_prompt, user_query, model, max_tokens)
    
    # Cache the response
    llm_cache[cache_key] = response
    
    # Save cache to file (could be optimized to not save every time)
    try:
        with open(cache_file, 'w') as f:
            json.dump(llm_cache, f)
    except Exception as e:
        logging.error(f"Warning: Could not save cache to file: {str(e)}")
    
    return response

def preprocess_text(text):
    """
    Preprocesses text for better search results.
    """
    # Lowercase
    text = str(text).lower()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords (optional - can affect search results)
    # stop_words = set(stopwords.words('english'))
    # tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize (optional - can affect search results)
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

def load_data():
    """
    Loads the theses database from the CSV file and handles the publication_date format.
    """
    try:
        df = pd.read_csv("merged_theses.csv")
        df = df.fillna('')  # Fill NaN values with empty strings
        
        # Create author full name
        df['author_full_name'] = df.apply(lambda x: f"{x['author1_fname']} {x['author1_mname']} {x['author1_lname']}".strip().replace('  ', ' '), axis=1).str.lower()
        
        # Create combined text field for searching
        df["combined_text"] = df[["advisor1", "advisor2", "advisor3", "Title", "keywords", "disciplines", "author_full_name", "department", "abstract"]].agg(" ".join, axis=1).str.lower()
        
        # Convert publication_date to datetime
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        
        logging.info(f"Loaded {len(df)} theses from database")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def prepare_embeddings(df):
    """
    Prepares the TF-IDF vectorizer and computes embeddings for the dataset.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
        logging.info("TF-IDF embeddings prepared successfully")
        return vectorizer, tfidf_matrix
    except Exception as e:
        logging.error(f"Error preparing embeddings: {str(e)}")
        raise

def expand_query(query):
    """
    Expands the query with related terms to improve search results.
    """
    expanded_terms = []
    
    # Expand with related academic terms
    academic_synonyms = {
        "paper": ["thesis", "dissertation", "research", "study"],
        "thesis": ["paper", "dissertation", "research", "study"],
        "research": ["study", "investigation", "analysis", "examination"],
        "by": ["from", "author", "written"],
        # Add more as needed
    }
    
    tokens = query.lower().split()
    for token in tokens:
        if token in academic_synonyms:
            expanded_terms.extend(academic_synonyms[token])
    
    expanded_query = query
    if expanded_terms:
        logging.info(f"Expanded query with terms: {expanded_terms}")
        expanded_query = query + " " + " ".join(expanded_terms)
    
    return expanded_query

def retrieve_relevant_theses(query, df, vectorizer, tfidf_matrix, top_k=5):
    """
    Retrieves relevant theses using a flexible approach that handles
    natural language queries without hard-coding specific patterns.
    """
    original_query = query
    query = query.lower().strip()

    # First try the direct match approach for the full query
    direct_matches = df[df['combined_text'].str.contains(re.escape(query), regex=True, case=False)]
    if not direct_matches.empty:
        return direct_matches.sort_values('publication_date', ascending=False)

    # If no direct matches, we'll try a more flexible tokenized approach
    tokens = query.split()

    # Extract potential names (all tokens of reasonable length)
    potential_names = []
    for i in range(len(tokens)):
        if len(tokens[i]) > 2:  # Skip very short words
            potential_names.append(tokens[i])

    # Look for any token matches in author_full_name with higher priority
    for name in potential_names:
        author_matches = df[df['author_full_name'].str.contains(re.escape(name), regex=True, case=False)]
        if not author_matches.empty:
            return author_matches.sort_values('publication_date', ascending=False)

    # Try keyword-based matching with all tokens
    for token in tokens:
        if len(token) > 3 and token not in ['thesis', 'the', 'and', 'for', 'from', 'with', 'about', 'that']:
            keyword_matches = df[df['keywords'].str.contains(re.escape(token), regex=True, case=False)]
            if not keyword_matches.empty:
                return keyword_matches.sort_values('publication_date', ascending=False)

    # Try title matching with significant tokens
    for token in tokens:
        if len(token) > 3 and token not in ['thesis', 'the', 'and', 'for', 'from', 'with', 'about', 'that']:
            title_matches = df[df['Title'].str.contains(re.escape(token), regex=True, case=False)]
            if not title_matches.empty:
                return title_matches.sort_values('publication_date', ascending=False)

    # If still no matches, create a semantic search filter
    query_vector = vectorizer.transform([original_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    similarity_indices = similarities.argsort()[0][-top_k:][::-1]
    result = df.iloc[similarity_indices].sort_values('publication_date', ascending=False)
    
    return result


def log_query(query, num_results, search_time, method="unknown"):
    """
    Logs query information for analysis.
    """
    logging.info(f"Query: {query} | Results: {num_results} | Time: {search_time:.2f}s | Method: {method}")
    
    # Also save to a structured format for analysis
    with open("query_metrics.csv", "a") as f:
        f.write(f"{datetime.now()},{query},{num_results},{search_time:.2f},{method}\n")

#def generate_recommendation(title, abstract, keywords, interactive_query):
    """
    Uses an LLM to generate a tailored recommendation based on the thesis abstract, keywords, and the user's query.
    """
    prompt = (f"Considering your interest in '{interactive_query}', here's why you should read the thesis titled '{title}', which discusses {keywords}: "
              "Based on its abstract {abstract}, this thesis offers insights that align with your query. Could you benefit from understanding more about this topic? "
              "Please provide a recommendation that highlights why this thesis is relevant to the query. Keep your recommendation concise, around 150 words.")

    return run_llm_with_cache(prompt)


def generate_recommendation(title, abstract, keywords):
    """
    Uses an LLM to generate a tailored recommendation based on the thesis abstract and keywords.
    """
    prompt = (f"Based on the abstract and keywords of the thesis titled '{title}', which discusses {keywords}, "
              "generate a recommendation reason that highlights why someone should read this thesis. Keep your recommendation to around 150 words.")

    return run_llm_with_cache(prompt)


def strip_html_tags(text):
    """
    Strips HTML tags from a given text.
    """
    return html.unescape(re.sub('<[^<]+?>', '', str(text)))

def apply_filters(results, filters=None):
    """
    Applies filters to the results dataframe.
    """
    if not filters:
        return results
    
    filtered = results.copy()
    
    if 'year' in filters and filters['year']:
        try:
            year = int(filters['year'])
            filtered = filtered[filtered['publication_date'].dt.year == year]
        except (ValueError, TypeError):
            logging.warning(f"Invalid year filter: {filters['year']}")
    
    if 'department' in filters and filters['department']:
        filtered = filtered[filtered['department'].str.contains(filters['department'], case=False)]
    
    if 'advisor' in filters and filters['advisor']:
        filtered = filtered[
            filtered['advisor1'].str.contains(filters['advisor'], case=False) |
            filtered['advisor2'].str.contains(filters['advisor'], case=False) |
            filtered['advisor3'].str.contains(filters['advisor'], case=False)
        ]
    
    return filtered

def get_user_feedback(thesis_id, query):
    """
    Collects user feedback on search results.
    """
    feedback = input("Was this result helpful? (y/n): ").lower().strip()
    if feedback in ['y', 'yes']:
        relevance = 1
    elif feedback in ['n', 'no']:
        relevance = 0
    else:
        return  # Invalid feedback
    
    # Store feedback in a CSV file
    with open("user_feedback.csv", "a") as f:
        f.write(f"{datetime.now()},{thesis_id},{query},{relevance}\n")
    
    print("Thank you for your feedback!")

def display_paginated_results(results, page_size=2):
    """
    Displays results in a paginated format.
    """
    if results.empty:
        print("No relevant theses found.")
        return
    
    total_results = len(results)
    total_pages = (total_results + page_size - 1) // page_size
    current_page = 1
    
    while True:
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_results)
        
        print(f"\nShowing results {start_idx + 1}-{end_idx} of {total_results} (Page {current_page}/{total_pages})")
        print("-" * 80)
        
        # Display current page of results
        for i, (index, row) in enumerate(results.iloc[start_idx:end_idx].iterrows(), start=1):
            advisors = ', '.join(filter(None, [row.get('advisor1', ''), row.get('advisor2', ''), row.get('advisor3', '')]))
            abstract = strip_html_tags(row['abstract'])
            publication_date = row['publication_date'].strftime('%Y-%m-%d') if not pd.isnull(row['publication_date']) else 'N/A'
            award = row['award'] if 'award' in row and row['award'] else 'None'
            keywords = row['keywords']
            
            # Generate or retrieve recommendation
            recommendation = generate_recommendation(row['Title'], abstract, keywords)
            
            print(f"Result #{start_idx + i}:")
            print(f"Title: {row['Title']}")
            print(f"Author: {row['author_full_name']}")
            print(f"Advisors: {advisors}")
            print(f"Department: {row['department']}")
            print(f"Publication Date: {publication_date}")
            print(f"Award: {award}")
            print(f"Keywords: {keywords}")
            print(f"Abstract: {abstract}")
            print(f"Recommendation: {recommendation}")
            print("-" * 80)
            
            # Optional: Get user feedback
            # get_user_feedback(index, query)
        
        if total_pages <= 1:
            break
            
        # Navigation options
        print("\nOptions: [n]ext page, [p]revious page, [q]uit pagination, [f]ilter results")
        choice = input("Enter choice: ").lower().strip()
        
        if choice == 'n' and current_page < total_pages:
            current_page += 1
        elif choice == 'p' and current_page > 1:
            current_page -= 1
        elif choice == 'f':
            # Apply filters
            year = input("Filter by year (press Enter to skip): ").strip()
            department = input("Filter by department (press Enter to skip): ").strip()
            advisor = input("Filter by advisor (press Enter to skip): ").strip()
            
            filters = {}
            if year:
                filters['year'] = year
            if department:
                filters['department'] = department
            if advisor:
                filters['advisor'] = advisor
                
            results = apply_filters(results, filters)
            total_results = len(results)
            total_pages = (total_results + page_size - 1) // page_size
            current_page = 1
            
            if results.empty:
                print("No results match your filters.")
                break
        elif choice == 'q':
            break
        else:
            print("Invalid choice or no more pages.")

def print_help():
    """
    Displays help information about available commands.
    """
    print("\nAvailable Commands:")
    print("  search [query]     - Search for theses matching your query")
    print("  recent             - Show the most recent theses")
    print("  popular            - Show popular theses based on user feedback")
    print("  filter [options]   - Filter results by year, department, etc.")
    print("  help               - Display this help message")
    print("  history            - Show your search history")
    print("  exit               - Exit the system")
    print("\nQuery Examples:")
    print("  - neuroscience")
    print("  - thesis by smith")
    print("  - research about climate change")
    print("  - papers from 2020")

def print_history(history):
    """
    Displays the user's search history.
    """
    if not history:
        print("No search history yet.")
        return
        
    print("\nSearch History:")
    for i, item in enumerate(history[-10:], 1):
        print(f"{i}. {item}")

def show_recent_theses(data, limit=10):
    """
    Shows the most recent theses in the database.
    """
    recent = data.sort_values('publication_date', ascending=False).head(limit)
    print(f"\nMost Recent {limit} Theses:")
    display_paginated_results(recent)

#def show_popular_theses(data, limit=10):
    """
    Shows popular theses based on user feedback.
    """
    # This would ideally use actual popularity metrics
    # For now, we'll just show some random theses as an example
    print(f"\nPopular Theses:")
    popular = data.sample(min(limit, len(data)))
    display_paginated_results(popular)

def interactive_query(data, vectorizer, tfidf_matrix):
    """
    Handles interactive querying of the theses database with improved UX.
    """
    print("Welcome to the Claremont McKenna Theses Query System.")
    print("Type 'help' for a list of commands or 'exit' to quit.")
    
    history = []
    
    while True:
        try:
            user_input = input("\nCommand> ").strip()
            
            # Add to history if not empty and not a command
            if user_input and not user_input.lower() in ['exit', 'help', 'history']:
                history.append(user_input)
            
            # Parse command
            if user_input.lower() == 'exit':
                print("Exiting system. Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
            elif user_input.lower() == 'history':
                print_history(history)
            elif user_input.lower() == 'recent':
                show_recent_theses(data)
            elif user_input.lower() == 'popular':
                show_popular_theses(data)
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    results = retrieve_relevant_theses(query, data, vectorizer, tfidf_matrix)
                    display_paginated_results(results, page_size=10)
                else:
                    print("Please provide a search query after 'search'.")
            elif user_input.lower().startswith('filter '):
                print("Please first perform a search, then use the filter option in pagination.")
            elif not user_input:
                # Empty input, just show prompt again
                continue
            else:
                # Treat as a search query by default
                results = retrieve_relevant_theses(user_input, data, vectorizer, tfidf_matrix)
                display_paginated_results(results, page_size=10)
        except KeyboardInterrupt:
            print("\nOperation cancelled. Type 'exit' to quit.")
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            print(f"An error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

def main():
    """
    Main function to run the thesis query system.
    """
    try:
        print("Loading thesis database... Please wait.")
        df = load_data()
        print("Preparing search index...")
        vectorizer, tfidf_matrix = prepare_embeddings(df)
        print("System ready!")
        interactive_query(df, vectorizer, tfidf_matrix)
    except Exception as e:
        logging.critical(f"Fatal error in main: {str(e)}")
        print(f"A critical error occurred: {str(e)}")
        print("Please check the log file for details.")

if __name__ == "__main__":
    main()
