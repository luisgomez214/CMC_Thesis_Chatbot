import os
import pandas as pd
import re
import html
import groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Groq client
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

def run_llm(system_prompt, user_query="", model='llama3-8b-8192', max_tokens=1000):
    """
    Sends a prompt to the LLM and returns the response.
    """
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

def load_data():
    """
    Loads the theses database from the CSV file and handles the publication_date format.
    """
    df = pd.read_csv("merged_theses.csv")
    df = df.fillna('').astype(str)  # Ensure all columns are string type before further processing
    df['keywords'] = df['keywords'].str.lower()  # Ensure keywords are in lowercase for matching
    df['author_full_name'] = df.apply(lambda x: f"{x['author1_fname']} {x['author1_mname']} {x['author1_lname']}".strip().replace('  ', ' '), axis=1).str.lower()
    df["combined_text"] = df[["advisor1", "advisor2", "advisor3", "Title", "keywords", "disciplines", "author_full_name", "department", "abstract"]].agg(" ".join, axis=1).str.lower()
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')  # Convert publication_date to datetime
    return df

def prepare_embeddings(df):
    """
    Prepares the TF-IDF vectorizer and computes embeddings for the dataset.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    return vectorizer, tfidf_matrix

def generate_recommendation(title, abstract, keywords):
    """
    Uses an LLM to generate a tailored recommendation based on the thesis abstract and keywords.
    """
    prompt = (f"Based on the abstract and keywords of the thesis titled '{title}', which discusses {keywords}, "
              "generate a recommendation reason that highlights why someone should read this thesis.")
    response = run_llm(prompt)  # Assuming run_llm is defined to interact with your LLM
    return response

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


#def retrieve_relevant_theses(query, df, vectorizer, tfidf_matrix, top_k=5):
    """
    Retrieves relevant theses based on broader search criteria, including keyword matching, and sorts by publication date.
    """
    query = query.lower().strip()  # Convert query to lower case
    # Matches based on full text or strict author initially
    matches = df[df['combined_text'].str.contains(re.escape(query), regex=True, case=False)]

    if matches.empty:
        print("No exact matches found. Trying keyword matching.")
        # Try matching based on keywords column, case insensitive
        keyword_matches = df[df['keywords'].str.contains(re.escape(query), case=False)]
        if not keyword_matches.empty:
            matches = keyword_matches
        else:
            print("No keyword matches found. Trying semantic matching.")
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, tfidf_matrix)
            top_indices = similarities.argsort()[0][-top_k:][::-1]
            matches = df.iloc[top_indices]

    if not matches.empty:
        sorted_results = matches.sort_values('publication_date', ascending=False) # Sort by publication_date
        return sorted_results
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no matches found

def strip_html_tags(text):
    """
    Strips HTML tags from a given text.
    """
    return html.unescape(re.sub('<[^<]+?>', '', text))


def interactive_query(data, vectorizer, tfidf_matrix):
    """
    Handles interactive querying of the theses database.
    """
    print("Welcome to the Claremont McKenna Theses Query System. Type 'exit' to quit.")
    while True:
        user_input = input("Enter your query: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting system.")
            break
        
        results = retrieve_relevant_theses(user_input, data, vectorizer, tfidf_matrix)
        
        if not results.empty:
            print("Here are the results for your query:")
            num_results = len(results)
            for index, row in results.iterrows():
                advisors = ', '.join(filter(None, [row.get('advisor1'), row.get('advisor2')]))
                abstract = strip_html_tags(row['abstract'])
                publication_date = row['publication_date'].strftime('%Y-%m-%d') if not pd.isnull(row['publication_date']) else 'N/A'
                award = row['award'] if 'award' in row and row['award'] else 'None'
                keywords = row['keywords']
                recommendation = generate_recommendation(row['Title'], abstract, keywords)
                
                print(f"Title: {row['Title']}\nAuthor: {row['author_full_name']}\nAdvisors: {advisors}\nDepartment: {row['department']}\nPublication Date: {publication_date}\nAward: {award}\nKeywords: {keywords}")
                print(f"Abstract: {abstract}")
                print(f"Recommendation: {recommendation}\n")
            print("-" * 80)

            print(f"A total of {num_results} theses were found for your query.")
            print("-" * 80)
        else:
            print("No relevant theses found.")
            print("- -" * 80)



def main():
    df = load_data()
    vectorizer, tfidf_matrix = prepare_embeddings(df)
    interactive_query(df, vectorizer, tfidf_matrix)

if __name__ == "__main__":
    main()
