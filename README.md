![deploy](https://github.com/luisgomez214/CMC_Thesis_Chatbot/actions/workflows/deploy.yml/badge.svg)


# CMC Thesis Chatbot

A smart, interactive chatbot that leverages Retrieval-Augmented Generation (RAG) to help explore the rich archive of Claremont McKenna College senior theses.

## Project Overview

The CMC Thesis Chatbot serves as a comprehensive tool for students, researchers, and faculty to efficiently search, analyze, and brainstorm based on Claremont McKenna College's senior thesis repository (up to Fall 2024). Built as a senior thesis project, this RAG-powered system combines a structured database of thesis metadata with the natural language capabilities of Groq's LLM to deliver intelligent, contextually relevant responses.

## Data Collection & Processing

- Data sourced directly from the [Claremont Colleges Library's institutional repository](https://scholarship.claremont.edu)
- Library-provided CSV files merged and cleaned to create a comprehensive dataset
- Organized in a SQLite database (`theses.db`) with optimized indexes and FTS5 full-text search capabilities

## Technology Stack

- **Backend**: Python for core logic and database interactions
- **Web Framework**: Flask for server implementation and routing
- **Database**: SQLite with FTS5 extension for performant full-text search
- **AI/ML**: Groq LLM API for natural language understanding and generation
- **Containerization**: Docker + Docker Compose for consistent deployment
- **Hosting**: AWS EC2 for cloud infrastructure
- **Domain & Security**: AWS Route 53 + SSL Certificate Manager for secure HTTPS access

## RAG System Architecture

The Retrieval-Augmented Generation approach ensures high-quality, factually grounded responses:

1. **Retrieval Component**: Queries the database for relevant thesis information based on user input
2. **Context Augmentation**: Enriches the prompt with retrieved metadata
3. **Generation**: Leverages Groq's LLM to produce natural language responses based on the retrieved context
4. **Response Formatting**: Structures information in a user-friendly format

This architecture minimizes hallucination by grounding the AI's responses in the actual thesis database.

## Deployment Architecture

The system follows a cloud-native deployment approach:

- **Production Environment**: AWS EC2 instance running Docker Compose
- **Domain**: Custom domain (`cmcthesischatbot.com`) configured via AWS Route 53
- **Security**: HTTPS secured with TLS certificates through AWS Certificate Manager
- **CI/CD**: Automated deployment pipeline via GitHub Actions

## Key Features

1. **Comprehensive Search**: Find theses by title, author, advisor, department, or keywords
2. **Analytical Queries**: Process counting and statistical queries (e.g., "How many theses in Government in 2020?")
3. **Thesis Ideation**: Generate thesis ideas with suggested CMC advisors based on topic or field
4. **Abstract Analysis**: Summarize and analyze thesis abstracts
5. **Faculty Insights**: Discover advisor relationships and expertise areas
6. **Contextual Follow-ups**: Reference previous search results in follow-up questions

## **Live Application**

The CMC Thesis Chatbot is available at: [https://cmcthesischatbot.com](https://cmcthesischatbot.com)

## Interface Examples

### Get Thesis Ideas / Outline
![Screenshot of Outline](screenshots/outline1.png) ![Screenshot](screenshots/outline2.png)

### Advisor Search
![Advisor Screenshot](screenshots/'advisor .png')

### Thesis Search
![Thesis Search Screenshot](screenshots/'thesis by search .png')


## Comparison With CHATGPT4o

### Get Thesis Ideas / Outline
![Screenshot of Outline](screenshots/check1.png)

Gives ideas not based on current thesis metadata. This could lead to repeated ideas.

### Advisor Search
![Advisor Screenshot](screenshots/check2.png)
Gives information over thesis/papers not in the metadata and from another source.


### Thesis Search
![Thesis Search Screenshot](screenshots/check3.png)
Both systems can summarize papers but ChatGPT cant have access to all thesis like my system.
Adding the actual paper icontent to the metadata will ensure better summarize. 


## Future Development Roadmap

- Enhanced query handling with fuzzy matching and spell correction
- Add more to metadata
- Responsive UI design for improved mobile experience
- Expansion to include theses from all Claremont Colleges
- User authentication for personalized session history
- Advanced LLM prompt engineering for more precise answers
- Production-grade WSGI implementation (NGINX + Gunicorn)

## Acknowledgments

Special thanks to:
1. My family (mom, grandma, grandpa, sister) for their unwavering support
2. Professor Mike Izbicki for guidance and mentorship
3. The CMC community for fostering an environment of academic excellence and innovation
4. Claremont Colleges Library for providing access to the thesis repository data

