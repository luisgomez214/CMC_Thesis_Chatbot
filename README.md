![deploy](https://github.com/luisgomez214/CMC_Thesis_Chatbot/actions/workflows/deploy.yml/badge.svg)

# CMC Thesis Chatbot

Welcome to the **CMC Thesis Chatbot** — a smart, interactive chatbot designed to help you explore senior theses from Claremont McKenna College using the power of a **Retrieval-Augmented Generation (RAG)** system backed by **Groq's LLM**.

---

## Project Overview

This project was built to help students, researchers, and faculty quickly **search**, **analyze**, and **brainstorm** based on Claremont McKenna College’s senior thesis archive (up to **Fall 2024**).

---

## Data Collection & Preparation

- The thesis data was sourced directly from the [Claremont Colleges Library’s institutional repository](https://scholarship.claremont.edu).
- I downloaded and **merged multiple CSV datasets** to create a comprehensive file of all CMC theses.
- The data was cleaned to standardize column formats, remove duplicates, and fill missing entries.
- I then loaded the cleaned data into a **SQLite database (`theses.db`)**, and built **indexes** and **FTS5 full-text search** tables to allow fast, flexible retrieval.

---

## What is a RAG System?

RAG stands for **Retrieval-Augmented Generation**. It's an AI architecture that combines two powerful techniques:

1. **Retrieval**: Find relevant context (e.g. thesis titles, keywords, abstracts, advisors) from a structured database.
2. **Augmented Generation**: Use a **Large Language Model (LLM)** — in this case, **Groq's LLM** — to generate intelligent, grounded answers using the retrieved data.

This ensures the chatbot doesn’t just hallucinate information — it responds based on real thesis records.

---

## Technologies Used

- **Python** for backend logic
- **Flask** for the web server and routing
- **SQLite** + **FTS5** for fast thesis queries
- **Groq LLM API** for natural language understanding and generation
- **Docker + Docker Compose** for containerized deployment
- **AWS EC2** for hosting
- **Route 53** + **SSL Certificate Manager** to map a secure HTTPS domain:  
  https://cmcthesischatbot.com

---

## Deployment

### Hosted on:

- **AWS EC2** instance running Docker Compose
- Custom domain (`cmcthesischatbot.com`) set up using **AWS Route 53**
- HTTPS secured with **TLS certificates** via AWS

### CI/CD

The deployment is automated using a GitHub Actions workflow defined in `.github/workflows/deploy.yml`.


1. Checks out the repo on push to `main`
2. Installs dependencies: Python, Docker, Docker Compose
3. Builds the Docker container using `docker-compose build`
4. Authenticates with AWS using GitHub secrets
5. Initializes the Elastic Beanstalk environment
6. Deploys the latest version using `eb deploy`

---

## Features

1. Search theses by title, author, advisor, department, or keywords  
2. Count queries like: _"How many theses in Government in 2020?"_  
3. Generate full thesis ideas with suggested CMC advisors  
4. Get summaries of specific thesis abstracts  
5. Ask for co-advisors for a professor  
6. Follow up on specific thesis results (e.g., _"What was the second one about?"_)

---

## Demo

> Add screenshots below to showcase your system's responses.

### Advisor Query
![Advisor Screenshot](screenshots/advisor_query.png)

### Departmental Thesis Ideas
![Economics Screenshot](screenshots/economics_ideas.png)

### Thesis Abstract Summary
![Abstract Screenshot](screenshots/abstract_example.png)

---

## Improvements & Future Work

- Handle edge cases more gracefully (e.g., typos, fuzzy matching)
- Improve UI styling and add responsive design for mobile
- Expand scope to include other Claremont Colleges
- Add login feature for personalized session history
- Improve LLM prompting for more focused answers
- Integrate BibTeX export or citations
- Convert to production WSGI stack (e.g., NGINX + Gunicorn)

---

## 🔗 Live App

Visit the chatbot here: [cmcthesischatbot.com](https://cmcthesischatbot.com)

---

## 🙏 Acknowledgments

1. Family (mom, grandma, grandpa, sister)
2. Mike Izbicki
3. CMC family
---


