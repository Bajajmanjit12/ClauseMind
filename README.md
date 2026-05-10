# ClauseMind  
### Semantic Decision Engine for Insurance and Legal Document Queries

ClauseMind is an AI-powered Retrieval-Augmented Generation (RAG) system designed to process insurance and legal documents and answer user queries using Large Language Models (LLMs).

The system enables users to ask natural language questions related to insurance policies, claims, waiting periods, exclusions, fraud clauses, and coverage conditions. It retrieves the most relevant clauses from uploaded documents and generates concise, context-aware answers.

---

# Problem Statement

Insurance and legal documents are typically:

- Long and unstructured
- Difficult to understand
- Filled with complex clauses and conditions
- Time-consuming to search manually

Users often struggle to quickly determine:
- Whether a treatment is covered
- Applicable waiting periods
- Policy exclusions
- Claim eligibility
- Fraud conditions
- Coverage limitations

ClauseMind addresses this problem by combining:
- Semantic search
- Vector databases
- Retrieval-Augmented Generation (RAG)
- Large Language Models

to provide fast and accurate policy assistance.

---

# Solution Overview

ClauseMind works in four major stages:

## 1. Document Processing
Insurance PDFs are parsed and converted into raw text.

## 2. Chunking
Large text is split into smaller semantic chunks for efficient retrieval.

## 3. Vector Embedding & Storage
Each chunk is converted into vector embeddings using Sentence Transformers and stored in a FAISS vector database.

## 4. Query Answering
When a user asks a question:
- the query is embedded,
- relevant chunks are retrieved from FAISS,
- Groq LLM generates a concise answer using retrieved context.

---

# System Architecture

```text
User Query
    ↓
Embedding Model
    ↓
FAISS Vector Search
    ↓
Relevant Policy Chunks
    ↓
Groq LLM
    ↓
Final Answer
```

---

# Features

- Natural language policy querying
- Semantic document retrieval
- FAISS vector similarity search
- Insurance clause understanding
- Concise AI-generated responses
- PDF-based knowledge ingestion
- Fast retrieval pipeline
- Flask-based web interface
- Modular architecture

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Flask | Web framework |
| FAISS | Vector similarity search |
| Sentence Transformers | Embedding generation |
| Groq API | LLM inference |
| PyPDF2 | PDF text extraction |
| LangChain | Text chunking utilities |

---

# Project Structure

```text
ClauseMind/
│
├── app.py
├── create_vectorstore.py
├── extract_text.py
├── query_llm.py
├── query_vectorspace.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sample_dataset1.pdf
│
├── static/
│   └── styles.css
│
├── templates/
│   └── index.html
│
└── faiss_index/
    ├── index.faiss
    └── chunks.pkl
```

---

# Core Components

## app.py
Main Flask application.

Responsibilities:
- Handles frontend requests
- Retrieves relevant chunks
- Sends context to Groq LLM
- Displays final response

---

## create_vectorstore.py
Creates the vector database.

Responsibilities:
- Extracts PDF text
- Splits text into chunks
- Generates embeddings
- Stores vectors in FAISS

---

## extract_text.py
Handles PDF parsing and chunking.

Responsibilities:
- Extract raw text from PDFs
- Create semantic chunks

---

## query_llm.py
Responsible for LLM interaction.

Responsibilities:
- Retrieve relevant context
- Send prompts to Groq
- Generate final answers

---

## query_vectorspace.py
Used for testing semantic similarity retrieval directly from FAISS.

---

# Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ClauseMind.git
```

---

## 2. Move into Project Directory

```bash
cd ClauseMind
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Setting Up Groq API Key

## Step 1 — Open Groq Console

https://console.groq.com/keys

---

## Step 2 — Sign In

Create an account or log in.

---

## Step 3 — Generate API Key

Click:

```text
Create API Key
```

Copy generated key.

---

## Step 4 — Create `.env` File

Create a file named:

```text
.env
```

Add:

```env
GROQ_API_KEY=your_api_key_here
```

Example:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
```

---

# Creating Vector Database

Run:

```bash
python create_vectorstore.py
```

This process:
- extracts PDF text,
- creates semantic chunks,
- generates embeddings,
- stores vectors in FAISS.

---

# Running the Application

Start Flask server:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# Example Queries

```text
Is knee surgery covered in first 2 months?
```

```text
What is the waiting period for hospitalization?
```

```text
What happens if fraud is detected?
```

```text
Is appendix surgery covered?
```

---

# Example Output

```text
Knee surgery is not covered before completion of the waiting period.
```

---

# How Retrieval-Augmented Generation (RAG) Works

Traditional LLMs rely only on pretrained knowledge.

ClauseMind uses RAG architecture:

1. Retrieve relevant policy clauses
2. Provide clauses as context to LLM
3. Generate grounded response

This reduces:
- hallucinations
- irrelevant answers
- generic chatbot behavior

and improves:
- factual accuracy
- policy relevance
- domain-specific reasoning

---

# Challenges Faced

- Poor chunk retrieval
- Random PDF chunk splitting
- Hallucinated LLM responses
- Long verbose answers
- API model deprecations
- Keras/TensorFlow dependency conflicts
- GitHub secret scanning issues

---

# Optimizations Implemented

- Recursive semantic chunking
- Embedding normalization
- Improved prompt engineering
- Concise response formatting
- Reduced hallucinations
- Removed unnecessary source chunk display
- Optimized retrieval pipeline

---

# Future Improvements

- Multi-document support
- OCR support for scanned PDFs
- JSON structured decision output
- Claim approval engine
- Streamlit deployment
- User authentication
- Legal contract analysis
- Multi-language support
- Real-time policy upload

---

# Sample Use Cases

- Insurance claim assistance
- Policy clause retrieval
- Legal document analysis
- Contract search engine
- Healthcare policy support
- Automated underwriting assistance

---

# Security Notes

- `.env` is excluded using `.gitignore`
- API keys should never be pushed to GitHub
- Vector databases are stored locally

---

# Author

## Manjit Bajaj

AI/ML & Full Stack Enthusiast  
Focused on LLMs, RAG systems, IoT, and intelligent automation.

---

# License

This project is intended for educational and research purposes.
