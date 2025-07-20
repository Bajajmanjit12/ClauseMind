# ClauseMind 
Semantic Decision Engine for Insurance and Legal Document Queries

## 🚀 Overview

ClauseMind is an AI-powered system that allows users to input natural language queries such as:

> "46-year-old male, knee surgery, Pune, 3-month-old policy"

The system then:
- Parses the query into structured data
- Retrieves relevant clauses from insurance documents, contracts, or emails
- Applies rules or LLM reasoning to give a decision
- Returns a structured JSON with:
  - ✅ Decision (Approved/Rejected)
  - 💰 Amount
  - 📌 Justification (Clause references)

---

## 🔧 Tech Stack

- 🧠 OpenAI GPT-4 / Claude / Gemini
- 🔍 FAISS / Chroma for vector search
- 📄 PyMuPDF / python-docx for file parsing
- 🧠 LangChain (optional for RAG pipelines)
- 💻 Streamlit (for UI)
- 🐍 Python 3.10+

---

---

## 🧪 Sample Input

> "32F, appendix surgery, 2-month policy, Mumbai"

### 🔁 Output

```json
{
  "decision": "Rejected",
  "amount": 0,
  "justification": "Clause 4.2: Surgery not covered in policy period < 3 months."
}
```
## 📦 Folder Structure

ClauseMind/
├── app/               # Streamlit or Flask frontend
├── parser/            # Query parser using LLM
├── retriever/         # Embedding and retrieval (RAG)
├── decision/          # Logic/rule-based evaluator
├── docs/              # Sample policy/contract documents
├── test_queries/      # Example queries for testing
├── utils/             # File loaders, embedding helpers, etc.
├── requirements.txt   # Python dependencies
└── README.md
