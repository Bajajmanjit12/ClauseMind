import pickle
import os
import faiss

from flask import Flask, render_template, request

from sentence_transformers import SentenceTransformer

from groq import Groq

from dotenv import load_dotenv


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found!")


# =========================
# FLASK APP
# =========================

app = Flask(__name__)


# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# LOAD FAISS INDEX
# =========================

print("Loading FAISS index...")

faiss_index = faiss.read_index(
    "faiss_index/index.faiss"
)


# =========================
# LOAD DOCUMENT CHUNKS
# =========================

with open("faiss_index/chunks.pkl", "rb") as f:

    documents = pickle.load(f)


# =========================
# GROQ CLIENT
# =========================

client = Groq(api_key=GROQ_API_KEY)


# =========================
# RETRIEVE RELEVANT CHUNKS
# =========================

def get_top_chunks(query, top_k=2):

    query_vec = embedding_model.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(query_vec)

    D, I = faiss_index.search(query_vec, top_k)

    results = []

    for idx in I[0]:

        if idx < len(documents):

            chunk = documents[idx].strip()

            # ignore broken chunks
            if len(chunk) > 40:

                results.append(chunk)

    return results


# =========================
# ASK GROQ
# =========================

def ask_groq_with_context(query):

    top_chunks = get_top_chunks(query)

    context = "\n\n".join(top_chunks)


    prompt = f"""
You are an insurance policy assistant.

Answer ONLY from the provided context.

STRICT RULES:
- Give answer in 1 or 2 short sentences only.
- No bullet points.
- No paragraph explanation.
- No extra details.
- No greetings.
- No formatting.
- No assumptions.
- If answer not found, say:
  "Information not found in policy."

Context:
{context}

User Question:
{query}

Short Answer:
"""


    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": "You answer insurance policy questions briefly and precisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_tokens=60
    )

    answer = response.choices[0].message.content.strip()

    return answer


# =========================
# HOME ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])

def home():

    answer = None

    if request.method == "POST":

        query = request.form.get("query")

        if query:

            answer = ask_groq_with_context(query)

    return render_template(
        "index.html",
        answer=answer
    )


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
