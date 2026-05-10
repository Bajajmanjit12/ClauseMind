# Purpose

# Sends:

# user query
# retrieved clauses

# to GPT/Gemini/Claude.

# query_llm.py
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in .env")

client = Groq(api_key=GROQ_API_KEY)

INDEX_DIR = "faiss_index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# load faiss index and chunks
index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

# embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def get_top_chunks(query: str, top_k: int = 3):
    q_vec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

def ask_groq(query: str):
    top_chunks = get_top_chunks(query, top_k=3)
    context = "\n\n".join(top_chunks)
    prompt = (
        "You are an insurance policy assistant. Use ONLY the context to answer the question. "
        "If the context doesn't have the answer, say you couldn't find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # Groq chat completion
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"   # change if needed per Groq models available
    )
    # response structure: resp.choices[0].message.content (mirrors earlier examples)
    try:
        return resp.choices[0].message.content
    except Exception:
        # fallback reading: some Groq versions return different dicts
        return resp.choices[0]["message"]["content"]

if __name__ == "__main__":
    while True:
        q = input("Ask (or type 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        answer = ask_groq(q)
        print("\nAnswer:\n", answer)
        print("\nTop context chunks:\n")
        for c in get_top_chunks(q, top_k=3):
            print("----\n", c[:600].strip(), "\n")
