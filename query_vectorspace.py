# Purpose

# Search relevant clauses from FAISS.

# This is RETRIEVAL step.

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

query = "What is insurance?"
docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    print(f"📄 Chunk {i}:\n", doc.page_content, "\n")
