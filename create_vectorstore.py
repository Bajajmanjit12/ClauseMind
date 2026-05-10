import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer

from extract_text import extract_text_from_pdf, chunk_text


# =========================
# PATHS
# =========================

PDF_PATH = "data/sample_dataset1.pdf"

FAISS_DIR = "faiss_index"

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "index.faiss"
)

CHUNKS_PATH = os.path.join(
    FAISS_DIR,
    "chunks.pkl"
)


# =========================
# CREATE DIRECTORY
# =========================

os.makedirs(FAISS_DIR, exist_ok=True)


# =========================
# LOAD MODEL
# =========================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# EXTRACT PDF TEXT
# =========================

print("Extracting text from PDF...")

pdf_text = extract_text_from_pdf(PDF_PATH)


# =========================
# CREATE CHUNKS
# =========================

print("Creating chunks...")

chunks = chunk_text(
    pdf_text,
    chunk_size=300,
    overlap=80
)

print(f"Total chunks: {len(chunks)}")


# =========================
# CREATE EMBEDDINGS
# =========================

print("Generating embeddings...")

embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
).astype("float32")


# normalize embeddings
faiss.normalize_L2(embeddings)


# =========================
# CREATE INDEX
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# =========================
# SAVE INDEX
# =========================

faiss.write_index(index, INDEX_PATH)

print("FAISS index saved.")


# =========================
# SAVE CHUNKS
# =========================

with open(CHUNKS_PATH, "wb") as f:

    pickle.dump(chunks, f)

print("Chunks saved.")


print("\n✅ VECTOR STORE CREATED")