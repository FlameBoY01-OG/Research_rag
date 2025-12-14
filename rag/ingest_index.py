import pickle
from ingest.embed import get_embedding
from ingest.load_pdf import load_pdfs
from ingest.chunk import chunk_pdf_documents
from rag.vectorstore import FaissVectorStore


INDEX_DIR = "rag/index"
FAISS_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH = f"{INDEX_DIR}/metadata.pkl"


def main():
    print("Loading PDFs...")
    documents = load_pdfs("data/papers")

    if not documents:
        raise RuntimeError("No PDFs found in data/papers")

    print("Chunking documents...")
    chunks = chunk_pdf_documents(documents)

    print("Building FAISS index...")
    DIM = 768
    store = FaissVectorStore(dim=DIM)

    for i, c in enumerate(chunks, 1):
        emb = get_embedding(c["text"])
        store.add(emb, c)

        if i % 20 == 0:
            print(f"Embedded {i}/{len(chunks)} chunks")

    print("Saving FAISS index...")
    store.save(FAISS_PATH)

    print("Saving metadata...")
    with open(META_PATH, "wb") as f:
        pickle.dump(store.metadata, f)

    print("Ingestion complete ✅")


if __name__ == "__main__":
    main()
