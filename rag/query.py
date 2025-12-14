import pickle
from ingest.embed import get_embedding
from rag.vectorstore import FaissVectorStore
from rag.generator import generate_answer


INDEX_DIR = "rag/index"
FAISS_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH = f"{INDEX_DIR}/metadata.pkl"


def main():
    print("Loading vector database...")

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    store = FaissVectorStore(dim=768)
    store.load(FAISS_PATH, metadata)

    print("Ready to answer questions 🚀")

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            print("Empty question. Try again.")
            continue

        query_embedding = get_embedding(question)
        top_chunks = store.search(query_embedding, top_k=3)

        context = "\n\n".join(
            [
                f"[{c['source']} | page {c['page']}]\n{c['text']}"
                for c in top_chunks
            ]
        )

        answer = generate_answer(context, question)

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES:")
        for c in top_chunks:
            print(f"- {c['source']} (page {c['page']})")


if __name__ == "__main__":
    main()
