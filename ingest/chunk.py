def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Splits text into overlapping character-based chunks.

    Args:
        text (str): Input text
        chunk_size (int): Number of characters per chunk
        overlap (int): Number of overlapping characters

    Returns:
        list[str]: List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

        if start < 0:
            start = 0

    return chunks


def chunk_documents(documents, chunk_size: int = 500, overlap: int = 100):
    """
    Splits multiple documents into chunks while preserving source metadata.

    Args:
        documents (list[dict]): List of documents with keys:
            - "source": document name
            - "text": document content
        chunk_size (int): Characters per chunk
        overlap (int): Overlap between chunks

    Returns:
        list[dict]: List of chunks with source information
    """
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)

        for chunk in text_chunks:
            all_chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    # Simple local test (single text file)
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(c)
