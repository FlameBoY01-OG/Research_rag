def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """
    Splits text into overlapping chunks.

    Args:
        text: Input text
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


if __name__ == "__main__":
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(c)
