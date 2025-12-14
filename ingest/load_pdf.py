from pathlib import Path
from pypdf import PdfReader


def load_pdfs(pdf_dir="data/papers"):
    documents = []

    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        reader = PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        documents.append({
            "source": pdf_path.name,
            "text": text
        })

    return documents
