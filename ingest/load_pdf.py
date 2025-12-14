from pathlib import Path
from pypdf import PdfReader


def load_pdfs(pdf_dir="data/papers"):
    documents = []

    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "source": pdf_path.name,
                    "page": page_num + 1,
                    "text": text
                })

    return documents
