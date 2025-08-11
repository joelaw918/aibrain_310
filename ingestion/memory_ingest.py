import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

from PIL import Image, ImageOps
import pytesseract

from pathlib import Path

import nltk
import json  

# Download punkt tokenizer for chunking if not already
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def read_image(file_path):
    try:
        image = Image.open(file_path)
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        threshold = 128
        image = image.point(lambda p: 255 if p > threshold else 0)
        text = pytesseract.image_to_string(image, lang='eng')
        print(f"[INFO] OCR output for {file_path.name}: '{text.strip()}'")
        return text.strip()
    except Exception as e:
        print(f"[WARN] Error reading image {file_path}: {e}")
        return ""

def read_pdf(file_path):
    try:
        return extract_pdf_text(file_path).strip()
    except Exception as e:
        print(f"[WARN] Error reading PDF {file_path}: {e}")
        return ""

def read_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        print(f"[WARN] Error reading DOCX {file_path}: {e}")
        return ""

def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[WARN] Error reading TXT {file_path}: {e}")
        return ""

def chunk_text(text, max_tokens=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk).split()) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def load_files_from_folder(folder_path):
    folder = Path(folder_path).resolve()
    print(f"[INFO] Loading files from: {folder}")
    documents = []
    metadata = []

    for file_path in folder.iterdir():
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix == '.pdf':
                text = read_pdf(file_path)
            elif suffix == '.docx':
                text = read_docx(file_path)
            elif suffix == '.txt':
                text = read_txt(file_path)
            elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                text = read_image(file_path)
            else:
                print(f"[INFO] Skipping unsupported file type: {file_path.name}")
                continue

            if text:
                print(f"[DEBUG] Extracted {len(text)} chars from {file_path.name}")
                chunks = chunk_text(text)
                print(f"[INFO] File '{file_path.name}' split into {len(chunks)} chunks.")

                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        "source": file_path.name,
                        "chunk_id": i,
                        "text": chunk
                    })
            else:
                print(f"[INFO] No text extracted from {file_path.name}")

    return documents, metadata

def main():
    base_dir = Path(__file__).parent.resolve()
    data_folder = base_dir.parent / "data"
    index_folder = base_dir.parent / "index"
    index_folder.mkdir(exist_ok=True)

    documents, metadata = load_files_from_folder(data_folder)
    if not documents:
        print("[ERROR] No supported files found.")
        return

    print(f"[INFO] Generating embeddings for {len(documents)} chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_folder / "faiss_index.index"))

    with open(index_folder / "chunks_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Indexed {len(documents)} chunks from {len(set(m['source'] for m in metadata))} files.")

if __name__ == "__main__":
    main()
