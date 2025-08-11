# AI Brain – Document Memory & Search

This project implements **Phases 1–5** of a local Retrieval-Augmented Generation (RAG) pipeline:

- Document ingestion & chunking
- Embedding generation
- FAISS vector index creation
- Optional OCR for images
- Vector search testing

It’s now ready for **Phase 6** (FastAPI + Chat UI).

---

## 📂 Project Structure

```text
aibrain/
│
├── ingestion/
│   └── memory_ingest.py        # Loads docs, chunks text, generates embeddings, saves FAISS index
│
├── search/
│   └── memory_search.py        # Loads FAISS index, allows interactive search
│
├── index/
│   ├── faiss_index.index       # Saved FAISS vector index
│   └── chunks_metadata.json    # Chunk metadata (source file, chunk_id, etc.)
│
├── data/                       # Your input documents/images
│   ├── *.pdf
│   ├── *.docx
│   ├── *.txt
│   └── *.jpg / *.png
│
└── README.md                   # This file

---

## 📦 Installation

```bash
# 1. Clone this repo
git clone <your_repo_url>
cd aibrain

# 2. Create a virtual environment (recommended)
conda create -n aibrain_310 python=3.10
conda activate aibrain_310

# 3. Install dependencies
pip install -r requirements.txt

requirements.txt example:

sentence-transformers
faiss-cpu
pdfminer.six
python-docx
Pillow
pytesseract
nltk

⚙️ Usage
1️⃣ Ingest Documents
Place your .pdf, .docx, .txt, and image files (.jpg, .png, etc.) in the data/ folder.

Run ingestion:

python ingestion/memory_ingest.py

2️⃣ Search the Memory

Run:
python search/memory_search.py

Enter your query (or 'exit' to quit): AI

Example:

Top matches:
ai.docx (chunk 0), distance: 0.9128
ai.pdf (chunk 0), distance: 0.9128
sample.txt (chunk 0), distance: 0.9128

🔍 How It Works
Document Ingestion

Reads PDFs, DOCX, TXT, and images

OCR for scanned images

Chunks text into manageable sections

Embedding Generation

Uses sentence-transformers model (all-MiniLM-L6-v2)

Converts text chunks into numerical vectors

FAISS Index

Stores vectors in a FAISS index (IndexFlatL2)

Enables fast similarity search

Search

Encodes the query to a vector

Retrieves the top-k most relevant chunks

🛠 Next Steps (Phase 6)
Add FastAPI backend for API access

Implement chat interface (HTML/JS)

Integrate with local LLM for end-to-end RAG

📜 License
MIT License

