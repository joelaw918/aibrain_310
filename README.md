# 🧠 AI Brain – Document & Image Memory Search

A simple system to **ingest documents and images**, chunk and embed their text content, store it in a **FAISS** vector index, and perform **semantic search**.

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
```text
---
## ⚙️ Installation
Clone the repository

git clone https://github.com/yourusername/aibrain.git
cd aibrain
Create & activate environment

conda create -n aibrain_310 python=3.10 -y
conda activate aibrain_310
Install dependencies

pip install -r requirements.txt

---

## 📥 Ingest Data
Place your .pdf, .docx, .txt, and .jpg/.png files into the data/ folder, then run:

python ingestion/memory_ingest.py
This will:

Extract text (via PDFMiner, python-docx, direct read, or Tesseract OCR for images).

Chunk text into smaller pieces.

Generate embeddings using SentenceTransformers.

Save the FAISS index and metadata into the index/ folder.

---

##🔍 Search the Memory
Run:

python search/memory_search.py
Example:

Enter your query (or 'exit' to quit): AI
Top matches:
ai.docx (chunk 0), distance: 0.9128
ai.pdf (chunk 0), distance: 0.9128
sample.txt (chunk 0), distance: 0.9128

---

##📦 Dependencies
sentence-transformers – For embeddings

faiss – For vector search

pdfminer.six – For PDF text extraction

python-docx – For Word document parsing

Pillow – For image handling

pytesseract – For OCR

nltk – For sentence tokenization

---


##🚀 Next Steps
Improve OCR preprocessing for better accuracy on images.

Add support for more file formats.

Implement a web UI for easier searching.
---

##📝 License
MIT License – feel free to use and modify.


---
