import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path="faiss_index.index"):
    print(f"[INFO] Loading FAISS index from {index_path}...")
    return faiss.read_index(index_path)

def load_metadata(metadata_path="chunks_metadata.json"):
    print(f"[INFO] Loading chunk metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_query(query, model):
    return model.encode([query]).astype("float32")

def search_index(index, query_embedding, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_faiss_index()
    metadata = load_metadata()

    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting...")
            break

        query_embedding = embed_query(query, model)
        distances, indices = search_index(index, query_embedding, top_k=3)

        print(f"\nTop {len(indices)} results:")

        for rank, (dist, idx) in enumerate(zip(distances, indices), start=1):
            if idx == -1 or idx >= len(metadata):
                continue
            chunk_info = metadata[idx]
            print(f"\nRank {rank}:")
            print(f"Source file: {chunk_info['source']}")
            print(f"Chunk ID: {chunk_info['chunk_id']}")
            print(f"Distance: {dist:.4f}")
            print(f"Text:\n{chunk_info.get('text', 'No text available')}\n{'-'*40}")

if __name__ == "__main__":
    main()
