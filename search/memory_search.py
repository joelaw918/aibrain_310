import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path

def load_index_and_metadata():
    base_dir = Path(__file__).parent.resolve()
    index_folder = base_dir.parent / "index"

    index = faiss.read_index(str(index_folder / "faiss_index.index"))
    with open(index_folder / "chunks_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

def search(query, model, index, metadata, top_k=5):
    query_embedding = model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")
    k = min(top_k, index.ntotal)  # avoid requesting too many neighbors
    distances, indices = index.search(query_vector, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            results.append({
                "source": metadata[idx]["source"],
                "chunk_id": metadata[idx]["chunk_id"],
                "distance": float(dist)
            })
    return results

def main():
    base_dir = Path(__file__).parent.resolve()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_index_and_metadata()
    print("[DEBUG] Index ntotal:", index.ntotal)
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = search(query, model, index, metadata)
        print("\nTop matches:")
        for res in results:
            print(f"{res['source']} (chunk {res['chunk_id']}), distance: {res['distance']:.4f}")

if __name__ == "__main__":
    main()
