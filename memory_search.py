import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def main():

    # Load FAISS index
    index = faiss.read_index("faiss_index.index")

    # Load metadata
    with open("chunks_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        # Embed the query
        query_vec = model.encode([query]).astype("float32")
    
        # Search top 3
        k = 3
        D, I = index.search(query_vec, k)
    
        print("\nTop results:")
        for rank, idx in enumerate(I[0]):
            if idx < len(metadata):
                m = metadata[idx]
                print(f"\nResult {rank+1}:")
                print(f"  Source: {m['source']}")
                print(f"  Chunk ID: {m['chunk_id']}")
                print(f"  Distance: {D[0][rank]:.4f}")
                print(f"  Text: {m['text'][:200]}{'...' if len(m['text'])>200 else ''}")
            else:
                print(f"[WARN] No metadata for index {idx}")
                
        print()

if __name__ == "__main__":
    main()