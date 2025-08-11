from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a small, fast embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
documents = [
    "Artificial Intelligence is fascinating.",
    "I love machine learning.",
    "Privacy is critical for AI applications.",
    "Python is a great programming language.",
    "NVIDIA GPUs accelerate deep learning."
]

# Generate embeddings
embeddings = model.encode(documents)

# Convert to float32 numpy array (required by FAISS)
embeddings = np.array(embeddings).astype('float32')

# Build FAISS index (flat index for simplicity)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query example
query = "How to keep AI private?"
query_vec = model.encode([query]).astype('float32')

# Search top 2 most similar docs
D, I = index.search(query_vec, k=2)

print("Query:", query)
print("Top results:")
for i, idx in enumerate(I[0]):
    print(f"{i+1}. {documents[idx]} (distance: {D[0][i]:.4f})")
