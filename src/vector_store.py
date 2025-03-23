import faiss
import numpy as np

# Create and store the FAISS vector store
def create_vector_store(embeddings):
    embeddings_np = np.array(embeddings)
    dimension = embeddings_np.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    # Save FAISS index
    faiss.write_index(index, "embeddings/vector_store.index")

    return index

# Load FAISS vector store
def load_vector_store():
    return faiss.read_index("embeddings/vector_store.index")
