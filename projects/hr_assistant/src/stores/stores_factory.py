from .faiss_vector_store import FAISSVectorStore
from .chroma_vector_store import ChromaVectorStore

def get_vector_store(store_type: str, persist_directory: str, embeddings):
    if store_type == "faiss":
        return FAISSVectorStore(persist_directory, embeddings)
    elif store_type == "chroma":
        return ChromaVectorStore(persist_directory, embeddings)
    else:
        raise ValueError(f"Invalid store type: {store_type}")