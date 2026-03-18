import os
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore
from .vector_store_adapter import VectorStoreAdapter

class FAISSVectorStore(VectorStoreAdapter):
    def db_exists(self) -> bool:
        # FAISS requires the directory to exist and contain its core files
        if not os.path.exists(self.persist_directory):
            return False
        return self.collection_exists("index") # Defaulting to standard index check

    def collection_exists(self, collection_name: str) -> bool:
        # In a file-based FAISS setup, we check for the specific index files
        index_path = os.path.join(self.persist_directory, f"{collection_name}.faiss")
        pkl_path = os.path.join(self.persist_directory, f"{collection_name}.pkl")
        return os.path.exists(index_path) and os.path.exists(pkl_path)

    def get_store(self, collection_name: str, documents: List[Document]) -> VectorStore:
        """
        Loads the FAISS index and performs a duplicate-aware sync.
        """
        if self.collection_exists(collection_name):
            # allow_dangerous_deserialization is required for pickle-based loading
            store = FAISS.load_local(
                self.persist_directory, 
                self.embeddings, 
                index_name=collection_name,
                allow_dangerous_deserialization=True
            )
            
            # Identify existing IDs in the internal docstore (same stable ID scheme)
            existing_ids = set()
            for doc in store.docstore._dict.values():
                if "id" in doc.metadata:
                    existing_ids.add(doc.metadata["id"])
            
            # Assign stable IDs to documents and filter for truly new ones
            docs_with_ids = []
            for doc in documents:
                chunk_id = doc.metadata.get("id") or self._make_chunk_id(doc)
                doc.metadata["id"] = chunk_id
                if chunk_id not in existing_ids:
                    docs_with_ids.append(doc)
            
            new_docs = docs_with_ids
            
            if new_docs:
                print(f"Adding {len(new_docs)} new documents to FAISS index: {collection_name}")
                store.add_documents(new_docs)
                store.save_local(self.persist_directory, index_name=collection_name)
            else:
                print(f"FAISS index '{collection_name}' is already up to date.")
        else:
            # Build the index if it doesn't exist
            print(f"Initializing new FAISS index: {collection_name}")
            for doc in documents:
                doc.metadata["id"] = doc.metadata.get("id") or self._make_chunk_id(doc)
            store = FAISS.from_documents(documents, self.embeddings)
            store.save_local(self.persist_directory, index_name=collection_name)
            
        return store