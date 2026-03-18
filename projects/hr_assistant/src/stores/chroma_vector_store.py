from .vector_store_adapter import VectorStoreAdapter
from langchain_community.vectorstores import Chroma
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore
import chromadb
import os

class ChromaVectorStore(VectorStoreAdapter):
    def __init__(self, persist_directory: str, embeddings):
        super().__init__(persist_directory, embeddings)
        self.store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

    def db_exists(self) -> bool:
        # Check for the existence of the database directory
        return os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0

    def collection_exists(self, collection_name: str) -> bool:
        result = False
        if self.db_exists():
            client = chromadb.PersistentClient(path=self.persist_directory)
            result = collection_name in [c.name for c in client.list_collections()]
            
        return result

    def get_store(self, collection_name: str, documents: List[Document]) -> VectorStore :
        client = chromadb.PersistentClient(path=self.persist_directory)
        collection = client.get_or_create_collection(name=collection_name)
        
        # Generate stable IDs from chunk metadata (document_name, page_number, chunk_number)
        new_ids = [doc.metadata.get("id") or self._make_chunk_id(doc) for doc in documents]
        
        # Check what already exists in the DB
        existing_data = collection.get(ids=new_ids)
        existing_ids = set(existing_data['ids'])
        
        # Filter out only the documents that are NOT in the database
        docs_to_add = [
            doc for doc, doc_id in zip(documents, new_ids) 
            if doc_id not in existing_ids
        ]
        ids_to_add = [doc_id for doc_id in new_ids if doc_id not in existing_ids]

        # Initialize the LangChain wrapper
        store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

        # Only add if there is new content to avoid duplicates
        if docs_to_add:
            print(f"Adding {len(docs_to_add)} new documents to {collection_name}")
            store.add_documents(documents=docs_to_add, ids=ids_to_add)
        else:
            print(f"All documents already exist in {collection_name}. Skipping.")

        return store