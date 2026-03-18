from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import VectorStore
from src.utils import get_chunk_id

class VectorStoreAdapter(ABC):
    def __init__(self, persist_directory: str, embeddings):
        self.persist_directory = persist_directory
        self.embeddings = embeddings

    @abstractmethod
    def db_exists(self) -> bool:
        """Check if the physical database files exist on disk."""
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a specific collection exists within the DB."""
        pass

    @abstractmethod
    def get_store(self, collection_name: str, documents: List[Document]) -> VectorStore:
        """Retrieve the store, or initialize it with documents if provided."""
        pass

    def _make_chunk_id(self, doc: Document) -> str:
        """Generate a stable, unique ID for a chunk from its metadata or content."""
        # Use metadata-based stable identifiers if available
        doc_name = doc.metadata.get("document_name")
        page = doc.metadata.get("page_number")
        chunk = doc.metadata.get("chunk_number")
        
        if doc_name is not None and page is not None and chunk is not None:
            return f"{doc_name}_{page}_{chunk}"
        
        return get_chunk_id(doc.page_content)