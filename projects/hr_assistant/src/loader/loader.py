import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import get_chunk_id
from langchain_core.documents import Document

class Loader:
    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not self._validate_directory_path(file_path):
            raise ValueError(f"Directory path {file_path} is not valid")

        # Support both text and pdf loading
        self.loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader
        }
        # Support both txt and pdf files
        self.loaders_list = [
            DirectoryLoader(path=self.file_path, glob="**/*.pdf", loader_cls=self._create_loader),
            DirectoryLoader(path=self.file_path, glob="**/*.txt", loader_cls=self._create_loader),
        ]

    def _validate_directory_path(self, file_path):
        result = True
        if not os.path.isdir(file_path):
            print(f"Directory path {file_path} does not exist")
            result = False
        if not os.listdir(file_path):
            print(f"Directory path {file_path} is empty")
            result = False
        return result

    def _create_loader(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = self.loaders.get(ext, TextLoader)
        return loader_class(file_path)

    def load(self):
            # Gather all raw document segments (PDFs will have many, TXTs will have one)
            raw_documents = []
            for loader in self.loaders_list:
                raw_documents.extend(loader.load())

            # Normalize: Group segments by their source file. The pypdfloader breaks
            # each page out as a document. This can be an issue when a paragraph spans
            # pages. This is not an issue for the txt loader. 
            grouped_by_source = {}
            for doc in raw_documents:
                source = doc.metadata.get("source", "unknown")
                if source not in grouped_by_source:
                    grouped_by_source[source] = []
                grouped_by_source[source].append(doc)

            all_chunks = []

            # Process each complete file as a single semantic stream
            for source, segments in grouped_by_source.items():
                # Join segments (pages) with a newline to preserve continuous flow
                full_content = "\n\n".join([seg.page_content for seg in segments])
                
                # Create a single "Master Document" for this file
                # We preserve the metadata from the first segment as a template
                master_doc = Document(
                    page_content=full_content,
                    metadata=segments[0].metadata
                )

                # Split the FULL document content
                chunks = self.text_splitter.split_documents([master_doc])
                
                document_name = os.path.basename(source)
                for i, chunk in enumerate(chunks, start=1):
                    # Inject IDs and Metadata required by your VectorStoreAdapter
                    chunk.metadata["document_name"] = document_name
                    chunk.metadata["chunk_number"] = i
                    # Create a hash based on the content
                    chunk.metadata["id"] = get_chunk_id(chunk.page_content)
                    all_chunks.append(chunk)

            return all_chunks