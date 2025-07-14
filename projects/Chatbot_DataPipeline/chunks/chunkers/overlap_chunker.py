from typing import List
from chunkers.base import AbstractChunker
import copy
import json
import jsonschema
from jsonschema import validate
from pathlib import Path
import logging

class OverlapChunks(AbstractChunker):
    def __init__(self, logger: logging.Logger, chunk_size: int = 1024, overlap_percentage: int = 20):
        self.logger = logger
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.overlap_words = int(chunk_size * self.overlap_percentage / 100)
        self.schema = self._load_schema(Path("../extract_and_normalize/schemas/sustainable_agriculture.schema.json"))

    def _load_schema(self, schema_path: Path):
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        return schema

    def generate_chunks(self, text: str, metadata: dict = None) -> List[dict]:
        """Generate overlapping chunks from a text based on the chunker-size and metadata
              chunker-size depends on the number of words for a given embedding model
           """
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap_words

        for index, start in enumerate(range(0, len(words), step)):
            chunk_text = ' '.join(words[start:start + self.chunk_size])
            chunk_schema = copy.deepcopy(metadata) if metadata else {}
            chunk_schema["chunk_id"] = f"{chunk_schema["doc_id"]}_{index}"
            

            chunk = {
                "title": chunk_schema.get("title", "Unknown"),
                "source_url": chunk_schema.get("source_url", "Unknown"),
                "date_published": chunk_schema.get("date_published", "Unknown"),
                "language": chunk_schema.get("language", "Unknown"),
                "region_or_country": chunk_schema.get("region_or_country", "Unknown"),
                "document_type": chunk_schema.get("document_type", "Unknown"),
                "sustainability_dimensions": chunk_schema.get("sustainability_dimensions", []),
                "key_topics": chunk_schema.get("key_topics", []),
                "contains_harmful_practices": chunk_schema.get("contains_harmful_practices", "Unknown"),
                "intended_audience": chunk_schema.get("intended_audience", []),
                "source_name": chunk_schema.get("source_name", "Unknown"),
                "doc_id": chunk_schema.get("doc_id", ""),
                "chunk_id": chunk_schema["chunk_id"],
                "section_title": chunk_schema.get("section_title", "Unknown Section"),
                "text": chunk_text,
            }

            try:
                validate(instance=chunk, schema=self.schema)
            except jsonschema.exceptions.ValidationError as e:
                self.logger.error(f"Validation failed for chunk {index}: {e.message}")
                raise ValueError(f"Validation failed for chunk {index}: {e.message}")
            final_text_to_overlap = ' '.join(chunk_text.split()[-self.overlap_words:])
            
            chunks.append(chunk)

        return chunks, final_text_to_overlap
    

