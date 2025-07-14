from typing import List, Dict, Tuple
from uuid import uuid4
from chunkers.base import AbstractChunker
import copy
import json
import jsonschema
from jsonschema import validate
from pathlib import Path
import logging

class OverlapParagraphChunks(AbstractChunker):
    def __init__(self, logger: logging.Logger, chunk_size: int = 1024, overlap_percentage: int = 20, min_chunk_size: int = 100, max_lookahead: int = 3, min_words_to_include: int = 15):
        self.logger = logger
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.overlap_words = int(chunk_size * self.overlap_percentage / 100)
        self.min_chunk_size = min_chunk_size
        self.max_lookahead = max_lookahead
        self.min_words_to_include = min_words_to_include
        self.schema = self._load_schema(Path("schemas/sustainable_agriculture.schema.json"))

    def _load_schema(self, schema_path: Path):
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        return schema

    def _create_chunk(self, chunk_text: str, chunk_idx: int, chunk_schema: dict) -> dict:
        """Create a chunk with only schema-defined fields"""
        section_title = chunk_schema.get("section_title", "Unknown Section")
        # Do not invlude section title in text if its Unknown or contains what looks like a page number from a headrt
        text = chunk_text if section_title == "Unknown Section" or section_title.startswith("Page") else f'{section_title}\n\n{chunk_text}'
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
                    "chunk_index": chunk_idx,  # Preserve existing chunk_index behavior
                    "section_title": section_title,
                    "text": text,
                    "paragraph_id": chunk_schema["paragraph_id"],
                    "paragraph_index": chunk_schema["paragraph_index"]
                }
                
        try:
            validate(instance=chunk, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"Validation failed for chunk: {e.message}")
            raise ValueError(f"Validation failed for chunk: {e.message}") 
        
        return chunk

    def _merge_paragraphs(self, paragraphs: List[str], start_idx: int) -> Tuple[str, int]:
        """Attempt to merge paragraphs starting from start_idx to meet min_chunk_size.
        Returns the merged text and the number of paragraphs merged.
        
        """
        current_text = paragraphs[start_idx]
        paragraphs_merged = 1  # Start with 1 as we always include the first paragraph
        best_merged_text = current_text  # Keep track of the best merge we've found
        
        # Look ahead up to max_lookahead paragraphs
        for i in range(1, min(self.max_lookahead, len(paragraphs) - start_idx)):
            next_paragraph = paragraphs[start_idx + i]
            
            # Try merging with the next paragraph
            merged_text = current_text + "\n\n" + next_paragraph
            words = merged_text.split()
            
            # If we meet min_chunk_size, this is our best option so far
            if len(words) >= self.min_chunk_size:
                best_merged_text = merged_text
                paragraphs_merged = i + 1
                break
                
            # If we don't meet min_chunk_size, keep track of this merge
            # but continue looking for a better one
            current_text = merged_text
            best_merged_text = merged_text  # Update best merge with current attempt
            
        return best_merged_text, paragraphs_merged

    def generate_chunks(self, text: str, metadata: dict, merge_paragraphs: bool = False) -> List[dict]:
        """Generate overlapping chunks from a text based on the chunker-size and metadata
           chunker-size depends on the number of words for a given embedding model
        """
        chunks = []
        paragraphs = text.split('\n\n')
        final_text_to_overlap = ""
        i = 0
        
        skipped_chunks = 0

        while i < len(paragraphs):
            paragraph_id = str(uuid4())  # Generate unique ID for this paragraph
            
            # Check if we need to merge paragraphs
            current_paragraph = paragraphs[i]
            words = current_paragraph.split()
            
            if (len(words) < self.min_chunk_size) and (merge_paragraphs):
                # Try to merge with subsequent paragraphs
                merged_text, paragraphs_merged = self._merge_paragraphs(paragraphs, i)
                paragraph_text = final_text_to_overlap + merged_text
                i += paragraphs_merged
            else:
                paragraph_text = final_text_to_overlap + current_paragraph
                i += 1
                
            words = paragraph_text.split()
            
            # Check if paragraph needs to be split
            if len(words) <= self.chunk_size:
                # Paragraph fits in one chunk
                chunk_text = ' '.join(words)
                chunk_schema = copy.deepcopy(metadata) if metadata else {}
                chunk_schema.update({
                    "chunk_id": str(uuid4()),
                    "paragraph_id": paragraph_id,
                    "paragraph_index": 0
                })

                chunk = self._create_chunk(chunk_text, len(chunks), chunk_schema)               
                final_text_to_overlap = ""

                # Only append if the chunk has a reasonable size to include in the final text
                if (len(words) > self.min_words_to_include):
                    chunks.append(chunk)
                else:
                    #self.logger.debug(f'Skipping chunk "{chunk["text"].replace('\n', ' ')}" with {len(words)} words')
                    skipped_chunks += 1
            else:
                # Paragraph needs to be split
                step = self.chunk_size - self.overlap_words
                split_count = 0
                
                for start in range(0, len(words), step):
                    chunk_text = ' '.join(words[start:start + self.chunk_size])
                    chunk_schema = copy.deepcopy(metadata) if metadata else {}
                    chunk_schema.update({
                        "chunk_id": str(uuid4()),
                        "paragraph_id": paragraph_id,
                        "paragraph_index": split_count
                    })

                    chunk = self._create_chunk(chunk_text, len(chunks), chunk_schema)                   
                    chunks.append(chunk)
                    final_text_to_overlap = ' '.join(chunk_text.split()[-self.overlap_words:])
                    split_count += 1

        return chunks, final_text_to_overlap, skipped_chunks
    

