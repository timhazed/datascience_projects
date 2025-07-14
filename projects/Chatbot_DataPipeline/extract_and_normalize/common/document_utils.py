import json
import os
import uuid
from urllib.parse import urlparse
from os.path import basename, splitext, join
import re
import unicodedata
import hashlib

class DocumentUtils:
    """Utility class for document processing operations."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def get_base_filename_from_url(self, url):
        """
        Generate a unique, filesystem-friendly identifier from a URL.
        
        This method creates a unique identifier by combining the domain name,
        path components, and original filename. It handles potential collisions
        by including a hash of the full URL when necessary.
        
        Args:
            url (str): The URL of the PDF document
            
        Returns:
            str: A unique identifier suitable for filesystem use
            
        Raises:
            Exception: If URL parsing fails
        """
        result = None
        try:
            parsed_url = urlparse(url)
            
            # Extract domain with TLD 
            domain_parts = parsed_url.netloc.split('.')
            
            if len(domain_parts) >= 2:
                # For domains like example.org, fao.org, etc.
                domain_name = domain_parts[-2]  # e.g., 'fao' from 'fao.org'
                tld = domain_parts[-1]         # e.g., 'org' from 'fao.org'
                domain = f"{domain_name}_{tld}"  # e.g., 'fao_org'
            else:
                # Fallback for unusual domains
                domain = parsed_url.netloc
            
            # Get original filename with and without extension for proper path filtering
            original_filename_with_ext = basename(parsed_url.path)
            original_filename = splitext(original_filename_with_ext)[0]
            
            # Clean the original filename (remove special chars, limit length)
            clean_filename = re.sub(r'[^\w\-]', '_', original_filename)
            clean_filename = clean_filename[:50] if len(clean_filename) > 50 else clean_filename
            
            # Extract path components, excluding the filename part
            # Use both with and without extension to properly filter
            path_parts = []
            path_segments = parsed_url.path.split('/')
            for p in path_segments:
                if (p and p != original_filename_with_ext and 
                      p != original_filename and 
                      not p.endswith('.pdf')):
                    # Only take meaningful path segments (exclude common patterns like /documents/ etc.)
                    if not re.match(r'^(docs?|documents|publications|files|pdfs?|downloads?)$', p, re.IGNORECASE):
                        path_parts.append(p)
            
            # Limit to last 2 meaningful segments
            path_parts = path_parts[-2:] if len(path_parts) > 2 else path_parts
            clean_path = '_'.join([re.sub(r'[^\w\-]', '_', p) for p in path_parts])
            
            # Create a base identifier combining domain, path, and filename
            if clean_path:
                base_id = f"{domain}_{clean_path}_{clean_filename}"
            else:
                base_id = f"{domain}_{clean_filename}"
                
            base_id = re.sub(r'_{2,}', '_', base_id)  # Replace multiple underscores with single
            
            # Add a short hash for additional uniqueness
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            unique_id = f"{base_id}_{url_hash}"
            
            result = unique_id
        except Exception as e:
            self.logger.exception(f"Failed to generate unique ID from URL: {url}")
            # Fallback to a hash-only approach if parsing fails
            result = f"doc_{hashlib.md5(url.encode()).hexdigest()[:16]}"

        return result

    def prepare_output_directory(self, base_filename, output_dir):
        """Create output directory and clean previous files for same document."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Remove existing files for this document
            for filename in os.listdir(output_dir):
                if filename.startswith(base_filename + "_"):
                    os.remove(join(output_dir, filename))
        except Exception as e:
            self.logger.exception("Failed to prepare output directory")
            raise

    def build_base_payload(self, metadata, url, inferred_title):
        """Construct standardized metadata payload from extracted data."""
        try:
            return {
                "title": metadata.get("title") or inferred_title,
                "source_url": url,
                "date_published": metadata.get("date_published"),
                "language": metadata.get("language"),
                "region_or_country": metadata.get("region_or_country"),
                "document_type": metadata.get("document_type"),
                "sustainability_dimensions": metadata.get("sustainability_dimensions"),
                "key_topics": metadata.get("key_topics"),
                "contains_harmful_practices": metadata.get("contains_harmful_practices"),
                "intended_audience": [a.lower() for a in metadata.get("intended_audience", [])],
                "source_name": self.infer_source_name_from_url(url),
                "doc_id": str(uuid.uuid5(uuid.NAMESPACE_URL, url))
            }
        except Exception as e:
            self.logger.exception("Failed to build base payload")
            raise

    def infer_source_name_from_url(self, url):
        """Determine document source organization from URL domain."""
        result = "Unknown"
        try:
            netloc = urlparse(url).netloc.lower()
            # Known organization mappings
            if "fao" in netloc:
                result = "FAO"
            elif "worldbank" in netloc:
                result = "World Bank"
            elif "cimmyt" in netloc:
                result = "CIMMYT"
            elif "undp" in netloc:
                result = "UNDP"
            result = netloc.replace("www.", "")
        except Exception as e:
            self.logger.warning("Failed to infer source name from URL")
        
        return result

    def save_base_payload(self, output_dir, base_filename, base_payload):
        """Save metadata payload as formatted JSON file."""
        try:
            payload_output = join(output_dir, f"{base_filename}_base_payload.json")
            with open(payload_output, "w", encoding="utf-8") as f:
                json.dump(base_payload, f, indent=2, ensure_ascii=False)

            return payload_output
            
        except Exception as e:
            self.logger.exception("Failed to save base payload JSON")
            raise

    def save_text_chunks(self, output_dir, base_filename, sections, prefix="page"):
        """Save each document section as separate text file."""
        chunks_output = join(output_dir, f"{base_filename}_page_")
        try:
            for i, section in enumerate(sections):
                if not section["text"]:
                    continue
                
                # Use the page_number from section, or fallback to (i+1) for 1-based indexing
                identifier = section.get("page_number", i + 1)
                section_title = section.get("section_title", f"Page {identifier}")
                
                # Include section title as header in the content
                formatted_content = f"--- {section_title} ---\n\n{section['text']}"

                # Save with page number in filename for consistent referencing
                with open(join(output_dir, f"{base_filename}_{prefix}_{identifier}.txt"), "w", encoding="utf-8") as f:
                    f.write(formatted_content)
                                    
            return chunks_output
        except Exception as e:
            self.logger.exception("Failed to save one or more text chunks")
            raise

    # Strip control characters but keep punctuation and line breaks
    def _strip_control_chars(self, s):
            return ''.join(
                c for c in s
                if unicodedata.category(c)[0] != "C" or c == "\n"
            )
    
    # Reconstruct paragraphs
    def _reconstruct_paragraphs(self, text):
        """
        Intelligently reconstructs paragraph structure from text that may have been broken
        during PDF extraction or processing.
        
        Args:
            text (str): Text with potentially broken paragraph structure
            
        Returns:
            str: Text with properly formatted paragraphs separated by double newlines
        """
        # First check if text already has well-defined paragraphs (separated by double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            # Text already has multiple paragraphs - preserve existing structure
            return "\n\n".join(paragraphs)

        # For text without clear paragraph breaks, we need to infer paragraph structure
        lines = text.splitlines()
        reconstructed = []  # Will hold our reconstructed paragraphs
        buffer = ""         # Temporary buffer to accumulate lines into a paragraph

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                # Empty line encountered - treat as paragraph break if buffer has content
                if buffer:
                    reconstructed.append(buffer.strip())
                    buffer = ""
                continue

            # Check if this line might end a paragraph by examining punctuation and next line
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            ends_with_punct = re.search(r"[.?!:]$", stripped)  # Line ends with sentence-ending punctuation
            starts_upper = next_line[:1].isupper()             # Next line starts with uppercase letter
            
            # Add current line to paragraph buffer with space separator
            # (leading space will be stripped later)
            buffer += " " + stripped

            # Heuristic: If line ends with punctuation and next line starts with uppercase,
            # it's likely a paragraph break
            if ends_with_punct and starts_upper:
                reconstructed.append(buffer.strip())
                buffer = ""

        # Add any remaining text in buffer to the reconstructed paragraphs
        if buffer:
            reconstructed.append(buffer.strip())

        # Join paragraphs with double newlines for clear paragraph separation
        return "\n\n".join(reconstructed) 
       
    def clean_text_block(self, text, max_newlines=2):
        """
        Normalize and clean extracted PDF text for semantic processing,
        while preserving paragraph-level structure via double line breaks.
        
        Args:
            text (str): Raw text extracted from PDF
            max_newlines (int): Maximum number of consecutive newlines to preserve
                                (default: 2 for paragraph separation)
        
        Returns:
            str: Cleaned text with normalized whitespace and preserved paragraph structure
        """
        result = ""

        if text:
            # Remove form feeds and null bytes
            text = re.sub(r"[\f\x00]", "", text)
            
            # Strip control characters
            text = self._strip_control_chars(text)
            
            # Convert all whitespace-only lines to newlines
            text = re.sub(r"^\s+$", "\n", text, flags=re.MULTILINE)
            
            # Normalize spaces (collapse multiple spaces into one)
            text = re.sub(r" +", " ", text)
            
            # Remove spaces at the beginning or end of lines
            text = re.sub(r"^ +| +$", "", text, flags=re.MULTILINE)
            
            # Convert runs of 3+ newlines into exactly max_newlines (default: 2 for paragraphs)
            text = re.sub(r"\n{%d,}" % (max_newlines + 1), "\n" * max_newlines, text)
            
            # Reconstruct paragraphs
            result = self._reconstruct_paragraphs(text)

        return result







