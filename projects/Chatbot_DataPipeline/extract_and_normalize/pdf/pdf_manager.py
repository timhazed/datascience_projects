import requests
import fitz 
import os
from common.metadata_generator import MetadataGenerator
from common.schema_prompt_builder import SchemaPromptBuilder
from nltk.tokenize import word_tokenize
from common.document_utils import DocumentUtils
from urllib.parse import urlparse
import arxiv
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

LOW_VALUE_PHRASES = ["©", "photo by", "image", "figure", "page", "source:"]

class PdfManager:
    def __init__(self, url, config, logger):
        self.url = url
        self.config = config
        self.logger = logger
        self.doc_utils = DocumentUtils(self.logger)

        self.elsevier = False

        # Get actual url if its an arxiv url
        self.url, self.domain = self._get_url(url)
        self.headless_domains = self.config.get("pdf_extractor", {}).get("headless_domains", [])
        

    def _get_url(self, url):
        result = url
        # Parse the URL
        parsed_url = urlparse(url)

        # Extract the domain name
        domain = parsed_url.netloc  # This gives 'arxiv.org'
        arxiv_result = 'arxiv.org' == domain
        self.elsevier = 'linkinghub.elsevier.com' == domain

        if arxiv_result:
            # Extract the arXiv ID from the path
            path = parsed_url.path
            arxiv_id = path.split('/')[-1]

            try:
                search = arxiv.Search(id_list=[arxiv_id], max_results=1)
                result_iter = next(search.results())
                
                # Make sure result_iter is not None
                if result_iter is None:
                    raise ValueError(f"ArXiv API returned None for ID: {arxiv_id}")
                
                # Check if it has pdf_url attribute
                if hasattr(result_iter, 'pdf_url'):
                    result = result_iter.pdf_url
                    self.logger.info(f"Retrieved arXiv PDF URL: {result}")
                else:
                    raise AttributeError(f"ArXiv result does not have 'pdf_url' attribute for ID: {arxiv_id}")
                    
            except StopIteration:
                self.logger.error(f"No results found for arXiv ID: {arxiv_id}")
                raise ValueError(f"No results found in ArXiv API for ID: {arxiv_id}")
            except Exception as e:
                self.logger.error(f"Error fetching arXiv paper {arxiv_id}: {str(e)}")
                raise

        return result, domain
    
    def is_url(self):
        parsed = urlparse(self.url)
        return parsed.scheme in ("http", "https")

    def extract_pdf(self, filename):
        self.logger.info("Reading and chunking PDF content by page...")
        
        # Open the PDF with fitz
        doc = fitz.open(filename)
        metadata_title = doc.metadata.get("title") or "Untitled Document"
        sections = []
        seen_titles = set()
        page_num = 1

        for page in doc:
            try:
                page_dict = page.get_text("dict")
                text = page.get_text("text")

                if not text.strip():
                    self.logger.info(f"Skipping empty page {page_num}")
                    page_num += 1
                    continue

                cleaned_text = self.doc_utils.clean_text_block(text)
                word_count = len(word_tokenize(cleaned_text))

                # Filter based on token count and low-value phrase patterns to minimize near empty output pages
                if word_count < 30:
                    lowered = cleaned_text.lower()
                    if all(phrase in lowered for phrase in LOW_VALUE_PHRASES):
                        self.logger.info(f"Skipping caption-only page {page_num} ({word_count} words)")
                        page_num += 1
                        continue
                    elif word_count < 10:
                        self.logger.info(f"Skipping nearly empty page {page_num} ({word_count} words)")
                        page_num += 1
                        continue

                section_title = self.find_best_section_title(page_dict, seen_titles, self.config)
                section_title = section_title or f"Page {page_num}"
                seen_titles.add(section_title)

                sections.append({
                    "page_number": page_num,
                    "section_title": section_title[:128],
                    "text": cleaned_text
                })

            except Exception as e:
                self.logger.exception(f"Error processing page {page_num}: {str(e)}")
                raise

            page_num += 1

        doc.close()

        return metadata_title, sections

    def _infer_referer(self, pdf_url):
        """Strip /pdf suffix to get the article page as referer."""
        result = pdf_url
        parsed = urlparse(self.url)
        if parsed.path.endswith("/pdf"):
            referer_path = parsed.path[:-4]
            result = pdf_url.replace(parsed.path, referer_path)

        return result

    def _download_elsevier_pdf(self):
        """
        Download a PDF from ScienceDirect using the official Elsevier API and extract its sections.
        Requires ELSEVIER_API_KEY in the environment.
        """
        try:
            # Extract PII from the ScienceDirect URL
            parsed_url = urlparse(self.url)
            path_parts = parsed_url.path.split("/")
            if "pii" not in path_parts:
                raise ValueError("URL does not contain a PII. Cannot download PDF.")
            
            pii_index = path_parts.index("pii") + 1
            if pii_index >= len(path_parts):
                raise ValueError("Invalid PII in URL.")
            
            pii = path_parts[pii_index]

            # Retrieve API key from environment variable
            api_key = os.getenv("ELSEVIER_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing ELSEVIER_API_KEY in environment variables.")

            # Build API URL with apiKey as a query parameter
            api_url = f"https://api.elsevier.com/content/article/pii/{pii}?apiKey={api_key}"

            self.logger.info(f"Downloading Elsevier PDF for PII: {pii}")

            headers = {
                "Accept": "application/pdf",
                "X-ELS-APIKey": api_key,  # Still send header as good practice
            }

            response = requests.get(api_url, headers=headers, stream=True)
            if response.status_code != 200:
                self.logger.error(f"Failed to download Elsevier PDF: HTTP {response.status_code}")
                raise ConnectionError(f"PDF download failed with status code {response.status_code}")

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as temp_pdf:
                temp_pdf.write(response.content)
                temp_pdf.flush()

                metadata_title, sections = self.extract_pdf(temp_pdf.name)

            self.logger.info(f"Successfully downloaded and extracted Elsevier PDF with {len(sections)} sections.")
            return metadata_title, sections

        except Exception as e:
            self.logger.exception(f"Error downloading Elsevier PDF: {str(e)}")
            raise


    def download_headless_pdf(self):
        """Download a PDF from a URL requiring a headless browser (Playwright)."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            referer_url = self._infer_referer(self.url)
            print(f"[INFO] Visiting article page: {referer_url}")
            page.goto(referer_url)

            # Wait and click the PDF link
            with page.expect_download() as download_info:
                page.click("a[href$='/pdf']")  # Clicks the download link
            download = download_info.value

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as temp_pdf:
                # Write PDF content to the temporary file 
                download.save_as(temp_pdf.name)
                metadata_title, sections = self.extract_pdf(temp_pdf.name)

            return metadata_title, sections
        
    # Method to download a PDF from a URL, read its content, and extract sections based on page content
    # Uses the find_best_section_title function to determine section titles
    def extract_pdf_sections(self):
        try:
            # Is this a url or a local file?    
            if self.is_url():
                # Does the url require a headless browser?
                if self.domain in self.headless_domains and self.url.endswith("/pdf"):
                    self.logger.info(f"Downloading PDF using a headless browser from URL: {self.url}")
                    metadata_title, sections = self.download_headless_pdf()
                elif self.elsevier:
                    self.logger.info(f"Downloading PDF from elsevier URL: {self.url}")
                    metadata_title, sections = self._download_elsevier_pdf()
                else:
                    self.logger.info(f"Downloading PDF from URL: {self.url}")

                    # First try with no header if that fails, add a referer header to avoid being blocked by the server
                    response = requests.get(self.url)

                    if response.status_code != 200:
                        self.logger.info(f"Could not download PDF with no header: HTTP {response.status_code}")

                        # Add a referer header to avoid being blocked by the server
                        headers = { "User-Agent": "Mozilla/5.0", "Referer": self.url }
                        response = requests.get(self.url, headers=headers)

                        if response.status_code != 200:
                            self.logger.error(f"Failed to download PDF: HTTP {response.status_code}")
                            raise ConnectionError("PDF download failed")
                                    
                    # Use tempfile to create a temporary file that auto-deletes when closed
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as temp_pdf:
                        # Write PDF content to the temporary file
                        temp_pdf.write(response.content)
                        temp_pdf.flush()  # Ensure all data is written
                        
                        metadata_title, sections = self.extract_pdf(temp_pdf.name)
            else:
                metadata_title, sections = self.extract_pdf(self.url)
             
            if not sections:
                self.logger.error("No valid sections extracted from PDF")
                raise ValueError("PDF processing yielded no valid content")

            return metadata_title, sections

        except Exception as e:
            self.logger.exception(f"Error processing PDF: {str(e)}")
            raise

    # Function to find the best section title from a page's text dictionary
    # Considers font size, word count, capitalization, and ignores phrases specified in the config
    def find_best_section_title(self, page_dict, seen_titles, config):
        best_title = None
        largest_font_size = 0
        
        # Get list of phrases to ignore from config (e.g., "table of contents")
        ignore_phrases = [s.lower() for s in config.get("section_title_filter", {}).get("ignore_if_contains", [])]

        # Traverse the hierarchical structure of the PDF page (blocks -> lines -> spans)
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    candidate = span["text"].strip()
                    words = candidate.split()

                    if not words:
                        continue

                    # Calculate ratio of capitalized words for title detection
                    capitalized_words = [w for w in words if w[0].isupper()]
                    candidate_lc = candidate.lower()

                    # Apply multiple heuristics to identify a good section title:
                    # 1. Larger font size than previous candidates
                    # 2. Reasonable length (not too short or long)
                    # 3. Contains at least 3 words
                    # 4. At least 40% of words are capitalized
                    # 5. Doesn't contain any ignored phrases
                    # 6. Not already seen in previous pages
                    if (
                        span["size"] > largest_font_size
                        and 12 <= len(candidate) <= 128
                        and len(words) >= 3
                        and len(capitalized_words) / len(words) >= 0.4
                        and not any(bad in candidate_lc for bad in ignore_phrases)
                        and candidate not in seen_titles
                    ):
                        largest_font_size = span["size"]
                        best_title = candidate

        return best_title

    # Function to build a payload by extracting sections from a PDF and generating metadata
    # Saves the metadata and text chunks to the specified output directory
    def build_payload(self):
        # Load configuration and environment variables
        llm_document_chunk_size = self.config.get("llm_document_chunk_size", 2048)
        output_dir = self.config.get("output_directory", "output")

        # Extract base filename from URL for output files 
        base_filename = self.doc_utils.get_base_filename_from_url(self.url)

        # Create output directory and clean up any existing files for this document
        self.doc_utils.prepare_output_directory(base_filename, output_dir)

        # Download PDF and extract its sections
        inferred_title, sections = self.extract_pdf_sections()

        # Combine all text from all sections
        full_text = "\n\n".join([s["text"] for s in sections if s["text"]])

        # Create schema prompt builder and metadata generator
        schema_path = self.config.get("schema_path", "schemas/base_payload.schema.json")
        prompt_builder = SchemaPromptBuilder(schema_path, self.logger)
        metadata_gen = MetadataGenerator(self.config, self.doc_utils, self.logger, prompt_builder)
        metadata = metadata_gen.generate_metadata(self.url, full_text, llm_document_chunk_size)

        # Build the base payload with the generated metadata
        base_payload = self.doc_utils.build_base_payload(metadata, self.url, inferred_title)

        # Save the base payload as JSON and page chunks
        payload_output = self.doc_utils.save_base_payload(output_dir, base_filename, base_payload)
        chunks_output = self.doc_utils.save_text_chunks(output_dir, base_filename, sections, prefix="page")

        return base_filename, chunks_output, payload_output