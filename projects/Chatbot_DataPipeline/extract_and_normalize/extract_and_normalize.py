import validators
import argparse
from urllib.parse import urlparse
from extract_and_normalize_pdf import run_pdf_extraction
from extract_and_normalize_html import run_html_extraction
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sustainable agriculture metadata from a document")
    parser.add_argument("--url", required=True, help="URL to the document or local file path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--load-env", default=False, help="Load environment variables when not using docker")
    parser.add_argument("--result-file", default=None, help="Path to write results JSON for Prefect integration")

    args = parser.parse_args()

    # Determine if we should use PDF extraction
    use_pdf_extractor = False
    
    # Local file check
    if Path(args.url).exists():
        use_pdf_extractor = True
    # URL checks
    elif validators.url(args.url):
        url_lower = args.url.lower()
        domain = urlparse(url_lower).netloc
        # arxiv has a special way of handling pdfs, mdpi is just odd where some urls do not end in pdf but could be:
        # https://www.mdpi.com/2073-4395/14/10/2423/pdf?version=1729328266
        use_pdf_extractor = url_lower.endswith("pdf") or domain in ["arxiv.org", "www.mdpi.com", "mdpi.com", "openknowledge.fao.org", "linkinghub.elsevier.com"]

    # Run appropriate extractor
    if use_pdf_extractor:
        run_pdf_extraction(args)
    else:
        run_html_extraction(args)

