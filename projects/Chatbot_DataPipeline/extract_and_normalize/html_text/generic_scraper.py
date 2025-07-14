import re
import requests
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_log, after_log

class GenericScraper:
    """
    A generic web scraper that can handle both static and JavaScript-rendered websites.
    This class is designed to be configurable for any website structure.
    """
       
    def __init__(self, base_url, config, logger):
        """
        Initialize the scraper with the configuration.
        
        Args:
            base_url (str): The base URL of the website to scrape
            config (dict): Configuration dictionary to use directly
            logger (logging.Logger): Logger instance for logging
        """
        
        self.base_url = base_url

        # Load configuration
        self.config = config
        # Set headers for requests
        self.headers = self.config.get("html_parameters", {}).get("headers", {})
        
        # Set up logger
        self.logger = logger

  
    def _make_absolute_url(self, url, base_url=None):
        """Convert a relative URL to an absolute URL"""
        result = url
        actual_base = base_url or self.base_url
        if actual_base:
            if not urlparse(url).netloc:
                result = urljoin(actual_base, url)
                self.logger.debug(f"Converting relative URL '{url}' to absolute: '{result}'")
        return result
    
    def _should_use_js_rendering(self, url, page_type="topic_page"):
        """
        Determine if JavaScript rendering should be used for a given URL.
        
        This method evaluates whether to use JavaScript rendering based on two criteria:
        1. The configuration setting for the specific page type
        2. URL pattern matching against the configured js_patterns list
        
        Args:
            url (str): The URL to evaluate for JavaScript rendering
            page_type (str, optional): The type of page being scraped. Defaults to "topic_page".
            
        Returns:
            bool: True if JavaScript rendering should be used, False otherwise
        
        Note:
            The method first checks configuration settings. If not enabled globally,
            it then tries to match the URL against patterns in the js_patterns list.
        """
        result = False
        if self.config.get("html_parameters", {}).get(page_type, {}).get("use_js_rendering", False):
            result = True
            self.logger.debug(f"Using JS rendering for {url} based on configuration for {page_type}")
        else:
            for pattern in self.config.get("html_parameters", {}).get("rendering", {}).get("js_patterns", []):
                if re.search(pattern, url):
                    result = True
                    self.logger.debug(f"Using JS rendering for {url} based on pattern match: {pattern}")
                    break
                else:
                    self.logger.debug(f"No JS rendering for {url} — did not match pattern: {pattern}")
        return result
    
    def _render_with_playwright(self, url):
        """
        Render a webpage using Playwright to handle JavaScript.
        
        Args:
            url (str): The URL to render
            
        Returns:
            str: The HTML content after JavaScript execution
        """
        self.logger.info(f"Rendering page with Playwright: {url}")
        rendering_config = self.config.get("html_parameters", {}).get("rendering", {})
        wait_for = rendering_config.get("wait_for_selector", "")
        wait_time = rendering_config.get("wait_time", 5000)
        should_scroll = rendering_config.get("scroll", True)
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=self.headers.get("User-Agent"),
                viewport={'width': 1920, 'height': 1080},
                #javascript_enabled=True
            )
            
            try:
                page = context.new_page()
                
                # Add request interception for performance
                page.route("**/*.{png,jpg,jpeg,gif,svg,css,font}", lambda route: route.abort())
                
                self.logger.debug(f"Navigating to {url} with Playwright")
                response = page.goto(url, wait_until="networkidle", timeout=30000)
                if not response.ok:
                    self.logger.warning(f"Page {url} returned status {response.status}")
                
                if wait_for:
                    self.logger.debug(f"Waiting for selector: {wait_for}")
                    page.wait_for_selector(wait_for, timeout=wait_time)
                
                if should_scroll:
                    self.logger.debug("Scrolling page to load dynamic content")
                    # Improved scrolling with dynamic content detection
                    last_height = page.evaluate("document.body.scrollHeight")
                    while True:
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(1000)
                        new_height = page.evaluate("document.body.scrollHeight")
                        if new_height == last_height:
                            break
                        last_height = new_height
                
                # Wait for any remaining dynamic content
                page.wait_for_timeout(1000)
                
                self.logger.debug("Page rendered successfully")
                return page.content()
                
            finally:
                context.close()
                browser.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before=before_log(logging.getLogger("KnowledgeCrawler"), logging.WARNING),
        after=after_log(logging.getLogger("KnowledgeCrawler"), logging.DEBUG)
    )
    def _fetch_page(self, url, use_js):
        """
        Fetch a webpage and parse its content into a BeautifulSoup object.
        
        This method handles both JavaScript-rendered and static content based on the 
        use_js parameter. It automatically retries failed requests using tenacity 
        with exponential backoff.
        
        Args:
            url (str): The URL of the webpage to fetch
            use_js (bool): Whether to use JavaScript rendering via Playwright (True)
                           or standard requests (False)
        
        Returns:
            tuple: A tuple containing:
                - soup (BeautifulSoup): Parsed HTML content
                - final_url (str): The final URL after any redirects
        
        Raises:
            Exception: Any exceptions not handled by the retry mechanism
        
        Note:
            - The method is decorated with @retry to automatically retry on failure
            - When using JavaScript rendering, URL redirects are not tracked
            - Standard requests will track and report redirects
        """
        if use_js:
            self.logger.info(f"Using JavaScript rendering for {url}")
            html = self._render_with_playwright(url)
            final_url = url  # Playwright doesn't provide redirect info directly
        else:
            self.logger.info(f"Fetching {url} with standard request")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            html = response.text
            final_url = response.url
            if final_url != url:
                self.logger.info(f"URL redirected to {final_url}")
        
        soup = BeautifulSoup(html, 'html.parser')
        self.logger.debug(f"Successfully parsed HTML from {url}")
        return soup, final_url
    
    def _get_page_content(self, url, use_js=None, retries=3):
        """
        Get the HTML content of a page with retry mechanism.
        """
        if use_js is None:
            use_js = self._should_use_js_rendering(url)
            
        try:
            return self._fetch_page(url, use_js)
        except Exception as e:
            self.logger.error(f"Error fetching {url} after {retries} attempts: {e}", exc_info=True)
            return None, url
       
    def _process_element_text(self, element):
        """
        Extract and format text content from an HTML element while preserving structure.
        
        This method processes HTML content by:
        1. Finding all headings (h1-h4) and paragraphs
        2. Preserving document structure with proper line breaks
        3. Removing redundant text where paragraphs repeat heading content
        4. Joining all processed text with newlines
        
        Args:
            element (bs4.element.Tag): The BeautifulSoup element to process
                
        Returns:
            str: Cleaned and formatted text content with preserved structure
        
        Note:
            - Headings are preserved as separate lines
            - If a paragraph starts with the same text as the preceding heading,
              the redundant first sentence is removed
            - Empty elements are skipped
        """
        lines = []
        last_heading = None  # Initialize at the start

        for sub in element.find_all(["h1", "h2", "h3", "h4", "p"], recursive=True):
            text = sub.get_text(separator=" ", strip=True)
            if not text:
                continue

            # Check if it's a heading
            if sub.name in ["h1", "h2", "h3", "h4"]:
                lines.append(text)
                last_heading = text.lower()
                continue

            # Check if paragraph starts with repeated heading and remove first sentence if it does
            if last_heading and text.lower().startswith(last_heading):
                sentences = re.split(r'(?<=[.!?])\s+', text)
                if len(sentences) > 1 and sentences[0].lower().startswith(last_heading):
                    text = ' '.join(sentences[1:])

            lines.append(text)

        return "\n".join(lines)
    
    def _is_boilerplate(self, text, text_config):
        """Check if text is boilerplate content"""
        result = False
        boilerplate_patterns = text_config.get("boilerplate_patterns", [])

        if boilerplate_patterns:
            text_lower = text.lower()
            for pattern in boilerplate_patterns:
                if pattern in text_lower:
                    self.logger.debug(f"Detected boilerplate text with pattern: '{pattern}'")
                    result = True
                    break

        return result
    
    def scrape_topic_page(self, url):
        """
        Scrape a topic page to extract its content and structure.
        
        This method orchestrates the entire scraping process, including:
        1. Determining if JavaScript rendering is needed
        2. Fetching and parsing the page content
        3. Extracting structured data (title, content, subtopics, images, links)
        4. Formatting and cleaning the extracted text
        
        The extraction process is modular, with separate methods handling specific 
        components like title, content, subtopics, images, and links.
        
        Args:
            url (str): The URL of the topic page to scrape
            
        Returns:
            dict: A structured dictionary containing the extracted content with keys:
                - url: The final URL after any redirects
                - title: The page title (if found)
                - content: The main content text (if found)
                - subtopics: List of subtopic dictionaries with title and content (if found)
                - images: List of image dictionaries with url and alt text (if enabled)
                - links: List of link dictionaries with url and text (if enabled)
                
        Returns None if:
            - No URL is provided
            - The page content cannot be retrieved
            - Critical errors occur during extraction
        
        Note:
            Configuration from the class instance determines extraction behavior,
            including selectors, formatting options, and which components to extract.
        """
        self.logger.info(f"Starting to scrape topic page: {url}")
        result = None

        if not url:
            self.logger.error("No URL provided for scraping")
            return None

        use_js = self._should_use_js_rendering(url)
        soup, final_url = self._get_page_content(url, use_js)

        if not soup:
            self.logger.error(f"Failed to retrieve content from {url}")
            return None

        result = {"url": final_url}
        topic_config = self.config.get("html_parameters", {}).get("topic_page", {})
        text_config = self.config.get("html_parameters", {}).get("text_formatting", {})
        extraction_config = self.config.get("html_parameters", {}).get("extraction", {})

        self._extract_title(soup, topic_config, result)
        content_element = self._extract_content(soup, topic_config, extraction_config, result)
        self._extract_subtopics(content_element or soup, topic_config, text_config, result)
        self._extract_images(content_element, final_url, extraction_config, result)
        self._extract_links(content_element, final_url, extraction_config, result)

        self.logger.info(f"Completed scraping of {url}")
        return result


    def _extract_title(self, soup, topic_config, result):
        """Extracts the title from the page using the configured selector."""
        if topic_config.get("title_selector"):
            title_element = soup.select_one(topic_config["title_selector"])
            if title_element:
                result["title"] = title_element.get_text(strip=True)
                self.logger.info(f"Extracted title: {result['title'][:50]}...")
            else:
                self.logger.warning(f"No title found using selector: {topic_config['title_selector']}")

    def _extract_content(self, soup, topic_config, extraction_config, result):
        """
        Extracts main content using prioritized selectors, validating size.
        """
        content_element = None
        content_selectors = topic_config.get("content_selectors", [])
        min_content_length = self.config.get("html_parameters", {}).get("text_formatting", {}).get("min_content_length", 500)

        for selector in content_selectors:
            elements = soup.select(selector)
            content_blocks = [el for el in elements if el and el.get_text(strip=True)]
            if not content_blocks:
                continue

            self.logger.info(f"Matched {len(content_blocks)} blocks using selector: {selector}")
            content_html = "\n".join(str(el) for el in content_blocks)
            candidate_element = BeautifulSoup(content_html, "html.parser")

            # Run standardized formatting and cleaning
            candidate_text = self._process_element_text(candidate_element)

            if len(candidate_text) >= min_content_length:
                self.logger.info(f"Selector {selector} succeeded with {len(candidate_text)} characters.")
                content_element = candidate_element
                result["content"] = candidate_text
                break
            else:
                self.logger.warning(f"Selector {selector} matched but content too small ({len(candidate_text)} chars). Trying next selector.")

        if content_element and extraction_config.get("text", True):
            # Remove unwanted visual clutter elements
            for unwanted in content_element.select("nav, footer, header, .sidebar, .social-share, .newsletter, .related-articles"):
                unwanted.decompose()
            self.logger.info("Removed unwanted elements from content")
        else:
            self.logger.warning(f"No sufficient content found using selectors: {content_selectors}")

        return content_element

    def _extract_subtopics(self, base_element, topic_config, text_config, result):
        """
        Identifies subtopic headers (e.g., h2/h3) and their associated text blocks.
        Builds structured subtopic list with titles and adjacent paragraphs.
        """
        selector = topic_config.get("subtopics_selector")
        if not selector:
            return

        subtopic_elements = base_element.select(selector)
        self.logger.info(f"Found {len(subtopic_elements)} subtopic elements")

        subtopics = []
        for subtopic_element in subtopic_elements:
            subtopic = {}

            # Determine subtopic title
            title_selector = topic_config.get("subtopic_title_selector")
            title_element = subtopic_element.select_one(title_selector) if title_selector else None
            subtopic["title"] = title_element.get_text(strip=True) if title_element else subtopic_element.get_text(strip=True)

            # Collect nearby content following the subtopic header
            content_text = []
            next_element = subtopic_element.find_next_sibling()
            while next_element and (not next_element.name or next_element.name not in selector.split(", ")):
                if next_element.name in topic_config.get("block_elements", []):
                    text = next_element.get_text(strip=True)
                    if text:
                        content_text.append(text)
                next_element = next_element.find_next_sibling()

            if content_text:
                subtopic["content"] = text_config.get("paragraph_separator", "\n\n").join(content_text)

            if subtopic.get("title") or subtopic.get("content"):
                subtopics.append(subtopic)
                self.logger.info(f"Added subtopic: {subtopic.get('title', '')[:30]}...")

        if subtopics:
            result["subtopics"] = subtopics
            self.logger.info(f"Extracted {len(subtopics)} subtopics")


    def _extract_images(self, content_element, base_url, extraction_config, result):
        """Extracts all images within the content block, if enabled."""
        if extraction_config.get("images") and content_element:
            images = []
            for img in content_element.select("img"):
                src = img.get("src")
                alt = img.get("alt", "")
                if src:
                    images.append({
                        "url": self._make_absolute_url(src, base_url),
                        "alt": alt
                    })
            if images:
                result["images"] = images
                self.logger.info(f"Extracted {len(images)} images")


    def _extract_links(self, content_element, base_url, extraction_config, result):
        """Extracts all links within the content block, if enabled."""
        if extraction_config.get("links") and content_element:
            links = []
            for link in content_element.select("a"):
                href = link.get("href")
                text = link.get_text(strip=True)
                if href and not href.startswith(("#", "javascript:", "mailto:")):
                    links.append({
                        "url": self._make_absolute_url(href, base_url),
                        "text": text
                    })
            if links:
                result["links"] = links
                self.logger.info(f"Extracted {len(links)} links")