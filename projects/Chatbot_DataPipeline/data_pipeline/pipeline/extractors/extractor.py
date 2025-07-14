from pathlib import Path
import os
import validators
from typing import Optional
from prefect import task
from pipeline.config_manager import ConfigManager
from pipeline.tool_executor import ToolExecutor

class Extractor:
    """
    Extracts and normalizes content from documents using a Docker container.
    Handles both URLs and local files regardless of format (PDF, HTML, etc).
    """
    
    def __init__(self, config_manager: ConfigManager, tool_executor: ToolExecutor, logger):
        """
        Initialize the Docker extractor with configuration.
        
        Args:
            config_manager: Configuration manager instance
            tool_executor: Tool executor instance
            logger: Logger instance to use throughout the class
        """
        self.logger = logger
        self.config_manager = config_manager
        self.config = config_manager.get_tool_config("extractor")
        self.tool_executor = tool_executor
        
        # Docker service configuration
        self.service_url = self.config.get("service_url", "http://localhost:8000")
        self.extract_endpoint = self.config.get("endpoints", {}).get("extract", "/extract")
        
        self.logger.info(f"Initialized Docker extractor with service URL: {self.service_url}")
        
        # Get Docker configuration
        self.docker_config = self.config_manager.get_docker_config("extractor")
        
        # Create output directory
        self.output_base = Path(self.docker_config.get('output_dir', 'output'))
        self.output_base.mkdir(parents=True, exist_ok=True)
    
    def is_enabled(self) -> bool:
        """
        Check if extraction is enabled in configuration.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.config_manager.is_tool_enabled("extractor")
    
    def can_process(self, path_or_url: str) -> bool:
        """
        Check if this extractor can process the given URL or file path.
        Simply checks if the URL ends with .pdf or is a valid URL.
        
        Args:
            path_or_url: URL or file path to check
            
        Returns:
            True if this extractor can process the input
        """
        result = False
        try:
            # Check if it's a valid URL
            if validators.url(path_or_url):
                result = True
            # Check if it's a container path (starting with /app/)
            elif path_or_url.startswith('/app/input'):
                # For container paths, just verify it ends with a supported extension
                result = path_or_url.lower().endswith('.pdf')
                
        except Exception as e:
            self.logger.warning(f"Error checking if extractor can process {path_or_url}: {e}")
            
        return result
    
    @task(name="extract-document")
    def extract(self, url_or_path: str, idx: int = 0, total: int = 1) -> Optional[Path]:
        """
        Extract content from a document.
        
        Args:
            url_or_path: URL or file path to extract content from
            idx: Index of the document in batch (for logging)
            total: Total number of documents (for logging)
            
        Returns:
            Path to the extracted document directory or None if extraction failed
        """
        output_dir = None
        doc_id = f"doc_{idx:04d}"
        
        # Process extraction only if enabled
        if self.is_enabled():
            # Get Docker volumes using Facade methods
            output_volume = self.config_manager.get_output_volume("extractor")
            input_volume = self.config_manager.get_input_volume("extractor")
            extra_volumes = self.config_manager.get_extra_volumes("extractor")
            
            # Validate output volume and get host path
            host_path = self.config_manager.get_host_path_from_volume(output_volume)
            
            if host_path:
                # Create extracted subdirectory
                extracted_dir = Path(host_path) / "extracted"
                extracted_dir.mkdir(parents=True, exist_ok=True)
                
                # Set up paths
                output_dir = extracted_dir / doc_id
                os.makedirs(output_dir, exist_ok=True)
                docker_input_path = url_or_path
                docker_output_path = f"/app/output/extracted/{doc_id}"
                
                self.logger.info(f"Extracting from: {url_or_path} [{idx+1}/{total}]")
                
                # Get Docker image and configuration
                image_name = self.config_manager.get_docker_image("extractor")
                env_file = self.config_manager.get_env_file("extractor")
                
                # Prepare arguments
                args = [
                    "--url", docker_input_path,
                    "--output", docker_output_path
                ]
                
                # Add config if specified
                config_path = self.config_manager.get_config_path("extractor")
                if config_path:
                    args.extend(["--config", config_path])
                
                # Execute the extractor
                result = self.tool_executor.execute_tool(
                    image_name=image_name,
                    args=args,
                    input_volume=input_volume,
                    output_volume=output_volume,
                    env_file=env_file,
                    extra_volumes=extra_volumes,
                    tool_name="extractor",
                    timeout=self.config_manager.get_timeout("extraction"),
                    doc_id=doc_id
                )
                
                # Check result
                if result.get("status") != "success":
                    self.logger.error(f"Failed to extract: {url_or_path}. Error: {result.get('error', 'Unknown error')}")
                    output_dir = None
                else:
                    self.logger.info(f"Successfully extracted: {url_or_path} [{idx+1}/{total}]")
            else:
                self.logger.error("Invalid output_volume configuration in extractor tool config")
        else:
            self.logger.info(f"Extractor is disabled, skipping: {url_or_path}")
        
        return output_dir