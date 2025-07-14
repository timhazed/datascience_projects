from pathlib import Path
from typing import Optional, Dict, Any
from prefect import task
from pipeline.tool_executor import ToolExecutor 
from pipeline.config_manager import ConfigManager

class InsertDB:
    """
    Inserts chunked documents into a vector database for retrieval.
    """
    
    def __init__(self, config_manager: ConfigManager, tool_executor: ToolExecutor, logger):
        """
        Initialize the database inserter with configuration.
        
        Args:
            config_manager: Configuration manager instance
            tool_executor: Tool executor instance
            logger: Logger instance to use throughout the class
        """
        self.logger = logger
        self.config_manager = config_manager
        self.tool_executor = tool_executor
        
        # Get configuration
        self.config = config_manager.get_tool_config("insert_db")
        
        # Get Docker configuration
        self.docker_config = self.config_manager.get_docker_config("insert_db")
    
    def is_enabled(self) -> bool:
        """
        Check if database insertion is enabled in configuration.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.config_manager.is_tool_enabled("insert_db")
    
    @task(name="insert-to-database")
    def process(self, input_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Process chunked content by inserting it into the database.
        
        Args:
            input_dir: Directory containing chunked content
            
        Returns:
            Dictionary with insertion results or None if insertion failed
        """
        result = None
        
        # Process only if all conditions are met
        if self.is_enabled() and input_dir is not None and input_dir.exists():
            # Use input directory name as document ID
            doc_id = input_dir.name
            self.logger.info(f"Inserting document {doc_id} into database")
            
            # Get Docker image and configuration
            image_name = self.config_manager.get_docker_image("insert_db")
            env_file = self.config_manager.get_env_file("insert_db")
            
            # Get volume mappings from ConfigManager Facade methods
            input_volume = self.config_manager.get_input_volume("insert_db")
            output_volume = self.config_manager.get_output_volume("insert_db")
            extra_volumes = self.config_manager.get_extra_volumes("insert_db")
            
            # Map input path in container
            container_input_path = f"/app/input/{input_dir.name}"
            
            # Prepare arguments
            args = [
                "--input", container_input_path,
            ]
            
            # Add merge_paragraphs if specified and True
            collection_name = self.config.get("collection_name", None)
            if collection_name:
                args.extend(["--collection-name", collection_name])
            else:
                raise ValueError("collection_name must be set")
            
            # Add config if specified
            config_path = self.config_manager.get_config_path("insert_db")
            if config_path:
                args.extend(["--config", config_path])
            
            # Execute the database inserter
            execution_result = self.tool_executor.execute_tool(
                image_name=image_name,
                args=args,
                input_volume=input_volume,
                output_volume=output_volume,
                env_file=env_file,
                extra_volumes=extra_volumes,
                tool_name="insert_db",
                timeout=self.config_manager.get_timeout("db_insertion"),
                doc_id=doc_id
            )
            
            # Check result
            if execution_result.get("status") == "success":
                self.logger.info(f"Successfully inserted document {doc_id} into database")
                # Set insertion stats
                result = {
                    "document_id": doc_id,
                    "chunks_inserted": execution_result.get("chunks_inserted", 0),
                    "status": "success"
                }
            else:
                self.logger.error(f"Failed to insert document {doc_id} into database. Error: {execution_result.get('error', 'Unknown error')}")
                result = execution_result
        else:
            # Log the reason why processing was skipped
            if not self.is_enabled():
                self.logger.info("Database insertion is disabled, skipping")
            elif input_dir is None:
                self.logger.warning("Input directory is None")
            elif not input_dir.exists():
                self.logger.warning(f"Input directory does not exist: {input_dir}")
        
        return result 