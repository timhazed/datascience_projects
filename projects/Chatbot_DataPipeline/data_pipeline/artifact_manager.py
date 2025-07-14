from datetime import datetime
from typing import Dict, Any, Optional
from prefect.artifacts import create_markdown_artifact, create_table_artifact

class ArtifactManager:
    """
    Manages the creation and persistence of pipeline execution artifacts.
    Provides methods to save pipeline results as Prefect artifacts.
    """
    
    def __init__(self, logger):
        self.logger = logger
    
    def save_pipeline_results(self, results: Dict[str, Any], csv_path: str, config_path: Optional[str] = None) -> None:
        """
        Save pipeline execution results as Prefect artifacts.
        
        Args:
            results: Dictionary containing pipeline execution results
            csv_path: Path to the CSV file used for pipeline input
            config_path: Path to the configuration file used (optional)
        """
        
        # Enhance results with additional metadata
        enhanced_results = {
            **results,
            "timestamp": datetime.now().isoformat(),
            "csv_path": csv_path,
            "config_path": str(config_path) if config_path else "default"
        }
        
        # Create structured table artifact
        self._create_table_artifact(enhanced_results)
        
        # Create human-readable markdown artifact
        self._create_markdown_artifact(enhanced_results)
        
        self.logger.info(f"Pipeline execution results persisted as Prefect artifacts")

    
    def _create_table_artifact(self, results: Dict[str, Any]) -> None:
        """Create a structured table artifact from the results dictionary"""
        try:
            # Format results for Prefect 2.11 compatibility - convert dict to list of records
            # Each record is a dict with "key" and "value" fields
            records = [{"key": k, "value": str(v)} for k, v in results.items()]
            
            # Create a simple table from the records
            create_table_artifact(
                key=f"pipeline-results-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                table=records,
                description="ETL Pipeline execution results"
            )
            self.logger.info("Successfully created table artifact")
        except Exception as e:
            self.logger.error(f"Failed to create table artifact: {e}")
            # Log the results directly as a fallback
            self.logger.info(f"Pipeline results: {results}")
    
    def _create_markdown_artifact(self, results: Dict[str, Any]) -> None:
        """Create a human-readable markdown artifact from the results dictionary"""
        total = results.get("total_documents", 0)
        successful_extractions = results.get("successful_extractions", 0)
        successful_chunks = results.get("successful_chunks", 0)
        successful_vectors = results.get("successful_vectors", 0)
        
        # Handle division by zero
        extract_pct = (successful_extractions/total*100) if total > 0 else 0
        chunk_pct = (successful_chunks/total*100) if total > 0 else 0  
        vector_pct = (successful_vectors/total*100) if total > 0 else 0
        
        markdown = f"""
        # Pipeline Run Summary
        
        ## Configuration
        - **Timestamp**: {results.get("timestamp", "N/A")}
        - **CSV Path**: `{results.get("csv_path", "N/A")}`
        - **Config Path**: `{results.get("config_path", "N/A")}`
        
        ## Results
        - **Total Documents**: {total}
        - **Successfully Extracted**: {successful_extractions} ({extract_pct:.1f}% of total)
        - **Successfully Chunked**: {successful_chunks} ({chunk_pct:.1f}% of total)
        - **Successfully Inserted to DB**: {successful_vectors} ({vector_pct:.1f}% of total)
        """
        
        self.logger.info(f"Pipeline Run Summary:\n{markdown}")

        try:
            create_markdown_artifact(
                key=f"pipeline-summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                markdown=markdown,
                description="Human-readable pipeline summary"
            )
            self.logger.info("Successfully created markdown artifact")
        except Exception as e:
            self.logger.error(f"Failed to create markdown artifact: {e}")