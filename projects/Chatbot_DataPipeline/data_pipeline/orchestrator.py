from pathlib import Path
import os
import csv
import argparse
import time
from typing import List, Optional
from prefect import flow, get_run_logger
from artifact_manager import ArtifactManager
from pipeline import ConfigManager, ToolExecutor
from pipeline.extractors import Extractor
from pipeline.processors import Chunker, InsertDB

def get_urls_from_csv(csv_path: str) -> List[str]:
    urls = []
    with open(csv_path, "r") as f:
        csv_reader = csv.reader(f)
        # Skip the header row if it exists
        next(csv_reader, None)
        for row in csv_reader:
            if row and len(row) > 0 and row[0].strip():
                urls.append(row[0].strip())
    return urls

@flow(name="ETL-Pipeline")
def main_pipeline(csv_path: str, config_path: str):
    """
    Main ETL pipeline that processes documents from CSV, extracts, chunks, and inserts them into the database
    
    Args:
        csv_path: Path to CSV file containing URLs to process
        config_path: Optional path to configuration file
    """
    # Configure logging
    logger = get_run_logger()
    logger.info(f"Starting ETL pipeline with input: {csv_path}")
    
    # Initialize managers
    config_file = Path(config_path)
    config_manager = ConfigManager(config_file, logger)
    tool_executor = ToolExecutor(config_manager, logger)
    
    # Initialize extractors and processors
    extractor = Extractor(config_manager, tool_executor, logger)
    chunker = Chunker(config_manager, tool_executor, logger)
    insert_db = InsertDB(config_manager, tool_executor, logger)
    
    # Log which tools are enabled from configuration
    extractor_enabled = extractor.is_enabled()
    chunker_enabled = chunker.is_enabled()
    insert_db_enabled = insert_db.is_enabled()

    logger.info(f"Enabled tools from configuration: extractor: {extractor_enabled}, chunker: {chunker_enabled}, insert_db: {insert_db_enabled}")
    
    # Load input data - handle CSV without proper headers
    urls = get_urls_from_csv(csv_path)
    
    total = len(urls)
    logger.info(f"Processing {total} URLs from {csv_path}")
    
    # Process each document through the pipeline
    extracted_paths = []
    if extractor_enabled:
        logger.info("Starting extraction phase...")
        for idx, url in enumerate(urls, 1):
            # Extract content using the extractor
            if extractor.can_process(url):
                extracted_path = extractor.extract(url, idx, total)
                extracted_paths.append(extracted_path)
            else:
                logger.warning(f"Extractor cannot process URL: {url}")
    else:
        logger.info("Skipping extraction phase (disabled in config).")

        # For debugging, you can provide extracted_paths for chunker to process 
        extracted_paths = config_manager.get_extracted_paths()
    
    chunked_paths = []
    if chunker_enabled and extracted_paths:
        logger.info("Starting chunking phase...")
        for extracted_path in extracted_paths:
            if chunker.can_process(extracted_path):
                chunked_path = chunker.process(Path(extracted_path))
                chunked_paths.append(chunked_path)
            else:
                logger.warning(f"Chunker cannot process extracted path: {extracted_path}")
    else:
        if not chunker_enabled:
            logger.info("Skipping chunking phase (disabled in config).")

            # For debugging, you can provide chunked_paths for insert_db to process 
            chunked_paths = config_manager.get_chunked_paths()
        elif not extracted_paths:
            logger.warning("No extracted content to chunk. Skipping chunking phase.")
    
    db_results = []
    if insert_db_enabled and chunked_paths:
        logger.info("Starting database insertion phase...")
        for chunked_path in chunked_paths:
            result = insert_db.process(Path(chunked_path))
            db_results.append(result)
    else:
        if not insert_db_enabled:
            logger.info("Skipping database insertion phase (disabled in config).")
        elif not chunked_paths:
            logger.warning("No chunked content to insert. Skipping database insertion phase.")
    
    # Summarize results
    successful_extractions = sum(1 for path in extracted_paths if path is not None)
    successful_chunks = sum(1 for path in chunked_paths if path is not None)
    successful_db_inserts = sum(1 for result in db_results if result is not None and result.get("status") == "success")
    
    # Comprehensive pipeline completion log message
    completion_msg = []
    if extractor_enabled:
        completion_msg.append(f"{successful_extractions}/{total} extracted")
    if chunker_enabled:
        completion_msg.append(f"{successful_chunks}/{successful_extractions} chunked")
    if insert_db_enabled:
        completion_msg.append(f"{successful_db_inserts}/{successful_chunks} inserted into DB")
    
    logger.info(f"Pipeline complete: {' | '.join(completion_msg)}")
    
    # Create results dictionary with tool statuses
    results = {
        "total_documents": total,
        "successful_extractions": successful_extractions, 
        "successful_chunks": successful_chunks,
        "successful_db_inserts": successful_db_inserts,
        "extractor_enabled": extractor_enabled,
        "chunker_enabled": chunker_enabled,
        "insert_db_enabled": insert_db_enabled,
        "completion_time": time.time()
    }
    
    # Save results as artifacts while still in flow context
    artifact_manager = ArtifactManager(logger)
    artifact_manager.save_pipeline_results(results, csv_path, config_path)

    logger.info("Pipeline execution complete. Results persisted as Prefect artifacts.")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL pipeline for document processing")
    parser.add_argument("--csv", required=True, help="CSV file with URLs")
    parser.add_argument("--config", default="config/pipeline_config.yaml", help="Optional path to configuration file")
    
    args = parser.parse_args()
    
    # Run the pipeline
    results = main_pipeline(args.csv, args.config)
