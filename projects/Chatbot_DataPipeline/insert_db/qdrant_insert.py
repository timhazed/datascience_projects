import argparse
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import logging
from qdrant_manager import QdrantManager
import json
from datetime import datetime, timezone

# Configure logging to display messages with timestamps and severity levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QdrantInsert")

# Default configuration values
DEFAULT_CONFIG = {
    "qdrant": {
        "qdrant_url": "http://localhost:6333",
        "qdrant_collection": "agriculture_docs",
        "embedding_model": "BAAI/bge-large-zh-v1.5",
        "embedding_dimension": 1024
    }
}

# Function to load configuration from a YAML file and environment variables
# Raises an error if the .env file is not found or QDRANT_API_KEY is missing
def load_config_and_env(config_path="config/config.yaml", load_env=False):
    logger.info("Loading configuration from YAML...")
    try:
        # Open and parse the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Initialize qdrant section if not present
        if "qdrant" not in config:
            config["qdrant"] = {}

        # Merge default values with user configuration
        for key, default_value in DEFAULT_CONFIG["qdrant"].items():
            if key not in config["qdrant"]:
                config["qdrant"][key] = default_value
                logger.info(f"Using default value for {key}: {default_value}")

        if load_env:
            # Expand user home directory if present in the env_path
            env_path = Path(config.get("env_path")).expanduser()

            # Validate that the .env file exists
            if not env_path or not os.path.isfile(env_path):
                logger.error(f".env file not found at {env_path}")
                raise FileNotFoundError(".env path is invalid or not found in config.yaml")

            logger.info(f"Loading environment variables from: {env_path}")
            
            # Load environment variables from the .env file
            load_dotenv(env_path)

            # Validate required environment variables
            if not os.getenv("QDRANT_API_KEY"):
                raise ValueError("Missing required environment variable: QDRANT_API_KEY")

        return config
    
    except Exception as e:
        logger.exception(f"Error loading config or env: {e}")
        raise

def write_results_file(results_file: str, success: bool, metrics: dict):
    """
    Write results to a JSON file for Prefect integration
    
    Args:
        results_file: Path to write results
        success: Whether the operation was successful
        metrics: Dictionary of operation metrics or error information
    """
    if success:
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "metrics": metrics
        }
    else:
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "metrics": {
                "status": "error",
                "url": metrics.get("url", "unknown"),
                "error": metrics.get("error", "Unknown error"),
                "error_type": metrics.get("error_type", "Unknown")
            }
        }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

def process_chunked_data(args):
    """
    Process chunked data from the input directory and insert into Qdrant.
    
    Args:
        args: Command line arguments containing input directory path and optional results file
    """
    try:
        # Load configuration
        config = load_config_and_env(config_path=args.config, load_env=args.load_env)
        qdrant_manager = QdrantManager(config["qdrant"], args.collection_name, logger)

        # Validate input directory
        input_dir = Path(args.input)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {args.input}")

        # Find JSON file in the directory
        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {args.input}")
        if len(json_files) > 1:
            logger.warning(f"Multiple JSON files found in {args.input}, using first one: {json_files[0]}")

        # Load and process the JSON file
        json_file = json_files[0]
        logger.info(f"Processing JSON file: {json_file}")
        
        with open(json_file, "r") as f:
            document = json.load(f)

        # Insert document into Qdrant and get metrics
        metrics = qdrant_manager.add_documents(document)
        logger.info("Successfully processed and inserted document into Qdrant")
        
        # Write results if results file is specified
        if args.result_file:
            write_results_file(args.result_file, True, metrics)
            
        return metrics["chunks_inserted"]

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing chunked data: {error_msg}")
        
        # Write results if results file is specified
        if args.result_file:
            write_results_file(args.result_file, False, {"url": str(args.input)})
            
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert chunked data into Qdrant")
    parser.add_argument("--input", required=True, help="Input directory containing chunked data JSON file")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--result-file", default=None, help="Path to write results JSON for Prefect integration")
    parser.add_argument("--collection-name", required=True, help="Collection name")
    parser.add_argument("--load-env", default=False, help="Load environment variables when not using docker")

    try:
        args = parser.parse_args()
        process_chunked_data(args)

    except Exception as e:
        logger.exception(f"qdrant_insert.py error processing chunked data: {e}")
        exit(1)
