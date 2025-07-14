import argparse
import yaml
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from pdf import PdfManager
from common.results_manager import write_results

# Configure logging to display messages with timestamps and severity levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ExtractAndNormalizePdf")

# Function to load configuration from a YAML file and environment variables
# Raises an error if the .env file is not found
def load_config_and_env(config_path="config/config.yaml", load_env=True):
    logger.info("Loading configuration from YAML...")
    try:
        # Open and parse the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

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

        return config
    except Exception as e:
        logger.exception(f"Error loading config or env: {e}")
        raise


def run_pdf_extraction(args):
    try:
        # Load configuration and environment variables
        config = load_config_and_env(args.config, args.load_env)

        # Set the output directory
        config["output_directory"] = args.output
        
        pdf_manager = PdfManager(args.url, config, logger)
        base_filename, chunks_output, payload_output = pdf_manager.build_payload()
        
        # Write results to JSON file for Prefect integration
        if args.result_file:
            write_results(args.result_file, args.url, args.output, base_filename, chunks_output, payload_output, logger)

    except Exception as e:
        logger.error(f"Failed to generate metadata: {e}")
        
        # Write error results if result-file is provided
        if args.result_file:
            write_results(args.result_file, args.url, args.output, None, None, None, logger, is_success=False, error=e)
                
        exit(1)