import argparse
import yaml
import logging
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
from html_text import HtmlTextManager
from common.results_manager import write_results

# Configure logging to display messages with timestamps and severity levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ExtractAndNormalizeHtml")

# Function to load configuration from a YAML file and environment variables
# Raises an error if the .env file is not found
def validate_config(config, logger):
    """
    Validate that all required configuration fields are present and correctly populated.
    Raises ValueError with descriptive message if configuration is invalid.
    
    Args:
        config (dict): The configuration dictionary to validate
        logger (logging.Logger): Logger instance for logging validation results
        
    Returns:
        dict: The validated configuration
    """
    logger.debug("Beginning configuration validation")
    
    # Check top-level html_parameters section
    if "html_parameters" not in config:
        logger.error("Missing required top-level 'html_parameters' section in configuration")
        raise ValueError("Missing required top-level 'html_parameters' section in configuration")
    
    html_params = config["html_parameters"]
    
    # Validate topic_page configuration
    if "topic_page" not in html_params:
        logger.error("Missing required 'topic_page' section in html_parameters")
        raise ValueError("Missing required 'topic_page' section in html_parameters")
    
    topic_page = html_params["topic_page"]
    required_topic_fields = {
        "title_selector": str,
        "content_selectors": list,
        "subtopics_selector": str,
        "block_elements": list,
        "list_elements": list
    }
    
    for field, field_type in required_topic_fields.items():
        if field not in topic_page:
            logger.error(f"Missing required field '{field}' in topic_page configuration")
            raise ValueError(f"Missing required field '{field}' in topic_page configuration")
        if not isinstance(topic_page[field], field_type):
            logger.error(f"Field '{field}' must be of type {field_type.__name__}")
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")
            
    # Validate rendering configuration
    if "rendering" not in html_params:
        logger.error("Missing required 'rendering' section in html_parameters")
        raise ValueError("Missing required 'rendering' section in html_parameters")
    
    rendering = html_params["rendering"]
    required_rendering_fields = {
        "wait_time": int,
        "scroll": bool,
        "js_patterns": list
    }
    
    for field, field_type in required_rendering_fields.items():
        if field not in rendering:
            logger.error(f"Missing required field '{field}' in rendering configuration")
            raise ValueError(f"Missing required field '{field}' in rendering configuration")
        if not isinstance(rendering[field], field_type):
            logger.error(f"Field '{field}' must be of type {field_type.__name__}")
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")
    
    # Validate text_formatting configuration
    if "text_formatting" not in html_params:
        logger.error("Missing required 'text_formatting' section in html_parameters")
        raise ValueError("Missing required 'text_formatting' section in html_parameters")
    
    text_formatting = html_params["text_formatting"]
    required_text_fields = {
        "paragraph_separator": str,
        "header_separator": str,
        "list_item_separator": str,
        "preserve_headers": bool,
        "min_paragraph_length": int,
        "clean_whitespace": bool
    }
    
    for field, field_type in required_text_fields.items():
        if field not in text_formatting:
            logger.error(f"Missing required field '{field}' in text_formatting configuration")
            raise ValueError(f"Missing required field '{field}' in text_formatting configuration")
        if not isinstance(text_formatting[field], field_type):
            logger.error(f"Field '{field}' must be of type {field_type.__name__}")
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")
    
    # Validate extraction configuration
    if "extraction" not in html_params:
        logger.error("Missing required 'extraction' section in html_parameters")
        raise ValueError("Missing required 'extraction' section in html_parameters")
    
    extraction = html_params["extraction"]
    required_extraction_fields = {
        "images": bool,
        "links": bool,
        "text": bool
    }
    
    for field, field_type in required_extraction_fields.items():
        if field not in extraction:
            logger.error(f"Missing required field '{field}' in extraction configuration")
            raise ValueError(f"Missing required field '{field}' in extraction configuration")
        if not isinstance(extraction[field], field_type):
            logger.error(f"Field '{field}' must be of type {field_type.__name__}")
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")
    
    # Validate headers configuration
    if "headers" not in html_params:
        logger.error("Missing required 'headers' section in html_parameters")
        raise ValueError("Missing required 'headers' section in html_parameters")
    
    headers = html_params["headers"]
    if "User-Agent" not in headers or not isinstance(headers["User-Agent"], str):
        logger.error("Missing or invalid 'User-Agent' in headers configuration")
        raise ValueError("Missing or invalid 'User-Agent' in headers configuration")
    
    # Validate rate_limit configuration
    if "rate_limit" not in html_params:
        logger.error("Missing required 'rate_limit' section in html_parameters")
        raise ValueError("Missing required 'rate_limit' section in html_parameters")
    
    rate_limit = html_params["rate_limit"]
    required_rate_limit_fields = {
        "enabled": bool,
        "requests_per_second": (int, float),
        "delay": (int, float)
    }
    
    for field, field_type in required_rate_limit_fields.items():
        if field not in rate_limit:
            logger.error(f"Missing required field '{field}' in rate_limit configuration")
            raise ValueError(f"Missing required field '{field}' in rate_limit configuration")
        
        if isinstance(field_type, tuple):
            if not any(isinstance(rate_limit[field], t) for t in field_type):
                type_names = " or ".join(t.__name__ for t in field_type)
                logger.error(f"Field '{field}' must be of type {type_names}")
                raise ValueError(f"Field '{field}' must be of type {type_names}")
        elif not isinstance(rate_limit[field], field_type):
            logger.error(f"Field '{field}' must be of type {field_type.__name__}")
            raise ValueError(f"Field '{field}' must be of type {field_type.__name__}")
    
    logger.info("Configuration validation successful")
    return config

def load_config_and_env(config_path="config/config.yaml", load_env=True):
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path (str): Path to the YAML configuration file
        logger (logging.Logger): Logger instance for logging load results
        
    Returns:
        dict: The validated configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.debug("YAML file loaded successfully")
        
        # Validate the configuration
        validated_config = validate_config(config, logger)
        logger.info("Configuration loaded and validated successfully")

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

        return validated_config
    
    except FileNotFoundError:
        logger.critical(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.critical(f"Invalid YAML in configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.critical(f"Invalid configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading configuration: {e}", exc_info=True)
        sys.exit(1)

def run_html_extraction(args):
    try:
        # Load configuration and environment variables
        config = load_config_and_env(args.config, args.load_env)

        # Set the output directory
        config["output_directory"] = args.output

        html_manager = HtmlTextManager(args.url, config, logger)
        base_filename, chunks_output, payload_output = html_manager.build_payload()
        
        # Write results to JSON file for Prefect integration
        if args.result_file:
            write_results(args.result_file, args.url, args.output, base_filename, chunks_output, payload_output, logger)

    except Exception as e:
        logger.error(f"Failed to generate metadata: {e}")
        
        # Write error results if result-file is provided
        if args.result_file:
            write_results(args.result_file, args.url, args.output, None, None, None, logger, is_success=False, error=e)
                
        exit(1)