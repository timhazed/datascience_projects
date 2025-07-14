import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QdrantQuery")

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

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
        logger.error(f"Error loading config: {str(e)}")
        raise


def connect_to_qdrant(config: dict):
    url = config["qdrant_url"]
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(
            url=url,
            api_key=api_key)
    return client


from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

def delete_all_points(client: QdrantClient, collection_name: str):
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(must=[])  # Empty filter matches all points
    )
    logger.info(f"✅ All points deleted from collection '{collection_name}'")


def main():
    parser = argparse.ArgumentParser(description="Delete Qdrant collection")
    parser.add_argument("--collection", default="ws1-test", help="Qdrant collection name")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)
        qdrant_config = config["qdrant"]

        # Initialize Qdrant client
        client = connect_to_qdrant(qdrant_config)

        # Uncomment for debugging to clear collection
        delete_all_points(client, args.collection)   

    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise

if __name__ == "__main__":
    main()
