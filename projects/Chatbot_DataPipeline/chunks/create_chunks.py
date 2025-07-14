import os
import json
import re
from chunkers.overlap_paragraph_chunker import OverlapParagraphChunks
import argparse
from pathlib import Path
import logging
import yaml
# Configure logging to display messages with timestamps and severity levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CreateChunks")

def load_config(config_file):
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Dictionary with configuration
        """
        config_data = {}
        
        try:
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_file}")
            else:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}

                config_data["chunk_size"] = int(config_data.get("chunk_size", 1024))
                config_data["overlap_percentage"] = int(config_data.get("overlap_percentage", 20))
                config_data["min_chunk_size"] = int(config_data.get("min_chunk_size", 100))
                config_data["max_lookahead"] = int(config_data.get("max_lookahead", 3))
                config_data["chunker_technique"] = config_data.get("chunker_technique", "overlap")
                config_data["min_words_to_include"] = int(config_data.get("min_words_to_include", 15))
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error loading configuration: {str(e)}")
        
        return config_data

def validating_file(dir_path):
    if not os.path.isdir(dir_path):
        logger.error(f"Invalid directory path: {dir_path}")
        raise ValueError(f"Invalid directory path: {dir_path}") 
    if not any(file.endswith('.json') for file in os.listdir(dir_path)):
        logger.error(f"No JSON files found in the directory: {dir_path}")
        raise ValueError(f"No JSON files found in the directory: {dir_path}")
    if not any(file.endswith('.txt') for file in os.listdir(dir_path)):
        logger.error(f"No text files found in the directory: {dir_path}")
        raise ValueError(f"No text files found in the directory: {dir_path}")
    return True

def get_text(file_path):
    """read text from a .tex file """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def get_metadata(dir_path):
    """read metadata from a .json file """
    json_file = next((f for f in os.listdir(dir_path) if f.endswith('.json')), None)
    if json_file:
        with open(os.path.join(dir_path, json_file), 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def create_chunks_from_text_file(dir_path, chunker, merge_paragraphs=False):
    """Creating chunks from text files in a directory"""

    file = next((f for f in os.listdir(dir_path) if f.endswith('.json')), None)
    file_prefix = re.match(r'(.+?)_base_payload.json', file).group(1)

    all_chunks = []
    
    metadata = get_metadata(dir_path)

    logger.info(f"Processing source_url: {metadata['source_url']} with merge_paragraphs: {merge_paragraphs}")
    
    final_text_to_overlap = ""

    chunker = chunker
    sorted_pages = sorted([int(re.search(r'_page_(\d+)\.txt', file).group(1)) for file in os.listdir(dir_path) if file .startswith(file_prefix) and file.endswith('txt')])
    chunk_index = 0

    total_skipped_chunks = 0
    for index in sorted_pages:
        
        file = f"{file_prefix}_page_{index}.txt"

        file_metadata = metadata.copy()
        
        page_text = get_text(f"{dir_path}/{file}")
        """ Populate the section title """
        file_metadata["section_title"] = page_text.split('\n')[0].strip('- ')
        text_without_first_line = re.sub(r'^[^\n]*\n', '', page_text)
        text = final_text_to_overlap + text_without_first_line
        
        chunks, final_text_to_overlap, skipped_chunks = chunker.generate_chunks(text, file_metadata, merge_paragraphs)
        total_skipped_chunks += skipped_chunks
        final_text_to_overlap = final_text_to_overlap.replace('\n', ' ')

        for chunk in chunks:
            chunk["chunk_index"] = chunk_index
            chunk_index += 1

        all_chunks.extend(chunks)

    # Calculate chunk statistics
    if all_chunks:
        chunk_word_counts = [len(chunk["text"].split()) for chunk in all_chunks]
        chunks_smaller_than_min_size = len([chunk for chunk in chunk_word_counts if chunk < chunker.min_chunk_size])
        total_words = sum(chunk_word_counts)
        avg_words = total_words / len(chunk_word_counts)
        min_words = min(chunk_word_counts)
        max_words = max(chunk_word_counts)
        logger.info(f"Chunk statistics:\n\tTotal chunks: {len(all_chunks)}\n\tTotal words: {total_words}\n\tAvg words per chunk: {avg_words:.1f}\n\tMin words per chunk: {min_words}\n\tMax words per chunk: {max_words}" +
                    f"\n\tChunks smaller than min size: {chunks_smaller_than_min_size}\n\tChunks skipped: {total_skipped_chunks}")

    return all_chunks

def store_chunks(chunks, file_path):
    """Storing chunks in a JSON file, creating directory if it doesn't exist"""
    # Convert file_path to Path object and get its parent directory
    path = Path(file_path)
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
   
    with open(file_path, "w") as f:
        json.dump([chunk for chunk in chunks], f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chunks from text files and ingest metadata from a JSON file")
    parser.add_argument("--input", required=True, help="input file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument("--merge_paragraphs", action='store_true', help="merge paragraphs")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    
    try:
        args = parser.parse_args()
        
        files_to_chunk_dir = args.input
        chunk_store_file = args.output + "/chunks.json"

        config_data = load_config(Path(args.config))

        # include as many case as chunk techniques are defined (script in the chunkers folder) 
        if config_data["chunker_technique"] == "overlap":
            chunker = OverlapParagraphChunks(logger, config_data["chunk_size"], config_data["overlap_percentage"], config_data["min_chunk_size"], 
                                             config_data["max_lookahead"], config_data["min_words_to_include"])
        else:
            logger.error(f'Unknown chunker technique: {config_data["chunker_technique"]}')
            raise ValueError(f'Unknown chunker technique: {config_data["chunker_technique"]}')
        
        file_validation = validating_file(files_to_chunk_dir)

        chunks = create_chunks_from_text_file(files_to_chunk_dir, chunker, args.merge_paragraphs)
        store_chunks(chunks, chunk_store_file)

    except Exception as e:
        logger.error(f"Chunker failed to generate chunks: {e}")
            
        exit(1)