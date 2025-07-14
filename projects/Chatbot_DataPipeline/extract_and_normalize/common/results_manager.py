import json
from pathlib import Path

def write_results(result_file_path, url, output_dir, base_filename, chunks_output, payload_output, logger, is_success=True, error=None):
    """
    Write extraction results to a JSON file for Prefect integration
    
    Args:
        result_file_path (str): Path to write the results JSON file
        url (str): The URL of the processed document
        output_dir (str): The output directory where document artifacts are stored
        base_filename (str): The base filename of the processed document
        chunks_output (str): The path to the chunks output file
        payload_output (str): The path to the payload output file
        logger (logging.Logger): Logger instance
        is_success (bool): Flag indicating if processing succeeded
        error (Exception, optional): Exception object if processing failed
    
    Returns:
        dict: The result data that was written to file
    """
    if not result_file_path:
        return None
    
    try:
        output_dir_path = Path(output_dir)
                
        if is_success:
            result_data = {
                "status": "success",
                "url": url,
                "output_path": str(output_dir_path),
                "document_id": base_filename,
                "metadata_path": str(output_dir_path / payload_output),
                "chunks_path": str(output_dir_path / chunks_output)
            }
        else:
            result_data = {
                "status": "error",
                "url": url,
                "error": str(error),
                "error_type": type(error).__name__ if error else "Unknown"
            }
        
        # Ensure parent directory exists
        Path(result_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write result data to specified file
        with open(result_file_path, "w") as f:
            json.dump(result_data, f)
            
        logger.info(f"Result data written to {result_file_path}")
        return result_data
        
    except Exception as write_error:
        logger.error(f"Failed to write result data: {write_error}")
        return None 