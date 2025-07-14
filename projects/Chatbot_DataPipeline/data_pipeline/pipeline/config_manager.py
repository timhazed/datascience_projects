import re
from pathlib import Path
import yaml
from typing import Dict, Any, Optional, List
import os

# Regex pattern for Docker volume validation
# Format: host_path:container_path where container_path must be absolute
DOCKER_VOLUME_PATTERN = r'^([^:]+):(/[^:]*)?$'

class ConfigManager:
    """
    Manages configuration for data pipeline and tools.
    """
    
    def __init__(self, config_file: Path, logger):
        """
        Initialize the config manager.
        
        Args:
            config_file: Path to the main configuration file
        """
        self.config_file = config_file
        self.logger = logger
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file: Path) -> Dict[str, Any]:
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
                self.logger.error(f"Configuration file not found: {config_file}")
            else:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, IOError) as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
        
        return config_data
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool configuration
        """
        tools_config = self.config.get('tools', {})
        return tools_config.get(tool_name, {})
    
    def get_docker_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get Docker configuration for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Docker configuration for the tool
        """
        tool_config = self.get_tool_config(tool_name)
        return tool_config.get("docker", {})
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """
        Check if a tool is enabled in the configuration.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            True if the tool is enabled, False otherwise.
        """
        tool_config = self.get_tool_config(tool_name)
        # Default to True if not specified - this means tools are enabled by default
        is_enabled = tool_config.get("enabled", True)
            
        return is_enabled
    
    def get_timeout(self, operation: str) -> int:
        """
        Get timeout for a specific operation.
        
        Args:
            operation: Name of the operation (extraction, chunking, db_insertion)
            
        Returns:
            Timeout in seconds
        """
        timeouts = self.config.get("timeouts", {})
        return timeouts.get(operation, 600)  # Default to 10 minutes
    
    def get_env_file(self, tool_name: str) -> Optional[str]:
        """
        Get environment file path for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Path to environment file or None if not specified
        """
        docker_config = self.get_docker_config(tool_name)
        env_file = docker_config.get("env_file")
        
        # Handle tilde expansion for home directory
        if env_file and isinstance(env_file, str) and env_file.startswith("~"):
            from os.path import expanduser
            env_file = expanduser(env_file)
            
        return env_file
        
    def get_docker_volumes(self, tool_name: str) -> Dict[str, str]:
        """
        Get all Docker volume configurations for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with volume configurations (input_volume, output_volume, etc.)
        """
        docker_config = self.get_docker_config(tool_name)
        volumes = {
            'input_volume': docker_config.get('input_volume', ''),
            'output_volume': docker_config.get('output_volume', ''),
            'extra_volumes': docker_config.get('extra_volumes', [])
        }
        return volumes
        
    def validate_docker_volume(self, volume_string: str) -> bool:
        """
        Validate a Docker volume string has the correct format using regex.
        
        Args:
            volume_string: Docker volume string in format "host_path:container_path"
            
        Returns:
            True if the format is valid, False otherwise
        """
        is_valid = False
        
        # Only process if we have a non-empty string
        if volume_string and isinstance(volume_string, str):
            # Use regex to validate the format
            match = re.match(DOCKER_VOLUME_PATTERN, volume_string)
            if match:
                host_path, container_path = match.groups()
                # Ensure both parts are non-empty
                if host_path and container_path:
                    is_valid = True
                
        return is_valid
        
    def get_host_path_from_volume(self, volume_string: str) -> Optional[str]:
        """
        Extract the host path from a Docker volume string.
        
        Args:
            volume_string: Docker volume string in format "host_path:container_path"
            
        Returns:
            Host path or None if the format is invalid
        """
        host_path = None
        
        # First validate the volume string
        if self.validate_docker_volume(volume_string):
            host_path = volume_string.split(':', 1)[0]
            
            # Expand user home directory if needed
            if host_path.startswith('~'):
                from os.path import expanduser
                host_path = expanduser(host_path)
                
            # Convert relative paths to absolute if needed
            if not os.path.isabs(host_path):
                host_path = os.path.abspath(host_path)
        else:
            self.logger.error(f"Invalid Docker volume format: {volume_string}. Expected 'host_path:container_path' where container_path starts with /")
            
        return host_path
        
    # Facade Pattern methods for Docker configuration
    def get_docker_image(self, tool_name: str) -> str:
        """
        Get Docker image name for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Docker image name with default fallback
        """
        docker_config = self.get_docker_config(tool_name)
        return docker_config.get('image', f'{tool_name}-default')
    
    def get_input_volume(self, tool_name: str) -> str:
        """
        Get input volume for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Input volume string or empty string if not configured
        """
        docker_config = self.get_docker_config(tool_name)
        return docker_config.get('input_volume', '')
    
    def get_output_volume(self, tool_name: str) -> str:
        """
        Get output volume for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Output volume string or empty string if not configured
        """
        docker_config = self.get_docker_config(tool_name)
        return docker_config.get('output_volume', '')
    
    def get_extra_volumes(self, tool_name: str) -> List[str]:
        """
        Get extra volumes for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of extra volume strings or empty list if not configured
        """
        docker_config = self.get_docker_config(tool_name)
        return docker_config.get('extra_volumes', [])
    
    def get_config_path(self, tool_name: str) -> Optional[str]:
        """
        Get config path for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Config path or None if not configured
        """
        docker_config = self.get_docker_config(tool_name)
        return docker_config.get('config_path') 
    
    def get_log_dir(self) -> str:
        """
        Get log directory from configuration.
        
        Returns:
            Log directory path
        """
        return self.config.get('log_dir', './logs')
    
    def get_extracted_paths(self) -> List[str]:
        """
        Get extracted_paths from configuration.
                   
        Returns:
            List of extracted paths of []
        """
        tool_config = self.get_tool_config("extractor")
        return tool_config.get("extracted_paths", [])
    
    def get_chunked_paths(self) -> List[str]:
        """
        Get extracted_paths from configuration.
                   
        Returns:
            List of extracted paths of []
        """
        tool_config = self.get_tool_config("chunker")
        return tool_config.get("chunked_paths", [])
