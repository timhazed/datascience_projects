import logging

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import main components for convenience
from pipeline.config_manager import ConfigManager
from pipeline.tool_executor import ToolExecutor

# Import subpackages
from pipeline import extractors
from pipeline import processors

__all__ = [
    'ConfigManager',
    'EnvironmentManager',
    'ToolExecutor',
    'extractors',
    'processors'
]
