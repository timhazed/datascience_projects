from pathlib import Path
import subprocess
import time
from typing import Dict, Any, List, Optional
from prefect import task
from pipeline.config_manager import ConfigManager

class ToolExecutor:
    """
    Executes tools as Docker containers.
    """
    
    def __init__(self, config_manager: ConfigManager, logger):
        """
        Initialize the tool executor.
        
        Args:
            logger: Logger instance to use throughout the class
        """
        self.config_manager = config_manager
        self.logger = logger
    
    @task(
        retries=2,
        retry_delay_seconds=10,
    )
    def execute_tool(
        self,
        image_name: str,
        args: List[str],
        input_volume: str,
        output_volume: str,
        env_file: Optional[str] = None,
        extra_volumes: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        tool_name: str = "docker-tool",
        timeout: int = 1200,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool as a Docker container
        
        Args:
            image_name: Docker image name
            args: Command arguments to pass to the container
            input_volume: Host path for input volume
            output_volume: Host path for output volume
            env_file: Path to environment file
            extra_volumes: Additional volumes to mount
            env_vars: Additional environment variables
            tool_name: Name of tool (for logging)
            timeout: Timeout in seconds
            doc_id: Document ID to include in log filenames for better traceability
            
        Returns:
            Execution results dictionary
        """
        # Create log directory
        log_dir = Path(self.config_manager.get_log_dir()) / tool_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Format timestamp in a human-readable format
        formatted_time = time.strftime("%Y%m%d-%H%M%S")
        
        # Include doc_id in log filename if provided
        if doc_id:
            log_file = log_dir / f"{tool_name}_{doc_id}_{formatted_time}.log"
        else:
            log_file = log_dir / f"{tool_name}_{formatted_time}.log"
        
        # Build the docker command
        cmd = ["docker", "run", "--rm"]
        
        # Add environment file if provided
        if env_file:
            cmd.extend(["--env-file", env_file])
        
        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Always add PYTHONUNBUFFERED for better logging
        cmd.extend(["-e", "PYTHONUNBUFFERED=1"])
        
        # Add volume mounts
        cmd.extend(["-v", input_volume])

        if output_volume:
            cmd.extend(["-v", output_volume])
        
        if extra_volumes:
            for volume in extra_volumes:
                cmd.extend(["-v", volume])
        
        # Add image name and arguments
        cmd.append(image_name)
        cmd.extend(args)
        
        # Log the command
        self.logger.info(f"Executing {tool_name} with Docker: {' '.join(cmd)} timeout: {timeout}")
        
        # Execute the Docker command
        start_time = time.time()
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            
            # Log output
            with open(log_file, "w") as f:
                f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Check if the command was successful
            if process.returncode == 0:
                self.logger.info(f"Successfully executed {tool_name} in {duration:.2f} seconds")
                return {
                    "status": "success",
                    "tool": tool_name,
                    "image": image_name,
                    "duration": duration,
                    "log_path": str(log_file),
                    "exit_code": 0,
                    "output": process.stdout
                }
            else:
                self.logger.error(f"{tool_name} failed with exit code {process.returncode}")
                self.logger.error(f"Error output: {process.stderr}")
                return {
                    "status": "failure",
                    "tool": tool_name,
                    "image": image_name,
                    "duration": duration,
                    "log_path": str(log_file),
                    "exit_code": process.returncode,
                    "error": f"{tool_name} execution failed: {process.stderr}",
                    "output": process.stdout
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"{tool_name} execution timed out after {timeout} seconds")
            with open(log_file, "w") as f:
                f.write("EXECUTION TIMED OUT")
            return {
                "status": "failure",
                "tool": tool_name,
                "image": image_name,
                "error": f"Process timed out after {timeout} seconds",
                "exit_code": -1,
                "log_path": str(log_file)
            }
        except Exception as e:
            self.logger.exception(f"Error executing {tool_name}: {e}")
            with open(log_file, "a") as f:
                f.write(f"\nEXCEPTION: {str(e)}")
            return {
                "status": "failure",
                "tool": tool_name,
                "image": image_name,
                "error": str(e),
                "exit_code": -1,
                "log_path": str(log_file)
            } 