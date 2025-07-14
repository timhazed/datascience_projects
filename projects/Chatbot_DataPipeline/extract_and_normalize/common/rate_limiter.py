import time
import threading
import math
from typing import Optional, Tuple
import tiktoken
import logging
class RateLimiter:
    """
    Rate limiter for API calls to prevent throttling.
    Supports both request-based and token-based rate limiting with exponential backoff.
    """
    
    def __init__(self, logger: logging.Logger, calls_per_minute: int, tokens_per_minute: int, model_name: str):
        """
        Initialize the rate limiter with required parameters.
        
        Args:
            calls_per_minute: Maximum API calls per minute (must be > 0)
            tokens_per_minute: Maximum tokens per minute (must be > 0)
            model_name: Name of the LLM model being used
        """
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be greater than 0")
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be greater than 0")
        if not model_name:
            raise ValueError("model_name cannot be empty")
            
        # Request-based rate limiting
        self.logger = logger
        self.calls_per_minute = calls_per_minute
        self.call_interval = 60.0 / self.calls_per_minute
        self.last_call_time = 0
        
        # Token-based rate limiting
        self.tokens_per_minute = tokens_per_minute
        self.token_interval = 60.0 / self.tokens_per_minute
        self.token_bucket = self.tokens_per_minute
        self.last_token_refill_time = time.time()
        
        # Adaptive rate limiting
        self.consecutive_429s = 0
        self.last_429_time = 0
        self.rate_limit_multiplier = 1.0
        
        # Token counting
        self.model_name = model_name
        try:
            # Try to use the appropriate encoding for the model
            if "llama" in model_name.lower():
                # For Llama models, use cl100k_base which is the closest approximation
                self.encoding = tiktoken.get_encoding("cl100k_base")
            else:
                # Default to cl100k_base for other models
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.info(f"Warning: Could not initialize tiktoken, falling back to basic token counting: {e}")
            self.encoding = None
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count using tiktoken if available, falling back to basic estimation.
        """
        if not text:
            return 0
            
        if self.encoding:
            try:
                # Use tiktoken for accurate token counting
                return len(self.encoding.encode(text))
            except Exception as e:
                self.logger.info(f"Warning: tiktoken token counting failed, falling back to basic estimation: {e}")
        
        # Fallback to basic estimation if tiktoken fails
        # This is a more conservative estimate than before
        words = len(text.split())
        punctuation = sum(1 for char in text if char in '.,;:!?()[]{}"\'')
        
        # Conservative estimate: assume 1.5 tokens per word and 1 token per punctuation
        # This is more conservative than the previous 1.3 multiplier
        return max(1, int(words * 1.5 + punctuation))
    
    def _refill_token_bucket(self):
        """Refill token bucket based on elapsed time."""
        if self.tokens_per_minute:
            current_time = time.time()
            elapsed = current_time - self.last_token_refill_time
            
            # Calculate tokens to add based on elapsed time
            tokens_to_add = int(elapsed * (self.tokens_per_minute / 60.0))
            
            if tokens_to_add > 0:
                self.token_bucket = min(self.token_bucket + tokens_to_add, self.tokens_per_minute)
                self.last_token_refill_time = current_time
    
    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff time based on consecutive 429s."""
        if self.consecutive_429s == 0:
            return 0
            
        # Exponential backoff with jitter
        base_delay = min(2 ** self.consecutive_429s, 60)  # Cap at 60 seconds
        jitter = base_delay * 0.1  # Add 10% jitter
        return base_delay + jitter
    
    def handle_429(self):
        """Handle a 429 response by adjusting rate limits and backoff."""
        with self.lock:
            current_time = time.time()
            
            # Reset consecutive 429s if it's been more than 5 minutes since last 429
            if current_time - self.last_429_time > 300:
                self.consecutive_429s = 0
                self.rate_limit_multiplier = 1.0
            
            self.consecutive_429s += 1
            self.last_429_time = current_time
            
            # Reduce rate limits by 20% for each consecutive 429
            self.rate_limit_multiplier *= 0.8
            
            # Apply the multiplier to both rate limits
            if self.calls_per_minute:
                self.calls_per_minute = max(1, int(self.calls_per_minute * self.rate_limit_multiplier))
                self.call_interval = 60.0 / self.calls_per_minute
                
            if self.tokens_per_minute:
                self.tokens_per_minute = max(1, int(self.tokens_per_minute * self.rate_limit_multiplier))
                self.token_interval = 60.0 / self.tokens_per_minute
    
    def wait(self, text: Optional[str] = None) -> int:
        """
        Wait if necessary to comply with rate limits.
        
        Args:
            text: Text to estimate token count for token-based rate limiting
            
        Returns:
            int: Estimated token count
        """
        # If no rate limits are set, return immediately
        if not self.calls_per_minute and not self.tokens_per_minute:
            return 0 if not text else self._estimate_token_count(text)
            
        with self.lock:
            # Calculate backoff time if we've had recent 429s
            backoff_time = self._calculate_backoff()
            
            # Handle request-based rate limiting
            current_time = time.time()
            elapsed_since_last_call = current_time - self.last_call_time
            request_wait_time = max(0, self.call_interval - elapsed_since_last_call) if self.calls_per_minute else 0
            
            # Handle token-based rate limiting if enabled and text is provided
            token_wait_time = 0
            estimated_tokens = 0
            if self.tokens_per_minute and text:
                estimated_tokens = self._estimate_token_count(text)
                
                # Refill the token bucket based on elapsed time
                self._refill_token_bucket()
                
                # If not enough tokens, calculate wait time
                if estimated_tokens > self.token_bucket:
                    tokens_needed = estimated_tokens - self.token_bucket
                    token_wait_time = tokens_needed * self.token_interval
                
                # Update token bucket
                self.token_bucket = max(0, self.token_bucket - estimated_tokens)
            
            # Wait for the maximum of all wait times
            wait_time = max(backoff_time, request_wait_time, token_wait_time)
            if wait_time > 0:
                time.sleep(wait_time)
                
            self.last_call_time = time.time()
            
            return estimated_tokens 