"""
Common Utilities Module

This module provides utility functions used across the leukemia detection project.
"""

import os
import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
from PIL import Image


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    # Suppress some verbose libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def ensure_dir(directory: Union[str, Path]) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return os.path.getsize(file_path) / (1024 * 1024)


def get_available_memory_gb() -> float:
    """
    Get available system memory in GB.
    
    Returns:
        Available memory in GB
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        warnings.warn("psutil not available, cannot check memory")
        return float('inf')


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple = (Exception,)
) -> Callable:
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                    time.sleep(delay)
            
        return wrapper
    return decorator


def validate_image_dimensions(
    image: Union[np.ndarray, Image.Image],
    min_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Validate image dimensions.
    
    Args:
        image: Image to validate
        min_size: Minimum (width, height) tuple
        max_size: Maximum (width, height) tuple
        
    Returns:
        True if valid, False otherwise
    """
    if isinstance(image, Image.Image):
        width, height = image.size
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            height, width = image.shape
        elif len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            return False
    else:
        return False
    
    if min_size and (width < min_size[0] or height < min_size[1]):
        return False
    
    if max_size and (width > max_size[0] or height > max_size[1]):
        return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize file path for cross-platform compatibility.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path string
    """
    return str(Path(path).resolve())


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Get hash of file contents.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)
        
    Returns:
        Hexadecimal hash string
    """
    import hashlib
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def batch_process(
    items: List[Any],
    batch_size: int,
    process_func: Callable,
    *args,
    **kwargs
) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to apply to each batch
        *args: Additional arguments for process_func
        **kwargs: Additional keyword arguments for process_func
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch, *args, **kwargs)
        
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {seconds:.0f}s"
    
    hours = minutes // 60
    minutes = minutes % 60
    
    return f"{hours}h {minutes}m {seconds:.0f}s"


def create_progress_callback(
    total_items: int,
    description: str = "Processing"
) -> Callable[[int], None]:
    """
    Create a progress callback function.
    
    Args:
        total_items: Total number of items to process
        description: Description for progress display
        
    Returns:
        Progress callback function
    """
    def progress_callback(completed: int) -> None:
        percentage = (completed / total_items) * 100
        logger = logging.getLogger(__name__)
        logger.info(f"{description}: {completed}/{total_items} ({percentage:.1f}%)")
    
    return progress_callback


def validate_file_extension(
    file_path: Union[str, Path],
    valid_extensions: List[str]
) -> bool:
    """
    Validate file extension.
    
    Args:
        file_path: Path to file
        valid_extensions: List of valid extensions (e.g., ['.png', '.jpg'])
        
    Returns:
        True if extension is valid, False otherwise
    """
    file_ext = Path(file_path).suffix.lower()
    return file_ext in [ext.lower() for ext in valid_extensions]


def load_json_with_comments(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file that may contain comments.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    import json
    import re
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove single-line comments
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    
    # Remove multi-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    return json.loads(content)


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries recursively.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dictionaries(result[key], value)
            else:
                result[key] = value
    
    return result


def find_files_by_pattern(
    directory: Union[str, Path],
    pattern: str,
    recursive: bool = True
) -> List[Path]:
    """
    Find files matching a pattern.
    
    Args:
        directory: Directory to search in
        pattern: File pattern (e.g., '*.png')
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current += increment
        self._log_progress()
    
    def set_progress(self, current: int) -> None:
        """Set current progress."""
        self.current = current
        self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current progress."""
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                estimated_total = elapsed * self.total / self.current
                remaining = estimated_total - elapsed
                eta = format_duration(remaining)
            else:
                eta = "Unknown"
            
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({percentage:.1f}%) - ETA: {eta}"
            )
    
    def finish(self) -> None:
        """Mark progress as finished."""
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"{self.description} completed in {format_duration(elapsed)}"
        )


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            import psutil
            self.psutil = psutil
            self.available = True
        except ImportError:
            self.psutil = None
            self.available = False
            self.logger.warning("psutil not available, resource monitoring disabled")
    
    def log_system_info(self) -> None:
        """Log current system resource usage."""
        if not self.available:
            return
        
        # Memory info
        memory = self.psutil.virtual_memory()
        self.logger.info(f"Memory: {memory.percent}% used ({format_bytes(memory.used)}/{format_bytes(memory.total)})")
        
        # CPU info
        cpu_percent = self.psutil.cpu_percent(interval=1)
        self.logger.info(f"CPU: {cpu_percent}% used")
        
        # Disk info
        disk = self.psutil.disk_usage('/')
        self.logger.info(f"Disk: {disk.percent}% used ({format_bytes(disk.used)}/{format_bytes(disk.total)})")
    
    def check_memory_usage(self, threshold_percent: float = 90.0) -> bool:
        """
        Check if memory usage exceeds threshold.
        
        Args:
            threshold_percent: Memory usage threshold percentage
            
        Returns:
            True if usage exceeds threshold
        """
        if not self.available:
            return False
        
        memory = self.psutil.virtual_memory()
        if memory.percent > threshold_percent:
            self.logger.warning(f"High memory usage: {memory.percent}%")
            return True
        
        return False