"""Utility functions for InfoTree."""

import random
import time
from typing import TypeVar, Callable, Any
from functools import wraps

T = TypeVar('T')


def exponential_backoff_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> Callable:
    """Decorator for exponential backoff retry with jitter.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter to delays
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt >= max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    actual_delay = min(delay, max_delay)
                    
                    # Add jitter
                    if jitter:
                        actual_delay *= (0.5 + random.random())
                    
                    time.sleep(actual_delay)
                    delay *= 2
            
            # If all retries failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


def truncate_text(text: str, max_chars: int, ellipsis: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_chars: Maximum character count
        ellipsis: String to append when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(ellipsis)] + ellipsis


def calculate_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculate Intersection over Union (IoU) for two spans.
    
    Args:
        start1: Start offset of first span
        end1: End offset of first span
        start2: Start offset of second span
        end2: End offset of second span
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_length = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_length = union_end - union_start
    
    # Avoid division by zero
    if union_length == 0:
        return 0.0
    
    return intersection_length / union_length


def generate_node_id(prefix: str, index: int) -> str:
    """Generate a node ID.
    
    Args:
        prefix: Prefix for the ID (e.g., 'leaf', 'internal')
        index: Numeric index
        
    Returns:
        Node ID string
    """
    return f"{prefix}_{index}"


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4


def batch_list(items: list, batch_size: int) -> list:
    """Split a list into batches.
    
    Args:
        items: List to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
