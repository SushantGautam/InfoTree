"""Windowing module for splitting text into overlapping windows."""

from typing import List
from .models import Window
from .config import InfoTreeConfig


class Windower:
    """Creates overlapping windows over text."""
    
    def __init__(self, config: InfoTreeConfig):
        """Initialize windower with configuration.
        
        Args:
            config: InfoTreeConfig instance
        """
        self.config = config
        self.window_chars = config.window_chars
        self.overlap_chars = config.overlap_chars
    
    def create_windows(self, text: str) -> List[Window]:
        """Create overlapping windows over the input text.
        
        Args:
            text: Input text string
            
        Returns:
            List of Window objects covering the entire text
        """
        if not text:
            return []
        
        windows = []
        text_length = len(text)
        wid = 0
        start = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.window_chars, text_length)
            
            # Extract window text
            window_text = text[start:end]
            
            # Create window
            window = Window(
                wid=wid,
                start=start,
                end=end,
                text=window_text
            )
            windows.append(window)
            
            # If we've reached the end, break
            if end >= text_length:
                break
            
            # Move to next window with overlap
            start = end - self.overlap_chars
            wid += 1
        
        return windows
    
    def get_window_count(self, text: str) -> int:
        """Calculate how many windows will be created.
        
        Args:
            text: Input text string
            
        Returns:
            Number of windows
        """
        if not text:
            return 0
        
        text_length = len(text)
        step = self.window_chars - self.overlap_chars
        
        if step <= 0:
            return 1
        
        # Calculate number of windows needed
        count = 1 + max(0, (text_length - self.window_chars + step - 1) // step)
        return count
