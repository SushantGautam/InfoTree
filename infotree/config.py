"""Configuration management for InfoTree pipeline."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class InfoTreeConfig(BaseModel):
    """Configuration for InfoTree pipeline."""
    
    model_config = ConfigDict(frozen=True)
    
    # OpenAI API configuration (for LLM)
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Model name for LLM calls")
    max_tokens: int = Field(default=4096, description="Maximum tokens for LLM responses")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    
    # Embedding API configuration (can be separate from LLM API)
    embedding_base_url: Optional[str] = Field(default=None, description="Embedding API base URL (defaults to base_url if not set)")
    embedding_api_key: Optional[str] = Field(default=None, description="Embedding API key (defaults to api_key if not set)")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    
    # Windowing parameters
    window_chars: int = Field(default=6000, description="Window size in characters")
    overlap_chars: int = Field(default=800, description="Overlap size in characters")
    
    # Node extraction parameters
    min_node_chars: int = Field(default=300, description="Minimum node size in characters")
    max_node_chars: int = Field(default=1200, description="Maximum node size in characters")
    
    # Deduplication parameters
    iou_threshold: float = Field(default=0.85, description="IoU threshold for deduplication")
    
    # Clustering parameters
    max_children: int = Field(default=10, description="Maximum children per internal node")
    max_depth: int = Field(default=4, description="Maximum tree depth")
    
    # Retry parameters
    max_retries: int = Field(default=3, description="Maximum retries for failed operations")
    retry_delay: float = Field(default=1.0, description="Initial retry delay in seconds")
    
    # Batch processing
    embedding_batch_size: int = Field(default=100, description="Batch size for embedding generation")
    
    # Parallel processing
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent async API requests")
    chunker: object = Field(..., description="Chunker instance for text chunking")
    
    @field_validator("overlap_chars")
    @classmethod
    def validate_overlap(cls, v, info):
        window = info.data.get("window_chars", 0)
        if v >= window:
            raise ValueError("overlap_chars must be less than window_chars")
        return v
    
    @field_validator("iou_threshold")
    @classmethod
    def validate_iou(cls, v):
        if not 0 < v <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        return v
    
    @field_validator("max_children")
    @classmethod
    def validate_max_children(cls, v):
        if v < 2:
            raise ValueError("max_children must be at least 2")
        return v
    
    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, v):
        if v < 1:
            raise ValueError("max_depth must be at least 1")
        return v
