"""LLM-based node extraction from windows."""

import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
from openai import AsyncOpenAI
from .models import Window, LeafNode, ExtractionResult
from .config import InfoTreeConfig
from .utils import exponential_backoff_retry, generate_node_id


class NodeExtractor:
    """Extracts atomic nodes from text windows using LLM."""
    
    def __init__(self, config: InfoTreeConfig):
        """Initialize node extractor.
        
        Args:
            config: InfoTreeConfig instance
        """
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=0  # We handle retries ourselves
        )
        self.node_counter = 0
    
    async def extract_nodes_from_window(self, window: Window, original_text: str) -> ExtractionResult:
        """Extract atomic nodes from a window.
        
        Args:
            window: Window object
            original_text: Original full text (for validation)
            
        Returns:
            ExtractionResult containing extracted nodes
        """
        try:
            # Call LLM with retry logic
            @exponential_backoff_retry(
                max_retries=self.config.max_retries,
                initial_delay=self.config.retry_delay
            )
            async def call_llm():
                return await self._call_extraction_llm(window)
            
            relative_nodes = await call_llm()
            
            # Convert to absolute offsets
            leaf_nodes = self._convert_to_leaf_nodes(
                relative_nodes, 
                window, 
                original_text
            )
            
            return ExtractionResult(
                nodes=leaf_nodes,
                window_id=window.wid,
                success=True
            )
            
        except Exception as e:
            return ExtractionResult(
                nodes=[],
                window_id=window.wid,
                success=False,
                error=str(e)
            )
    
    async def _call_extraction_llm(self, window: Window) -> List[Dict[str, int]]:
        """Call LLM to extract node boundaries.
        
        Args:
            window: Window object
            
        Returns:
            List of dicts with 'start' and 'end' keys (relative offsets)
        """
        prompt = self._build_extraction_prompt(window)
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a text segmentation assistant. Extract atomic text segments suitable for indexing."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=self.config.max_tokens,
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            result = json.loads(content)
            if "nodes" in result:
                return result["nodes"]
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
                result = json.loads(content)
                if "nodes" in result:
                    return result["nodes"]
                return result
            raise
    
    def _build_extraction_prompt(self, window: Window) -> str:
        """Build prompt for node extraction.
        
        Args:
            window: Window object
            
        Returns:
            Prompt string
        """
        prompt = f"""You are tasked with segmenting the following text into atomic indexing nodes.

RULES:
1. Each node should be a contiguous span of text
2. Nodes must fully cover the input text with NO GAPS
3. Prefer breaking at blank lines or sentence boundaries
4. Each node should be between {self.config.min_node_chars} and {self.config.max_node_chars} characters
5. Nodes should represent coherent semantic units (paragraph-like)

TEXT TO SEGMENT (length: {len(window.text)} chars):
{window.text}

OUTPUT FORMAT:
Return a JSON array called "nodes" where each element has:
- "start": relative character offset (0-based)
- "end": relative character offset (exclusive)

Example:
{{
  "nodes": [
    {{"start": 0, "end": 450}},
    {{"start": 450, "end": 890}}
  ]
}}

Ensure the nodes fully cover the text from 0 to {len(window.text)}.
"""
        return prompt
    
    def _convert_to_leaf_nodes(
        self, 
        relative_nodes: List[Dict[str, int]], 
        window: Window,
        original_text: str
    ) -> List[LeafNode]:
        """Convert relative offsets to absolute leaf nodes.
        
        Args:
            relative_nodes: List of dicts with relative 'start' and 'end'
            window: Window object
            original_text: Original full text
            
        Returns:
            List of LeafNode objects
        """
        leaf_nodes = []
        
        for node_data in relative_nodes:
            rel_start = node_data["start"]
            rel_end = node_data["end"]
            
            # Convert to absolute offsets
            abs_start = window.start + rel_start
            abs_end = window.start + rel_end
            
            # Validate bounds
            if abs_start < 0 or abs_end > len(original_text):
                continue
            if abs_start >= abs_end:
                continue
            
            # Extract text
            text = original_text[abs_start:abs_end]
            
            # Create leaf node
            node_id = generate_node_id("leaf", self.node_counter)
            self.node_counter += 1
            
            leaf = LeafNode(
                node_id=node_id,
                start=abs_start,
                end=abs_end,
                text=text
            )
            leaf_nodes.append(leaf)
        
        return leaf_nodes
    
    def extract_nodes_from_windows(
        self, 
        windows: List[Window], 
        original_text: str
    ) -> List[LeafNode]:
        """Extract nodes from multiple windows concurrently.
        
        Args:
            windows: List of Window objects
            original_text: Original full text
            
        Returns:
            List of all extracted LeafNode objects
        """
        return asyncio.run(self._extract_nodes_async(windows, original_text))
    
    async def _extract_nodes_async(self, windows: List[Window], original_text: str) -> List[LeafNode]:
        """Async implementation of extract_nodes_from_windows."""
        all_nodes = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def extract_with_semaphore(window):
            async with semaphore:
                return await self.extract_nodes_from_window(window, original_text)
        
        # Create tasks for all windows
        tasks = [extract_with_semaphore(window) for window in windows]
        
        # Execute with progress bar
        with tqdm(total=len(windows), desc="Extracting nodes", unit="window", leave=True) as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result.success:
                        all_nodes.extend(result.nodes)
                    else:
                        print(f"Warning: Failed to extract nodes from window {result.window_id}: {result.error}")
                except Exception as e:
                    print(f"Warning: Exception extracting nodes: {e}")
                pbar.update(1)
        
        return all_nodes
