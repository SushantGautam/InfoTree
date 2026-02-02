"""Embedding generation for leaf nodes."""

import asyncio
from typing import List
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
from .models import LeafNode
from .config import InfoTreeConfig
from .utils import exponential_backoff_retry, batch_list


class EmbeddingGenerator:
    """Generates embeddings for text nodes."""
    
    def __init__(self, config: InfoTreeConfig):
        """Initialize embedding generator.
        
        Args:
            config: InfoTreeConfig instance
        """
        self.config = config
        
        # Use separate embedding API configuration if provided, otherwise use main API config
        embedding_base_url = config.embedding_base_url or config.base_url
        embedding_api_key = config.embedding_api_key or config.api_key
        
        self.client = AsyncOpenAI(
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            timeout=config.timeout,
            max_retries=0
        )
    
    def generate_embeddings(self, nodes: List[LeafNode]) -> List[LeafNode]:
        """Generate embeddings for all leaf nodes concurrently.
        
        Args:
            nodes: List of LeafNode objects
            
        Returns:
            Same list with embeddings populated
        """
        if not nodes:
            return nodes
        
        return asyncio.run(self._generate_embeddings_async(nodes))
    
    async def _generate_embeddings_async(self, nodes: List[LeafNode]) -> List[LeafNode]:
        """Async implementation of generate_embeddings."""
        # Process in batches
        batches = batch_list(nodes, self.config.embedding_batch_size)
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def generate_with_semaphore(batch):
            async with semaphore:
                await self._generate_batch_embeddings(batch)
        
        # Create tasks for all batches
        tasks = [generate_with_semaphore(batch) for batch in batches]
        
        # Execute with progress bar
        with tqdm(total=len(batches), desc="Generating embeddings", unit="batch", leave=True) as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as e:
                    print(f"Warning: Exception during async embedding generation: {e}")
                pbar.update(1)
        
        return nodes
    
    async def _generate_batch_embeddings(self, nodes: List[LeafNode]):
        """Generate embeddings for a batch of nodes.
        
        Args:
            nodes: Batch of LeafNode objects
        """
        # Extract texts
        texts = [node.text for node in nodes]
        
        # Call embedding API with retry
        @exponential_backoff_retry(
            max_retries=self.config.max_retries,
            initial_delay=self.config.retry_delay
        )
        async def call_embedding_api():
            return await self._call_embedding_api(texts)
        
        try:
            embeddings = await call_embedding_api()
            
            # Assign embeddings to nodes
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
                
        except Exception as e:
            print(f"Warning: Failed to generate embeddings for batch: {e}")
            # Assign zero embeddings as fallback
            embedding_dim = 1536  # Default for text-embedding-3-small
            for node in nodes:
                node.embedding = [0.0] * embedding_dim
    
    async def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embedding API.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input=texts
        )
        
        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def get_embedding_matrix(self, nodes: List[LeafNode]) -> np.ndarray:
        """Get embeddings as a numpy matrix.
        
        Args:
            nodes: List of LeafNode objects with embeddings
            
        Returns:
            Numpy array of shape (n_nodes, embedding_dim)
        """
        embeddings = [node.embedding for node in nodes if node.embedding is not None]
        
        if not embeddings:
            return np.array([])
        
        return np.array(embeddings)
    
    def compute_similarity(self, node1: LeafNode, node2: LeafNode) -> float:
        """Compute cosine similarity between two nodes.
        
        Args:
            node1: First LeafNode
            node2: Second LeafNode
            
        Returns:
            Cosine similarity (0 to 1)
        """
        if node1.embedding is None or node2.embedding is None:
            return 0.0
        
        vec1 = np.array(node1.embedding)
        vec2 = np.array(node2.embedding)
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
