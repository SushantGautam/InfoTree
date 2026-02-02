"""LLM-based labeling for tree nodes."""

import json
import asyncio
from typing import List
from tqdm import tqdm
from openai import AsyncOpenAI
from .models import TreeNode, LeafNode, InternalNode
from .config import InfoTreeConfig
from .utils import exponential_backoff_retry, truncate_text


class NodeLabeler:
    """Generates labels for tree nodes using LLM."""
    
    def __init__(self, config: InfoTreeConfig):
        """Initialize node labeler.
        
        Args:
            config: InfoTreeConfig instance
        """
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=0
        )
        self.total_nodes = 0
        self.labeled_nodes = 0
        self.pbar = None
    
    def label_tree(self, root: TreeNode):
        """Recursively label all nodes in tree.
        
        Args:
            root: Root TreeNode to label
        """
        # Count total nodes first
        self.total_nodes = self._count_nodes(root)
        self.labeled_nodes = 0
        
        # Create progress bar
        self.pbar = tqdm(total=self.total_nodes, desc="Labeling nodes", unit="node", leave=True)
        
        try:
            asyncio.run(self._label_node_recursive(root))
        finally:
            self.pbar.close()
            self.pbar = None
    
    def _count_nodes(self, node: TreeNode) -> int:
        """Count total nodes in tree.
        
        Args:
            node: TreeNode to start from
            
        Returns:
            Total node count
        """
        count = 1
        if isinstance(node, InternalNode):
            for child in node.children:
                count += self._count_nodes(child)
        return count
    
    async def _label_node_recursive(self, node: TreeNode):
        """Recursively label a node and its children.
        
        Args:
            node: TreeNode to label
        """
        if isinstance(node, LeafNode):
            # Label leaf nodes with truncated content
            node.label = self._generate_leaf_label(node)
            # Update progress
            if self.pbar is not None:
                self.pbar.update(1)
        
        elif isinstance(node, InternalNode):
            # First label all children concurrently
            if node.children:
                await asyncio.gather(*[self._label_node_recursive(child) for child in node.children])
            
            # Then label this internal node based on children
            node.label = await self._generate_internal_label(node)
            # Update progress
            if self.pbar is not None:
                self.pbar.update(1)
    
    def _generate_leaf_label(self, node: LeafNode) -> str:
        """Generate label for a leaf node.
        
        Args:
            node: LeafNode to label
            
        Returns:
            Label string
        """
        # For leaf nodes, use first few words as label
        text = node.text.strip()
        words = text.split()[:8]
        label = " ".join(words)
        
        if len(label) > 60:
            label = label[:57] + "..."
        
        return label
    
    async def _generate_internal_label(self, node: InternalNode) -> str:
        """Generate label for an internal node using LLM.
        
        Args:
            node: InternalNode to label
            
        Returns:
            Label string
        """
        try:
            # Collect sample snippets from children
            snippets = self._collect_child_snippets(node)
            
            # Call LLM with retry
            @exponential_backoff_retry(
                max_retries=self.config.max_retries,
                initial_delay=self.config.retry_delay
            )
            async def call_llm():
                return await self._call_labeling_llm(snippets)
            
            label = await call_llm()
            return label
            
        except Exception as e:
            print(f"Warning: Failed to generate label for node {node.node_id}: {e}")
            return "Unlabeled Section"
    
    def _collect_child_snippets(self, node: InternalNode) -> List[str]:
        """Collect representative text snippets from children.
        
        Args:
            node: InternalNode
            
        Returns:
            List of text snippets
        """
        snippets = []
        max_snippet_length = 200
        
        for child in node.children[:10]:  # Limit to first 10 children
            if isinstance(child, LeafNode):
                snippet = truncate_text(child.text, max_snippet_length)
                snippets.append(snippet)
            elif isinstance(child, InternalNode):
                # Use child's label if available
                if child.label:
                    snippets.append(f"[Section: {child.label}]")
                else:
                    # Try to get snippet from first leaf
                    first_leaf = self._get_first_leaf(child)
                    if first_leaf:
                        snippet = truncate_text(first_leaf.text, max_snippet_length)
                        snippets.append(snippet)
        
        return snippets
    
    def _get_first_leaf(self, node: TreeNode) -> LeafNode:
        """Get the first leaf node in subtree.
        
        Args:
            node: TreeNode
            
        Returns:
            First LeafNode or None
        """
        if isinstance(node, LeafNode):
            return node
        
        if isinstance(node, InternalNode) and node.children:
            return self._get_first_leaf(node.children[0])
        
        return None
    
    async def _call_labeling_llm(self, snippets: List[str]) -> str:
        """Call LLM to generate label.
        
        Args:
            snippets: List of representative text snippets
            
        Returns:
            Label string
        """
        snippets_text = "\n\n".join(f"Snippet {i+1}:\n{s}" for i, s in enumerate(snippets))
        
        prompt = f"""You are tasked with creating a concise index-style label for a section of text.

RULES:
1. Label must be 3-8 words
2. Use noun phrase format (no full sentences)
3. Describe what ALL snippets have in common or the overall theme
4. Be specific and descriptive
5. Do NOT speculate or add concepts not present in the text
6. Do NOT use generic labels like "Various Topics" or "Text Section"

REPRESENTATIVE SNIPPETS FROM SECTION:
{snippets_text}

Return ONLY the label text, nothing else.
"""
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating concise, descriptive index labels."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=50,
        )
        
        label = response.choices[0].message.content.strip()
        
        # Clean up label
        label = label.strip('"\'')
        
        # Ensure it's not too long
        if len(label) > 80:
            words = label.split()[:8]
            label = " ".join(words)
        
        return label
