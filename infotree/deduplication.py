"""Deduplication module for removing overlapping nodes."""

from typing import List, Set
from tqdm import tqdm
from .models import LeafNode
from .config import InfoTreeConfig
from .utils import calculate_iou


class Deduplicator:
    """Removes duplicate nodes based on IoU threshold."""
    
    def __init__(self, config: InfoTreeConfig):
        """Initialize deduplicator.
        
        Args:
            config: InfoTreeConfig instance
        """
        self.config = config
        self.iou_threshold = config.iou_threshold
    
    def deduplicate(self, nodes: List[LeafNode]) -> List[LeafNode]:
        """Remove duplicate nodes using IoU-based clustering.
        
        Args:
            nodes: List of LeafNode objects (possibly with duplicates)
            
        Returns:
            List of deduplicated LeafNode objects
        """
        if not nodes:
            return []
        
        # Sort by start offset for efficient processing
        sorted_nodes = sorted(nodes, key=lambda n: n.start)
        
        # Track which nodes to keep
        unique_nodes = []
        skip_indices: Set[int] = set()
        
        for i, node1 in tqdm(enumerate(sorted_nodes), total=len(sorted_nodes), 
                              desc="Deduplicating nodes", unit="node", leave=True):
            if i in skip_indices:
                continue
            
            # Find all duplicates of this node
            duplicates = [node1]
            
            for j in range(i + 1, len(sorted_nodes)):
                if j in skip_indices:
                    continue
                
                node2 = sorted_nodes[j]
                
                # If node2 starts after node1 ends, no more overlaps possible
                if node2.start >= node1.end:
                    break
                
                # Calculate IoU
                iou = calculate_iou(node1.start, node1.end, node2.start, node2.end)
                
                if iou >= self.iou_threshold:
                    duplicates.append(node2)
                    skip_indices.add(j)
            
            # Select the best representative from duplicates
            best_node = self._select_best_node(duplicates)
            unique_nodes.append(best_node)
        
        return unique_nodes
    
    def _select_best_node(self, duplicates: List[LeafNode]) -> LeafNode:
        """Select the best node from a list of duplicates.
        
        Strategy: prefer the node with the most complete span (longest)
        
        Args:
            duplicates: List of duplicate LeafNode objects
            
        Returns:
            Best LeafNode
        """
        if len(duplicates) == 1:
            return duplicates[0]
        
        # Select the longest node
        return max(duplicates, key=lambda n: n.end - n.start)
    
    def get_coverage_stats(self, nodes: List[LeafNode], text_length: int) -> dict:
        """Calculate coverage statistics for nodes.
        
        Args:
            nodes: List of LeafNode objects
            text_length: Length of original text
            
        Returns:
            Dictionary with coverage statistics
        """
        if not nodes:
            return {
                "coverage_chars": 0,
                "coverage_percent": 0.0,
                "gaps": [],
                "overlaps": []
            }
        
        # Sort by start offset
        sorted_nodes = sorted(nodes, key=lambda n: n.start)
        
        # Calculate coverage
        covered_chars = 0
        gaps = []
        overlaps = []
        
        prev_end = 0
        for node in sorted_nodes:
            # Check for gap
            if node.start > prev_end:
                gaps.append((prev_end, node.start))
            
            # Check for overlap
            if node.start < prev_end:
                overlaps.append((node.start, min(prev_end, node.end)))
            
            # Add to coverage (avoiding double counting overlaps)
            covered_chars += max(0, node.end - max(node.start, prev_end))
            prev_end = max(prev_end, node.end)
        
        # Check for gap at the end
        if prev_end < text_length:
            gaps.append((prev_end, text_length))
        
        coverage_percent = (covered_chars / text_length * 100) if text_length > 0 else 0
        
        return {
            "coverage_chars": covered_chars,
            "coverage_percent": coverage_percent,
            "gaps": gaps,
            "overlaps": overlaps
        }
