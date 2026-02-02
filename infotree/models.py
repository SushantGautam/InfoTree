"""Data models for InfoTree."""

from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class TreeNode(ABC):
    """Abstract base class for tree nodes."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        pass
    
    @abstractmethod
    def get_start_offset(self) -> int:
        """Get the earliest character offset covered by this node."""
        pass


@dataclass
class LeafNode(TreeNode):
    """Leaf node representing an atomic text span."""
    
    node_id: str
    start: int  # Absolute character offset in original text
    end: int    # Absolute character offset in original text
    text: str   # The actual text span
    label: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert leaf node to dictionary representation."""
        return {
            "type": "leaf",
            "node_id": self.node_id,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
    
    def get_start_offset(self) -> int:
        """Get the start offset of this leaf node."""
        return self.start
    
    def __hash__(self):
        return hash((self.start, self.end))
    
    def __eq__(self, other):
        if not isinstance(other, LeafNode):
            return False
        return self.start == other.start and self.end == other.end


@dataclass
class InternalNode(TreeNode):
    """Internal node representing a cluster of child nodes."""
    
    node_id: str
    label: Optional[str] = None
    children: List[TreeNode] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert internal node to dictionary representation."""
        return {
            "type": "internal",
            "node_id": self.node_id,
            "label": self.label,
            "children": [child.to_dict() for child in self.children],
        }
    
    def get_start_offset(self) -> int:
        """Get the earliest start offset among all children."""
        if not self.children:
            return 0
        return min(child.get_start_offset() for child in self.children)
    
    def add_child(self, child: TreeNode):
        """Add a child node."""
        self.children.append(child)
    
    def sort_children(self):
        """Sort children by their start offset."""
        self.children.sort(key=lambda x: x.get_start_offset())


@dataclass
class Window:
    """Represents a text window with overlap."""
    
    wid: int           # Window ID
    start: int         # Absolute start offset
    end: int           # Absolute end offset
    text: str          # Window text
    
    def __repr__(self):
        return f"Window(wid={self.wid}, start={self.start}, end={self.end}, len={len(self.text)})"


@dataclass
class InfoTree:
    """Complete information tree with metadata."""
    
    root: TreeNode
    original_text: str
    config: Dict[str, Any]
    leaf_count: int = 0
    total_nodes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        return {
            "root": self.root.to_dict(),
            "metadata": {
                "leaf_count": self.leaf_count,
                "total_nodes": self.total_nodes,
                "text_length": len(self.original_text),
                "config": self.config,
            }
        }
    
    def get_all_leaves(self) -> List[LeafNode]:
        """Get all leaf nodes in the tree."""
        leaves = []
        
        def traverse(node: TreeNode):
            if isinstance(node, LeafNode):
                leaves.append(node)
            elif isinstance(node, InternalNode):
                for child in node.children:
                    traverse(child)
        
        traverse(self.root)
        return leaves
    
    def validate(self) -> bool:
        """Validate tree structure and coverage."""
        leaves = self.get_all_leaves()
        
        # Check that all leaves have valid offsets
        for leaf in leaves:
            if leaf.start < 0 or leaf.end > len(self.original_text):
                return False
            if leaf.start >= leaf.end:
                return False
            if self.original_text[leaf.start:leaf.end] != leaf.text:
                return False
        
        # Check for no orphan spans (simplified - just check we have leaves)
        if not leaves:
            return False
        
        return True


@dataclass
class ExtractionResult:
    """Result from node extraction process."""
    
    nodes: List[LeafNode]
    window_id: int
    success: bool = True
    error: Optional[str] = None
