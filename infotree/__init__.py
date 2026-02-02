"""InfoTree: Window-based LLM Information Tree for Indexing"""

__version__ = "0.1.0"

from .pipeline import InfoTreePipeline
from .models import InfoTree, TreeNode, LeafNode, InternalNode
from .config import InfoTreeConfig
from .cli import main as cli_main

__all__ = [
    "InfoTreePipeline",
    "InfoTree",
    "TreeNode",
    "LeafNode",
    "InternalNode",
    "InfoTreeConfig",
    "cli_main",
]
