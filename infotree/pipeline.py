"""Main pipeline orchestrator for InfoTree."""

import json
from typing import Optional
from .config import InfoTreeConfig
from .models import InfoTree, LeafNode
from .windowing import Windower
from .extraction import NodeExtractor
from .deduplication import Deduplicator
from .embeddings import EmbeddingGenerator
from .clustering import HierarchicalClusterer
from .labeling import NodeLabeler
from .validation import TreeValidator


class InfoTreePipeline:
    """Main pipeline for building information trees."""
    
    def __init__(self, config: InfoTreeConfig):
        """Initialize pipeline with configuration.
        
        Args:
            config: InfoTreeConfig instance
        """
        self.config = config
        
        # Initialize components
        self.windower = Windower(config)
        self.extractor = NodeExtractor(config)
        self.deduplicator = Deduplicator(config)
        self.embedder = EmbeddingGenerator(config)
        self.clusterer = HierarchicalClusterer(config)
        self.labeler = NodeLabeler(config)
        self.validator = TreeValidator()
    
    def process(self, text: str, validate: bool = True) -> InfoTree:
        """Process text and build information tree.
        
        Args:
            text: Raw input text
            validate: Whether to validate the resulting tree
            
        Returns:
            InfoTree object
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        print(f"\n{'='*60}")
        print("INFO TREE PIPELINE")
        print(f"{'='*60}")
        print(f"Input text length: {len(text)} characters")
        
        # Step 1: Create windows
        print("\n[1/7] Creating windows...")
        windows = self.windower.create_windows(text)
        print(f"  ✓ Created {len(windows)} windows")
        
        # Step 2: Extract nodes from windows
        print("\n[2/7] Extracting nodes from windows...")
        all_nodes = self.extractor.extract_nodes_from_windows(windows, text)
        print(f"  ✓ Extracted {len(all_nodes)} raw nodes")
        
        # Step 3: Deduplicate nodes
        print("\n[3/7] Deduplicating nodes...")
        unique_nodes = self.deduplicator.deduplicate(all_nodes)
        print(f"  ✓ Deduplicated to {len(unique_nodes)} unique nodes")
        
        # Check coverage
        coverage = self.deduplicator.get_coverage_stats(unique_nodes, len(text))
        print(f"  ✓ Coverage: {coverage['coverage_percent']:.2f}%")
        
        # Step 4: Generate embeddings
        print("\n[4/7] Generating embeddings...")
        unique_nodes = self.embedder.generate_embeddings(unique_nodes)
        print(f"  ✓ Generated embeddings for {len(unique_nodes)} nodes")
        
        # Step 5: Build hierarchical tree
        print("\n[5/7] Building hierarchical tree...")
        root = self.clusterer.build_tree(unique_nodes)
        depth = self.clusterer.get_tree_depth(root)
        total_nodes, leaf_count = self.clusterer.count_nodes(root)
        print(f"  ✓ Built tree with depth {depth}")
        print(f"  ✓ Total nodes: {total_nodes}, Leaf nodes: {leaf_count}")
        
        # Step 6: Label nodes
        print("\n[6/7] Labeling nodes...")
        self.labeler.label_tree(root)
        print(f"  ✓ Labeled all nodes")
        
        # Step 7: Create InfoTree
        tree = InfoTree(
            root=root,
            original_text=text,
            config=self.config.model_dump(),
            leaf_count=leaf_count,
            total_nodes=total_nodes
        )
        
        # Validate if requested
        if validate:
            print("\n[7/7] Validating tree...")
            validation_results = self.validator.validate_tree(tree)
            
            if validation_results["valid"]:
                print("  ✓ Tree validation passed")
            else:
                print("  ✗ Tree validation failed")
                print(f"  Errors: {len(validation_results['errors'])}")
            
            if validation_results["warnings"]:
                print(f"  Warnings: {len(validation_results['warnings'])}")
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}\n")
        
        return tree
    
    def process_and_export(
        self, 
        text: str, 
        output_path: Optional[str] = None,
        validate: bool = True
    ) -> InfoTree:
        """Process text and export tree to JSON.
        
        Args:
            text: Raw input text
            output_path: Path to save JSON output (optional)
            validate: Whether to validate the tree
            
        Returns:
            InfoTree object
        """
        tree = self.process(text, validate=validate)
        
        if output_path:
            self.export_tree(tree, output_path)
            print(f"Exported tree to: {output_path}")
        
        return tree
    
    def export_tree(self, tree: InfoTree, output_path: str):
        """Export tree to JSON file.
        
        Args:
            tree: InfoTree to export
            output_path: Path to save JSON file
        """
        tree_dict = tree.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)
    
    def print_tree(self, tree: InfoTree, max_depth: Optional[int] = None):
        """Print tree structure in readable format.
        
        Args:
            tree: InfoTree to print
            max_depth: Maximum depth to print (None for all)
        """
        print(f"\n{'='*60}")
        print("TREE STRUCTURE")
        print(f"{'='*60}")
        print(f"Total nodes: {tree.total_nodes}")
        print(f"Leaf nodes: {tree.leaf_count}")
        print()
        
        self._print_node(tree.root, depth=0, max_depth=max_depth)
        print(f"{'='*60}\n")
    
    def _print_node(self, node, depth: int, max_depth: Optional[int], prefix: str = ""):
        """Recursively print node structure.
        
        Args:
            node: TreeNode to print
            depth: Current depth
            max_depth: Maximum depth to print
            prefix: Prefix for indentation
        """
        from .models import LeafNode, InternalNode
        
        if max_depth is not None and depth > max_depth:
            return
        
        indent = "  " * depth
        
        if isinstance(node, LeafNode):
            label = node.label or "Unlabeled"
            print(f"{indent}└─ [LEAF] {label[:60]}")
            print(f"{indent}   Span: [{node.start}:{node.end}] ({node.end - node.start} chars)")
        
        elif isinstance(node, InternalNode):
            label = node.label or "Unlabeled Section"
            print(f"{indent}└─ [{label}] ({len(node.children)} children)")
            
            for i, child in enumerate(node.children):
                self._print_node(child, depth + 1, max_depth)
