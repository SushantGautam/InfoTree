"""Validation module for tree structure."""

from typing import List, Dict, Any, Set
from .models import InfoTree, TreeNode, LeafNode, InternalNode


class TreeValidator:
    """Validates information tree structure and coverage."""
    
    def __init__(self):
        """Initialize validator."""
        pass
    
    def validate_tree(self, tree: InfoTree) -> Dict[str, Any]:
        """Validate complete tree structure.
        
        Args:
            tree: InfoTree to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Validate root exists
        if tree.root is None:
            results["valid"] = False
            results["errors"].append("Tree has no root node")
            return results
        
        # Collect all leaves
        leaves = tree.get_all_leaves()
        
        if not leaves:
            results["valid"] = False
            results["errors"].append("Tree has no leaf nodes")
            return results
        
        # Validate leaf nodes
        leaf_validation = self._validate_leaves(leaves, tree.original_text)
        results["errors"].extend(leaf_validation["errors"])
        results["warnings"].extend(leaf_validation["warnings"])
        if leaf_validation["errors"]:
            results["valid"] = False
        
        # Validate internal nodes
        internal_validation = self._validate_internal_nodes(tree.root, set())
        results["errors"].extend(internal_validation["errors"])
        results["warnings"].extend(internal_validation["warnings"])
        if internal_validation["errors"]:
            results["valid"] = False
        
        # Check coverage
        coverage = self._check_coverage(leaves, len(tree.original_text))
        results["stats"]["coverage"] = coverage
        
        if coverage["coverage_percent"] < 95.0:
            results["warnings"].append(
                f"Coverage is only {coverage['coverage_percent']:.1f}%"
            )
        
        # Check for large gaps
        if coverage["gaps"]:
            large_gaps = [g for g in coverage["gaps"] if g[1] - g[0] > 100]
            if large_gaps:
                results["warnings"].append(
                    f"Found {len(large_gaps)} large gaps in coverage"
                )
        
        return results
    
    def _validate_leaves(
        self, 
        leaves: List[LeafNode], 
        original_text: str
    ) -> Dict[str, List[str]]:
        """Validate all leaf nodes.
        
        Args:
            leaves: List of LeafNode objects
            original_text: Original text
            
        Returns:
            Dictionary with errors and warnings
        """
        result = {"errors": [], "warnings": []}
        text_length = len(original_text)
        
        for leaf in leaves:
            # Check offsets are valid
            if leaf.start < 0:
                result["errors"].append(
                    f"Leaf {leaf.node_id} has negative start offset: {leaf.start}"
                )
            
            if leaf.end > text_length:
                result["errors"].append(
                    f"Leaf {leaf.node_id} end offset {leaf.end} exceeds text length {text_length}"
                )
            
            if leaf.start >= leaf.end:
                result["errors"].append(
                    f"Leaf {leaf.node_id} has invalid span: [{leaf.start}, {leaf.end})"
                )
                continue
            
            # Check text matches
            expected_text = original_text[leaf.start:leaf.end]
            if leaf.text != expected_text:
                result["errors"].append(
                    f"Leaf {leaf.node_id} text does not match original at [{leaf.start}:{leaf.end}]"
                )
            
            # Check node size
            node_size = leaf.end - leaf.start
            if node_size < 50:
                result["warnings"].append(
                    f"Leaf {leaf.node_id} is very small ({node_size} chars)"
                )
        
        return result
    
    def _validate_internal_nodes(
        self, 
        node: TreeNode, 
        visited: Set[str]
    ) -> Dict[str, List[str]]:
        """Recursively validate internal nodes.
        
        Args:
            node: TreeNode to validate
            visited: Set of visited node IDs
            
        Returns:
            Dictionary with errors and warnings
        """
        result = {"errors": [], "warnings": []}
        
        # Check for cycles
        if node.node_id in visited:
            result["errors"].append(f"Cycle detected at node {node.node_id}")
            return result
        
        visited.add(node.node_id)
        
        if isinstance(node, InternalNode):
            # Check has children
            if not node.children:
                result["errors"].append(
                    f"Internal node {node.node_id} has no children"
                )
                return result
            
            # Check children count
            if len(node.children) > 20:
                result["warnings"].append(
                    f"Internal node {node.node_id} has many children ({len(node.children)})"
                )
            
            # Recursively validate children
            for child in node.children:
                child_result = self._validate_internal_nodes(child, visited.copy())
                result["errors"].extend(child_result["errors"])
                result["warnings"].extend(child_result["warnings"])
        
        return result
    
    def _check_coverage(
        self, 
        leaves: List[LeafNode], 
        text_length: int
    ) -> Dict[str, Any]:
        """Check text coverage by leaf nodes.
        
        Args:
            leaves: List of LeafNode objects
            text_length: Length of original text
            
        Returns:
            Dictionary with coverage statistics
        """
        if not leaves:
            return {
                "coverage_chars": 0,
                "coverage_percent": 0.0,
                "gaps": [(0, text_length)] if text_length > 0 else [],
                "overlaps": []
            }
        
        # Sort by start offset
        sorted_leaves = sorted(leaves, key=lambda n: n.start)
        
        # Track covered characters
        covered = set()
        gaps = []
        overlaps = []
        
        prev_end = 0
        for leaf in sorted_leaves:
            # Check for gap
            if leaf.start > prev_end:
                gaps.append((prev_end, leaf.start))
            
            # Check for overlap
            if leaf.start < prev_end:
                overlap_start = leaf.start
                overlap_end = min(prev_end, leaf.end)
                overlaps.append((overlap_start, overlap_end))
            
            # Add to covered set
            for i in range(leaf.start, leaf.end):
                covered.add(i)
            
            prev_end = max(prev_end, leaf.end)
        
        # Check for gap at end
        if prev_end < text_length:
            gaps.append((prev_end, text_length))
        
        coverage_chars = len(covered)
        coverage_percent = (coverage_chars / text_length * 100) if text_length > 0 else 0
        
        return {
            "coverage_chars": coverage_chars,
            "coverage_percent": coverage_percent,
            "gaps": gaps,
            "overlaps": overlaps
        }
    
    def print_validation_report(self, results: Dict[str, Any]):
        """Print validation report.
        
        Args:
            results: Validation results dictionary
        """
        print("\n" + "="*60)
        print("TREE VALIDATION REPORT")
        print("="*60)
        
        if results["valid"]:
            print("✓ Tree is VALID")
        else:
            print("✗ Tree is INVALID")
        
        if results["errors"]:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"  ✗ {error}")
        
        if results["warnings"]:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  ⚠ {warning}")
        
        if "coverage" in results["stats"]:
            cov = results["stats"]["coverage"]
            print(f"\nCoverage Statistics:")
            print(f"  Coverage: {cov['coverage_percent']:.2f}%")
            print(f"  Covered chars: {cov['coverage_chars']}")
            print(f"  Gaps: {len(cov['gaps'])}")
            print(f"  Overlaps: {len(cov['overlaps'])}")
        
        print("="*60 + "\n")
