"""Hierarchical clustering for tree construction."""

from typing import List, Union
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from .models import TreeNode, LeafNode, InternalNode
from .config import InfoTreeConfig
from .utils import generate_node_id


class HierarchicalClusterer:
    """Builds hierarchical tree structure from leaf nodes."""
    
    def __init__(self, config: InfoTreeConfig):
        """Initialize clusterer.
        
        Args:
            config: InfoTreeConfig instance
        """
        self.config = config
        self.max_children = config.max_children
        self.max_depth = config.max_depth
        self.internal_node_counter = 0
    
    def build_tree(self, leaf_nodes: List[LeafNode]) -> TreeNode:
        """Build hierarchical tree from leaf nodes.
        
        Args:
            leaf_nodes: List of LeafNode objects with embeddings
            
        Returns:
            Root TreeNode (may be InternalNode or single LeafNode)
        """
        if not leaf_nodes:
            raise ValueError("Cannot build tree from empty leaf list")
        
        # Single leaf case
        if len(leaf_nodes) == 1:
            return leaf_nodes[0]
        
        # If few enough nodes, create single parent
        if len(leaf_nodes) <= self.max_children:
            root = self._create_internal_node(leaf_nodes)
            return root
        
        # Otherwise, build tree recursively
        return self._build_tree_recursive(leaf_nodes, depth=0)
    
    def _build_tree_recursive(
        self, 
        nodes: List[TreeNode], 
        depth: int
    ) -> TreeNode:
        """Recursively build tree using clustering.
        
        Args:
            nodes: List of TreeNode objects
            depth: Current depth in tree
            
        Returns:
            Root TreeNode for this subtree
        """
        # Base cases
        if len(nodes) == 1:
            return nodes[0]
        
        if len(nodes) <= self.max_children or depth >= self.max_depth:
            return self._create_internal_node(nodes)
        
        # Cluster nodes
        clusters = self._cluster_nodes(nodes)
        
        # Build internal nodes for each cluster
        cluster_roots = []
        for cluster in clusters:
            if len(cluster) == 1:
                cluster_roots.append(cluster[0])
            else:
                child_root = self._build_tree_recursive(cluster, depth + 1)
                cluster_roots.append(child_root)
        
        # Create parent node
        parent = self._create_internal_node(cluster_roots)
        return parent
    
    def _cluster_nodes(self, nodes: List[TreeNode]) -> List[List[TreeNode]]:
        """Cluster nodes based on embeddings.
        
        Args:
            nodes: List of TreeNode objects
            
        Returns:
            List of clusters (each cluster is a list of nodes)
        """
        # Extract embeddings
        embeddings = self._get_embeddings(nodes)
        
        if embeddings.shape[0] <= self.max_children:
            return [nodes]
        
        # Determine number of clusters
        n_clusters = min(
            self.max_children,
            max(2, len(nodes) // (self.max_children // 2))
        )
        
        # For large datasets, use mini-batch k-means for speed
        # Otherwise use agglomerative clustering for quality
        if len(nodes) > 5000:
            from sklearn.cluster import MiniBatchKMeans
            clustering = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(1024, len(nodes) // 10)
            )
            labels = clustering.fit_predict(embeddings)
        else:
            # Perform agglomerative clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
        
        # Group nodes by cluster
        clusters = [[] for _ in range(n_clusters)]
        for node, label in zip(nodes, labels):
            clusters[label].append(node)
        
        # Remove empty clusters and sort by earliest offset
        clusters = [c for c in clusters if c]
        for cluster in clusters:
            cluster.sort(key=lambda n: n.get_start_offset())
        
        return clusters
    
    def _get_embeddings(self, nodes: List[TreeNode]) -> np.ndarray:
        """Get embedding matrix for nodes.
        
        Args:
            nodes: List of TreeNode objects
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        for node in nodes:
            if isinstance(node, LeafNode):
                if node.embedding is not None:
                    embeddings.append(node.embedding)
                else:
                    # Fallback: zero vector
                    embeddings.append([0.0] * 1536)
            elif isinstance(node, InternalNode):
                # For internal nodes, use average of children's embeddings
                child_embeddings = self._get_embeddings(node.children)
                if len(child_embeddings) > 0:
                    avg_embedding = np.mean(child_embeddings, axis=0)
                    embeddings.append(avg_embedding.tolist())
                else:
                    embeddings.append([0.0] * 1536)
        
        return np.array(embeddings)
    
    def _create_internal_node(self, children: List[TreeNode]) -> InternalNode:
        """Create an internal node with given children.
        
        Args:
            children: List of child TreeNode objects
            
        Returns:
            InternalNode
        """
        node_id = generate_node_id("internal", self.internal_node_counter)
        self.internal_node_counter += 1
        
        # Sort children by start offset
        sorted_children = sorted(children, key=lambda n: n.get_start_offset())
        
        node = InternalNode(
            node_id=node_id,
            children=sorted_children
        )
        
        return node
    
    def get_tree_depth(self, node: TreeNode) -> int:
        """Calculate depth of tree.
        
        Args:
            node: Root TreeNode
            
        Returns:
            Maximum depth
        """
        if isinstance(node, LeafNode):
            return 0
        
        if isinstance(node, InternalNode):
            if not node.children:
                return 0
            return 1 + max(self.get_tree_depth(child) for child in node.children)
        
        return 0
    
    def count_nodes(self, node: TreeNode) -> tuple:
        """Count total and leaf nodes in tree.
        
        Args:
            node: Root TreeNode
            
        Returns:
            Tuple of (total_nodes, leaf_nodes)
        """
        if isinstance(node, LeafNode):
            return (1, 1)
        
        if isinstance(node, InternalNode):
            total = 1
            leaves = 0
            for child in node.children:
                child_total, child_leaves = self.count_nodes(child)
                total += child_total
                leaves += child_leaves
            return (total, leaves)
        
        return (0, 0)
