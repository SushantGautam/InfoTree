"""Unit tests for InfoTree components."""

import pytest
from infotree.config import InfoTreeConfig
from infotree.models import LeafNode, InternalNode, Window
from infotree.windowing import Windower
from infotree.deduplication import Deduplicator
from infotree.utils import calculate_iou, generate_node_id, truncate_text


def test_config_validation():
    """Test configuration validation."""
    # Valid config
    config = InfoTreeConfig(
        api_key="test-key",
        window_chars=1000,
        overlap_chars=200
    )
    assert config.window_chars == 1000
    assert config.overlap_chars == 200
    
    # Invalid config (overlap >= window)
    with pytest.raises(ValueError):
        InfoTreeConfig(
            api_key="test-key",
            window_chars=1000,
            overlap_chars=1000
        )


def test_windowing():
    """Test window creation."""
    config = InfoTreeConfig(
        api_key="test-key",
        window_chars=100,
        overlap_chars=20
    )
    windower = Windower(config)
    
    text = "a" * 250
    windows = windower.create_windows(text)
    
    assert len(windows) > 1
    assert windows[0].start == 0
    assert windows[0].end == 100
    assert windows[-1].end == len(text)


def test_leaf_node_equality():
    """Test leaf node equality."""
    node1 = LeafNode(node_id="leaf_1", start=0, end=100, text="test")
    node2 = LeafNode(node_id="leaf_2", start=0, end=100, text="test")
    node3 = LeafNode(node_id="leaf_3", start=10, end=100, text="test")
    
    assert node1 == node2  # Same span
    assert node1 != node3  # Different span


def test_calculate_iou():
    """Test IoU calculation."""
    # Perfect overlap
    assert calculate_iou(0, 100, 0, 100) == 1.0
    
    # No overlap
    assert calculate_iou(0, 100, 200, 300) == 0.0
    
    # Partial overlap
    iou = calculate_iou(0, 100, 50, 150)
    assert 0 < iou < 1


def test_deduplication():
    """Test deduplication logic."""
    config = InfoTreeConfig(
        api_key="test-key",
        iou_threshold=0.85
    )
    deduplicator = Deduplicator(config)
    
    nodes = [
        LeafNode(node_id="leaf_1", start=0, end=100, text="a" * 100),
        LeafNode(node_id="leaf_2", start=5, end=105, text="a" * 100),  # High overlap
        LeafNode(node_id="leaf_3", start=200, end=300, text="b" * 100),
    ]
    
    unique = deduplicator.deduplicate(nodes)
    assert len(unique) == 2  # Should keep only 2 nodes


def test_internal_node_sorting():
    """Test internal node child sorting."""
    leaf1 = LeafNode(node_id="leaf_1", start=100, end=200, text="b")
    leaf2 = LeafNode(node_id="leaf_2", start=0, end=100, text="a")
    leaf3 = LeafNode(node_id="leaf_3", start=200, end=300, text="c")
    
    internal = InternalNode(node_id="internal_1", children=[leaf1, leaf2, leaf3])
    internal.sort_children()
    
    assert internal.children[0].get_start_offset() == 0
    assert internal.children[1].get_start_offset() == 100
    assert internal.children[2].get_start_offset() == 200


def test_utility_functions():
    """Test utility functions."""
    # Test truncate_text
    text = "a" * 100
    truncated = truncate_text(text, 50)
    assert len(truncated) <= 50
    
    # Test generate_node_id
    node_id = generate_node_id("test", 42)
    assert node_id == "test_42"


def test_tree_validation():
    """Test tree validation logic."""
    from infotree.models import InfoTree
    from infotree.validation import TreeValidator
    
    text = "Hello world! " * 100
    
    leaf1 = LeafNode(node_id="leaf_1", start=0, end=50, text=text[0:50])
    leaf2 = LeafNode(node_id="leaf_2", start=50, end=100, text=text[50:100])
    
    root = InternalNode(node_id="root", children=[leaf1, leaf2])
    
    tree = InfoTree(
        root=root,
        original_text=text,
        config={},
        leaf_count=2,
        total_nodes=3
    )
    
    validator = TreeValidator()
    results = validator.validate_tree(tree)
    
    assert results["valid"]
    assert len(results["errors"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
