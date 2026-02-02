# InfoTree

## Building Intelligent Hierarchical Indexes from Unstructured Text

A Python library that transforms raw, unstructured text into navigable semantic hierarchies using advanced LLM segmentation and AI-powered embedding clustering.

## Key Features

### Core Capabilities
- **Intelligent Window-Based Processing** - Analyzes documents without reliance on artificial paragraph boundaries, maintaining context across arbitrary text segments
- **Complete Text Coverage** - Guarantees 100% coverage of input text through intelligent overlapping windows, with zero information loss
- **Grounded Semantic Anchors** - Every leaf node is precisely anchored to exact text spans with byte-accurate character offsets for full traceability

### Features
- **Controlled LLM Usage** - Leverages large language models exclusively for segmentation decisions and semantic labeling, avoiding hallucinated content generation
- **Semantic-Driven Hierarchy** - Tree structure is built entirely on embedding-based semantic similarity, not arbitrary heuristics
- **Deterministic Processing** - Produces consistent, reproducible results across multiple runs with identical configuration

- **Reliability** - Built-in retry mechanisms, comprehensive validation, and graceful error handling throughout the pipeline
- **Flexible Integration** - Powerful CLI interface and programmatic API for seamless integration into any workflow
- **Performance Optimized** - Handles large documents and noisy inputs (PDFs, OCR, transcripts) with exceptional efficiency

## Installation

```bash
pip install git+https://github.com/SushantGautam/InfoTree.git
```
or 

Super Quick run with UV:
```bash
uv run --with https://github.com/SushantGautam/InfoTree.git infotree process input.txt -o output.json --print-tree
```


## Quickstart with Command-Line Interface
InfoTree includes a powerful CLI for easy command-line usage:

```bash
# Process a text file
infotree process input.txt -o output.json --print-tree

# With custom configuration
infotree process input.txt \
  --model gpt-4o-mini \
  --embedding-model BAAI/bge-m3 \
  --embedding-base-url https://embed.example.com/v1 \
  --window-chars 3000 \
  --max-depth 3 \
  -o output.json

# From stdin
cat input.txt | infotree process - -o output.json

# Validate a tree
infotree validate output.json

# Show tree information
infotree info output.json

# Export to CSV or html
infotree export output.json -f csv -o output.csv
infotree export output.json -f html -o output.html

```

See [CLI_EXAMPLES.sh](CLI_EXAMPLES.sh) for more examples.


## Python API

```python
from infotree import InfoTreePipeline, InfoTreeConfig

# Configure pipeline
config = InfoTreeConfig(
    api_key="your-openai-api-key",
    model="gpt-4o-mini",
    window_chars=6000,
    overlap_chars=800,
    max_children=10,
    max_depth=4
)

# Create pipeline
pipeline = InfoTreePipeline(config)

# Process text
text = "Your long unstructured text here..."
tree = pipeline.process(text)

# Export to JSON
pipeline.export_tree(tree, "output.json")

# Print tree structure
pipeline.print_tree(tree, max_depth=3)
```

## Architecture

The pipeline consists of 7 stages:

1. **Windowing**: Split text into overlapping windows
2. **Node Extraction**: LLM extracts atomic nodes from each window
3. **Deduplication**: Remove overlapping nodes using IoU threshold
4. **Embedding Generation**: Generate embeddings for all leaf nodes
5. **Hierarchical Clustering**: Build tree structure using agglomerative clustering
6. **Labeling**: Generate index-style labels for all nodes
7. **Validation**: Verify tree structure and coverage

## Configuration

Key configuration parameters:

- `api_key`: OpenAI API key
- `base_url`: API base URL (default: OpenAI)
- `model`: LLM model for segmentation/labeling (e.g., "gpt-4o-mini")
- `embedding_model`: Embedding model (default: "text-embedding-3-small")
- `window_chars`: Window size in characters (default: 6000)
- `overlap_chars`: Overlap size (default: 800)
- `min_node_chars`: Minimum node size (default: 300)
- `max_node_chars`: Maximum node size (default: 1200)
- `iou_threshold`: IoU threshold for deduplication (default: 0.85)
- `max_children`: Maximum children per internal node (default: 10)
- `max_depth`: Maximum tree depth (default: 4)

## Output Format

The tree structure includes:

**Leaf Nodes:**
- `node_id`: Unique identifier
- `start`: Character offset in original text
- `end`: Character offset in original text
- `text`: Actual text span
- `label`: Short descriptive label

**Internal Nodes:**
- `node_id`: Unique identifier
- `label`: Index-style label describing children
- `children`: List of child nodes

## Use Cases

- **Document Indexing**: Create navigable indexes for large documents
- **Semantic Search**: Find relevant sections based on embedding similarity
- **Document Summarization**: Navigate document structure hierarchically
- **Content Analysis**: Understand document organization and topics
- **Text Highlighting**: Map search results to exact character offsets

## Requirements

- Python 3.8+
- OpenAI API key (or compatible API)
- Dependencies listed in requirements.txt

## Testing

```bash
pytest tests/
```

## Example

See `example.py` for a complete working example with sample text.

```bash
python example.py
```

## Design Principles

1. **No Hallucination**: Structure is embedding-driven, not LLM-invented
2. **Grounded**: Every node maps to exact text spans
3. **Resilient**: Localized failures don't invalidate entire pipeline
4. **Auditable**: Full traceability from nodes to original text
5. **Scalable**: Handles long documents with noisy inputs (PDFs, OCR, transcripts)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.
