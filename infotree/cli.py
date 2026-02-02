"""Command-line interface for InfoTree."""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .config import InfoTreeConfig
from .pipeline import InfoTreePipeline


def load_env_file():
    """Load environment variables from .env file if it exists."""
    if load_dotenv is None:
        return
    
    # Try to load from current directory
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        return
    
    # Try to load from home directory
    home_env = Path.home() / ".env"
    if home_env.exists():
        load_dotenv(home_env)


def get_config_from_args(args) -> InfoTreeConfig:
    """Create InfoTreeConfig from CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        InfoTreeConfig instance
    """
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set --api-key or OPENAI_API_KEY environment variable")
    
    # Get embedding API key (optional, defaults to main API key)
    embedding_api_key = args.embedding_api_key or os.getenv("EMBEDDING_MODEL_API_KEY")
    
    # Get model names from args or environment variables
    model = args.model if args.model else os.getenv("MODEL_NAME", "gpt-4o-mini")
    embedding_model = args.embedding_model if args.embedding_model else os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    
    # Helper function to get int from env with default
    def get_int_env(env_var: str, arg_value: int) -> int:
        if arg_value != parser_defaults.get(env_var.lower().replace('_', '-')):
            return arg_value
        env_val = os.getenv(env_var)
        return int(env_val) if env_val is not None else arg_value
    
    # Helper function to get float from env with default
    def get_float_env(env_var: str, arg_value: float) -> float:
        if arg_value != parser_defaults.get(env_var.lower().replace('_', '-')):
            return arg_value
        env_val = os.getenv(env_var)
        return float(env_val) if env_val is not None else arg_value
    
    # Store parser defaults for comparison
    parser_defaults = {
        'max-tokens': 4096,
        'timeout': 60,
        'window-chars': 6000,
        'overlap-chars': 800,
        'min-node-chars': 300,
        'max-node-chars': 1200,
        'iou-threshold': 0.85,
        'max-children': 10,
        'max-depth': 4,
        'max-retries': 3,
        'retry-delay': 1.0,
        'embedding-batch-size': 100,
        'max-concurrent-requests': 10,
    }
    
    config = InfoTreeConfig(
        api_key=api_key,
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=model,
        embedding_model=embedding_model,
        embedding_base_url=args.embedding_base_url or os.getenv("EMBEDDING_BASE_URL"),
        embedding_api_key=embedding_api_key,
        max_tokens=get_int_env("MAX_TOKENS", args.max_tokens),
        timeout=get_int_env("TIMEOUT", args.timeout),
        window_chars=get_int_env("WINDOW_CHARS", args.window_chars),
        overlap_chars=get_int_env("OVERLAP_CHARS", args.overlap_chars),
        min_node_chars=get_int_env("MIN_NODE_CHARS", args.min_node_chars),
        max_node_chars=get_int_env("MAX_NODE_CHARS", args.max_node_chars),
        iou_threshold=get_float_env("IOU_THRESHOLD", args.iou_threshold),
        max_children=get_int_env("MAX_CHILDREN", args.max_children),
        max_depth=get_int_env("MAX_DEPTH", args.max_depth),
        max_retries=get_int_env("MAX_RETRIES", args.max_retries),
        retry_delay=get_float_env("RETRY_DELAY", args.retry_delay),
        embedding_batch_size=get_int_env("EMBEDDING_BATCH_SIZE", args.embedding_batch_size),
        max_concurrent_requests=get_int_env("MAX_CONCURRENT_REQUESTS", getattr(args, 'max_concurrent_requests', 10)),
    )
    return config


def cmd_process(args):
    """Process text file and build tree.
    
    Args:
        args: Parsed command-line arguments
    """
    # Load input text
    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    
    if not text or not text.strip():
        print("Error: Input text is empty", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded text: {len(text)} characters")
    
    # Create config and pipeline
    config = get_config_from_args(args)
    pipeline = InfoTreePipeline(config)
    
    # Process
    tree = pipeline.process(text, validate=args.validate)
    
    # Export if output specified
    if args.output:
        pipeline.export_tree(tree, args.output)
        print(f"\nExported to: {args.output}")
    
    # Print tree if requested
    if args.print_tree:
        max_depth = args.print_depth if args.print_depth else None
        pipeline.print_tree(tree, max_depth=max_depth)
    
    # Print stats
    if not args.quiet:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Leaf nodes: {tree.leaf_count}")
        print(f"Total nodes: {tree.total_nodes}")
        print(f"Original text: {len(tree.original_text)} characters")
        print(f"{'='*60}")


def cmd_validate(args):
    """Validate a tree JSON file.
    
    Args:
        args: Parsed command-line arguments
    """
    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree_dict = json.load(f)
    
    print(f"Loaded tree: {args.tree_json}")
    
    # Basic validation
    if "root" not in tree_dict:
        print("Error: Invalid tree JSON - missing root", file=sys.stderr)
        sys.exit(1)
    
    def count_nodes(node):
        count = 1
        if node.get("type") == "internal" and "children" in node:
            for child in node["children"]:
                count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(tree_dict["root"])
    
    print(f"Tree structure:")
    print(f"  Total nodes: {total_nodes}")
    if "metadata" in tree_dict:
        meta = tree_dict["metadata"]
        print(f"  Leaf nodes: {meta.get('leaf_count', 'Unknown')}")
        print(f"  Text length: {meta.get('text_length', 'Unknown')} characters")
    
    print("\n✓ Tree validation successful")


def cmd_info(args):
    """Show information about a tree.
    
    Args:
        args: Parsed command-line arguments
    """
    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree_dict = json.load(f)
    
    print(f"Tree: {args.tree_json}\n")
    
    if "metadata" in tree_dict:
        meta = tree_dict["metadata"]
        config = meta.get("config", {})
        
        print("Configuration:")
        print(f"  Model: {config.get('model', 'Unknown')}")
        print(f"  Embedding Model: {config.get('embedding_model', 'Unknown')}")
        print(f"  Window Size: {config.get('window_chars', 'Unknown')} chars")
        print(f"  Overlap: {config.get('overlap_chars', 'Unknown')} chars")
        
        print(f"\nMetadata:")
        print(f"  Total Nodes: {meta.get('total_nodes', 'Unknown')}")
        print(f"  Leaf Nodes: {meta.get('leaf_count', 'Unknown')}")
        print(f"  Text Length: {meta.get('text_length', 'Unknown')} characters")
    
    # Count tree depth
    def get_depth(node, current=0):
        if node.get("type") == "leaf":
            return current
        if node.get("type") == "internal" and "children" in node:
            return max((get_depth(child, current + 1) for child in node["children"]), default=current)
        return current
    
    depth = get_depth(tree_dict["root"])
    print(f"  Tree Depth: {depth}")


def cmd_export(args):
    """Export tree to different formats.
    
    Args:
        args: Parsed command-line arguments
    """
    with open(args.input, "r", encoding="utf-8") as f:
        tree_dict = json.load(f)
    
    if args.format == "json":
        output_path = args.output or args.input.replace(".json", "_formatted.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)
        print(f"Exported to: {output_path}")
    
    elif args.format == "csv":
        output_path = args.output or args.input.replace(".json", ".csv")
        import csv
        
        leaves = []
        
        def collect_leaves(node):
            if node.get("type") == "leaf":
                leaves.append(node)
            elif node.get("type") == "internal" and "children" in node:
                for child in node["children"]:
                    collect_leaves(child)
        
        collect_leaves(tree_dict["root"])
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "label", "start", "end", "length", "text_preview"])
            for leaf in leaves:
                text_preview = leaf.get("text", "")[:100].replace("\n", " ")
                writer.writerow([
                    leaf.get("node_id", ""),
                    leaf.get("label", ""),
                    leaf.get("start", ""),
                    leaf.get("end", ""),
                    leaf.get("end", 0) - leaf.get("start", 0),
                    text_preview
                ])
        
        print(f"Exported {len(leaves)} leaf nodes to: {output_path}")

    elif args.format == "html":
        output_path = args.output or args.input.replace(".json", ".html")

        def _safe_json_for_html(data):
            json_str = json.dumps(data, ensure_ascii=False)
            return json_str.replace("</", "<\\/")

        tree_json = _safe_json_for_html(tree_dict)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>InfoTree Visualization</title>
    <style>
        :root {{
            color-scheme: light dark;
            --bg: #0f172a;
            --panel: #111827;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --highlight: #fbbf24;
            --highlight-bg: rgba(251, 191, 36, 0.2);
            --border: #1f2937;
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            background: var(--bg);
            color: var(--text);
        }}
        header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            background: var(--panel);
            font-weight: 600;
        }}
        .layout {{
            display: grid;
            grid-template-columns: 350px 6px 1fr;
            gap: 0;
            height: calc(100vh - 58px);
        }}
        .tree-panel {{
            overflow: auto;
            padding: 16px 20px;
            border-right: 1px solid var(--border);
            width: 350px;
            min-width: 200px;
        }}
        .resize-handle {{
            width: 6px;
            cursor: ew-resize;
            background: var(--border);
            position: relative;
            z-index: 10;
            transition: background-color 0.2s ease;
        }}
        .resize-handle:hover {{
            background: var(--accent);
        }}
        .document-panel {{
            overflow: auto;
            padding: 16px 20px;
            border-right: 1px solid var(--border);
            background: var(--bg);
            font-size: 14px;
            line-height: 1.6;
        }}
        .document-text {{
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'Monaco', 'Courier New', monospace;
            color: var(--text);
        }}
        .context-label {{
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 8px;
            padding: 4px 8px;
            background: var(--panel);
            border-radius: 4px;
        }}
        .highlight {{
            background-color: var(--highlight-bg);
            color: var(--highlight);
            padding: 2px 0;
            border-radius: 2px;
            font-weight: 600;
        }}
        .details-panel {{
            overflow: auto;
            padding: 16px 20px;
        }}
        .node-label {{
            cursor: pointer;
            display: inline-block;
            padding: 2px 6px;
            border-radius: 6px;
            max-width: 100%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.2s ease;
        }}
        .node-label:hover {{
            background: rgba(56, 189, 248, 0.15);
        }}
        .node-label.active {{
            background: rgba(251, 191, 36, 0.3);
            color: var(--highlight);
            font-weight: 600;
        }}
        .node-label.leaf {{
            color: #a7f3d0;
        }}
        .node-label.leaf.active {{
            background: rgba(167, 243, 208, 0.3);
            color: #a7f3d0;
        }}
        .toggle {{
            cursor: pointer;
            color: var(--muted);
            margin-right: 6px;
            font-weight: 700;
            display: inline-block;
            width: 16px;
            text-align: center;
        }}
        ul {{
            list-style: none;
            padding-left: 18px;
            margin: 4px 0;
            border-left: 1px dashed var(--border);
        }}
        li {{
            margin: 2px 0;
        }}
        .meta {{
            color: var(--muted);
            font-size: 12px;
            margin-top: 6px;
            padding: 8px;
            background: var(--panel);
            border-radius: 6px;
        }}
        pre {{
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 8px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow: auto;
        }}
        @media (max-width: 1400px) {{
            .layout {{
                grid-template-columns: 300px 6px 1fr;
            }}
            .details-panel {{
                display: none;
            }}
        }}
        @media (max-width: 900px) {{
            .layout {{
                grid-template-columns: 350px 6px 1fr;
            }}
            .document-panel {{
                display: none;
            }}
            .details-panel {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <header>InfoTree Visualization - Click nodes to view and highlight document chunks</header>
    <div class="layout">
        <div class="tree-panel">
            <ul id="tree-root"></ul>
        </div>
        <div class="resize-handle" id="resize-handle"></div>
        <div class="document-panel">
            <div class="document-text" id="document-text"></div>
        </div>
        <div class="details-panel">
            <h2 id="details-title">Select a node</h2>
            <div id="details-meta" class="meta"></div>
            <pre id="details-text"></pre>
        </div>
    </div>

    <script id="tree-data" type="application/json">{tree_json}</script>
    <script>
        const data = JSON.parse(document.getElementById('tree-data').textContent);
        const root = data.root;
        const treeRoot = document.getElementById('tree-root');
        const detailsTitle = document.getElementById('details-title');
        const detailsMeta = document.getElementById('details-meta');
        const detailsText = document.getElementById('details-text');
        const documentText = document.getElementById('document-text');
        
        const CONTEXT_CHARS = 500;
        
        function getLeafRanges(node) {{
            if (node.type === 'leaf' && typeof node.start === 'number' && typeof node.end === 'number') {{
                return [{{start: node.start, end: node.end}}];
            }}
            if (node.children && node.children.length > 0) {{
                return node.children.flatMap(child => getLeafRanges(child));
            }}
            return [];
        }}
        
        function extractFullText(node) {{
            if (node.text && node.type === 'leaf') {{
                return node.text;
            }}
            if (node.children && node.children.length > 0) {{
                return node.children.map(child => extractFullText(child)).join('');
            }}
            return '';
        }}
        
        const fullDocumentText = extractFullText(root);
        
        function showContextAroundRanges(ranges) {{
            if (!ranges || ranges.length === 0) {{
                documentText.innerHTML = '<div class="context-label">No content to display</div>';
                return;
            }}
            
            // Show context around the first chunk
            const firstRange = ranges[0];
            const contextStart = Math.max(0, firstRange.start - CONTEXT_CHARS);
            const contextEnd = Math.min(fullDocumentText.length, firstRange.end + CONTEXT_CHARS);
            
            const before = fullDocumentText.substring(contextStart, firstRange.start);
            const highlighted = fullDocumentText.substring(firstRange.start, firstRange.end);
            const after = fullDocumentText.substring(firstRange.end, contextEnd);
            
            let html = '<div class="context-label">';
            html += '📍 Showing context around chunk 1 of ' + ranges.length;
            html += ' [chars ' + firstRange.start + '-' + firstRange.end + ']';
            if (ranges.length > 1) {{
                html += ' (' + (ranges.length - 1) + ' more chunks below)';
            }}
            html += '</div>';
            html += '<div class="document-text">';
            
            // Show ... if there's context before
            if (contextStart > 0) {{
                html += '<span style="color: var(--muted);">... </span>';
            }}
            
            html += escapeHtml(before);
            html += '<span class="highlight">' + escapeHtml(highlighted) + '</span>';
            html += escapeHtml(after);
            
            // Show ... if there's context after
            if (contextEnd < fullDocumentText.length) {{
                html += '<span style="color: var(--muted);"> ...</span>';
            }}
            
            html += '</div>';
            
            // Show additional chunks if there are multiple
            if (ranges.length > 1) {{
                html += '<div class="context-label" style="margin-top: 16px;">Other chunks:</div>';
                for (let i = 1; i < Math.min(ranges.length, 4); i++) {{
                    const range = ranges[i];
                    const preview = fullDocumentText.substring(range.start, Math.min(range.start + 100, range.end));
                    html += '<div style="margin: 8px 0; padding: 8px; background: var(--panel); border-radius: 4px;">';
                    html += '<div class="context-label" style="margin: 0 0 4px 0;">Chunk ' + (i + 1) + ' [' + range.start + ':' + range.end + ']</div>';
                    html += '<div class="document-text" style="font-size: 12px; max-height: 60px; overflow: hidden;">' + escapeHtml(preview);
                    if (range.end - range.start > 100) {{
                        html += '...';
                    }}
                    html += '</div></div>';
                }}
                if (ranges.length > 4) {{
                    html += '<div style="color: var(--muted); padding: 8px; font-size: 12px;">... and ' + (ranges.length - 4) + ' more chunks</div>';
                }}
            }}
            
            documentText.innerHTML = html;
        }}
        
        function escapeHtml(text) {{
            const map = {{
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            }};
            return text.replace(/[&<>"']/g, m => map[m]);
        }}
        
        function showDetails(node) {{
            detailsTitle.textContent = node.label || node.node_id || node.type || 'Node';
            const parts = [];
            if (node.node_id) parts.push('ID: ' + node.node_id);
            if (node.type) parts.push('Type: ' + node.type);
            
            const ranges = getLeafRanges(node);
            if (ranges.length > 0) {{
                parts.push('Chunks: ' + ranges.length);
                const spans = ranges.map(r => '[' + r.start + ':' + r.end + ']').join(', ');
                parts.push('Spans: ' + spans);
                const totalLength = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
                parts.push('Total: ' + totalLength + ' chars');
                showContextAroundRanges(ranges);
            }} else {{
                showContextAroundRanges([]);
            }}
            
            detailsMeta.textContent = parts.join(' • ');
            detailsText.textContent = node.text || '';
        }}

        function createNodeElement(node) {{
            const li = document.createElement('li');

            const hasChildren = node.children && node.children.length > 0;

            if (hasChildren) {{
                const toggle = document.createElement('span');
                toggle.className = 'toggle';
                toggle.textContent = '+';
                li.appendChild(toggle);

                const childrenList = document.createElement('ul');
                childrenList.style.display = 'none';
                li.appendChild(childrenList);

                let rendered = false;
                const expand = () => {{
                    if (!rendered) {{
                        node.children.forEach(child => {{
                            childrenList.appendChild(createNodeElement(child));
                        }});
                        rendered = true;
                    }}
                    const isOpen = childrenList.style.display === 'block';
                    childrenList.style.display = isOpen ? 'none' : 'block';
                    toggle.textContent = isOpen ? '+' : '−';
                    
                    // Scroll parent node to top when expanded
                    if (!isOpen) {{
                        li.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    }}
                }};

                toggle.addEventListener('click', expand);
                li._expand = expand;
            }} else {{
                const spacer = document.createElement('span');
                spacer.className = 'toggle';
                spacer.textContent = '•';
                li.appendChild(spacer);
            }}

            const label = document.createElement('span');
            label.className = 'node-label' + (node.type === 'leaf' ? ' leaf' : '');
            label.textContent = node.label || node.node_id || node.type || 'Node';
            label.title = node.label || node.node_id || '';
            label.addEventListener('click', () => {{
                // Remove active class from all labels
                document.querySelectorAll('.node-label.active').forEach(el => {{
                    el.classList.remove('active');
                }});
                
                // Add active class to clicked label
                label.classList.add('active');
                
                if (li._expand && node.type !== 'leaf') {{
                    li._expand();
                }}
                showDetails(node);
            }});
            li.appendChild(label);

            return li;
        }}

        treeRoot.appendChild(createNodeElement(root));
        showDetails(root);

        // Resizable tree-panel logic
        const treePanel = document.querySelector('.tree-panel');
        const resizeHandle = document.getElementById('resize-handle');
        const layout = document.querySelector('.layout');
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;

        resizeHandle.addEventListener('mousedown', function(e) {{
            isResizing = true;
            startX = e.clientX;
            startWidth = treePanel.offsetWidth;
            document.body.style.cursor = 'ew-resize';
            document.body.style.userSelect = 'none';
        }});

        document.addEventListener('mousemove', function(e) {{
            if (!isResizing) return;
            const delta = e.clientX - startX;
            let newWidth = startWidth + delta;
            newWidth = Math.max(200, Math.min(600, newWidth));
            treePanel.style.width = newWidth + 'px';
            layout.style.gridTemplateColumns = newWidth + 'px 6px 1fr';
        }});

        document.addEventListener('mouseup', function() {{
            if (isResizing) {{
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }}
        }});
    </script>
        </body>
        </html>
        """

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Exported to: {output_path}")

    else:
        print(f"Error: Unknown format: {args.format}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    # Load .env file if it exists
    load_env_file()
    
    parser = argparse.ArgumentParser(
        prog="infotree",
        description="Window-based LLM Information Tree for Indexing"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process text and build tree")
    process_parser.add_argument("input", help="Input text file (or - for stdin)")
    process_parser.add_argument("-o", "--output", help="Output JSON file")
    process_parser.add_argument("--print-tree", action="store_true", help="Print tree structure")
    process_parser.add_argument("--print-depth", type=int, help="Max depth for tree printing")
    process_parser.add_argument("--no-validate", dest="validate", action="store_false", 
                              default=True, help="Skip validation")
    process_parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    
    # API configuration
    process_parser.add_argument("--api-key", help="OpenAI API key (or OPENAI_API_KEY env var)")
    process_parser.add_argument("--base-url", help="LLM API base URL (or OPENAI_BASE_URL env var)")
    process_parser.add_argument("--model", default=None, help="LLM model (or MODEL_NAME env var)")
    process_parser.add_argument("--embedding-model", default=None, 
                              help="Embedding model (or EMBEDDING_MODEL_NAME env var)")
    process_parser.add_argument("--embedding-base-url", help="Embedding API base URL (or EMBEDDING_BASE_URL env var)")
    process_parser.add_argument("--embedding-api-key", help="Embedding API key (or EMBEDDING_MODEL_API_KEY env var)")
    
    # Processing parameters
    process_parser.add_argument("--window-chars", type=int, default=6000, help="Window size (or WINDOW_CHARS env var)")
    process_parser.add_argument("--overlap-chars", type=int, default=800, help="Overlap size (or OVERLAP_CHARS env var)")
    process_parser.add_argument("--min-node-chars", type=int, default=300, help="Min node size (or MIN_NODE_CHARS env var)")
    process_parser.add_argument("--max-node-chars", type=int, default=1200, help="Max node size (or MAX_NODE_CHARS env var)")
    process_parser.add_argument("--iou-threshold", type=float, default=0.85, 
                              help="IoU threshold for deduplication (or IOU_THRESHOLD env var)")
    process_parser.add_argument("--max-children", type=int, default=10, 
                              help="Max children per node (or MAX_CHILDREN env var)")
    process_parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth (or MAX_DEPTH env var)")
    
    # Other parameters
    process_parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (or MAX_TOKENS env var)")
    process_parser.add_argument("--timeout", type=int, default=60, help="Request timeout (or TIMEOUT env var)")
    process_parser.add_argument("--max-retries", type=int, default=3, help="Max retries (or MAX_RETRIES env var)")
    process_parser.add_argument("--retry-delay", type=float, default=1.0, help="Retry delay (or RETRY_DELAY env var)")
    process_parser.add_argument("--embedding-batch-size", type=int, default=100, 
                              help="Embedding batch size (or EMBEDDING_BATCH_SIZE env var)")
    process_parser.add_argument("--max-concurrent-requests", type=int, default=10,
                              help="Max concurrent async API requests (or MAX_CONCURRENT_REQUESTS env var)")
    
    process_parser.set_defaults(func=cmd_process)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a tree JSON file")
    validate_parser.add_argument("tree_json", help="Tree JSON file")
    validate_parser.set_defaults(func=cmd_validate)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show tree information")
    info_parser.add_argument("tree_json", help="Tree JSON file")
    info_parser.set_defaults(func=cmd_info)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export tree to different format")
    export_parser.add_argument("input", help="Input tree JSON file")
    export_parser.add_argument("-f", "--format", choices=["json", "csv", "html"], default="json",
                             help="Output format")
    export_parser.add_argument("-o", "--output", help="Output file")
    export_parser.set_defaults(func=cmd_export)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
