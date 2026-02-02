#!/bin/bash
# Example CLI usage commands for InfoTree

# Display help
infotree --help

# Get help for a specific command
infotree process --help

# Process a text file with default settings
infotree process input.txt -o output.json --print-tree

# Process with custom API configuration
infotree process input.txt \
  --api-key "your-api-key" \
  --model "gpt-4o-mini" \
  --embedding-model "BAAI/bge-m3" \
  --embedding-base-url "https://embed.example.com/v1" \
  --output output.json

# Process with custom window and tree parameters
infotree process input.txt \
  --window-chars 3000 \
  --overlap-chars 500 \
  --max-children 8 \
  --max-depth 3 \
  --output output.json

# Process from stdin
cat input.txt | infotree process - -o output.json

# Process and print tree structure (limited depth)
infotree process input.txt --print-tree --print-depth 2

# Process without validation (faster)
infotree process input.txt --no-validate -o output.json

# Process in quiet mode (less output)
infotree process input.txt -q -o output.json

# Validate a tree JSON file
infotree validate output.json

# Show information about a tree
infotree info output.json

# Export tree to CSV format
infotree export output.json -f csv -o output.csv

# Export tree to formatted JSON
infotree export output.json -f json -o output_formatted.json

# Using environment variables for API keys
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://llm.example.com/v1"
export EMBEDDING_BASE_URL="https://embed.example.com/v1"
export EMBEDDING_MODEL_API_KEY="embed-key"

infotree process input.txt \
  --model gpt-4o-mini \
  --embedding-model BAAI/bge-m3 \
  -o output.json
