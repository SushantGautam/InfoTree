#!/usr/bin/env python3
"""Example usage of InfoTree pipeline."""

import os
from dotenv import load_dotenv
from infotree import InfoTreePipeline, InfoTreeConfig

# Load environment variables from .env file
load_dotenv()


def main():
    """Run example pipeline."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set the OPENAI_API_KEY environment variable.")
        print("You can get an API key from: https://platform.openai.com/account/api-keys")
        print("\nOptions:")
        print("  1. Create a .env file with: OPENAI_API_KEY=your-actual-api-key")
        print("  2. Or export OPENAI_API_KEY='your-actual-api-key'")
        print("\nThen run: python example.py")
        return 1
    
    # Example text (you can replace with your own)
    sample_text = """
    Climate change is one of the most pressing challenges facing humanity today. The Earth's 
    average temperature has risen by approximately 1.1 degrees Celsius since the pre-industrial 
    era, primarily due to human activities such as burning fossil fuels, deforestation, and 
    industrial processes. This warming trend has far-reaching consequences for ecosystems, 
    weather patterns, and human societies worldwide.
    
    The greenhouse effect is the primary mechanism driving climate change. When sunlight reaches 
    Earth's surface, some of it is reflected back to space while the rest is absorbed, warming 
    the planet. The Earth then radiates this energy as infrared radiation. Greenhouse gases in 
    the atmosphere, such as carbon dioxide, methane, and nitrous oxide, trap some of this heat, 
    preventing it from escaping into space. While this natural process is essential for life on 
    Earth, human activities have dramatically increased the concentration of these gases, 
    intensifying the greenhouse effect.
    
    The impacts of climate change are already being felt across the globe. Rising sea levels 
    threaten coastal communities and island nations. More frequent and severe weather events, 
    including hurricanes, droughts, and floods, are causing widespread damage and displacement. 
    Changes in temperature and precipitation patterns are affecting agriculture, water resources, 
    and biodiversity. Arctic ice is melting at an alarming rate, disrupting ecosystems and 
    contributing to sea level rise.
    
    Mitigation strategies focus on reducing greenhouse gas emissions. Transitioning to renewable 
    energy sources like solar, wind, and hydroelectric power is crucial. Improving energy 
    efficiency in buildings, transportation, and industry can significantly reduce emissions. 
    Protecting and restoring forests, which absorb carbon dioxide, is another important strategy. 
    Many countries have committed to achieving net-zero emissions by mid-century, but current 
    efforts fall short of what is needed to limit warming to 1.5 degrees Celsius.
    
    Adaptation measures are also necessary to cope with the changes already underway. Building 
    resilient infrastructure, developing drought-resistant crops, and implementing early warning 
    systems for extreme weather events can help communities adapt to new conditions. Coastal 
    defenses, such as sea walls and mangrove restoration, can protect against rising sea levels. 
    International cooperation and funding are essential to support vulnerable nations in their 
    adaptation efforts.
    
    Individual actions also play a role in addressing climate change. Reducing energy consumption, 
    choosing sustainable transportation options, adopting plant-based diets, and supporting 
    climate-friendly policies can collectively make a difference. Public awareness and education 
    are crucial for driving the behavioral changes needed to build a sustainable future. While 
    the challenge is immense, coordinated global action can still avert the worst impacts of 
    climate change and create a more sustainable world for future generations.
    """
    
    # Configuration
    config = InfoTreeConfig(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),  # Optional: separate embedding API
        embedding_api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),  # Optional: separate embedding API key
        window_chars=1000,
        overlap_chars=200,
        max_children=8,
        max_depth=3,
    )
    
    # Create pipeline
    pipeline = InfoTreePipeline(config)
    
    # Process text
    print("Processing sample text about climate change...")
    tree = pipeline.process(sample_text, validate=True)
    
    # Print tree structure
    pipeline.print_tree(tree, max_depth=3)
    
    # Export to JSON
    output_path = "infotree_output.json"
    pipeline.export_tree(tree, output_path)
    print(f"\nTree exported to: {output_path}")
    
    # Print some statistics
    print(f"\nTree Statistics:")
    print(f"  Total nodes: {tree.total_nodes}")
    print(f"  Leaf nodes: {tree.leaf_count}")
    print(f"  Original text length: {len(tree.original_text)} characters")
    
    # Example: Access leaf nodes
    leaves = tree.get_all_leaves()
    print(f"\nFirst 3 leaf nodes:")
    for i, leaf in enumerate(leaves[:3]):
        print(f"\n  Leaf {i+1}:")
        print(f"    Label: {leaf.label}")
        print(f"    Span: [{leaf.start}:{leaf.end}]")
        print(f"    Text preview: {leaf.text[:100]}...")


if __name__ == "__main__":
    main()
