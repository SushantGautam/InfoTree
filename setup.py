from setuptools import setup, find_packages

setup(
    name="infotree",
    version="0.1.0",
    description="Window-based LLM Information Tree for Indexing",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tenacity>=8.2.0",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "infotree=infotree.cli:main",
        ],
    },
    python_requires=">=3.8",
)
