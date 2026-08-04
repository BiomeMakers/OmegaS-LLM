from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="omega-s",
    version="0.1.0",
    author="Alberto Acedo",
    description=(
        "Omega-S: A Functional Resilience Index for Catastrophic Forgetting "
        "via Hutchinson Trace Estimation on Weight Adjacency Matrices"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiomeMakers/OmegaS-LLM",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "deep learning", "regularization", "topology", "LLM",
        "LoRA", "FSDP", "Hutchinson", "graph neural networks",
        "structured pruning", "weight monopolies"
    ],
)
