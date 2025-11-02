# setup.py
from setuptools import setup, find_packages

setup(
    name="importancescore",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dynamic LoRA fine-tuning with importance-based adaptive rank allocation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/importancescore",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
