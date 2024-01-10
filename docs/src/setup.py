import os
from setuptools import setup, find_packages
import setuptools


setup(
    name="tsllm",  # Replace with your own username
    version="0.1.0",
    description="TS_LLM: AlphaZero-like tree-search learning framework for LLMs",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="tmp",
    author_email="tmp",
    packages=setuptools.find_packages(),
    classifiers=[],
    keywords="large language model, tree search, reinforcement learning, value function",
    python_requires='>=3.10',
)