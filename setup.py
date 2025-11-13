from setuptools import setup, find_packages

setup(
    name="ragcraft",
    version="0.2.0",
    description="RAG pipeline with built-in evaluation",
    author="chu2bard",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0",
        "numpy>=1.24",
        "faiss-cpu>=1.7",
        "tiktoken>=0.5",
        "pydantic>=2.0",
    ],
)
