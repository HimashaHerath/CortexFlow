from setuptools import setup, find_packages

setup(
    name="adaptive_context",
    version="0.1.0",
    description="Memory optimization system for local LLMs running via Ollama",
    author="AdaptiveContext Team",
    author_email="adaptivecontext@example.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.22.0",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.0.2",
        "sqlitedict>=2.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 