"""
Benchmark datasets for evaluating RAG systems.

This module provides functions to load datasets for benchmarking.
"""
from typing import Dict, List, Any
import os
import json

# Dataset registry
DATASETS = {
    "hotpotqa": "Multi-hop reasoning dataset from HotPotQA",
    "complex": "Complex query dataset with relationship questions",
    "long_context": "Long context conversation dataset for memory evaluation",
    "memory_test": "Memory retention test dataset"
}

def available_datasets() -> List[str]:
    """
    Get list of available datasets.
    
    Returns:
        List of dataset names
    """
    return list(DATASETS.keys())


def load_hotpotqa_sample() -> Dict[str, Any]:
    """
    Load a sample of the HotPotQA dataset for multi-hop question testing.
    
    Returns:
        Dictionary with dataset information
    """
    # Sample HotPotQA multi-hop questions and contexts
    sample_data = {
        "name": "HotPotQA Sample",
        "description": "Multi-hop reasoning questions from HotPotQA",
        "context": [
            "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
            "Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0.",
            "Guido van Rossum was employed by Google from 2005 to 2012, where he spent half of his time developing the Python language.",
            "After leaving Google, Van Rossum worked at Dropbox from 2013 to 2019.",
            "Silicon Valley is a region in Northern California that serves as a global center for high technology and innovation.",
            "Many major tech companies have their headquarters in Silicon Valley, including Google, Apple, and Facebook.",
            "Google LLC is an American multinational technology company that specializes in Internet-related services and products.",
            "Google's headquarters (the Googleplex) is located in Mountain View, California, which is part of Silicon Valley.",
            "TensorFlow is an open-source machine learning framework developed by the Google Brain team.",
            "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab.",
            "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "Facebook (now Meta Platforms, Inc.) was founded by Mark Zuckerberg and his college roommates in 2004."
        ],
        "queries": [
            {
                "query": "Who created Python?",
                "answers": ["Guido van Rossum created Python in the late 1980s and released it in 1991."],
                "hop_count": 1
            },
            {
                "query": "What companies has Guido van Rossum worked for?",
                "answers": ["Guido van Rossum worked for Google from 2005 to 2012 and Dropbox from 2013 to 2019."],
                "hop_count": 2
            },
            {
                "query": "What is the relationship between Python and Google?",
                "answers": ["Guido van Rossum, Python's creator, worked at Google from 2005 to 2012, where he spent half his time developing the Python language."],
                "hop_count": 2,
                "type": "relationship"
            },
            {
                "query": "What is the connection between Silicon Valley and Python?",
                "answers": ["Python's creator, Guido van Rossum, worked at Google, which is headquartered in Mountain View in Silicon Valley."],
                "hop_count": 3,
                "type": "relationship"
            },
            {
                "query": "How are Facebook and machine learning related?",
                "answers": ["Facebook's AI Research lab developed PyTorch, which is an open-source machine learning library."],
                "hop_count": 2,
                "type": "relationship"
            }
        ]
    }
    
    return sample_data


def load_complex_query_sample() -> Dict[str, Any]:
    """
    Load a sample dataset for complex query testing.
    
    Returns:
        Dictionary with dataset information
    """
    # Complex query dataset
    sample_data = {
        "name": "Complex Query Sample",
        "description": "Complex relationship and multi-hop queries",
        "context": [
            "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Transformers are a type of neural network architecture that was introduced in the paper 'Attention Is All You Need' by researchers at Google in 2017.",
            "The original Transformer architecture consists of an encoder and a decoder, each composed of multiple layers of self-attention and feed-forward neural networks.",
            "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique developed by Google.",
            "GPT (Generative Pre-trained Transformer) is an autoregressive language model developed by OpenAI.",
            "GPT is based on the transformer architecture but uses only the decoder portion, unlike BERT which uses the encoder portion.",
            "LlamaIndex (formerly GPT Index) is a data framework designed to help developers build LLM applications, particularly for working with custom data.",
            "LangChain is a framework for developing applications powered by language models, focusing on composability and extensibility.",
            "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs by incorporating relevant information retrieved from external sources.",
            "AdaptiveContext is a context management system that uses tiered memory and dynamic token allocation to optimize context for LLMs.",
            "Tokens are the basic units that LLMs process, typically representing parts of words or characters in the input text."
        ],
        "queries": [
            {
                "query": "What is the relationship between BERT and GPT?",
                "answers": ["Both BERT and GPT are based on the Transformer architecture, but BERT uses the encoder portion while GPT uses the decoder portion."],
                "hop_count": 2,
                "type": "relationship"
            },
            {
                "query": "How do LlamaIndex and LangChain relate to RAG systems?",
                "answers": ["Both LlamaIndex and LangChain are frameworks that can be used to build Retrieval-Augmented Generation (RAG) systems, helping integrate external knowledge with LLMs."],
                "hop_count": 2,
                "type": "relationship"
            },
            {
                "query": "What connects transformers, BERT, and GPT?",
                "answers": ["BERT and GPT are both based on the Transformer architecture; BERT uses the encoder portion while GPT uses the decoder portion of the Transformer."],
                "hop_count": 3,
                "type": "relationship"
            },
            {
                "query": "Explain the evolution from neural networks to transformers to modern LLMs.",
                "answers": ["Neural networks inspired the development of deep learning, which led to the Transformer architecture introduced in 2017. Transformers then enabled models like BERT and GPT, which are foundation models for modern LLMs."],
                "hop_count": 4
            },
            {
                "query": "What's the difference between AdaptiveContext and traditional RAG systems?",
                "answers": ["AdaptiveContext is a context management system that uses tiered memory and dynamic token allocation to optimize context for LLMs, while traditional RAG systems focus on retrieving and incorporating external information without the adaptive memory management."],
                "hop_count": 2,
                "type": "comparison"
            }
        ]
    }
    
    return sample_data


def load_long_context_sample() -> Dict[str, Any]:
    """
    Load a sample long context conversation dataset.
    
    Returns:
        Dictionary with dataset information
    """
    # Long context conversation sample
    sample_data = {
        "name": "Long Context Conversation Sample",
        "description": "Long conversations for memory retention testing",
        "conversations": [
            {
                "id": "conversation_1",
                "messages": [
                    {"role": "user", "content": "Let's discuss the history of artificial intelligence."},
                    {"role": "assistant", "content": "The history of artificial intelligence (AI) dates back to the 1950s when the field was formally founded. Key pioneers include Alan Turing, who proposed the Turing Test in 1950, and John McCarthy, who coined the term 'artificial intelligence' in 1956 at the Dartmouth Conference."},
                    {"role": "user", "content": "What were the major AI winters?"},
                    {"role": "assistant", "content": "There were two major AI winters. The first occurred in the 1970s after initial hype around AI failed to deliver practical results. Funding was cut as promises of general intelligence weren't fulfilled. The second AI winter happened in the late 1980s and early 1990s when expert systems failed to scale and deliver commercial value, leading to another round of reduced funding and interest."},
                    {"role": "user", "content": "How did deep learning revive interest in AI?"},
                    {"role": "assistant", "content": "Deep learning revived interest in AI primarily through breakthroughs in the 2010s. In 2012, Geoffrey Hinton's team won the ImageNet competition using a deep convolutional neural network called AlexNet, significantly outperforming traditional computer vision approaches. This success, combined with advances in computing power (especially GPUs), availability of large datasets, and algorithmic improvements, led to an explosion of progress in areas like image recognition, speech processing, and natural language understanding."},
                    {"role": "user", "content": "Let's switch topics. Can you tell me about quantum computing basics?"},
                    {"role": "assistant", "content": "Quantum computing uses quantum bits or qubits instead of classical bits. While classical bits represent either 0 or 1, qubits can exist in a superposition of both states simultaneously. This property, along with entanglement, allows quantum computers to perform certain calculations exponentially faster than classical computers. Key quantum algorithms include Shor's algorithm for factoring large numbers and Grover's algorithm for searching unsorted databases. However, quantum computers are still in early stages, facing challenges like qubit stability and error correction."},
                    {"role": "user", "content": "What are the main approaches to building quantum computers?"},
                    {"role": "assistant", "content": "The main approaches to building quantum computers include: 1) Superconducting qubits - Used by IBM, Google, and others, operating at near absolute zero temperatures. 2) Trapped ion qubits - Ions trapped by electromagnetic fields, known for high coherence times. 3) Photonic quantum computing - Using photons as qubits, operating at room temperature. 4) Topological quantum computing - A theoretical approach aimed at creating more stable qubits through topological properties. 5) Silicon spin qubits - Similar to traditional semiconductor technology, potentially easier to scale."}
                ],
                "queries": [
                    {
                        "query": "What did we discuss about AI winters?",
                        "answers": ["There were two major AI winters - the first in the 1970s due to unfulfilled promises and the second in the late 1980s/early 1990s when expert systems failed to scale."]
                    },
                    {
                        "query": "What quantum computing approaches did we talk about?",
                        "answers": ["The main approaches to quantum computing discussed were superconducting qubits, trapped ion qubits, photonic quantum computing, topological quantum computing, and silicon spin qubits."]
                    }
                ]
            }
        ]
    }
    
    return sample_data


def load_memory_test_sample() -> Dict[str, Any]:
    """
    Load a sample memory test dataset.
    
    Returns:
        Dictionary with dataset information
    """
    # Memory test sample
    sample_data = {
        "name": "Memory Test Sample",
        "description": "Conversations with memory flush tests",
        "tests": [
            {
                "id": "memory_test_1",
                "description": "Basic memory retention after flush",
                "setup_messages": [
                    {"role": "user", "content": "Remember that the capital of France is Paris, the capital of Japan is Tokyo, and the capital of Brazil is Brasília."},
                    {"role": "assistant", "content": "I'll remember that information. The capital of France is Paris, the capital of Japan is Tokyo, and the capital of Brazil is Brasília."}
                ],
                "pre_flush_query": {
                    "query": "What are the capitals of France, Japan, and Brazil?",
                    "expected_answer": "The capital of France is Paris, the capital of Japan is Tokyo, and the capital of Brazil is Brasília."
                },
                "post_flush_query": {
                    "query": "What are the capitals of France, Japan, and Brazil?",
                    "expected_answer": "The capital of France is Paris, the capital of Japan is Tokyo, and the capital of Brazil is Brasília."
                }
            },
            {
                "id": "memory_test_2",
                "description": "Complex information retention after flush",
                "setup_messages": [
                    {"role": "user", "content": "Remember this sequence: The red fox named Felix jumped over the lazy dog Charlie after eating the purple grapes from vineyard number 7."},
                    {"role": "assistant", "content": "I'll remember that sequence: The red fox named Felix jumped over the lazy dog Charlie after eating the purple grapes from vineyard number 7."}
                ],
                "pre_flush_query": {
                    "query": "What is the name of the fox, what did it jump over, what did it eat, and from where?",
                    "expected_answer": "The fox's name is Felix, it jumped over the lazy dog Charlie, it ate purple grapes from vineyard number 7."
                },
                "post_flush_query": {
                    "query": "What is the name of the fox, what did it jump over, what did it eat, and from where?",
                    "expected_answer": "The fox's name is Felix, it jumped over the lazy dog Charlie, it ate purple grapes from vineyard number 7."
                }
            }
        ]
    }
    
    return sample_data


def load_dataset(name: str) -> Dict[str, Any]:
    """
    Load a benchmark dataset.
    
    Args:
        name: Dataset name
        
    Returns:
        Dictionary with dataset information
        
    Raises:
        ValueError: If dataset not found
    """
    name = name.lower()
    
    # Check if we have a custom dataset file
    dataset_path = os.path.join("benchmark", "data", f"{name}.json")
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    # Use built-in sample datasets
    if name == "hotpotqa":
        return load_hotpotqa_sample()
    elif name == "complex":
        return load_complex_query_sample()
    elif name == "long_context":
        return load_long_context_sample()
    elif name == "memory_test":
        return load_memory_test_sample()
    else:
        available = ", ".join(available_datasets())
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}") 