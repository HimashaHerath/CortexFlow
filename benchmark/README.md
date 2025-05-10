# AdaptiveContext Benchmark Framework

This directory contains tools for benchmarking and evaluating AdaptiveContext against other RAG and context management systems.

## Implementation Status

- ✅ Core benchmark framework
- ✅ AdaptiveContext adapter
- ✅ Retrieval quality metrics
- ✅ Memory efficiency metrics
- ✅ Query complexity metrics
- ✅ Visualization tools
- ✅ LlamaIndex adapter (implemented but dependency issues remain)
- ✅ LangChain adapter (fully implemented and benchmarked)
- ✅ Activeloop DeepMemory adapter (fully implemented and benchmarked)
- ✅ Qdrant adapter (implemented but dependency issues remain)
- ⬜ Custom datasets (sample datasets implemented)

## Supported Comparison Systems

- [AdaptiveContext](https://github.com/yourusername/adaptivecontext) - Multi-tiered memory with GraphRAG
- [LlamaIndex](https://github.com/jerryjliu/llama_index) - Document indexing and retrieval system
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM-powered applications
- [Activeloop DeepMemory](https://github.com/activeloopai/deeplake) - Vector database with conversation memory
- [Qdrant Vector Database](https://github.com/qdrant/qdrant) - Vector similarity search engine

## Current Results Summary

Our current benchmark results (using llama3.2 model) show:

### Retrieval Quality
- **LangChain**: Best precision (0.410), Best F1 score (0.375), Fastest (4.84s)
- **Activeloop DeepMemory**: Best recall (0.448), but slowest (44.21s)
- **AdaptiveContext**: Lower precision (0.085) and moderate speed (11.34s)

### Memory Efficiency
- **Activeloop DeepMemory**: Some knowledge retention (0.150), but higher token usage
- **AdaptiveContext**: Most efficient with token usage (0.750)
- **LangChain**: Moderate efficiency (0.681)

### Query Complexity
- **Activeloop DeepMemory**: Best for complex queries (0.670 score)
- **AdaptiveContext**: Moderate complexity handling (0.486)
- **LangChain**: Similar to AdaptiveContext (0.486)

## Running Benchmarks

### Prerequisites

```bash
pip install -r benchmark/requirements.txt
```

### Running the Benchmarks

```bash
cd benchmark
./run_comparison.sh
```

### Fixing Dependencies

If you encounter dependency issues:

```bash
# For LlamaIndex
pip install llama-index==0.9.48 llama-index-core==0.10.39 llama-index-llms-ollama==0.1.1

# For Activeloop DeepMemory
pip install deeplake==3.8.3 langchain-community==0.1.1

# For Qdrant
pip install qdrant-client==1.7.0 langchain-community==0.1.1
```

## Visualization

Benchmark results are saved to:
- Raw data: `../benchmark_results/benchmark_results.json`
- Summary: `../benchmark_results/benchmark_comparison.md`
- Implementation status: `../benchmark_results/implementation_status.md`

## Extending the Benchmark

### Adding New Systems

To add a new system for comparison:

1. Create a new adapter in `benchmark/adapters/`
2. Implement the `BenchmarkSystemAdapter` interface
3. Register the adapter in `benchmark/registry.py`

### Future Work

- Complete benchmarking of all implemented systems
- Add more specialized datasets for testing
- Implement visualizations for benchmark results
- Test with more powerful LLM models 