# Changelog

All notable changes to the CortexFlow project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Advanced graph traversal capabilities:
  - Weighted path algorithms that consider relation importance and confidence scores
  - Bidirectional search for efficient complex queries
  - Constrained path finding (e.g., finding paths only through specific relation types)
  - Graph contraction algorithms to handle large knowledge graphs efficiently
  - Hierarchical graph abstraction for improved traversal performance
- New example script demonstrating the advanced graph traversal features

### Fixed
- Memory tier limit update function
- Embedding caching to avoid redundant computation
- Optimized BM25 indexing for large knowledge bases

## [0.5.0] - 2023-10-15

### Added
- GraphRAG implementation with entity and relation extraction
- Semantic role labeling for improved relation extraction
- Coreference resolution for better entity linking
- Support for n-ary relations in knowledge graphs
- Support for temporal and confidence attributes in graph relationships

### Changed
- Restructured knowledge store API for better extensibility
- Improved entity extraction with domain-specific recognition

### Fixed
- Memory leaks in long-running graph queries
- Entity extraction performance issues with large documents

## [0.4.0] - 2023-08-20

### Added
- Chain of Agents architecture for collaborative reasoning
- Self-reflection capabilities for verifying knowledge relevance
- Dynamic memory tier weighting for adaptive token allocation

### Changed
- Improved summarization algorithms
- Enhanced entity extraction performance

## [0.3.0] - 2023-06-10

### Added
- Vector-based knowledge retrieval for semantic search
- Importance-based information retention using ML classification
- Progressive context compression with extractive summarization

### Changed
- Refactored memory management for better performance
- Updated documentation with new examples

## [0.2.0] - 2023-04-05

### Added
- Knowledge store for long-term information persistence
- Multi-tier memory management system
- Basic entity extraction capabilities

### Changed
- Renamed project from AdaptiveContext to CortexFlow
- Improved token counting accuracy

## [0.1.0] - 2023-02-15

### Added
- Initial release with basic context management
- Rule-based importance classification
- Simple memory tiers implementation 