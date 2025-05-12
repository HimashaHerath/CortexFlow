# Changelog

All notable changes to CortexFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project renamed from AdaptiveContext to CortexFlow
- Compatibility layer for existing AdaptiveContext users
- Enhanced entity extraction in GraphRAG
- Improved error handling and fallback in Chain of Agents
- Streaming response support
- Command line interface
- Flexible ontology system with inheritance hierarchies (OntologyClass, RelationType)
- Support for n-ary relationships beyond simple triples
- Metadata framework for tracking provenance, confidence, and temporal information
- Enhanced relation typing with inheritance through the ontology system

### Fixed
- Memory tier limit update function
- Embedding caching to avoid redundant computation
- Optimized BM25 indexing for large knowledge bases

## [0.1.0] - 2025-05-12

### Added
- Initial release with multi-tier memory architecture
- Chain of Agents framework
- GraphRAG for complex multi-hop queries
- Dynamic memory weighting
- Self-reflection and verification 