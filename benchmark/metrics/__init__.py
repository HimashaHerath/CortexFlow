"""
Metrics package for CortexFlow evaluation framework.

This package contains various metrics for evaluating the performance
and capabilities of the CortexFlow knowledge graph system.
"""

from benchmark.metrics.consistency_metrics import (
    belief_revision_impact,
    evaluate_knowledge_consistency,
    jaccard_similarity,
    temporal_consistency_score,
    track_knowledge_growth,
)
from benchmark.metrics.multi_hop_metrics import (
    benchmark_multi_hop_reasoning,
    evaluate_reasoning_chain,
    multi_hop_reasoning_score,
    normalize_path,
    path_order_accuracy,
    path_overlap_score,
)

__all__ = [
    # Multi-hop metrics
    "normalize_path",
    "path_overlap_score",
    "path_order_accuracy",
    "multi_hop_reasoning_score",
    "evaluate_reasoning_chain",
    "benchmark_multi_hop_reasoning",
    # Consistency metrics
    "jaccard_similarity",
    "temporal_consistency_score",
    "track_knowledge_growth",
    "belief_revision_impact",
    "evaluate_knowledge_consistency",
]
