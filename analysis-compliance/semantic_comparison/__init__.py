"""
Semantic Comparison Module
AI-powered contract comparison and benchmarking
"""

from .contract_comparison import (
    ClauseEmbeddingManager,
    ContractComparator,
    SemanticComparisonResult,
    ComparisonType,
    DeviationType,
)

from .benchmarking import (
    ContractBenchmarker,
    BenchmarkStandard,
    BenchmarkResult,
    BenchmarkType,
)

__all__ = [
    "ClauseEmbeddingManager",
    "ContractComparator",
    "SemanticComparisonResult",
    "ComparisonType",
    "DeviationType",
    "ContractBenchmarker",
    "BenchmarkStandard",
    "BenchmarkResult",
    "BenchmarkType",
]
