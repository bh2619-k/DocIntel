"""
Real-Time Document Intelligence & Compliance Engine
Analysis & Compliance Module Initialization
Version: 1.0
Date: 2026-02-03
"""

from .rule_engine.deterministic_rules import (
    ComplianceRule,
    RuleEngine,
    RuleConflict,
    RulePerformanceOptimizer,
)

from .anomaly_detection.anomaly_engine import (
    StatisticalAnalyzer,
    PatternRecognitionAnalyzer,
    OutlierDetector,
)

from .anomaly_detection.comprehensive_detector import ComprehensiveAnomalyDetector

from .risk_scoring.composite_risk import (
    CompositeRiskScorer,
    OperationalRiskAssessor,
    RegulatoryRiskAssessor,
)

from .semantic_comparison.contract_comparison import (
    ClauseEmbeddingManager,
    ContractComparator,
)

from .semantic_comparison.benchmarking import (
    ContractBenchmarker,
    BenchmarkStandard,
    BenchmarkResult,
)

from .audit_trail.audit_manager import AuditTrailManager, AuditEvent, ComplianceReport

__version__ = "1.0.0"
__author__ = "AI Document Intelligence Team"

__all__ = [
    # Rule Engine
    "ComplianceRule",
    "RuleEngine",
    "RuleConflict",
    "RulePerformanceOptimizer",
    # Anomaly Detection
    "StatisticalAnalyzer",
    "PatternRecognitionAnalyzer",
    "OutlierDetector",
    "ComprehensiveAnomalyDetector",
    # Risk Scoring
    "CompositeRiskScorer",
    "OperationalRiskAssessor",
    "RegulatoryRiskAssessor",
    # Semantic Comparison
    "ClauseEmbeddingManager",
    "ContractComparator",
    # Benchmarking
    "ContractBenchmarker",
    "BenchmarkStandard",
    "BenchmarkResult",
    # Audit Trail
    "AuditTrailManager",
    "AuditEvent",
    "ComplianceReport",
]

# Module-level configuration
DEFAULT_CONFIG = {
    "rule_engine": {
        "max_rules_per_category": 100,
        "performance_optimization": True,
        "conflict_detection": True,
    },
    "anomaly_detection": {
        "statistical_threshold": 0.95,
        "pattern_sensitivity": 0.8,
        "outlier_contamination": 0.1,
        "enable_temporal": True,
        "enable_contextual": True,
    },
    "risk_scoring": {
        "composite_weights": {
            "compliance": 0.3,
            "financial": 0.25,
            "operational": 0.25,
            "regulatory": 0.2,
        }
    },
    "semantic_comparison": {
        "similarity_threshold": 0.8,
        "enable_benchmarking": True,
        "clause_embedding_dim": 384,
    },
    "audit_trail": {
        "storage_path": "./audit_logs",
        "retention_days": 2555,  # 7 years for SOX
        "enable_blockchain": True,
        "compression": True,
    },
}


def get_version_info():
    """Get version and component information"""
    return {
        "version": __version__,
        "components": {
            "rule_engine": "Deterministic compliance rule processing",
            "anomaly_detection": "7-feature anomaly detection system",
            "risk_scoring": "5-feature risk assessment engine",
            "semantic_comparison": "AI-powered contract comparison",
            "benchmarking": "Industry standard benchmarking",
            "audit_trail": "Compliance audit and reporting",
        },
        "capabilities": [
            "Real-time compliance checking",
            "Multi-framework anomaly detection",
            "Composite risk assessment",
            "Semantic contract analysis",
            "Regulatory benchmarking",
            "Tamper-proof audit trails",
            "Automated compliance reporting",
        ],
    }


def validate_configuration(config: dict = None) -> bool:
    """Validate module configuration"""
    if config is None:
        config = DEFAULT_CONFIG

    required_sections = [
        "rule_engine",
        "anomaly_detection",
        "risk_scoring",
        "semantic_comparison",
        "audit_trail",
    ]

    for section in required_sections:
        if section not in config:
            return False

    return True
