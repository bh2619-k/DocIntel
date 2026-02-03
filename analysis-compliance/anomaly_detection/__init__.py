"""
Anomaly Detection Module
Multi-feature anomaly detection system
"""

from .anomaly_engine import (
    StatisticalAnalyzer,
    PatternRecognitionAnalyzer,
    OutlierDetector,
)

from .comprehensive_detector import ComprehensiveAnomalyDetector

try:
    from .temporal_contextual import TemporalAnalyzer, ContextualAnalyzer
    from .severity_alerting import SeverityScorer, AlertManager

    HAS_EXTENDED_FEATURES = True
except ImportError:
    HAS_EXTENDED_FEATURES = False

__all__ = [
    "StatisticalAnalyzer",
    "PatternRecognitionAnalyzer",
    "OutlierDetector",
    "ComprehensiveAnomalyDetector",
]

if HAS_EXTENDED_FEATURES:
    __all__.extend(
        ["TemporalAnalyzer", "ContextualAnalyzer", "SeverityScorer", "AlertManager"]
    )
