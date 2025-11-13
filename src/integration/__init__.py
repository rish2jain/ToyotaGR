"""
RaceIQ Pro Integration Engine

This package provides the core integration layer that connects tactical
driver coaching with strategic race planning - the key differentiator of
RaceIQ Pro.

Main Components:
- IntegrationEngine: Connects anomaly detection to pit strategy optimization
- RecommendationBuilder: Formats insights for dashboard display

Example Usage:
    from integration import IntegrationEngine, RecommendationBuilder

    # Initialize
    engine = IntegrationEngine()
    builder = RecommendationBuilder()

    # Connect anomaly to strategy
    impact = engine.connect_anomaly_to_strategy(
        anomaly=detected_anomaly,
        tire_model=tire_model,
        strategy_optimizer=strategy
    )

    # Build recommendations
    integrated_recs = builder.build_integrated_recommendation(
        tactical=tactical_recommendations,
        strategic=strategic_recommendations,
        integration_engine=engine
    )
"""

from .intelligence_engine import (
    IntegrationEngine,
    AnomalyImpact,
    SectionImpactAnalysis,
    IntegratedInsight
)

from .recommendation_builder import (
    RecommendationBuilder,
    TacticalRecommendation,
    StrategicRecommendation,
    IntegratedRecommendation,
    format_recommendation_json,
    export_recommendations_csv
)

__all__ = [
    'IntegrationEngine',
    'AnomalyImpact',
    'SectionImpactAnalysis',
    'IntegratedInsight',
    'RecommendationBuilder',
    'TacticalRecommendation',
    'StrategicRecommendation',
    'IntegratedRecommendation',
    'format_recommendation_json',
    'export_recommendations_csv'
]

__version__ = '0.1.0'
