"""
Integration Engine for RaceIQ Pro

This module connects tactical insights (driver performance, anomalies) with
strategic insights (pit strategy, tire management) to generate unified,
actionable recommendations that maximize race performance.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import timedelta
import numpy as np


@dataclass
class AnomalyImpact:
    """Impact assessment of a detected anomaly"""
    anomaly_id: str
    section: str
    lap_time_loss: float  # seconds per lap
    corrected_tire_degradation: float  # deg/lap after fix
    optimal_pit_lap: int
    position_gain_potential: int
    confidence: float


@dataclass
class SectionImpactAnalysis:
    """Analysis of how section improvements affect overall strategy"""
    section: str
    current_time: float
    potential_time: float
    time_gain_per_lap: float
    laps_remaining: int
    total_time_gain: float
    adjusted_pit_lap: int
    position_impact: int
    confidence: float


@dataclass
class IntegratedInsight:
    """Unified insight combining tactical and strategic elements"""
    insight_type: str  # 'anomaly_fix', 'section_improvement', 'unified'
    priority: int  # 1 (highest) to 5 (lowest)
    tactical_element: str
    strategic_element: str
    expected_impact: str
    action_items: List[str]
    confidence: float
    projected_position_gain: int
    chain_of_impact: str  # e.g., "Fix brake anomaly → 0.8s/lap → Delay pit to L16 → Gain P3"


class IntegrationEngine:
    """
    Core integration engine that connects tactical and strategic modules.

    This is the key differentiator for RaceIQ Pro - it doesn't just show
    data, it connects the dots between driver performance and race strategy.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration engine.

        Args:
            config: Optional configuration dictionary with thresholds and parameters
        """
        self.config = config or {}
        self.lap_time_threshold = self.config.get('lap_time_threshold', 0.1)  # seconds
        self.position_value = self.config.get('position_value', 2.0)  # seconds per position
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

    def connect_anomaly_to_strategy(
        self,
        anomaly: Dict[str, Any],
        tire_model: Any,
        strategy_optimizer: Any
    ) -> AnomalyImpact:
        """
        Connect anomaly detection to pit strategy optimization.

        When an anomaly is detected (e.g., driver braking too early in a section),
        this function:
        1. Estimates the lap time impact if the anomaly is corrected
        2. Recalculates tire degradation with improved pace
        3. Updates optimal pit window based on improved performance
        4. Calculates projected position gain

        Args:
            anomaly: Anomaly data with section, type, magnitude, baseline
            tire_model: Tire degradation model from strategic module
            strategy_optimizer: Pit strategy optimizer from strategic module

        Returns:
            AnomalyImpact object with integrated strategic implications
        """
        # Extract anomaly details
        section = anomaly.get('section', 'unknown')
        anomaly_type = anomaly.get('type', 'unknown')
        magnitude = anomaly.get('magnitude', 0.0)  # seconds lost
        baseline_time = anomaly.get('baseline_time', 0.0)
        current_time = anomaly.get('current_time', 0.0)
        lap = anomaly.get('lap', 0)

        # 1. Estimate lap time impact if anomaly is fixed
        # Assume fixing the anomaly returns section time to baseline
        section_time_gain = current_time - baseline_time  # seconds saved per lap

        # Calculate total lap time impact (conservative estimate: 60-80% of section gain)
        lap_time_gain = section_time_gain * 0.7

        # 2. Recalculate tire degradation with improved pace
        # Faster laps = more tire stress, but also means can push longer before pit
        current_deg_rate = tire_model.get_degradation_rate(lap)

        # Improved pace increases tire deg by ~5-10% but allows later pit stop
        # because we're gaining time on competitors
        pace_factor = 1.0 + (lap_time_gain * 0.01)  # 1% increase per 0.1s gain
        corrected_degradation = current_deg_rate * pace_factor

        # 3. Update optimal pit window
        current_pit_lap = strategy_optimizer.get_optimal_pit_lap()
        laps_remaining = strategy_optimizer.get_total_laps() - lap

        # With faster pace, can afford to delay pit stop slightly
        # Each 0.5s/lap gain allows ~1 lap delay in pit window
        lap_delay = int(lap_time_gain / 0.5)
        adjusted_pit_lap = min(
            current_pit_lap + lap_delay,
            strategy_optimizer.get_total_laps() - 3  # Must pit by lap N-3
        )

        # 4. Calculate position gain potential
        # Time gained per lap × laps remaining = total time gain
        total_time_gain = lap_time_gain * laps_remaining

        # Convert time gain to position gain (rough estimate: 2-3 seconds per position)
        position_gain = int(total_time_gain / self.position_value)

        # Calculate confidence based on anomaly strength and consistency
        confidence = self._calculate_anomaly_confidence(anomaly, tire_model)

        return AnomalyImpact(
            anomaly_id=anomaly.get('id', f"{section}_{lap}"),
            section=section,
            lap_time_loss=lap_time_gain,
            corrected_tire_degradation=corrected_degradation,
            optimal_pit_lap=adjusted_pit_lap,
            position_gain_potential=position_gain,
            confidence=confidence
        )

    def connect_section_improvement_to_strategy(
        self,
        section_improvement: Dict[str, Any],
        race_data: Dict[str, Any],
        strategy: Any
    ) -> SectionImpactAnalysis:
        """
        Connect section-level improvements to overall race strategy.

        Takes a section where the driver has improvement potential and calculates:
        1. How the improvement affects overall lap time
        2. How adjusted lap times affect pit window timing
        3. Strategic recommendations based on improved pace

        Args:
            section_improvement: Section analysis with current/potential times
            race_data: Current race state data
            strategy: Strategy optimizer instance

        Returns:
            SectionImpactAnalysis with strategic recommendations
        """
        section = section_improvement.get('section', 'unknown')
        current_section_time = section_improvement.get('current_time', 0.0)
        potential_section_time = section_improvement.get('potential_time', 0.0)
        improvement_type = section_improvement.get('type', 'general')  # brake, apex, throttle

        # 1. Calculate lap time impact
        section_time_gain = current_section_time - potential_section_time

        # Convert section improvement to lap improvement
        # Different improvement types have different translation factors
        translation_factors = {
            'brake': 0.8,  # Braking improvements translate well to lap time
            'apex': 0.9,   # Apex improvements are very impactful
            'throttle': 0.85,  # Throttle improvements compound through section
            'general': 0.7  # Conservative default
        }

        factor = translation_factors.get(improvement_type, 0.7)
        lap_time_gain_per_lap = section_time_gain * factor

        # 2. Adjust pit window based on improved pace
        current_lap = race_data.get('current_lap', 0)
        total_laps = race_data.get('total_laps', 0)
        laps_remaining = total_laps - current_lap

        # Total time gain over remaining laps
        total_time_gain = lap_time_gain_per_lap * laps_remaining

        # Improved pace allows strategic flexibility in pit timing
        current_pit_lap = strategy.get_optimal_pit_lap()

        # Each 0.3s/lap improvement allows ~1-2 lap flexibility in pit window
        pit_flexibility = int(lap_time_gain_per_lap / 0.3)

        # Decide whether to pit earlier (capitalize on pace) or later (extend stint)
        # If pace improvement > 0.5s/lap, extend stint to maximize advantage
        # If pace improvement < 0.5s/lap, pit on schedule
        if lap_time_gain_per_lap > 0.5:
            adjusted_pit_lap = current_pit_lap + pit_flexibility
        else:
            adjusted_pit_lap = current_pit_lap

        # Ensure pit lap is within valid window
        adjusted_pit_lap = max(
            current_lap + 2,  # Can't pit immediately
            min(adjusted_pit_lap, total_laps - 3)  # Must finish on fresh tires
        )

        # 3. Calculate position impact
        position_impact = int(total_time_gain / self.position_value)

        # Calculate confidence based on improvement consistency
        confidence = self._calculate_improvement_confidence(section_improvement, race_data)

        return SectionImpactAnalysis(
            section=section,
            current_time=current_section_time,
            potential_time=potential_section_time,
            time_gain_per_lap=lap_time_gain_per_lap,
            laps_remaining=laps_remaining,
            total_time_gain=total_time_gain,
            adjusted_pit_lap=adjusted_pit_lap,
            position_impact=position_impact,
            confidence=confidence
        )

    def generate_unified_recommendation(
        self,
        tactical_insights: List[Dict[str, Any]],
        strategic_insights: List[Dict[str, Any]]
    ) -> List[IntegratedInsight]:
        """
        Generate unified recommendations combining tactical and strategic insights.

        This is the key value proposition: connecting driver coaching with
        race strategy to create actionable, prioritized recommendations.

        Args:
            tactical_insights: List of tactical insights (anomalies, improvements)
            strategic_insights: List of strategic insights (pit timing, tire state)

        Returns:
            List of IntegratedInsight objects, prioritized by expected impact
        """
        integrated_insights = []

        # Process each tactical insight and connect to strategy
        for tactical in tactical_insights:
            insight_type = tactical.get('type', 'unknown')

            if insight_type == 'anomaly':
                # Connect anomaly to strategy
                integrated = self._integrate_anomaly_insight(tactical, strategic_insights)
                if integrated:
                    integrated_insights.append(integrated)

            elif insight_type == 'improvement_opportunity':
                # Connect improvement to strategy
                integrated = self._integrate_improvement_insight(tactical, strategic_insights)
                if integrated:
                    integrated_insights.append(integrated)

        # Also process strategic-only insights that don't have tactical connections
        for strategic in strategic_insights:
            if strategic.get('type') == 'pit_window_critical':
                integrated = self._create_strategic_insight(strategic)
                if integrated:
                    integrated_insights.append(integrated)

        # Prioritize recommendations by expected impact
        integrated_insights = self._prioritize_insights(integrated_insights)

        # Filter out low-confidence recommendations
        integrated_insights = [
            insight for insight in integrated_insights
            if insight.confidence >= self.confidence_threshold
        ]

        return integrated_insights

    def _integrate_anomaly_insight(
        self,
        anomaly: Dict[str, Any],
        strategic_insights: List[Dict[str, Any]]
    ) -> Optional[IntegratedInsight]:
        """Create integrated insight from anomaly detection."""
        section = anomaly.get('section', 'unknown')
        magnitude = anomaly.get('magnitude', 0.0)
        lap_impact = anomaly.get('lap_impact', magnitude * 0.7)

        # Find relevant strategic insight (pit timing)
        pit_strategy = next(
            (s for s in strategic_insights if s.get('type') == 'pit_timing'),
            None
        )

        if not pit_strategy:
            return None

        current_pit_lap = pit_strategy.get('optimal_lap', 0)
        laps_remaining = pit_strategy.get('laps_remaining', 0)

        # Calculate impact chain
        total_gain = lap_impact * laps_remaining
        position_gain = int(total_gain / self.position_value)
        adjusted_pit_lap = current_pit_lap + int(lap_impact / 0.5)

        # Build chain of impact description
        chain = (
            f"Fix {section} {anomaly.get('anomaly_type', 'issue')} → "
            f"Save {lap_impact:.2f}s/lap → "
            f"Delay pit to Lap {adjusted_pit_lap} → "
            f"Gain P{position_gain}"
        )

        # Create action items
        action_items = [
            f"DRIVER: {self._get_driver_coaching(anomaly)}",
            f"STRATEGY: Adjust pit window to Lap {adjusted_pit_lap}±2",
            f"MONITOR: Track {section} times for improvement"
        ]

        confidence = anomaly.get('confidence', 0.8) * 0.9  # Slight reduction for integration uncertainty

        return IntegratedInsight(
            insight_type='anomaly_fix',
            priority=self._calculate_priority(lap_impact, position_gain),
            tactical_element=f"{section} {anomaly.get('anomaly_type', 'anomaly')}",
            strategic_element=f"Pit Lap {adjusted_pit_lap}",
            expected_impact=f"{lap_impact:.2f}s/lap, P{position_gain} potential",
            action_items=action_items,
            confidence=confidence,
            projected_position_gain=position_gain,
            chain_of_impact=chain
        )

    def _integrate_improvement_insight(
        self,
        improvement: Dict[str, Any],
        strategic_insights: List[Dict[str, Any]]
    ) -> Optional[IntegratedInsight]:
        """Create integrated insight from improvement opportunity."""
        section = improvement.get('section', 'unknown')
        time_gain = improvement.get('time_gain_per_lap', 0.0)
        improvement_type = improvement.get('improvement_type', 'general')

        # Find relevant strategic insight
        pit_strategy = next(
            (s for s in strategic_insights if s.get('type') == 'pit_timing'),
            None
        )

        if not pit_strategy:
            return None

        current_pit_lap = pit_strategy.get('optimal_lap', 0)
        laps_remaining = pit_strategy.get('laps_remaining', 0)

        # Calculate impact
        total_gain = time_gain * laps_remaining
        position_gain = int(total_gain / self.position_value)

        # Determine pit adjustment strategy
        if time_gain > 0.5:
            adjusted_pit_lap = current_pit_lap + 2
            strategy_note = "Extend stint to maximize pace advantage"
        else:
            adjusted_pit_lap = current_pit_lap
            strategy_note = "Maintain current pit window"

        # Build chain of impact
        chain = (
            f"Improve {section} {improvement_type} → "
            f"Gain {time_gain:.2f}s/lap → "
            f"{strategy_note} → "
            f"Gain P{position_gain}"
        )

        # Create action items
        action_items = [
            f"DRIVER: {self._get_improvement_coaching(improvement)}",
            f"STRATEGY: {strategy_note} (Lap {adjusted_pit_lap})",
            f"ANALYZE: Compare {section} to reference lap"
        ]

        confidence = improvement.get('confidence', 0.75) * 0.85

        return IntegratedInsight(
            insight_type='section_improvement',
            priority=self._calculate_priority(time_gain, position_gain),
            tactical_element=f"{section} {improvement_type} improvement",
            strategic_element=strategy_note,
            expected_impact=f"{time_gain:.2f}s/lap, P{position_gain} potential",
            action_items=action_items,
            confidence=confidence,
            projected_position_gain=position_gain,
            chain_of_impact=chain
        )

    def _create_strategic_insight(
        self,
        strategic: Dict[str, Any]
    ) -> Optional[IntegratedInsight]:
        """Create insight from purely strategic consideration."""
        if strategic.get('type') != 'pit_window_critical':
            return None

        critical_lap = strategic.get('critical_lap', 0)
        reason = strategic.get('reason', 'tire degradation')

        chain = f"Pit window closing → Must pit by Lap {critical_lap} → {reason}"

        action_items = [
            f"STRATEGY: Prepare for pit stop by Lap {critical_lap}",
            f"TEAM: Confirm tire choice and fuel load",
            f"DRIVER: Manage tire temps approaching pit window"
        ]

        return IntegratedInsight(
            insight_type='strategic_critical',
            priority=1,  # Critical items get highest priority
            tactical_element="N/A - Strategic Only",
            strategic_element=f"Critical pit window: Lap {critical_lap}",
            expected_impact=f"Avoid {reason} impact",
            action_items=action_items,
            confidence=strategic.get('confidence', 0.9),
            projected_position_gain=0,
            chain_of_impact=chain
        )

    def _calculate_priority(self, time_gain: float, position_gain: int) -> int:
        """
        Calculate priority (1-5) based on expected impact.

        Priority 1: Highest impact (>1.0s/lap or >2 positions)
        Priority 2: High impact (0.5-1.0s/lap or 1-2 positions)
        Priority 3: Medium impact (0.3-0.5s/lap or 1 position)
        Priority 4: Low impact (0.1-0.3s/lap)
        Priority 5: Minimal impact (<0.1s/lap)
        """
        if time_gain > 1.0 or position_gain > 2:
            return 1
        elif time_gain > 0.5 or position_gain >= 1:
            return 2
        elif time_gain > 0.3:
            return 3
        elif time_gain > 0.1:
            return 4
        else:
            return 5

    def _prioritize_insights(
        self,
        insights: List[IntegratedInsight]
    ) -> List[IntegratedInsight]:
        """Sort insights by priority and expected impact."""
        return sorted(
            insights,
            key=lambda x: (x.priority, -x.projected_position_gain, -x.confidence)
        )

    def _calculate_anomaly_confidence(
        self,
        anomaly: Dict[str, Any],
        tire_model: Any
    ) -> float:
        """Calculate confidence score for anomaly-based recommendation."""
        base_confidence = anomaly.get('detection_confidence', 0.8)

        # Adjust based on consistency
        consistency = anomaly.get('consistency', 0.5)  # How often this anomaly appears

        # Adjust based on tire state (anomalies on fresh tires are more fixable)
        tire_life = tire_model.get_tire_life() if hasattr(tire_model, 'get_tire_life') else 0.5
        tire_factor = 1.0 if tire_life < 0.3 else 0.9 if tire_life < 0.6 else 0.8

        # Combined confidence
        confidence = base_confidence * (0.7 + 0.3 * consistency) * tire_factor

        return min(confidence, 0.95)  # Cap at 95%

    def _calculate_improvement_confidence(
        self,
        improvement: Dict[str, Any],
        race_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for improvement-based recommendation."""
        base_confidence = improvement.get('confidence', 0.75)

        # Adjust based on reference quality
        reference_quality = improvement.get('reference_quality', 0.8)

        # Adjust based on driver consistency
        driver_consistency = race_data.get('driver_consistency', 0.7)

        # Combined confidence
        confidence = base_confidence * reference_quality * (0.8 + 0.2 * driver_consistency)

        return min(confidence, 0.95)

    def _get_driver_coaching(self, anomaly: Dict[str, Any]) -> str:
        """Generate specific driver coaching based on anomaly type."""
        anomaly_type = anomaly.get('anomaly_type', 'unknown')
        section = anomaly.get('section', 'unknown')

        coaching_templates = {
            'brake_early': f"Brake later into {section} - carrying {anomaly.get('magnitude', 0):.2f}s",
            'brake_soft': f"Brake harder into {section} - increase initial pressure",
            'apex_early': f"Delay apex in {section} - wait for rotation",
            'apex_wide': f"Tighten line in {section} - hit apex marker",
            'throttle_late': f"Get on throttle earlier out of {section}",
            'throttle_soft': f"More aggressive throttle application in {section}",
            'default': f"Optimize {section} - review reference lap"
        }

        return coaching_templates.get(anomaly_type, coaching_templates['default'])

    def _get_improvement_coaching(self, improvement: Dict[str, Any]) -> str:
        """Generate specific driver coaching based on improvement opportunity."""
        improvement_type = improvement.get('improvement_type', 'general')
        section = improvement.get('section', 'unknown')
        time_gain = improvement.get('time_gain_per_lap', 0.0)

        coaching_templates = {
            'brake': f"Optimize braking in {section} - {time_gain:.2f}s available",
            'apex': f"Perfect apex in {section} - {time_gain:.2f}s available",
            'throttle': f"Maximize throttle in {section} - {time_gain:.2f}s available",
            'general': f"Improve {section} technique - {time_gain:.2f}s available"
        }

        return coaching_templates.get(improvement_type, coaching_templates['general'])
