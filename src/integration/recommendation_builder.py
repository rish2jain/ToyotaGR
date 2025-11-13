"""
Recommendation Builder for RaceIQ Pro

This module formats tactical and strategic insights into actionable,
prioritized recommendations suitable for dashboard display and real-time
decision making.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class TacticalRecommendation:
    """Driver coaching recommendation"""
    id: str
    priority: int  # 1-5
    section: str
    issue_type: str
    coaching_message: str
    time_gain_potential: float  # seconds per lap
    confidence: float
    reference_lap: Optional[str] = None
    telemetry_comparison: Optional[Dict[str, Any]] = None


@dataclass
class StrategicRecommendation:
    """Pit strategy recommendation"""
    id: str
    priority: int
    recommendation_type: str  # 'pit_now', 'pit_window', 'extend_stint', 'tire_choice'
    message: str
    optimal_lap: int
    window_start: int
    window_end: int
    expected_impact: str
    confidence: float
    confidence_interval: Dict[str, float]  # {'lower': 0.8, 'upper': 0.95}
    tire_state: Optional[Dict[str, Any]] = None


@dataclass
class IntegratedRecommendation:
    """Unified recommendation combining tactical and strategic"""
    id: str
    priority: int
    title: str
    tactical_component: str
    strategic_component: str
    chain_of_impact: str
    action_items: List[Dict[str, str]]  # [{'role': 'DRIVER', 'action': '...'}]
    expected_impact: Dict[str, Any]  # {'time_gain': 0.8, 'position_gain': 2}
    confidence: float
    status: str  # 'new', 'in_progress', 'completed', 'dismissed'
    timestamp: str
    display_format: Dict[str, Any]  # Formatting hints for dashboard


class RecommendationBuilder:
    """
    Builds formatted recommendations for dashboard display.

    Transforms raw insights from tactical, strategic, and integration modules
    into user-friendly, actionable recommendations with proper prioritization,
    confidence intervals, and formatting for real-time dashboard display.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the recommendation builder.

        Args:
            config: Optional configuration for formatting and thresholds
        """
        self.config = config or {}
        self.recommendation_counter = 0
        self.active_recommendations: List[IntegratedRecommendation] = []

    def build_tactical_recommendation(
        self,
        section_analysis: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> List[TacticalRecommendation]:
        """
        Create driver coaching recommendations from section analysis and anomalies.

        Prioritizes recommendations by time gain potential and presents them
        in a clear, actionable format for driver coaching.

        Args:
            section_analysis: Section-by-section performance analysis
            anomalies: List of detected anomalies

        Returns:
            List of TacticalRecommendation objects, sorted by priority
        """
        recommendations = []

        # Process anomalies into recommendations
        for anomaly in anomalies:
            rec = self._create_tactical_from_anomaly(anomaly, section_analysis)
            if rec:
                recommendations.append(rec)

        # Process section improvements
        sections = section_analysis.get('sections', [])
        for section in sections:
            if section.get('has_improvement_potential', False):
                rec = self._create_tactical_from_improvement(section, section_analysis)
                if rec:
                    recommendations.append(rec)

        # Sort by time gain potential (descending)
        recommendations.sort(
            key=lambda x: (x.priority, -x.time_gain_potential, -x.confidence)
        )

        return recommendations

    def build_strategic_recommendation(
        self,
        pit_strategy: Dict[str, Any],
        tire_state: Dict[str, Any]
    ) -> List[StrategicRecommendation]:
        """
        Create pit timing recommendations with confidence intervals.

        Transforms pit strategy analysis into clear, actionable recommendations
        with confidence intervals and expected impact.

        Args:
            pit_strategy: Pit strategy optimization results
            tire_state: Current tire degradation state

        Returns:
            List of StrategicRecommendation objects
        """
        recommendations = []

        # Main pit window recommendation
        if pit_strategy.get('optimal_lap'):
            rec = self._create_pit_window_recommendation(pit_strategy, tire_state)
            recommendations.append(rec)

        # Critical timing recommendation
        if pit_strategy.get('critical_window'):
            rec = self._create_critical_timing_recommendation(pit_strategy, tire_state)
            recommendations.append(rec)

        # Tire choice recommendation
        if pit_strategy.get('tire_choice_analysis'):
            rec = self._create_tire_choice_recommendation(pit_strategy, tire_state)
            recommendations.append(rec)

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)

        return recommendations

    def build_integrated_recommendation(
        self,
        tactical: List[TacticalRecommendation],
        strategic: List[StrategicRecommendation],
        integration_engine: Any
    ) -> List[IntegratedRecommendation]:
        """
        Combine tactical and strategic recommendations into unified insights.

        This is where the magic happens - connecting driver coaching with
        pit strategy to show the full chain of impact: "Fix X → Gain Y → Strategy Z"

        Args:
            tactical: List of tactical recommendations
            strategic: List of strategic recommendations
            integration_engine: IntegrationEngine instance

        Returns:
            List of IntegratedRecommendation objects showing connections
        """
        integrated_recommendations = []

        # Build integration matrix: which tactical items affect which strategic items
        integration_matrix = self._build_integration_matrix(tactical, strategic)

        # Create integrated recommendations for each connection
        for tactical_rec in tactical:
            # Find strategic recommendations affected by this tactical item
            affected_strategic = integration_matrix.get(tactical_rec.id, [])

            if affected_strategic:
                # Create integrated recommendation showing the connection
                integrated = self._create_integrated_recommendation(
                    tactical_rec,
                    affected_strategic,
                    integration_engine
                )
                integrated_recommendations.append(integrated)
            else:
                # Tactical-only recommendation (still valuable coaching)
                integrated = self._create_tactical_only_recommendation(tactical_rec)
                integrated_recommendations.append(integrated)

        # Add strategic-only recommendations (e.g., critical pit windows)
        for strategic_rec in strategic:
            if strategic_rec.priority == 1 and strategic_rec.recommendation_type == 'pit_now':
                # Critical strategic item - add as standalone
                integrated = self._create_strategic_only_recommendation(strategic_rec)
                integrated_recommendations.append(integrated)

        # Sort by priority and expected impact
        integrated_recommendations.sort(
            key=lambda x: (x.priority, -x.expected_impact.get('time_gain', 0))
        )

        # Store active recommendations
        self.active_recommendations = integrated_recommendations

        return integrated_recommendations

    def format_for_dashboard(
        self,
        recommendations: List[IntegratedRecommendation]
    ) -> Dict[str, Any]:
        """
        Format recommendations for dashboard display.

        Returns:
            Dictionary with formatted recommendations and display metadata
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'priority_breakdown': self._get_priority_breakdown(recommendations),
            'recommendations': [
                self._format_recommendation_for_display(rec)
                for rec in recommendations
            ],
            'summary': self._generate_summary(recommendations),
            'quick_actions': self._generate_quick_actions(recommendations[:3])  # Top 3
        }

    def _create_tactical_from_anomaly(
        self,
        anomaly: Dict[str, Any],
        section_analysis: Dict[str, Any]
    ) -> Optional[TacticalRecommendation]:
        """Create tactical recommendation from anomaly detection."""
        section = anomaly.get('section', 'unknown')
        anomaly_type = anomaly.get('anomaly_type', 'unknown')
        magnitude = anomaly.get('magnitude', 0.0)

        # Generate coaching message based on anomaly type
        coaching_messages = {
            'brake_early': f"You're braking {magnitude:.2f}s too early in {section}. Trust the grip and brake later.",
            'brake_soft': f"Increase initial brake pressure in {section} by ~{magnitude*100:.0f}%.",
            'apex_early': f"Wait for the apex in {section} - you're turning in {magnitude:.2f}s early.",
            'apex_wide': f"Tighten your line in {section} - {magnitude:.2f}s available by hitting apex.",
            'throttle_late': f"Get on throttle {magnitude:.2f}s earlier exiting {section}.",
            'throttle_soft': f"More aggressive throttle application in {section} - {magnitude:.2f}s available."
        }

        coaching = coaching_messages.get(
            anomaly_type,
            f"Optimize {section} - {magnitude:.2f}s available"
        )

        # Estimate lap time impact (conservative: 70% of section time)
        lap_time_gain = magnitude * 0.7

        return TacticalRecommendation(
            id=f"TAC_{self._get_next_id()}",
            priority=self._calculate_tactical_priority(lap_time_gain),
            section=section,
            issue_type=anomaly_type,
            coaching_message=coaching,
            time_gain_potential=lap_time_gain,
            confidence=anomaly.get('confidence', 0.8),
            reference_lap=anomaly.get('reference_lap_id'),
            telemetry_comparison=anomaly.get('telemetry_comparison')
        )

    def _create_tactical_from_improvement(
        self,
        section: Dict[str, Any],
        section_analysis: Dict[str, Any]
    ) -> Optional[TacticalRecommendation]:
        """Create tactical recommendation from improvement opportunity."""
        section_name = section.get('name', 'unknown')
        time_gain = section.get('time_gain_potential', 0.0)
        improvement_type = section.get('improvement_type', 'general')

        # Skip if time gain is too small
        if time_gain < 0.05:
            return None

        coaching_templates = {
            'brake': f"Optimize braking in {section_name} - compare to reference lap for brake point.",
            'apex': f"Focus on apex precision in {section_name} - {time_gain:.2f}s available.",
            'throttle': f"Earlier, more aggressive throttle in {section_name} - {time_gain:.2f}s on table.",
            'general': f"Study {section_name} vs reference - {time_gain:.2f}s improvement possible."
        }

        coaching = coaching_templates.get(improvement_type, coaching_templates['general'])

        return TacticalRecommendation(
            id=f"TAC_{self._get_next_id()}",
            priority=self._calculate_tactical_priority(time_gain),
            section=section_name,
            issue_type=f"{improvement_type}_improvement",
            coaching_message=coaching,
            time_gain_potential=time_gain,
            confidence=section.get('confidence', 0.75),
            reference_lap=section_analysis.get('reference_lap_id')
        )

    def _create_pit_window_recommendation(
        self,
        pit_strategy: Dict[str, Any],
        tire_state: Dict[str, Any]
    ) -> StrategicRecommendation:
        """Create pit window recommendation."""
        optimal_lap = pit_strategy.get('optimal_lap', 0)
        window_start = pit_strategy.get('window_start', optimal_lap - 2)
        window_end = pit_strategy.get('window_end', optimal_lap + 2)

        message = (
            f"Optimal pit window: Laps {window_start}-{window_end}. "
            f"Target Lap {optimal_lap} for best track position."
        )

        # Calculate confidence interval
        confidence = pit_strategy.get('confidence', 0.85)
        confidence_interval = {
            'lower': max(0.0, confidence - 0.1),
            'upper': min(1.0, confidence + 0.05)
        }

        expected_impact = f"Optimize track position, minimize time loss (~{pit_strategy.get('pit_loss', 22.0):.1f}s)"

        return StrategicRecommendation(
            id=f"STR_{self._get_next_id()}",
            priority=2,
            recommendation_type='pit_window',
            message=message,
            optimal_lap=optimal_lap,
            window_start=window_start,
            window_end=window_end,
            expected_impact=expected_impact,
            confidence=confidence,
            confidence_interval=confidence_interval,
            tire_state=tire_state
        )

    def _create_critical_timing_recommendation(
        self,
        pit_strategy: Dict[str, Any],
        tire_state: Dict[str, Any]
    ) -> StrategicRecommendation:
        """Create critical timing recommendation."""
        critical_lap = pit_strategy['critical_window'].get('latest_lap', 0)
        reason = pit_strategy['critical_window'].get('reason', 'tire degradation')

        message = (
            f"CRITICAL: Must pit by Lap {critical_lap}. "
            f"Reason: {reason}."
        )

        return StrategicRecommendation(
            id=f"STR_{self._get_next_id()}",
            priority=1,
            recommendation_type='pit_now',
            message=message,
            optimal_lap=critical_lap,
            window_start=critical_lap - 1,
            window_end=critical_lap,
            expected_impact=f"Avoid {reason} performance cliff",
            confidence=pit_strategy['critical_window'].get('confidence', 0.95),
            confidence_interval={'lower': 0.90, 'upper': 0.98},
            tire_state=tire_state
        )

    def _create_tire_choice_recommendation(
        self,
        pit_strategy: Dict[str, Any],
        tire_state: Dict[str, Any]
    ) -> StrategicRecommendation:
        """Create tire choice recommendation."""
        tire_analysis = pit_strategy['tire_choice_analysis']
        recommended_compound = tire_analysis.get('recommended_compound', 'medium')
        reason = tire_analysis.get('reason', 'optimal for conditions')

        message = f"Recommended tire compound: {recommended_compound.upper()}. {reason}."

        return StrategicRecommendation(
            id=f"STR_{self._get_next_id()}",
            priority=3,
            recommendation_type='tire_choice',
            message=message,
            optimal_lap=pit_strategy.get('optimal_lap', 0),
            window_start=0,
            window_end=0,
            expected_impact=tire_analysis.get('expected_impact', 'Optimal tire life'),
            confidence=tire_analysis.get('confidence', 0.8),
            confidence_interval={'lower': 0.75, 'upper': 0.85},
            tire_state=tire_state
        )

    def _build_integration_matrix(
        self,
        tactical: List[TacticalRecommendation],
        strategic: List[StrategicRecommendation]
    ) -> Dict[str, List[StrategicRecommendation]]:
        """
        Build matrix showing which tactical items affect which strategic items.

        Returns:
            Dict mapping tactical recommendation IDs to affected strategic recommendations
        """
        matrix = {}

        for tac in tactical:
            # Tactical improvements affect pit timing strategy
            affected = [
                strat for strat in strategic
                if strat.recommendation_type in ['pit_window', 'extend_stint']
            ]
            if affected:
                matrix[tac.id] = affected

        return matrix

    def _create_integrated_recommendation(
        self,
        tactical: TacticalRecommendation,
        strategic_list: List[StrategicRecommendation],
        integration_engine: Any
    ) -> IntegratedRecommendation:
        """Create integrated recommendation showing tactical-strategic connection."""
        # Use the first (highest priority) strategic recommendation
        strategic = strategic_list[0]

        # Build chain of impact
        time_gain = tactical.time_gain_potential
        laps_remaining = 20  # Could be passed in as parameter
        total_gain = time_gain * laps_remaining
        position_gain = int(total_gain / 2.0)  # Rough estimate

        chain = (
            f"Fix {tactical.section} {tactical.issue_type} → "
            f"Save {time_gain:.2f}s/lap → "
            f"Adjust pit to Lap {strategic.optimal_lap} → "
            f"Gain P{position_gain}"
        )

        title = f"{tactical.section} Optimization + Strategy Adjustment"

        # Build action items
        action_items = [
            {
                'role': 'DRIVER',
                'action': tactical.coaching_message,
                'priority': 'HIGH'
            },
            {
                'role': 'STRATEGY',
                'action': strategic.message,
                'priority': 'MEDIUM'
            },
            {
                'role': 'ENGINEER',
                'action': f"Monitor {tactical.section} times for improvement",
                'priority': 'LOW'
            }
        ]

        # Expected impact
        expected_impact = {
            'time_gain': time_gain,
            'total_time_gain': total_gain,
            'position_gain': position_gain,
            'confidence': min(tactical.confidence, strategic.confidence)
        }

        # Display format hints
        display_format = {
            'color': self._get_priority_color(tactical.priority),
            'icon': 'integration',
            'show_chain': True,
            'expanded_by_default': tactical.priority <= 2
        }

        return IntegratedRecommendation(
            id=f"INT_{self._get_next_id()}",
            priority=min(tactical.priority, strategic.priority),
            title=title,
            tactical_component=tactical.coaching_message,
            strategic_component=strategic.message,
            chain_of_impact=chain,
            action_items=action_items,
            expected_impact=expected_impact,
            confidence=expected_impact['confidence'],
            status='new',
            timestamp=datetime.now().isoformat(),
            display_format=display_format
        )

    def _create_tactical_only_recommendation(
        self,
        tactical: TacticalRecommendation
    ) -> IntegratedRecommendation:
        """Create recommendation for tactical-only insight."""
        title = f"{tactical.section} Performance Improvement"

        action_items = [
            {
                'role': 'DRIVER',
                'action': tactical.coaching_message,
                'priority': 'HIGH'
            },
            {
                'role': 'ENGINEER',
                'action': f"Review {tactical.section} telemetry vs reference",
                'priority': 'LOW'
            }
        ]

        expected_impact = {
            'time_gain': tactical.time_gain_potential,
            'total_time_gain': tactical.time_gain_potential * 20,
            'position_gain': int(tactical.time_gain_potential * 20 / 2.0),
            'confidence': tactical.confidence
        }

        display_format = {
            'color': self._get_priority_color(tactical.priority),
            'icon': 'driver',
            'show_chain': False,
            'expanded_by_default': tactical.priority <= 2
        }

        return IntegratedRecommendation(
            id=f"INT_{self._get_next_id()}",
            priority=tactical.priority,
            title=title,
            tactical_component=tactical.coaching_message,
            strategic_component="No strategic impact",
            chain_of_impact=f"Improve {tactical.section} → Save {tactical.time_gain_potential:.2f}s/lap",
            action_items=action_items,
            expected_impact=expected_impact,
            confidence=tactical.confidence,
            status='new',
            timestamp=datetime.now().isoformat(),
            display_format=display_format
        )

    def _create_strategic_only_recommendation(
        self,
        strategic: StrategicRecommendation
    ) -> IntegratedRecommendation:
        """Create recommendation for strategic-only insight."""
        title = "Critical Pit Window"

        action_items = [
            {
                'role': 'STRATEGY',
                'action': strategic.message,
                'priority': 'CRITICAL'
            },
            {
                'role': 'TEAM',
                'action': "Prepare pit crew and confirm tire choice",
                'priority': 'HIGH'
            }
        ]

        expected_impact = {
            'time_gain': 0.0,
            'total_time_gain': 0.0,
            'position_gain': 0,
            'confidence': strategic.confidence
        }

        display_format = {
            'color': 'red',
            'icon': 'warning',
            'show_chain': False,
            'expanded_by_default': True
        }

        return IntegratedRecommendation(
            id=f"INT_{self._get_next_id()}",
            priority=1,
            title=title,
            tactical_component="N/A",
            strategic_component=strategic.message,
            chain_of_impact=f"Critical pit window: Lap {strategic.optimal_lap}",
            action_items=action_items,
            expected_impact=expected_impact,
            confidence=strategic.confidence,
            status='new',
            timestamp=datetime.now().isoformat(),
            display_format=display_format
        )

    def _format_recommendation_for_display(
        self,
        rec: IntegratedRecommendation
    ) -> Dict[str, Any]:
        """Format a single recommendation for dashboard display."""
        return {
            'id': rec.id,
            'priority': rec.priority,
            'priority_label': self._get_priority_label(rec.priority),
            'title': rec.title,
            'chain_of_impact': rec.chain_of_impact,
            'tactical': rec.tactical_component,
            'strategic': rec.strategic_component,
            'action_items': rec.action_items,
            'expected_impact': {
                'time_per_lap': f"{rec.expected_impact['time_gain']:.2f}s",
                'total_time': f"{rec.expected_impact['total_time_gain']:.1f}s",
                'positions': f"P{rec.expected_impact['position_gain']}" if rec.expected_impact['position_gain'] > 0 else "N/A"
            },
            'confidence': f"{rec.confidence*100:.0f}%",
            'status': rec.status,
            'timestamp': rec.timestamp,
            'display': rec.display_format
        }

    def _generate_summary(self, recommendations: List[IntegratedRecommendation]) -> Dict[str, Any]:
        """Generate executive summary of recommendations."""
        total_time_gain = sum(r.expected_impact['total_time_gain'] for r in recommendations)
        max_position_gain = max(
            (r.expected_impact['position_gain'] for r in recommendations),
            default=0
        )

        return {
            'total_recommendations': len(recommendations),
            'total_time_gain_potential': f"{total_time_gain:.1f}s",
            'max_position_gain': f"P{max_position_gain}",
            'high_priority_count': len([r for r in recommendations if r.priority <= 2]),
            'avg_confidence': f"{np.mean([r.confidence for r in recommendations])*100:.0f}%" if recommendations else "N/A"
        }

    def _generate_quick_actions(self, top_recommendations: List[IntegratedRecommendation]) -> List[str]:
        """Generate quick action list from top recommendations."""
        actions = []
        for rec in top_recommendations:
            # Get the highest priority action from each recommendation
            driver_actions = [
                a['action'] for a in rec.action_items
                if a['role'] == 'DRIVER' and a['priority'] in ['CRITICAL', 'HIGH']
            ]
            if driver_actions:
                actions.append(driver_actions[0])
        return actions

    def _calculate_tactical_priority(self, time_gain: float) -> int:
        """Calculate priority (1-5) for tactical recommendation."""
        if time_gain > 0.8:
            return 1
        elif time_gain > 0.5:
            return 2
        elif time_gain > 0.3:
            return 3
        elif time_gain > 0.1:
            return 4
        else:
            return 5

    def _get_priority_label(self, priority: int) -> str:
        """Get human-readable priority label."""
        labels = {
            1: 'CRITICAL',
            2: 'HIGH',
            3: 'MEDIUM',
            4: 'LOW',
            5: 'MINIMAL'
        }
        return labels.get(priority, 'UNKNOWN')

    def _get_priority_color(self, priority: int) -> str:
        """Get color code for priority level."""
        colors = {
            1: 'red',
            2: 'orange',
            3: 'yellow',
            4: 'blue',
            5: 'gray'
        }
        return colors.get(priority, 'gray')

    def _get_priority_breakdown(self, recommendations: List[IntegratedRecommendation]) -> Dict[str, int]:
        """Get breakdown of recommendations by priority."""
        breakdown = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'minimal': 0
        }

        for rec in recommendations:
            if rec.priority == 1:
                breakdown['critical'] += 1
            elif rec.priority == 2:
                breakdown['high'] += 1
            elif rec.priority == 3:
                breakdown['medium'] += 1
            elif rec.priority == 4:
                breakdown['low'] += 1
            else:
                breakdown['minimal'] += 1

        return breakdown

    def _get_next_id(self) -> str:
        """Generate next recommendation ID."""
        self.recommendation_counter += 1
        return f"{self.recommendation_counter:04d}_{datetime.now().strftime('%H%M%S')}"


# Utility functions for external use

def format_recommendation_json(recommendation: IntegratedRecommendation) -> str:
    """Format recommendation as JSON string."""
    return json.dumps(asdict(recommendation), indent=2)


def export_recommendations_csv(recommendations: List[IntegratedRecommendation]) -> str:
    """Export recommendations as CSV format."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        'ID', 'Priority', 'Title', 'Tactical', 'Strategic',
        'Time Gain (s/lap)', 'Position Gain', 'Confidence', 'Status'
    ])

    # Data rows
    for rec in recommendations:
        writer.writerow([
            rec.id,
            rec.priority,
            rec.title,
            rec.tactical_component,
            rec.strategic_component,
            f"{rec.expected_impact['time_gain']:.2f}",
            rec.expected_impact['position_gain'],
            f"{rec.confidence:.2f}",
            rec.status
        ])

    return output.getvalue()
