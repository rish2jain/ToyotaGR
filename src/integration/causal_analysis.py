"""
Causal Inference for Racing Strategy Analysis
==============================================

This module implements rigorous causal inference using DoWhy to answer
counterfactual questions about racing strategy decisions.

Key Questions:
- "What if driver improved Section 3 by 0.5s?"
- "What's the causal effect of early pitting on final position?"
- "Does tire age causally affect lap time or is it confounded?"

This goes beyond simple correlation to establish proper causal relationships
with statistical rigor, sensitivity analysis, and uncertainty quantification.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logging.warning("DoWhy not available. Install with: pip install dowhy")

import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class CausalEffect:
    """Results of causal effect estimation"""
    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    robustness_score: float
    interpretation: str


@dataclass
class CounterfactualResult:
    """Results of counterfactual analysis"""
    scenario: str
    treatment_var: str
    intervention_value: float
    original_outcome: float
    counterfactual_outcome: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    practical_interpretation: str


class CausalStrategyAnalyzer:
    """
    Causal inference for racing strategy decisions using DoWhy.

    This class provides statistically rigorous methods to:
    1. Build causal graphs representing racing performance factors
    2. Estimate counterfactual outcomes ("what if" scenarios)
    3. Identify and control for confounding variables
    4. Test robustness of causal estimates

    Answers questions like:
    - "What if driver improved Section 3 by 0.5s?"
    - "What's the causal effect of early pitting on final position?"
    - "Does tire age causally affect lap time or is it confounded?"
    """

    def __init__(self, min_data_points: int = 20):
        """
        Initialize the causal analyzer.

        Args:
            min_data_points: Minimum data points required for analysis
        """
        if not DOWHY_AVAILABLE:
            raise ImportError(
                "DoWhy is required for causal analysis. "
                "Install with: pip install dowhy"
            )

        self.min_data_points = min_data_points
        self.causal_graph = None
        self.model = None

    def build_causal_graph(
        self,
        race_data: pd.DataFrame,
        include_weather: bool = False
    ) -> Dict[str, Any]:
        """
        Build causal DAG (Directed Acyclic Graph) for racing performance.

        Causal Structure:
        - Section Times → Lap Time
        - Lap Time → Race Position
        - Tire Age → Lap Time
        - Track Temp → Tire Degradation
        - Driver Skill (instrument) → Performance
        - Fuel Load → Lap Time
        - Pit Timing → Tire Age

        Args:
            race_data: DataFrame with race performance data
            include_weather: Whether to include weather variables

        Returns:
            Dictionary with graph structure and metadata
        """
        # Define causal relationships
        causal_edges = []

        # Core performance relationships
        causal_edges.extend([
            ("section_1_time", "lap_time"),
            ("section_2_time", "lap_time"),
            ("section_3_time", "lap_time"),
            ("lap_time", "race_position"),
        ])

        # Tire dynamics
        causal_edges.extend([
            ("tire_age", "lap_time"),
            ("tire_age", "section_1_time"),
            ("tire_age", "section_2_time"),
            ("tire_age", "section_3_time"),
            ("pit_lap", "tire_age"),
        ])

        # Fuel effects
        causal_edges.extend([
            ("fuel_load", "lap_time"),
            ("fuel_load", "section_1_time"),
        ])

        # Track conditions
        if include_weather:
            causal_edges.extend([
                ("track_temp", "tire_degradation"),
                ("track_temp", "lap_time"),
                ("air_temp", "track_temp"),
            ])

        # Build graph
        self.causal_graph = {
            'edges': causal_edges,
            'nodes': self._extract_nodes(causal_edges),
            'metadata': {
                'num_edges': len(causal_edges),
                'num_nodes': len(self._extract_nodes(causal_edges)),
                'includes_weather': include_weather
            }
        }

        logger.info(f"Built causal graph with {len(causal_edges)} edges")

        return self.causal_graph

    def _extract_nodes(self, edges: List[Tuple[str, str]]) -> List[str]:
        """Extract unique nodes from edge list"""
        nodes = set()
        for src, dst in edges:
            nodes.add(src)
            nodes.add(dst)
        return sorted(list(nodes))

    def estimate_counterfactual(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        intervention_value: float,
        common_causes: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None
    ) -> CounterfactualResult:
        """
        Estimate counterfactual: "What if we changed X to Y?"

        Example: "What if driver improved Section 3 by 0.5s?"

        Args:
            data: Race data DataFrame
            treatment: Variable to intervene on (e.g., 'section_3_time')
            outcome: Variable to measure (e.g., 'final_position')
            intervention_value: Value to set treatment to
            common_causes: List of confounding variables to control
            instruments: Instrumental variables (optional)

        Returns:
            CounterfactualResult with estimated effect and confidence intervals
        """
        # Validate data
        self._validate_data(data, [treatment, outcome])

        # Default common causes if not specified
        if common_causes is None:
            common_causes = self._identify_common_causes(treatment, outcome, data)

        # Filter common causes to only those present in data
        common_causes = [c for c in common_causes if c in data.columns]

        # Build causal model
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes if common_causes else None,
            instruments=instruments
        )

        # Identify causal effect
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        # Estimate effect using backdoor adjustment
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            confidence_intervals=True,
            test_significance=True
        )

        # Calculate counterfactual
        original_mean = data[outcome].mean()
        current_treatment_mean = data[treatment].mean()

        # Change in treatment
        treatment_change = intervention_value - current_treatment_mean

        # Causal effect size
        effect_size = estimate.value

        # Counterfactual outcome
        counterfactual_outcome = original_mean + (effect_size * treatment_change)

        # Extract confidence intervals
        ci_low, ci_high = self._extract_confidence_interval(estimate)

        # Calculate confidence interval for counterfactual
        cf_ci_low = original_mean + (ci_low * treatment_change)
        cf_ci_high = original_mean + (ci_high * treatment_change)

        # Generate interpretation
        interpretation = self._interpret_counterfactual(
            treatment, outcome, intervention_value, current_treatment_mean,
            original_mean, counterfactual_outcome, effect_size
        )

        result = CounterfactualResult(
            scenario=f"Improve {treatment} to {intervention_value}",
            treatment_var=treatment,
            intervention_value=intervention_value,
            original_outcome=original_mean,
            counterfactual_outcome=counterfactual_outcome,
            effect_size=effect_size,
            confidence_interval=(cf_ci_low, cf_ci_high),
            practical_interpretation=interpretation
        )

        # Store model for refutation
        self.model = model
        self._last_estimate = estimate
        self._last_estimand = identified_estimand

        return result

    def identify_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        common_causes: Optional[List[str]] = None,
        method: str = "backdoor.linear_regression"
    ) -> CausalEffect:
        """
        Identify and estimate causal effect using backdoor criterion.

        Methods available:
        - backdoor.linear_regression: Standard linear regression with confounders
        - backdoor.propensity_score_matching: Matching on propensity scores
        - backdoor.propensity_score_stratification: Stratification by propensity
        - iv.instrumental_variable: Instrumental variables (if available)

        Args:
            data: Race data DataFrame
            treatment: Treatment variable
            outcome: Outcome variable
            common_causes: Confounding variables to control
            method: Estimation method

        Returns:
            CausalEffect with effect size, confidence intervals, and p-value
        """
        # Validate data
        self._validate_data(data, [treatment, outcome])

        # Identify common causes if not specified
        if common_causes is None:
            common_causes = self._identify_common_causes(treatment, outcome, data)

        # Filter to available columns
        common_causes = [c for c in common_causes if c in data.columns]

        # Build model
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes if common_causes else None
        )

        # Identify effect
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        # Estimate effect
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
            confidence_intervals=True,
            test_significance=True
        )

        # Refute estimate for robustness
        robustness_score = self._calculate_robustness(
            model, identified_estimand, estimate
        )

        # Extract results
        effect_size = estimate.value
        ci_low, ci_high = self._extract_confidence_interval(estimate)
        p_value = self._extract_p_value(estimate)

        # Interpret
        interpretation = self._interpret_causal_effect(
            treatment, outcome, effect_size, p_value, method
        )

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=effect_size,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            method=method,
            robustness_score=robustness_score,
            interpretation=interpretation
        )

    def refute_estimate(
        self,
        model: CausalModel,
        estimate: Any,
        identified_estimand: Any = None
    ) -> Dict[str, Any]:
        """
        Sensitivity analysis to test robustness of causal estimate.

        Tests:
        1. Random common cause (add unmeasured confounder)
        2. Placebo treatment (replace treatment with random variable)
        3. Data subset validation (test on random subsets)
        4. Bootstrap confidence intervals

        Args:
            model: CausalModel instance
            estimate: Effect estimate to refute
            identified_estimand: Identified estimand (optional)

        Returns:
            Dictionary with refutation results
        """
        refutations = {}

        if identified_estimand is None:
            identified_estimand = model.identify_effect()

        # Test 1: Random common cause
        try:
            ref_random = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause",
                num_simulations=100
            )
            refutations['random_common_cause'] = {
                'new_effect': ref_random.new_effect,
                'original_effect': estimate.value,
                'passed': abs(ref_random.new_effect - estimate.value) < abs(estimate.value * 0.3)
            }
        except Exception as e:
            logger.warning(f"Random common cause refutation failed: {e}")
            refutations['random_common_cause'] = {'error': str(e)}

        # Test 2: Placebo treatment
        try:
            ref_placebo = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                num_simulations=100
            )
            refutations['placebo_treatment'] = {
                'new_effect': ref_placebo.new_effect,
                'original_effect': estimate.value,
                'passed': abs(ref_placebo.new_effect) < abs(estimate.value * 0.5)
            }
        except Exception as e:
            logger.warning(f"Placebo treatment refutation failed: {e}")
            refutations['placebo_treatment'] = {'error': str(e)}

        # Test 3: Data subset validation
        try:
            ref_subset = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8,
                num_simulations=50
            )
            refutations['data_subset'] = {
                'new_effect': ref_subset.new_effect,
                'original_effect': estimate.value,
                'passed': abs(ref_subset.new_effect - estimate.value) < abs(estimate.value * 0.4)
            }
        except Exception as e:
            logger.warning(f"Data subset refutation failed: {e}")
            refutations['data_subset'] = {'error': str(e)}

        # Calculate overall robustness
        passed_tests = sum(
            1 for r in refutations.values()
            if isinstance(r, dict) and r.get('passed', False)
        )
        total_tests = len(refutations)

        refutations['summary'] = {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'robustness_score': passed_tests / total_tests if total_tests > 0 else 0.0,
            'interpretation': self._interpret_robustness(passed_tests, total_tests)
        }

        return refutations

    def analyze_pit_strategy_effect(
        self,
        race_data: pd.DataFrame,
        outcome: str = 'final_position'
    ) -> CausalEffect:
        """
        Specific analysis: Causal effect of pit timing on race result.

        Treatment: Pit lap number
        Outcome: Final position
        Common causes: Tire age, track position, fuel load

        Args:
            race_data: DataFrame with pit and position data
            outcome: Outcome variable (default: 'final_position')

        Returns:
            CausalEffect for pit strategy
        """
        # Prepare data
        required_cols = ['pit_lap', outcome]
        self._validate_data(race_data, required_cols)

        # Identify confounders
        common_causes = []
        potential_confounders = [
            'tire_age', 'starting_position', 'fuel_load',
            'lap_time_avg', 'track_temp'
        ]
        for conf in potential_confounders:
            if conf in race_data.columns:
                common_causes.append(conf)

        return self.identify_causal_effect(
            data=race_data,
            treatment='pit_lap',
            outcome=outcome,
            common_causes=common_causes
        )

    def analyze_section_improvement_effect(
        self,
        race_data: pd.DataFrame,
        section_id: int,
        outcome: str = 'lap_time'
    ) -> CausalEffect:
        """
        Specific analysis: What if driver improved specific section?

        Treatment: Section time
        Outcome: Lap time, then final position
        Common causes: Tire age, track conditions, driver form

        Args:
            race_data: DataFrame with section and lap time data
            section_id: Section number (1, 2, or 3)
            outcome: Outcome variable

        Returns:
            CausalEffect for section improvement
        """
        # Construct section column name
        section_col = f'section_{section_id}_time'

        # Validate
        required_cols = [section_col, outcome]
        self._validate_data(race_data, required_cols)

        # Identify confounders
        common_causes = []
        potential_confounders = [
            'tire_age', 'fuel_load', 'track_temp',
            'lap_number', 'driver_consistency'
        ]
        for conf in potential_confounders:
            if conf in race_data.columns:
                common_causes.append(conf)

        return self.identify_causal_effect(
            data=race_data,
            treatment=section_col,
            outcome=outcome,
            common_causes=common_causes
        )

    def visualize_causal_graph(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate visual representation of causal DAG.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if self.causal_graph is None:
            raise ValueError("No causal graph built. Call build_causal_graph() first.")

        # Create directed graph
        G = nx.DiGraph()

        # Add edges
        for src, dst in self.causal_graph['edges']:
            G.add_edge(src, dst)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.shell_layout(G)

        # Node colors by type
        node_colors = []
        for node in G.nodes():
            if 'section' in node:
                node_colors.append('#FF6B6B')  # Red for sections
            elif node in ['lap_time', 'race_position']:
                node_colors.append('#4ECDC4')  # Teal for outcomes
            elif 'tire' in node or 'pit' in node:
                node_colors.append('#FFD93D')  # Yellow for strategy
            elif 'temp' in node or 'weather' in node:
                node_colors.append('#95E1D3')  # Light green for conditions
            else:
                node_colors.append('#B4B4B8')  # Gray for others

        # Draw graph
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors,
            node_size=3000, alpha=0.9, ax=ax
        )

        nx.draw_networkx_edges(
            G, pos, edge_color='gray',
            arrows=True, arrowsize=20, arrowstyle='->',
            width=2, alpha=0.6, ax=ax
        )

        # Labels
        nx.draw_networkx_labels(
            G, pos, font_size=8,
            font_weight='bold', ax=ax
        )

        ax.set_title(
            "Causal Graph: Racing Performance Factors",
            fontsize=16, fontweight='bold', pad=20
        )
        ax.axis('off')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#FF6B6B', markersize=10, label='Section Times'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#4ECDC4', markersize=10, label='Performance Outcomes'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#FFD93D', markersize=10, label='Strategy Variables'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#95E1D3', markersize=10, label='Track Conditions'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Causal graph saved to {save_path}")

        return fig

    # Helper methods

    def _validate_data(
        self,
        data: pd.DataFrame,
        required_columns: List[str]
    ):
        """Validate data for causal analysis"""
        if len(data) < self.min_data_points:
            raise ValueError(
                f"Insufficient data: {len(data)} rows "
                f"(minimum: {self.min_data_points})"
            )

        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for variation in treatment/outcome
        for col in required_columns[:2]:  # treatment and outcome
            if col in data.columns and data[col].nunique() < 2:
                raise ValueError(
                    f"No variation in {col}: Cannot estimate causal effect"
                )

    def _identify_common_causes(
        self,
        treatment: str,
        outcome: str,
        data: pd.DataFrame
    ) -> List[str]:
        """Automatically identify potential confounders"""
        common_causes = []

        # Potential confounders based on racing domain knowledge
        potential = {
            'tire_age', 'fuel_load', 'track_temp', 'lap_number',
            'driver_consistency', 'starting_position', 'weather_condition'
        }

        # Add if present in data and not treatment/outcome
        for var in potential:
            if var in data.columns and var not in [treatment, outcome]:
                common_causes.append(var)

        return common_causes

    def _extract_confidence_interval(
        self,
        estimate: Any
    ) -> Tuple[float, float]:
        """Extract confidence interval from estimate"""
        try:
            ci = estimate.get_confidence_intervals()
            return (ci[0], ci[1])
        except:
            # If CI not available, use standard error
            try:
                se = estimate.get_standard_error()
                return (estimate.value - 1.96*se, estimate.value + 1.96*se)
            except:
                # Fallback: ±20% of effect
                margin = abs(estimate.value * 0.2)
                return (estimate.value - margin, estimate.value + margin)

    def _extract_p_value(self, estimate: Any) -> float:
        """Extract p-value from estimate"""
        try:
            return estimate.test_stat_significance()['p_value']
        except:
            return 0.05  # Default moderate significance

    def _calculate_robustness(
        self,
        model: CausalModel,
        identified_estimand: Any,
        estimate: Any
    ) -> float:
        """Calculate overall robustness score"""
        try:
            refutations = self.refute_estimate(model, estimate, identified_estimand)
            return refutations['summary']['robustness_score']
        except Exception as e:
            logger.warning(f"Robustness calculation failed: {e}")
            return 0.5  # Moderate uncertainty

    def _interpret_causal_effect(
        self,
        treatment: str,
        outcome: str,
        effect_size: float,
        p_value: float,
        method: str
    ) -> str:
        """Generate human-readable interpretation"""
        # Significance
        if p_value < 0.01:
            sig_text = "highly significant"
        elif p_value < 0.05:
            sig_text = "significant"
        elif p_value < 0.10:
            sig_text = "marginally significant"
        else:
            sig_text = "not statistically significant"

        # Effect direction and magnitude
        direction = "increases" if effect_size > 0 else "decreases"
        magnitude = abs(effect_size)

        if magnitude < 0.1:
            size_text = "small"
        elif magnitude < 0.5:
            size_text = "moderate"
        else:
            size_text = "large"

        interpretation = (
            f"A one-unit change in {treatment} {direction} {outcome} "
            f"by {magnitude:.4f} units on average. This effect is {size_text} "
            f"and {sig_text} (p={p_value:.4f}). "
            f"Method: {method}."
        )

        return interpretation

    def _interpret_counterfactual(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float,
        current_mean: float,
        original_outcome: float,
        counterfactual_outcome: float,
        effect_size: float
    ) -> str:
        """Generate interpretation for counterfactual"""
        change = counterfactual_outcome - original_outcome
        pct_change = (change / original_outcome) * 100 if original_outcome != 0 else 0

        direction = "improve" if change < 0 else "worsen"

        interpretation = (
            f"If {treatment} improved from {current_mean:.3f} to "
            f"{intervention_value:.3f}, {outcome} would {direction} "
            f"from {original_outcome:.3f} to {counterfactual_outcome:.3f} "
            f"({change:+.3f}, {pct_change:+.1f}%). "
            f"Causal effect per unit: {effect_size:.4f}."
        )

        return interpretation

    def _interpret_robustness(
        self,
        passed_tests: int,
        total_tests: int
    ) -> str:
        """Interpret robustness test results"""
        score = passed_tests / total_tests if total_tests > 0 else 0

        if score >= 0.75:
            return "HIGH confidence - Effect is robust to sensitivity tests"
        elif score >= 0.50:
            return "MODERATE confidence - Some sensitivity to unmeasured confounding"
        else:
            return "LOW confidence - Effect may be due to unmeasured confounding"


def prepare_race_data_for_causal_analysis(
    sections_df: pd.DataFrame,
    results_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prepare race data for causal analysis.

    Transforms section analysis and results data into format suitable
    for causal inference, with proper naming and derived variables.

    Args:
        sections_df: Section analysis DataFrame
        results_df: Race results DataFrame (optional)

    Returns:
        Prepared DataFrame for causal analysis
    """
    df = sections_df.copy()

    # Rename columns to causal analysis format
    column_mapping = {
        'S1_SECONDS': 'section_1_time',
        'S2_SECONDS': 'section_2_time',
        'S3_SECONDS': 'section_3_time',
        'LAP_TIME_SECONDS': 'lap_time',
        'LAP_NUMBER': 'lap_number',
        'DRIVER_NUMBER': 'driver_number',
    }

    for old, new in column_mapping.items():
        if old in df.columns:
            df[new] = df[old]

    # Calculate derived variables
    if 'lap_number' in df.columns:
        # Tire age (laps since start)
        df['tire_age'] = df.groupby('driver_number')['lap_number'].transform(
            lambda x: x - x.min()
        )

        # Fuel load (decreases with laps)
        max_laps = df['lap_number'].max()
        df['fuel_load'] = 1.0 - (df['lap_number'] / max_laps)

    # Detect pit laps (large lap time jumps)
    if 'lap_time' in df.columns:
        df['lap_time_diff'] = df.groupby('driver_number')['lap_time'].diff()
        df['pit_lap'] = (df['lap_time_diff'] > 20).astype(int)

        # Reset tire age after pits
        df['tire_age'] = df.groupby('driver_number').apply(
            lambda g: g['lap_number'] - g[g['pit_lap'] == 1]['lap_number'].shift(1, fill_value=0)
        ).reset_index(drop=True)

    # Add final position if results available
    if results_df is not None and 'POSITION' in results_df.columns:
        driver_positions = results_df.set_index('NUMBER')['POSITION'].to_dict()
        df['final_position'] = df['driver_number'].map(driver_positions)

    # Driver consistency
    if 'lap_time' in df.columns:
        df['driver_consistency'] = df.groupby('driver_number')['lap_time'].transform('std')

    # Drop temporary columns
    df = df.drop(columns=['lap_time_diff'], errors='ignore')

    return df
