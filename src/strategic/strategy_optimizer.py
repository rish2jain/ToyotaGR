"""
Pit Strategy Optimization Module for RaceIQ Pro

This module provides advanced race strategy optimization using Monte Carlo simulation,
undercut/overcut analysis, and Bayesian uncertainty quantification.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class PitStrategyOptimizer:
    """
    Advanced pit strategy optimizer using Monte Carlo simulation and
    multi-objective optimization.

    Features:
    - Optimal pit window calculation balancing tire deg vs track position
    - Monte Carlo simulation for strategy robustness
    - Undercut/overcut opportunity analysis
    - Bayesian uncertainty quantification
    - Expected position gain/loss calculation
    """

    def __init__(self, pit_loss_seconds: float = 25.0,
                 simulation_iterations: int = 100,
                 uncertainty_model: str = 'gaussian'):
        """
        Initialize the pit strategy optimizer.

        Args:
            pit_loss_seconds: Estimated time loss for pit stop (default: 25s)
            simulation_iterations: Number of Monte Carlo iterations (default: 100)
            uncertainty_model: Type of uncertainty model ('gaussian' or 'bayesian')
        """
        self.pit_loss_seconds = pit_loss_seconds
        self.simulation_iterations = simulation_iterations
        self.uncertainty_model = uncertainty_model

    def _parse_lap_time(self, lap_time: str) -> float:
        """Convert lap time from MM:SS.SSS format to seconds."""
        if pd.isna(lap_time) or lap_time == '':
            return np.nan

        try:
            if isinstance(lap_time, (int, float)):
                return float(lap_time)

            parts = str(lap_time).split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(lap_time)
        except:
            return np.nan

    def _estimate_future_lap_times(self, tire_model, current_lap: int,
                                   race_length: int) -> np.ndarray:
        """
        Estimate future lap times based on tire degradation model.

        Args:
            tire_model: Fitted tire degradation model
            current_lap: Current lap number
            race_length: Total race laps

        Returns:
            Array of predicted lap times
        """
        future_laps = np.arange(current_lap, race_length + 1)

        if tire_model is None or 'fitted_curve' not in tire_model:
            # Use conservative estimate
            baseline = tire_model.get('baseline_lap_time', 100.0) if tire_model else 100.0
            degradation_rate = tire_model.get('degradation_rate', 0.05) if tire_model else 0.05

            laps_on_tires = future_laps - current_lap
            future_times = baseline + degradation_rate * laps_on_tires
        else:
            # Use fitted model
            if tire_model['model_type'] == 'polynomial':
                future_times = np.polyval(tire_model['fitted_curve'], future_laps)
            elif tire_model['model_type'] == 'exponential':
                a, b, c = tire_model['fitted_curve']
                future_times = a + b * np.exp(c * future_laps)
            else:
                # Spline or other - use linear extrapolation
                baseline = tire_model['baseline_lap_time']
                degradation_rate = tire_model['degradation_rate']
                laps_on_tires = future_laps - current_lap
                future_times = baseline + degradation_rate * laps_on_tires

        return future_times

    def _simulate_stint(self, pit_lap: int, race_length: int,
                       tire_model: Dict, noise_std: float = 0.2) -> float:
        """
        Simulate a single race stint with given pit lap strategy.

        Args:
            pit_lap: Lap number to pit
            race_length: Total race laps
            tire_model: Tire degradation model parameters
            noise_std: Standard deviation for lap time noise (seconds)

        Returns:
            Total race time
        """
        total_time = 0.0

        # Pre-pit stint on old tires
        baseline_lap = tire_model.get('baseline_lap_time', 100.0)
        degradation_rate = tire_model.get('degradation_rate', 0.05)

        for lap in range(1, pit_lap + 1):
            # Calculate lap time with degradation
            lap_time = baseline_lap + degradation_rate * lap
            # Add random noise
            lap_time += np.random.normal(0, noise_std)
            total_time += lap_time

        # Add pit stop time
        total_time += self.pit_loss_seconds

        # Post-pit stint on fresh tires
        laps_after_pit = race_length - pit_lap

        for lap in range(1, laps_after_pit + 1):
            # Fresh tires, restart degradation
            lap_time = baseline_lap + degradation_rate * lap
            # Add random noise
            lap_time += np.random.normal(0, noise_std)
            total_time += lap_time

        return total_time

    def calculate_optimal_pit_window(self, race_data: pd.DataFrame,
                                    tire_model: Dict,
                                    race_length: int = 25,
                                    current_lap: int = None) -> Dict:
        """
        Calculate optimal pit window using Monte Carlo simulation.

        Balances tire degradation vs track position to find expected optimal pit lap.

        Args:
            race_data: DataFrame with current race data
            tire_model: Tire degradation model from TireDegradationModel
            race_length: Total race laps (default: 25)
            current_lap: Current lap number (if None, use last lap in data)

        Returns:
            Dictionary with optimal pit window:
            - optimal_pit_lap: Recommended pit lap
            - confidence_interval: 95% confidence interval [lower, upper]
            - expected_time_gain: Expected time saved vs alternatives
            - pit_window: Recommended window [earliest, latest]
            - simulation_results: Full simulation data
        """
        # Determine current lap
        if current_lap is None:
            if 'LAP_NUMBER' in race_data.columns:
                current_lap = race_data['LAP_NUMBER'].max()
            elif 'lap_number' in race_data.columns:
                current_lap = race_data['lap_number'].max()
            else:
                current_lap = len(race_data)

        # Define candidate pit laps
        earliest_pit = max(5, current_lap + 1)  # Don't pit too early
        latest_pit = min(race_length - 3, race_length)  # Leave time to benefit

        candidate_laps = range(earliest_pit, latest_pit + 1)

        # Run Monte Carlo simulation for each candidate lap
        simulation_results = {}

        for pit_lap in candidate_laps:
            lap_times = []

            for _ in range(self.simulation_iterations):
                total_time = self._simulate_stint(
                    pit_lap, race_length, tire_model
                )
                lap_times.append(total_time)

            lap_times = np.array(lap_times)

            simulation_results[pit_lap] = {
                'mean_time': np.mean(lap_times),
                'std_time': np.std(lap_times),
                'median_time': np.median(lap_times),
                'percentile_5': np.percentile(lap_times, 5),
                'percentile_95': np.percentile(lap_times, 95),
                'samples': lap_times
            }

        # Find optimal pit lap (minimum expected time)
        optimal_pit_lap = min(simulation_results.keys(),
                             key=lambda k: simulation_results[k]['mean_time'])

        optimal_result = simulation_results[optimal_pit_lap]

        # Calculate confidence interval
        confidence_interval = [
            optimal_result['percentile_5'],
            optimal_result['percentile_95']
        ]

        # Calculate expected time gain vs worst alternative
        all_means = [r['mean_time'] for r in simulation_results.values()]
        worst_time = max(all_means)
        expected_time_gain = worst_time - optimal_result['mean_time']

        # Define pit window (optimal +/- 1 lap, or where times are within 0.5s)
        pit_window_candidates = []
        optimal_time = optimal_result['mean_time']

        for lap, result in simulation_results.items():
            if abs(result['mean_time'] - optimal_time) < 0.5:
                pit_window_candidates.append(lap)

        if pit_window_candidates:
            pit_window = [min(pit_window_candidates), max(pit_window_candidates)]
        else:
            pit_window = [optimal_pit_lap - 1, optimal_pit_lap + 1]

        # Bayesian uncertainty estimation (optional)
        if self.uncertainty_model == 'bayesian':
            posterior_mean, posterior_std = self._bayesian_estimate(
                optimal_result['samples']
            )
        else:
            posterior_mean = optimal_result['mean_time']
            posterior_std = optimal_result['std_time']

        return {
            'optimal_pit_lap': int(optimal_pit_lap),
            'confidence_interval': confidence_interval,
            'expected_time_gain': float(expected_time_gain),
            'pit_window': pit_window,
            'optimal_expected_time': float(optimal_result['mean_time']),
            'optimal_time_uncertainty': float(optimal_result['std_time']),
            'simulation_results': {
                k: {
                    'mean': float(v['mean_time']),
                    'std': float(v['std_time']),
                    'median': float(v['median_time'])
                }
                for k, v in simulation_results.items()
            },
            'current_lap': int(current_lap),
            'laps_until_window': int(pit_window[0] - current_lap),
            'posterior_mean': float(posterior_mean),
            'posterior_std': float(posterior_std)
        }

    def _bayesian_estimate(self, samples: np.ndarray) -> Tuple[float, float]:
        """
        Bayesian estimation of mean and uncertainty.

        Uses conjugate prior (normal-inverse-gamma) for analytical solution.

        Args:
            samples: Array of simulation samples

        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        # Prior parameters (weakly informative)
        prior_mean = np.mean(samples)
        prior_precision = 0.1  # Low precision = high uncertainty
        prior_alpha = 1.0
        prior_beta = 1.0

        # Sample statistics
        n = len(samples)
        sample_mean = np.mean(samples)
        sample_var = np.var(samples, ddof=1)

        # Posterior parameters
        posterior_precision = prior_precision + n
        posterior_mean = (prior_precision * prior_mean + n * sample_mean) / posterior_precision
        posterior_alpha = prior_alpha + n / 2
        posterior_beta = prior_beta + 0.5 * np.sum((samples - sample_mean) ** 2) + \
                        (n * prior_precision / posterior_precision) * \
                        (sample_mean - prior_mean) ** 2 / 2

        # Posterior standard deviation
        posterior_std = np.sqrt(posterior_beta / (posterior_alpha - 1))

        return posterior_mean, posterior_std

    def simulate_undercut_opportunity(self, race_data: pd.DataFrame,
                                     competitor_data: Optional[pd.DataFrame] = None,
                                     gap_to_competitor: float = 2.0,
                                     pit_lap_difference: int = 1) -> Dict:
        """
        Simulate undercut opportunity - pitting earlier than competitor.

        Args:
            race_data: DataFrame with your lap data
            competitor_data: Optional DataFrame with competitor lap data
            gap_to_competitor: Current gap to competitor in seconds (default: 2.0)
            pit_lap_difference: How many laps earlier to pit (default: 1)

        Returns:
            Dictionary with undercut analysis:
            - undercut_success_probability: Probability of overtaking
            - expected_gap_after_stops: Expected gap after both pit
            - time_gained_on_track: Time gained during undercut
            - recommendation: Strategic recommendation
        """
        # Parse race data
        df = race_data.copy()

        if 'LAP_TIME' in df.columns:
            df['lap_time'] = df['LAP_TIME'].apply(self._parse_lap_time)
        elif 'lap_time' in df.columns:
            df['lap_time'] = df['lap_time'].apply(self._parse_lap_time)

        # Estimate lap time advantage on fresh tires
        recent_laps = df['lap_time'].tail(5).values
        average_current_pace = np.mean(recent_laps)

        # Estimate fresh tire pace (typically 1-2 seconds faster)
        fresh_tire_advantage = 1.5  # Conservative estimate

        # Simulate undercut scenario
        simulations = []

        for _ in range(self.simulation_iterations):
            # Scenario: You pit first, competitor continues
            your_pit_lap = len(df) + 1
            competitor_pit_lap = your_pit_lap + pit_lap_difference

            # Your outlap on fresh tires (slower due to pit exit)
            your_outlap = average_current_pace + 5.0  # Pit exit delay

            # Competitor's laps while you pit
            competitor_laps_during_pit = []
            for i in range(pit_lap_difference + 1):
                # Competitor on old tires, degrading
                lap_time = average_current_pace + np.random.normal(0.1 * i, 0.3)
                competitor_laps_during_pit.append(lap_time)

            # Your flying laps on fresh tires
            your_fresh_laps = []
            for i in range(pit_lap_difference):
                lap_time = average_current_pace - fresh_tire_advantage + np.random.normal(0, 0.3)
                your_fresh_laps.append(lap_time)

            # Calculate time gained
            your_total_time = your_outlap + sum(your_fresh_laps)
            competitor_total_time = sum(competitor_laps_during_pit)

            time_gained = competitor_total_time - your_total_time

            # Account for pit stop time
            time_gained -= self.pit_loss_seconds

            # Final gap
            final_gap = gap_to_competitor - time_gained

            simulations.append({
                'time_gained': time_gained,
                'final_gap': final_gap,
                'overtake': final_gap < 0  # Negative gap = you're ahead
            })

        simulations_df = pd.DataFrame(simulations)

        # Calculate statistics
        undercut_success_prob = simulations_df['overtake'].mean()
        expected_gap = simulations_df['final_gap'].mean()
        expected_time_gained = simulations_df['time_gained'].mean()

        # Generate recommendation
        if undercut_success_prob > 0.7:
            recommendation = "STRONG UNDERCUT: High probability of success - pit now!"
        elif undercut_success_prob > 0.5:
            recommendation = "MODERATE UNDERCUT: Reasonable chance - consider track position"
        elif undercut_success_prob > 0.3:
            recommendation = "WEAK UNDERCUT: Low probability - may want to extend stint"
        else:
            recommendation = "NO UNDERCUT: Stay out - maintain track position"

        return {
            'undercut_success_probability': float(undercut_success_prob),
            'expected_gap_after_stops': float(expected_gap),
            'time_gained_on_track': float(expected_time_gained),
            'current_gap': float(gap_to_competitor),
            'confidence_interval_gap': [
                float(simulations_df['final_gap'].quantile(0.05)),
                float(simulations_df['final_gap'].quantile(0.95))
            ],
            'recommendation': recommendation,
            'simulation_count': self.simulation_iterations,
            'risk_assessment': self._assess_undercut_risk(undercut_success_prob, expected_gap)
        }

    def _assess_undercut_risk(self, success_prob: float, expected_gap: float) -> str:
        """Generate risk assessment for undercut attempt."""
        if success_prob > 0.7 and expected_gap < -0.5:
            return "LOW RISK - High confidence in position gain"
        elif success_prob > 0.5:
            return "MODERATE RISK - Competitive opportunity with some uncertainty"
        elif expected_gap > 2.0:
            return "HIGH RISK - Likely to lose track position"
        else:
            return "VERY HIGH RISK - Strong chance of losing significant time"

    def calculate_optimal_pit_window_with_uncertainty(self, race_data: pd.DataFrame,
                                                      tire_model: Dict,
                                                      race_length: int = 25) -> Dict:
        """
        Calculate optimal pit window with Bayesian uncertainty quantification.

        Uses conjugate prior approach:
        - Prior: Normal distribution over pit lap times
        - Likelihood: Observed lap time data from Monte Carlo simulations
        - Posterior: Updated belief with uncertainty quantification

        Args:
            race_data: DataFrame with current race data
            tire_model: Tire degradation model from TireDegradationModel
            race_length: Total race laps (default: 25)

        Returns:
            Dictionary containing:
            - optimal_lap: Best estimate for pit lap (posterior mean)
            - confidence_95: 95% credible interval (lower, upper)
            - confidence_90: 90% credible interval
            - confidence_80: 80% credible interval
            - posterior_mean: Mean of posterior distribution
            - posterior_std: Standard deviation of posterior
            - uncertainty: Relative uncertainty (std/mean)
            - credible_interval: Highest density interval
            - posterior_samples: Array of samples from posterior distribution
            - risk_assessment: Qualitative risk assessment
        """
        # Determine current lap
        if 'LAP_NUMBER' in race_data.columns:
            current_lap = race_data['LAP_NUMBER'].max()
        elif 'lap_number' in race_data.columns:
            current_lap = race_data['lap_number'].max()
        else:
            current_lap = len(race_data)

        # Prior parameters (from racing experience/historical data)
        prior_mean = race_length * 0.6  # Typically pit around 60% of race
        prior_std = 3.0  # Uncertain by ~3 laps

        # Define candidate pit laps
        earliest_pit = max(5, current_lap + 1)
        latest_pit = min(race_length - 3, race_length)
        candidate_laps = np.arange(earliest_pit, latest_pit + 1)

        # Run Monte Carlo simulations for likelihood
        simulation_results = self._run_simulations(
            race_data, tire_model, race_length, candidate_laps
        )

        # Update posterior using Bayesian inference
        posterior_mean, posterior_std = self._update_posterior(
            prior_mean, prior_std, simulation_results
        )

        # Calculate credible intervals
        confidence_95 = stats.norm.interval(0.95, posterior_mean, posterior_std)
        confidence_90 = stats.norm.interval(0.90, posterior_mean, posterior_std)
        confidence_80 = stats.norm.interval(0.80, posterior_mean, posterior_std)

        # Generate posterior samples for visualization
        posterior_samples = stats.norm.rvs(
            loc=posterior_mean,
            scale=posterior_std,
            size=1000,
            random_state=42
        )

        # Clip samples to valid lap range
        posterior_samples = np.clip(posterior_samples, earliest_pit, latest_pit)

        # Calculate relative uncertainty
        relative_uncertainty = posterior_std / posterior_mean

        # Risk assessment based on posterior spread
        risk_assessment = self._assess_uncertainty_risk(
            posterior_std, relative_uncertainty, simulation_results
        )

        return {
            'optimal_lap': int(np.round(posterior_mean)),
            'confidence_95': (int(confidence_95[0]), int(confidence_95[1])),
            'confidence_90': (int(confidence_90[0]), int(confidence_90[1])),
            'confidence_80': (int(confidence_80[0]), int(confidence_80[1])),
            'posterior_mean': float(posterior_mean),
            'posterior_std': float(posterior_std),
            'uncertainty': float(relative_uncertainty),
            'credible_interval': (int(confidence_95[0]), int(confidence_95[1])),
            'posterior_samples': posterior_samples.tolist(),
            'risk_assessment': risk_assessment,
            'current_lap': int(current_lap),
            'earliest_pit': int(earliest_pit),
            'latest_pit': int(latest_pit),
            'simulation_results': simulation_results
        }

    def _run_simulations(self, race_data: pd.DataFrame, tire_model: Dict,
                        race_length: int, candidate_laps: np.ndarray) -> Dict:
        """
        Run Monte Carlo simulations for each candidate pit lap.

        Args:
            race_data: Current race data
            tire_model: Tire degradation model
            race_length: Total race laps
            candidate_laps: Array of candidate pit laps

        Returns:
            Dictionary mapping lap numbers to race times
        """
        simulation_results = {}

        for pit_lap in candidate_laps:
            race_times = []

            for _ in range(self.simulation_iterations):
                total_time = self._simulate_stint(
                    pit_lap, race_length, tire_model
                )
                race_times.append(total_time)

            race_times = np.array(race_times)

            simulation_results[int(pit_lap)] = {
                'mean': float(np.mean(race_times)),
                'std': float(np.std(race_times)),
                'samples': race_times.tolist()
            }

        return simulation_results

    def _update_posterior(self, prior_mean: float, prior_std: float,
                         simulation_results: Dict) -> Tuple[float, float]:
        """
        Update posterior distribution using conjugate normal-normal model.

        Uses weighted average of prior and likelihood based on simulation results.

        Args:
            prior_mean: Prior mean for optimal pit lap
            prior_std: Prior standard deviation
            simulation_results: Dictionary of simulation results per lap

        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        # Convert simulation results to likelihood
        # Find lap with minimum expected time (maximum likelihood)
        optimal_lap = min(simulation_results.keys(),
                         key=lambda k: simulation_results[k]['mean'])

        # Likelihood parameters from simulation
        likelihood_mean = float(optimal_lap)

        # Estimate likelihood precision from simulation variance
        # Lower variance in optimal lap -> higher confidence
        optimal_std = simulation_results[optimal_lap]['std']

        # Calculate spread of optimal laps across simulations
        # (how sensitive is optimal choice to randomness)
        lap_means = [(lap, data['mean']) for lap, data in simulation_results.items()]
        lap_means_sorted = sorted(lap_means, key=lambda x: x[1])

        # Find laps within 1 second of optimal
        optimal_time = lap_means_sorted[0][1]
        competitive_laps = [lap for lap, time in lap_means if time - optimal_time < 1.0]

        if len(competitive_laps) > 1:
            likelihood_std = float(np.std(competitive_laps))
        else:
            likelihood_std = 1.0  # Very certain

        # Conjugate normal-normal update
        # Precision = 1 / variance
        prior_precision = 1 / (prior_std ** 2)
        likelihood_precision = 1 / (likelihood_std ** 2)

        # Posterior precision
        posterior_precision = prior_precision + likelihood_precision
        posterior_variance = 1 / posterior_precision
        posterior_std = np.sqrt(posterior_variance)

        # Posterior mean (weighted average)
        posterior_mean = (prior_precision * prior_mean +
                         likelihood_precision * likelihood_mean) / posterior_precision

        return posterior_mean, posterior_std

    def _assess_uncertainty_risk(self, posterior_std: float,
                                relative_uncertainty: float,
                                simulation_results: Dict) -> str:
        """
        Assess risk based on posterior uncertainty.

        Args:
            posterior_std: Posterior standard deviation
            relative_uncertainty: Relative uncertainty (std/mean)
            simulation_results: Simulation results

        Returns:
            Risk assessment string
        """
        if posterior_std < 1.0:
            risk_level = "LOW"
            explanation = "Optimal pit window is well-defined with high confidence"
        elif posterior_std < 2.0:
            risk_level = "MODERATE"
            explanation = "Reasonable confidence in pit window, some timing flexibility"
        elif posterior_std < 3.0:
            risk_level = "ELEVATED"
            explanation = "Significant uncertainty - monitor tire degradation closely"
        else:
            risk_level = "HIGH"
            explanation = "Large uncertainty - pit timing highly sensitive to conditions"

        # Check simulation result spread
        all_means = [data['mean'] for data in simulation_results.values()]
        time_spread = max(all_means) - min(all_means)

        if time_spread < 1.0:
            strategy_note = "Pit timing not critical - minimal time difference"
        elif time_spread < 3.0:
            strategy_note = "Moderate advantage from optimal timing"
        else:
            strategy_note = "Critical to hit optimal window - large time advantage"

        return {
            'risk_level': risk_level,
            'explanation': explanation,
            'strategy_note': strategy_note,
            'posterior_std': float(posterior_std),
            'relative_uncertainty': float(relative_uncertainty),
            'time_spread_seconds': float(time_spread)
        }

    def visualize_posterior_distribution(self, posterior_results: Dict) -> Dict:
        """
        Create visualization data for posterior distribution.

        Returns plotly-compatible data structures for:
        - Violin plot of posterior distribution
        - Confidence interval bars
        - Risk assessment visualization

        Args:
            posterior_results: Results from calculate_optimal_pit_window_with_uncertainty

        Returns:
            Dictionary with visualization data
        """
        posterior_samples = np.array(posterior_results['posterior_samples'])

        # Create histogram for distribution visualization
        hist, bin_edges = np.histogram(posterior_samples, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate theoretical PDF
        x_range = np.linspace(
            posterior_results['earliest_pit'],
            posterior_results['latest_pit'],
            100
        )
        pdf = stats.norm.pdf(
            x_range,
            posterior_results['posterior_mean'],
            posterior_results['posterior_std']
        )

        visualization_data = {
            'histogram': {
                'bin_centers': bin_centers.tolist(),
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'pdf': {
                'x': x_range.tolist(),
                'y': pdf.tolist()
            },
            'samples': posterior_samples.tolist(),
            'confidence_intervals': {
                '95%': posterior_results['confidence_95'],
                '90%': posterior_results['confidence_90'],
                '80%': posterior_results['confidence_80']
            },
            'optimal_lap': posterior_results['optimal_lap'],
            'posterior_mean': posterior_results['posterior_mean'],
            'risk_assessment': posterior_results['risk_assessment']
        }

        return visualization_data

    def analyze_overcut_opportunity(self, race_data: pd.DataFrame,
                                   competitor_pit_lap: int,
                                   tire_model: Dict,
                                   gap_to_competitor: float = 2.0) -> Dict:
        """
        Analyze overcut opportunity - staying out longer than competitor.

        Args:
            race_data: DataFrame with your lap data
            competitor_pit_lap: Lap when competitor pitted
            tire_model: Your tire degradation model
            gap_to_competitor: Current gap in seconds

        Returns:
            Dictionary with overcut analysis
        """
        current_lap = len(race_data)
        laps_staying_out = current_lap - competitor_pit_lap

        # Estimate your pace on old tires
        degradation_rate = tire_model.get('degradation_rate', 0.05)
        baseline = tire_model.get('baseline_lap_time', 100.0)

        your_lap_times = []
        for lap in range(laps_staying_out):
            lap_time = baseline + degradation_rate * (current_lap + lap)
            your_lap_times.append(lap_time)

        # Estimate competitor's pace on fresh tires
        competitor_fresh_pace = baseline  # Fresh tires

        # Calculate time delta
        your_total_time = sum(your_lap_times)
        competitor_total_time = competitor_fresh_pace * laps_staying_out

        time_delta = your_total_time - competitor_total_time

        # Adjust for pit stop time (competitor already paid this)
        effective_gap = gap_to_competitor + time_delta

        overcut_success = effective_gap < self.pit_loss_seconds

        if overcut_success:
            recommendation = "OVERCUT WORKING - Stay out to maximize advantage"
        else:
            recommendation = "OVERCUT NOT WORKING - Pit soon to minimize time loss"

        return {
            'overcut_success': bool(overcut_success),
            'laps_stayed_out': int(laps_staying_out),
            'time_delta': float(time_delta),
            'effective_gap': float(effective_gap),
            'time_until_critical': float(max(0, self.pit_loss_seconds - effective_gap)),
            'recommendation': recommendation
        }
