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
