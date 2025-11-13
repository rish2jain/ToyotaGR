"""
Multi-Driver Race Simulation Module for RaceIQ Pro

This module provides comprehensive multi-car race simulation with:
- Position changes and overtaking dynamics
- Undercut/overcut strategy analysis
- Team strategy optimization
- Realistic tire degradation and fuel effects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')


class MultiDriverRaceSimulator:
    """
    Simulate multi-car races with position changes and strategy interactions.

    Features:
    - Full race simulation with pit stops
    - Undercut/overcut scenario analysis
    - Team strategy optimization
    - Realistic tire degradation modeling
    - Fuel load effects
    - Overtaking logic
    """

    def __init__(self, race_length: int = 25, pit_loss_time: float = 25.0):
        """
        Initialize the multi-driver race simulator.

        Args:
            race_length: Total number of laps in the race (default: 25)
            pit_loss_time: Time loss for pit stop in seconds (default: 25.0)
        """
        self.race_length = race_length
        self.pit_loss_time = pit_loss_time
        self.track_position_penalty = 0.3  # Extra time needed to overtake on track
        self.fuel_effect = 0.03  # Seconds per lap gained from fuel burn (0.3s over 10 laps)

    def simulate_race(self, drivers_data: Dict, strategies: Dict) -> Dict:
        """
        Simulate full race with pit stops and position changes.

        Args:
            drivers_data: Dict of driver configs with structure:
                {
                    'driver_id': {
                        'name': str,
                        'base_lap_time': float (seconds),
                        'tire_deg_rate': float (seconds per lap),
                        'consistency': float (std dev of lap times, default 0.1)
                    }
                }
            strategies: Dict of pit strategies per driver:
                {
                    'driver_id': {
                        'pit_laps': [int, ...] (list of lap numbers to pit)
                    }
                }

        Returns:
            Dictionary with:
            - lap_by_lap: List of race states for each lap
            - final_results: Final positions and times
            - position_changes: History of position changes
            - strategy_effectiveness: Analysis of each driver's strategy
        """
        # Validate inputs
        if not drivers_data or not strategies:
            raise ValueError("drivers_data and strategies must not be empty")

        # Initialize race state
        race_state = self._initialize_race(drivers_data, strategies)
        lap_by_lap = []
        position_changes = []

        # Simulate each lap
        for lap in range(1, self.race_length + 1):
            # Simulate lap for each driver
            for driver_id in race_state['drivers']:
                driver = race_state['drivers'][driver_id]

                # Check if driver should pit this lap
                if lap in driver['strategy']['pit_laps'] and not driver['in_pit']:
                    driver['in_pit'] = True
                    driver['pit_lap_start'] = lap

                # Calculate lap time
                lap_time = self._simulate_lap_time(
                    driver=driver,
                    lap_number=lap,
                    total_laps=self.race_length
                )

                # Handle pit stop
                if driver['in_pit']:
                    lap_time += self.pit_loss_time
                    driver['in_pit'] = False
                    driver['tire_age'] = 0  # Fresh tires
                    driver['last_pit_lap'] = lap
                else:
                    driver['tire_age'] += 1

                # Update cumulative time
                driver['lap_times'].append(lap_time)
                driver['cumulative_time'] += lap_time
                driver['laps_completed'] += 1

            # Update positions based on cumulative times
            previous_positions = {d_id: d['position'] for d_id, d in race_state['drivers'].items()}
            race_state = self._update_positions(race_state)

            # Record position changes
            for driver_id, driver in race_state['drivers'].items():
                if driver['position'] != previous_positions[driver_id]:
                    position_changes.append({
                        'lap': lap,
                        'driver_id': driver_id,
                        'old_position': previous_positions[driver_id],
                        'new_position': driver['position']
                    })

            # Store lap state
            lap_state = {
                'lap': lap,
                'positions': {
                    driver_id: {
                        'position': driver['position'],
                        'cumulative_time': driver['cumulative_time'],
                        'last_lap_time': driver['lap_times'][-1],
                        'tire_age': driver['tire_age'],
                        'gap_to_leader': driver['cumulative_time'] - race_state['drivers'][race_state['leader']]['cumulative_time']
                    }
                    for driver_id, driver in race_state['drivers'].items()
                }
            }
            lap_by_lap.append(lap_state)

        # Calculate final results
        final_results = self._calculate_final_results(race_state)

        # Analyze strategy effectiveness
        strategy_effectiveness = self._analyze_strategy_effectiveness(
            lap_by_lap, final_results, strategies
        )

        return {
            'lap_by_lap': lap_by_lap,
            'final_results': final_results,
            'position_changes': position_changes,
            'strategy_effectiveness': strategy_effectiveness,
            'race_config': {
                'race_length': self.race_length,
                'pit_loss_time': self.pit_loss_time,
                'num_drivers': len(drivers_data)
            }
        }

    def simulate_undercut_scenario(self, driver_a_config: Dict, driver_b_config: Dict,
                                   pit_lap_a: int, pit_lap_b: int) -> Dict:
        """
        Simulate undercut: Driver A pits earlier, tries to pass B on fresh tires.

        Args:
            driver_a_config: Config for driver attempting undercut
                {'base_lap_time': float, 'tire_deg_rate': float}
            driver_b_config: Config for driver being undercut
            pit_lap_a: Lap when driver A pits (earlier)
            pit_lap_b: Lap when driver B pits (later)

        Returns:
            Dictionary with:
            - success: Did undercut work?
            - overtake_lap: When did position change occur? (None if no overtake)
            - time_delta: Final gap between drivers (negative if A ahead)
            - gap_evolution: Lap-by-lap gap
            - critical_laps: Analysis of key laps
        """
        if pit_lap_a >= pit_lap_b:
            raise ValueError("Driver A must pit earlier than Driver B for undercut")

        # Create driver data and strategies
        drivers_data = {
            'A': {
                'name': 'Driver A (Undercut)',
                'base_lap_time': driver_a_config.get('base_lap_time', 95.0),
                'tire_deg_rate': driver_a_config.get('tire_deg_rate', 0.05),
                'consistency': driver_a_config.get('consistency', 0.05)
            },
            'B': {
                'name': 'Driver B (Defending)',
                'base_lap_time': driver_b_config.get('base_lap_time', 95.0),
                'tire_deg_rate': driver_b_config.get('tire_deg_rate', 0.05),
                'consistency': driver_b_config.get('consistency', 0.05)
            }
        }

        strategies = {
            'A': {'pit_laps': [pit_lap_a]},
            'B': {'pit_laps': [pit_lap_b]}
        }

        # Run simulation
        result = self.simulate_race(drivers_data, strategies)

        # Analyze undercut success
        gap_evolution = []
        overtake_lap = None

        for lap_state in result['lap_by_lap']:
            gap_a_to_b = (lap_state['positions']['A']['cumulative_time'] -
                         lap_state['positions']['B']['cumulative_time'])
            gap_evolution.append({
                'lap': lap_state['lap'],
                'gap': gap_a_to_b,
                'position_a': lap_state['positions']['A']['position'],
                'position_b': lap_state['positions']['B']['position']
            })

            # Check for overtake
            if overtake_lap is None and lap_state['positions']['A']['position'] < lap_state['positions']['B']['position']:
                overtake_lap = lap_state['lap']

        final_gap = gap_evolution[-1]['gap']
        success = final_gap < 0  # Negative gap means A is ahead

        # Identify critical laps
        critical_laps = {
            'driver_a_pit': pit_lap_a,
            'driver_b_pit': pit_lap_b,
            'overtake_lap': overtake_lap,
            'gap_at_a_pit': next((g['gap'] for g in gap_evolution if g['lap'] == pit_lap_a), None),
            'gap_at_b_pit': next((g['gap'] for g in gap_evolution if g['lap'] == pit_lap_b), None),
            'final_gap': final_gap
        }

        return {
            'success': success,
            'overtake_lap': overtake_lap,
            'time_delta': final_gap,
            'gap_evolution': gap_evolution,
            'critical_laps': critical_laps,
            'summary': self._generate_undercut_summary(success, overtake_lap, final_gap, pit_lap_a, pit_lap_b)
        }

    def simulate_overcut_scenario(self, driver_a_config: Dict, driver_b_config: Dict,
                                  pit_lap_a: int, pit_lap_b: int) -> Dict:
        """
        Simulate overcut: Driver A stays out longer on old tires.

        Args:
            driver_a_config: Config for driver attempting overcut
            driver_b_config: Config for driver pitting earlier
            pit_lap_a: Lap when driver A pits (later)
            pit_lap_b: Lap when driver B pits (earlier)

        Returns:
            Dictionary with overcut analysis
        """
        if pit_lap_a <= pit_lap_b:
            raise ValueError("Driver A must pit later than Driver B for overcut")

        # Create driver data and strategies
        drivers_data = {
            'A': {
                'name': 'Driver A (Overcut)',
                'base_lap_time': driver_a_config.get('base_lap_time', 95.0),
                'tire_deg_rate': driver_a_config.get('tire_deg_rate', 0.05),
                'consistency': driver_a_config.get('consistency', 0.05)
            },
            'B': {
                'name': 'Driver B (Early Pit)',
                'base_lap_time': driver_b_config.get('base_lap_time', 95.0),
                'tire_deg_rate': driver_b_config.get('tire_deg_rate', 0.05),
                'consistency': driver_b_config.get('consistency', 0.05)
            }
        }

        strategies = {
            'A': {'pit_laps': [pit_lap_a]},
            'B': {'pit_laps': [pit_lap_b]}
        }

        # Run simulation
        result = self.simulate_race(drivers_data, strategies)

        # Analyze overcut success
        gap_evolution = []
        overtake_lap = None

        for lap_state in result['lap_by_lap']:
            gap_a_to_b = (lap_state['positions']['A']['cumulative_time'] -
                         lap_state['positions']['B']['cumulative_time'])
            gap_evolution.append({
                'lap': lap_state['lap'],
                'gap': gap_a_to_b,
                'position_a': lap_state['positions']['A']['position'],
                'position_b': lap_state['positions']['B']['position']
            })

            if overtake_lap is None and lap_state['positions']['A']['position'] < lap_state['positions']['B']['position']:
                overtake_lap = lap_state['lap']

        final_gap = gap_evolution[-1]['gap']
        success = final_gap < 0

        return {
            'success': success,
            'overtake_lap': overtake_lap,
            'time_delta': final_gap,
            'gap_evolution': gap_evolution,
            'laps_stayed_out': pit_lap_a - pit_lap_b,
            'summary': self._generate_overcut_summary(success, overtake_lap, final_gap, pit_lap_a, pit_lap_b)
        }

    def optimize_team_strategy(self, team_drivers: Dict, opponents: Dict,
                              objective: str = 'maximize_team_points') -> Dict:
        """
        Find optimal strategy when controlling multiple cars.

        Args:
            team_drivers: Dict of team driver configs
            opponents: Dict of opponent driver configs
            objective: Optimization objective:
                - 'maximize_team_points': Maximize combined points
                - 'guarantee_win': Ensure at least one car wins
                - 'block_opponents': Use team tactics to hinder opponents

        Returns:
            Dictionary with optimal team strategy
        """
        # Combine all drivers
        all_drivers = {**team_drivers, **opponents}

        # Define candidate strategies for team drivers
        candidate_pit_laps = [8, 10, 12, 14, 16]

        best_result = None
        best_score = -float('inf')
        best_strategies = None

        # Try different strategy combinations for team
        team_ids = list(team_drivers.keys())

        if len(team_ids) == 1:
            # Single car - optimize for best result
            for pit_lap in candidate_pit_laps:
                strategies = {team_ids[0]: {'pit_laps': [pit_lap]}}

                # Default strategies for opponents (mid-race pit)
                for opp_id in opponents:
                    strategies[opp_id] = {'pit_laps': [12]}

                result = self.simulate_race(all_drivers, strategies)
                score = self._calculate_team_score(result, team_ids, objective)

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_strategies = strategies

        elif len(team_ids) == 2:
            # Two cars - try split strategies
            for pit_lap_1 in candidate_pit_laps:
                for pit_lap_2 in candidate_pit_laps:
                    strategies = {
                        team_ids[0]: {'pit_laps': [pit_lap_1]},
                        team_ids[1]: {'pit_laps': [pit_lap_2]}
                    }

                    # Default strategies for opponents
                    for opp_id in opponents:
                        strategies[opp_id] = {'pit_laps': [12]}

                    result = self.simulate_race(all_drivers, strategies)
                    score = self._calculate_team_score(result, team_ids, objective)

                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_strategies = strategies
        else:
            # Multiple team cars - simplified optimization
            for pit_lap in candidate_pit_laps:
                strategies = {}
                for team_id in team_ids:
                    # Stagger pit stops slightly
                    offset = team_ids.index(team_id) * 2
                    strategies[team_id] = {'pit_laps': [min(pit_lap + offset, self.race_length - 3)]}

                for opp_id in opponents:
                    strategies[opp_id] = {'pit_laps': [12]}

                result = self.simulate_race(all_drivers, strategies)
                score = self._calculate_team_score(result, team_ids, objective)

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_strategies = strategies

        return {
            'optimal_strategies': best_strategies,
            'expected_result': best_result,
            'team_score': best_score,
            'team_positions': [best_result['final_results'][i]['driver_id']
                             for i in range(len(best_result['final_results']))
                             if best_result['final_results'][i]['driver_id'] in team_ids],
            'recommendation': self._generate_team_strategy_recommendation(
                best_strategies, team_ids, objective
            )
        }

    def _simulate_lap_time(self, driver: Dict, lap_number: int,
                          total_laps: int) -> float:
        """
        Calculate lap time with tire degradation and fuel effect.

        Args:
            driver: Driver state dictionary
            lap_number: Current lap number
            total_laps: Total race laps

        Returns:
            Simulated lap time in seconds
        """
        # Base lap time
        base_time = driver['base_lap_time']

        # Tire degradation effect (increases with tire age)
        tire_age = driver['tire_age']
        tire_effect = driver['tire_deg_rate'] * tire_age

        # Fuel effect (car gets lighter as race progresses)
        laps_completed = lap_number - 1
        fuel_remaining = 1.0 - (laps_completed / total_laps)
        fuel_benefit = (1.0 - fuel_remaining) * self.fuel_effect * total_laps / 10  # 0.3s over 10 laps

        # Random variation (driver consistency)
        consistency = driver.get('consistency', 0.1)
        random_variation = np.random.normal(0, consistency)

        # Calculate final lap time
        lap_time = base_time + tire_effect - fuel_benefit + random_variation

        # Ensure lap time is reasonable (minimum 90% of base time)
        lap_time = max(lap_time, base_time * 0.9)

        return lap_time

    def _update_positions(self, race_state: Dict) -> Dict:
        """
        Update race positions based on cumulative times.

        Args:
            race_state: Current race state

        Returns:
            Updated race state with new positions
        """
        # Sort drivers by cumulative time
        sorted_drivers = sorted(
            race_state['drivers'].items(),
            key=lambda x: x[1]['cumulative_time']
        )

        # Update positions
        for position, (driver_id, driver) in enumerate(sorted_drivers, start=1):
            race_state['drivers'][driver_id]['position'] = position

        # Update leader
        race_state['leader'] = sorted_drivers[0][0]

        return race_state

    def _check_overtake(self, driver_a_time: float, driver_b_time: float,
                       gap: float) -> bool:
        """
        Determine if overtake occurred.

        Args:
            driver_a_time: Driver A's lap time
            driver_b_time: Driver B's lap time
            gap: Current gap in seconds

        Returns:
            True if overtake successful
        """
        # Calculate time delta
        time_delta = driver_a_time - driver_b_time

        # Overtake occurs if gap closes to within 1.0 second
        # and pursuing driver is faster
        new_gap = gap + time_delta

        if new_gap < 1.0 and time_delta < 0:
            # Check if pursuing driver has enough pace advantage
            # Need extra pace to overcome track position advantage
            pace_advantage = abs(time_delta)
            return pace_advantage > self.track_position_penalty

        return False

    def _initialize_race(self, drivers_data: Dict, strategies: Dict) -> Dict:
        """Initialize race state with all drivers."""
        race_state = {
            'drivers': {},
            'leader': None,
            'current_lap': 0
        }

        for position, (driver_id, driver_config) in enumerate(drivers_data.items(), start=1):
            race_state['drivers'][driver_id] = {
                'id': driver_id,
                'name': driver_config.get('name', driver_id),
                'base_lap_time': driver_config['base_lap_time'],
                'tire_deg_rate': driver_config['tire_deg_rate'],
                'consistency': driver_config.get('consistency', 0.1),
                'position': position,
                'cumulative_time': 0.0,
                'lap_times': [],
                'tire_age': 0,
                'laps_completed': 0,
                'strategy': strategies[driver_id],
                'in_pit': False,
                'last_pit_lap': 0,
                'pit_lap_start': None
            }

        # Set initial leader (P1)
        race_state['leader'] = list(drivers_data.keys())[0]

        return race_state

    def _calculate_final_results(self, race_state: Dict) -> List[Dict]:
        """Calculate final race results."""
        sorted_drivers = sorted(
            race_state['drivers'].values(),
            key=lambda x: x['cumulative_time']
        )

        results = []
        leader_time = sorted_drivers[0]['cumulative_time']

        for position, driver in enumerate(sorted_drivers, start=1):
            gap = driver['cumulative_time'] - leader_time

            results.append({
                'position': position,
                'driver_id': driver['id'],
                'driver_name': driver['name'],
                'total_time': driver['cumulative_time'],
                'gap_to_leader': gap,
                'laps_completed': driver['laps_completed'],
                'pit_stops': len(driver['strategy']['pit_laps']),
                'avg_lap_time': driver['cumulative_time'] / driver['laps_completed'] if driver['laps_completed'] > 0 else 0
            })

        return results

    def _analyze_strategy_effectiveness(self, lap_by_lap: List[Dict],
                                       final_results: List[Dict],
                                       strategies: Dict) -> Dict:
        """Analyze effectiveness of each driver's strategy."""
        effectiveness = {}

        for result in final_results:
            driver_id = result['driver_id']
            strategy = strategies[driver_id]

            # Calculate positions gained/lost
            initial_position = 1  # Assume grid position based on entry order
            final_position = result['position']
            positions_delta = initial_position - final_position

            # Analyze pit timing
            pit_laps = strategy['pit_laps']
            pit_analysis = []

            for pit_lap in pit_laps:
                # Find position before and after pit
                lap_before = next((l for l in lap_by_lap if l['lap'] == pit_lap - 1), None)
                lap_after = next((l for l in lap_by_lap if l['lap'] == pit_lap + 1), None)

                if lap_before and lap_after:
                    pos_before = lap_before['positions'][driver_id]['position']
                    pos_after = lap_after['positions'][driver_id]['position']

                    pit_analysis.append({
                        'pit_lap': pit_lap,
                        'position_before': pos_before,
                        'position_after': pos_after,
                        'positions_lost': pos_after - pos_before
                    })

            effectiveness[driver_id] = {
                'final_position': final_position,
                'positions_delta': positions_delta,
                'pit_stops': len(pit_laps),
                'pit_laps': pit_laps,
                'pit_analysis': pit_analysis,
                'total_time': result['total_time'],
                'avg_lap_time': result['avg_lap_time']
            }

        return effectiveness

    def _calculate_team_score(self, result: Dict, team_ids: List[str],
                             objective: str) -> float:
        """Calculate team score based on objective."""
        final_results = result['final_results']

        if objective == 'maximize_team_points':
            # Points system: 25, 18, 15, 12, 10, 8, 6, 4, 2, 1 (F1-style)
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            total_points = 0

            for res in final_results:
                if res['driver_id'] in team_ids:
                    total_points += points_map.get(res['position'], 0)

            return total_points

        elif objective == 'guarantee_win':
            # Maximize chance of winning
            winner = final_results[0]['driver_id']
            if winner in team_ids:
                return 100.0
            else:
                # Score based on highest team position
                best_team_position = min(
                    (res['position'] for res in final_results if res['driver_id'] in team_ids),
                    default=999
                )
                return 100.0 / best_team_position

        elif objective == 'block_opponents':
            # Minimize opponent scores
            opponent_score = 0
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

            for res in final_results:
                if res['driver_id'] not in team_ids:
                    opponent_score += points_map.get(res['position'], 0)

            # Return negative opponent score (we want to minimize it)
            return -opponent_score

        return 0.0

    def _generate_undercut_summary(self, success: bool, overtake_lap: Optional[int],
                                  final_gap: float, pit_lap_a: int, pit_lap_b: int) -> str:
        """Generate human-readable undercut summary."""
        if success:
            if overtake_lap:
                return (f"Undercut SUCCESSFUL! Driver A overtook on lap {overtake_lap}, "
                       f"{pit_lap_b - overtake_lap} laps after pitting. "
                       f"Final advantage: {abs(final_gap):.2f}s")
            else:
                return f"Undercut SUCCESSFUL! Final advantage: {abs(final_gap):.2f}s"
        else:
            return (f"Undercut FAILED. Driver B held position. "
                   f"Final deficit: {final_gap:.2f}s")

    def _generate_overcut_summary(self, success: bool, overtake_lap: Optional[int],
                                 final_gap: float, pit_lap_a: int, pit_lap_b: int) -> str:
        """Generate human-readable overcut summary."""
        laps_stayed_out = pit_lap_a - pit_lap_b

        if success:
            return (f"Overcut SUCCESSFUL! Stayed out {laps_stayed_out} extra laps. "
                   f"Final advantage: {abs(final_gap):.2f}s")
        else:
            return (f"Overcut FAILED. Staying out {laps_stayed_out} laps not enough. "
                   f"Final deficit: {final_gap:.2f}s")

    def _generate_team_strategy_recommendation(self, strategies: Dict,
                                              team_ids: List[str],
                                              objective: str) -> str:
        """Generate team strategy recommendation."""
        team_pit_laps = [strategies[tid]['pit_laps'][0] for tid in team_ids]

        if len(team_ids) == 1:
            return f"Optimal single-car strategy: Pit on lap {team_pit_laps[0]}"

        elif len(team_ids) == 2:
            if team_pit_laps[0] == team_pit_laps[1]:
                return f"Optimal team strategy: Both cars pit together on lap {team_pit_laps[0]}"
            else:
                early_pit = min(team_pit_laps)
                late_pit = max(team_pit_laps)
                return (f"Optimal split strategy: First car pits lap {early_pit}, "
                       f"second car pits lap {late_pit} for undercut/overcut pressure")

        else:
            return f"Optimal team strategy: Staggered pit stops on laps {sorted(team_pit_laps)}"
