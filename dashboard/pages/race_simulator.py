"""
Race Simulator Page
Multi-driver race simulation with undercut/overcut analysis and team strategy optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from strategic.race_simulation import MultiDriverRaceSimulator


def show_race_simulator(data, track, race_num):
    """
    Display multi-driver race simulation interface.

    Args:
        data: Dictionary containing race dataframes
        track: Track name
        race_num: Race number
    """
    st.title(f"🏁 Multi-Driver Race Simulator: {track.replace('-', ' ').title()}")
    st.markdown("""
    Simulate multi-car races with realistic tire degradation, pit strategies, and position battles.
    Test undercut/overcut scenarios and optimize team strategies.
    """)

    # Create tabs for different simulation modes
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏎️ Race Animation",
        "⚡ Undercut Analyzer",
        "🎯 Strategy Optimizer",
        "🔧 What-If Scenarios"
    ])

    with tab1:
        show_race_animation()

    with tab2:
        show_undercut_analyzer()

    with tab3:
        show_strategy_optimizer()

    with tab4:
        show_whatif_scenarios()


def show_race_animation():
    """Display full race simulation with position changes over time."""
    st.header("Full Race Simulation")

    # Configuration section
    st.subheader("Race Configuration")

    col1, col2 = st.columns(2)

    with col1:
        race_length = st.number_input("Race Length (laps)", min_value=10, max_value=50, value=25)
        pit_loss_time = st.number_input("Pit Stop Time Loss (seconds)", min_value=15.0, max_value=35.0, value=25.0, step=0.5)

    with col2:
        num_drivers = st.selectbox("Number of Drivers", [2, 3, 4, 5, 6, 8, 10], index=2)

    st.markdown("---")

    # Driver configuration
    st.subheader("Driver Configuration")

    # Use preset scenarios or custom
    preset = st.selectbox(
        "Load Preset Scenario",
        ["Custom", "2024 Barber Race 1 (Top 5)", "Close Battle (3 drivers)", "Strategy Mix (5 drivers)"]
    )

    drivers_data = {}
    strategies = {}

    if preset == "2024 Barber Race 1 (Top 5)":
        # Realistic preset based on actual race data
        drivers_data = {
            '30': {'name': 'Car #30', 'base_lap_time': 93.5, 'tire_deg_rate': 0.04, 'consistency': 0.08},
            '32': {'name': 'Car #32', 'base_lap_time': 93.7, 'tire_deg_rate': 0.05, 'consistency': 0.10},
            '21': {'name': 'Car #21', 'base_lap_time': 93.9, 'tire_deg_rate': 0.045, 'consistency': 0.09},
            '14': {'name': 'Car #14', 'base_lap_time': 94.0, 'tire_deg_rate': 0.05, 'consistency': 0.11},
            '8': {'name': 'Car #8', 'base_lap_time': 94.2, 'tire_deg_rate': 0.048, 'consistency': 0.10}
        }
        strategies = {
            '30': {'pit_laps': [12]},
            '32': {'pit_laps': [10]},
            '21': {'pit_laps': [14]},
            '14': {'pit_laps': [11]},
            '8': {'pit_laps': [13]}
        }

    elif preset == "Close Battle (3 drivers)":
        drivers_data = {
            'A': {'name': 'Driver A', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05, 'consistency': 0.08},
            'B': {'name': 'Driver B', 'base_lap_time': 95.1, 'tire_deg_rate': 0.045, 'consistency': 0.09},
            'C': {'name': 'Driver C', 'base_lap_time': 95.2, 'tire_deg_rate': 0.048, 'consistency': 0.10}
        }
        strategies = {
            'A': {'pit_laps': [10]},
            'B': {'pit_laps': [12]},
            'C': {'pit_laps': [14]}
        }

    elif preset == "Strategy Mix (5 drivers)":
        drivers_data = {
            'A': {'name': 'Early Pit', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05, 'consistency': 0.10},
            'B': {'name': 'Mid Pit', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05, 'consistency': 0.10},
            'C': {'name': 'Late Pit', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05, 'consistency': 0.10},
            'D': {'name': 'Very Early', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05, 'consistency': 0.10},
            'E': {'name': 'Very Late', 'base_lap_time': 95.0, 'tire_deg_rate': 0.05, 'consistency': 0.10}
        }
        strategies = {
            'A': {'pit_laps': [10]},
            'B': {'pit_laps': [13]},
            'C': {'pit_laps': [16]},
            'D': {'pit_laps': [8]},
            'E': {'pit_laps': [18]}
        }

    else:  # Custom
        st.info("Configure custom drivers below:")

        for i in range(num_drivers):
            driver_id = f"D{i+1}"

            with st.expander(f"Driver {i+1} Configuration", expanded=(i < 2)):
                col1, col2, col3 = st.columns(3)

                with col1:
                    name = st.text_input(f"Name", value=f"Driver {i+1}", key=f"name_{i}")
                    base_lap = st.number_input(
                        f"Base Lap Time (s)",
                        min_value=85.0,
                        max_value=110.0,
                        value=95.0 + i * 0.2,
                        step=0.1,
                        key=f"base_{i}"
                    )

                with col2:
                    tire_deg = st.slider(
                        f"Tire Degradation (s/lap)",
                        min_value=0.01,
                        max_value=0.15,
                        value=0.05,
                        step=0.01,
                        key=f"deg_{i}"
                    )

                with col3:
                    consistency = st.slider(
                        f"Consistency (std dev)",
                        min_value=0.0,
                        max_value=0.3,
                        value=0.1,
                        step=0.01,
                        key=f"cons_{i}"
                    )
                    pit_lap = st.number_input(
                        f"Pit Lap",
                        min_value=5,
                        max_value=race_length - 3,
                        value=12,
                        key=f"pit_{i}"
                    )

                drivers_data[driver_id] = {
                    'name': name,
                    'base_lap_time': base_lap,
                    'tire_deg_rate': tire_deg,
                    'consistency': consistency
                }
                strategies[driver_id] = {'pit_laps': [pit_lap]}

    # Run simulation button
    st.markdown("---")

    if st.button("🏁 Run Race Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating race..."):
            # Create simulator
            simulator = MultiDriverRaceSimulator(
                race_length=race_length,
                pit_loss_time=pit_loss_time
            )

            # Run simulation
            try:
                result = simulator.simulate_race(drivers_data, strategies)

                # Store result in session state
                st.session_state['race_result'] = result

                st.success(f"✓ Simulation complete! {len(drivers_data)} drivers, {race_length} laps")

            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                return

    # Display results if available
    if 'race_result' in st.session_state:
        result = st.session_state['race_result']

        st.markdown("---")
        st.header("Race Results")

        # Final standings
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Final Classification")

            results_df = pd.DataFrame(result['final_results'])
            results_df['gap_to_leader'] = results_df['gap_to_leader'].apply(lambda x: f"+{x:.2f}s" if x > 0 else "")
            results_df['avg_lap_time'] = results_df['avg_lap_time'].apply(lambda x: f"{x:.2f}s")

            st.dataframe(
                results_df[['position', 'driver_name', 'gap_to_leader', 'pit_stops', 'avg_lap_time']],
                hide_index=True,
                use_container_width=True
            )

        with col2:
            st.subheader("Race Statistics")
            winner = result['final_results'][0]
            st.metric("Winner", winner['driver_name'])
            st.metric("Winning Margin", f"{result['final_results'][1]['gap_to_leader']:.2f}s")
            st.metric("Position Changes", len(result['position_changes']))

        # Position changes over time
        st.subheader("Position Changes Over Time")

        # Create position chart
        fig = create_position_chart(result['lap_by_lap'])
        st.plotly_chart(fig, use_container_width=True)

        # Gap evolution
        st.subheader("Gap to Leader Evolution")

        fig_gap = create_gap_chart(result['lap_by_lap'])
        st.plotly_chart(fig_gap, use_container_width=True)

        # Pit stop timeline
        st.subheader("Pit Stop Timeline")

        fig_pit = create_pit_timeline(result, strategies)
        st.plotly_chart(fig_pit, use_container_width=True)

        # Strategy effectiveness
        st.subheader("Strategy Effectiveness Analysis")

        effectiveness_df = pd.DataFrame(result['strategy_effectiveness']).T
        st.dataframe(effectiveness_df, use_container_width=True)

        # Download results
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Prepare CSV download
            lap_by_lap_df = prepare_lap_by_lap_csv(result['lap_by_lap'])
            csv = lap_by_lap_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Lap-by-Lap Data (CSV)",
                data=csv,
                file_name="race_simulation_results.csv",
                mime="text/csv"
            )

        with col2:
            results_csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Final Results (CSV)",
                data=results_csv,
                file_name="race_final_results.csv",
                mime="text/csv"
            )


def show_undercut_analyzer():
    """Display undercut scenario analysis."""
    st.header("Undercut Strategy Analyzer")

    st.markdown("""
    **Undercut Strategy:** Pit earlier than your competitor to gain track position on fresh tires.

    This tool simulates head-to-head battles to determine if an undercut will succeed.
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Driver A (Attempting Undercut)")
        base_lap_a = st.number_input("Base Lap Time (s)", min_value=90.0, max_value=100.0, value=95.0, key="under_a_base")
        tire_deg_a = st.slider("Tire Degradation (s/lap)", min_value=0.01, max_value=0.15, value=0.05, step=0.01, key="under_a_deg")
        pit_lap_a = st.number_input("Pit Lap", min_value=5, max_value=20, value=10, key="under_a_pit")

    with col2:
        st.subheader("Driver B (Defending Position)")
        base_lap_b = st.number_input("Base Lap Time (s)", min_value=90.0, max_value=100.0, value=95.0, key="under_b_base")
        tire_deg_b = st.slider("Tire Degradation (s/lap)", min_value=0.01, max_value=0.15, value=0.05, step=0.01, key="under_b_deg")
        pit_lap_b = st.number_input("Pit Lap", min_value=5, max_value=20, value=12, key="under_b_pit")

    race_length = st.slider("Race Length (laps)", min_value=15, max_value=40, value=25)

    if st.button("⚡ Analyze Undercut", type="primary", use_container_width=True):
        if pit_lap_a >= pit_lap_b:
            st.error("Driver A must pit earlier than Driver B for undercut analysis!")
            return

        with st.spinner("Analyzing undercut scenario..."):
            simulator = MultiDriverRaceSimulator(race_length=race_length)

            driver_a_config = {'base_lap_time': base_lap_a, 'tire_deg_rate': tire_deg_a}
            driver_b_config = {'base_lap_time': base_lap_b, 'tire_deg_rate': tire_deg_b}

            try:
                result = simulator.simulate_undercut_scenario(
                    driver_a_config, driver_b_config, pit_lap_a, pit_lap_b
                )

                st.session_state['undercut_result'] = result

            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                return

    # Display results
    if 'undercut_result' in st.session_state:
        result = st.session_state['undercut_result']

        st.markdown("---")
        st.header("Undercut Analysis Results")

        # Success metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            success_icon = "✅" if result['success'] else "❌"
            st.metric("Undercut Success", f"{success_icon} {'YES' if result['success'] else 'NO'}")

        with col2:
            if result['overtake_lap']:
                st.metric("Overtake Occurred", f"Lap {result['overtake_lap']}")
            else:
                st.metric("Overtake Occurred", "Never")

        with col3:
            gap_color = "inverse" if result['time_delta'] < 0 else "off"
            st.metric("Final Gap", f"{result['time_delta']:.2f}s", delta_color=gap_color)

        # Summary
        st.info(result['summary'])

        # Gap evolution chart
        st.subheader("Gap Evolution")

        gap_df = pd.DataFrame(result['gap_evolution'])

        fig = go.Figure()

        # Gap line
        fig.add_trace(go.Scatter(
            x=gap_df['lap'],
            y=gap_df['gap'],
            mode='lines+markers',
            name='Gap (A to B)',
            line=dict(width=3),
            marker=dict(size=6)
        ))

        # Mark pit stops
        fig.add_vline(x=result['critical_laps']['driver_a_pit'], line_dash="dash", line_color="blue",
                     annotation_text="Driver A Pits", annotation_position="top")
        fig.add_vline(x=result['critical_laps']['driver_b_pit'], line_dash="dash", line_color="red",
                     annotation_text="Driver B Pits", annotation_position="top")

        # Mark overtake
        if result['overtake_lap']:
            fig.add_vline(x=result['overtake_lap'], line_dash="dot", line_color="green",
                         annotation_text="Overtake!", annotation_position="bottom")

        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)

        fig.update_layout(
            title="Gap Evolution (Negative = Driver A Ahead)",
            xaxis_title="Lap Number",
            yaxis_title="Gap (seconds)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Critical laps analysis
        st.subheader("Critical Moments")

        critical_data = {
            'Event': ['Driver A Pits', 'Driver B Pits', 'Overtake', 'Race End'],
            'Lap': [
                result['critical_laps']['driver_a_pit'],
                result['critical_laps']['driver_b_pit'],
                result['overtake_lap'] if result['overtake_lap'] else 'N/A',
                gap_df['lap'].max()
            ],
            'Gap (A to B)': [
                f"{result['critical_laps']['gap_at_a_pit']:.2f}s" if result['critical_laps']['gap_at_a_pit'] else 'N/A',
                f"{result['critical_laps']['gap_at_b_pit']:.2f}s" if result['critical_laps']['gap_at_b_pit'] else 'N/A',
                'Position Change' if result['overtake_lap'] else 'N/A',
                f"{result['critical_laps']['final_gap']:.2f}s"
            ]
        }

        st.table(pd.DataFrame(critical_data))


def show_strategy_optimizer():
    """Display team strategy optimization."""
    st.header("Team Strategy Optimizer")

    st.markdown("""
    Optimize pit strategy when controlling multiple team cars against opponents.
    Find the best strategy to maximize team points or guarantee a win.
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team Configuration")
        num_team_cars = st.selectbox("Team Cars", [1, 2, 3], index=1)

        team_drivers = {}
        for i in range(num_team_cars):
            with st.expander(f"Team Car {i+1}", expanded=(i==0)):
                team_drivers[f"T{i+1}"] = {
                    'name': f"Team Car {i+1}",
                    'base_lap_time': st.number_input("Base Lap Time", value=94.5, key=f"team_{i}_base"),
                    'tire_deg_rate': st.slider("Tire Deg", 0.01, 0.15, 0.05, key=f"team_{i}_deg"),
                    'consistency': 0.08
                }

    with col2:
        st.subheader("Opponent Configuration")
        num_opponents = st.selectbox("Opponent Cars", [1, 2, 3, 4, 5], index=2)

        opponents = {}
        for i in range(num_opponents):
            with st.expander(f"Opponent {i+1}", expanded=False):
                opponents[f"O{i+1}"] = {
                    'name': f"Opponent {i+1}",
                    'base_lap_time': st.number_input("Base Lap Time", value=94.8, key=f"opp_{i}_base"),
                    'tire_deg_rate': st.slider("Tire Deg", 0.01, 0.15, 0.05, key=f"opp_{i}_deg"),
                    'consistency': 0.10
                }

    # Objective selection
    objective = st.selectbox(
        "Optimization Objective",
        ["maximize_team_points", "guarantee_win", "block_opponents"],
        format_func=lambda x: {
            "maximize_team_points": "Maximize Team Points",
            "guarantee_win": "Guarantee Win",
            "block_opponents": "Block Opponents"
        }[x]
    )

    if st.button("🎯 Optimize Team Strategy", type="primary", use_container_width=True):
        with st.spinner("Optimizing strategy..."):
            simulator = MultiDriverRaceSimulator(race_length=25)

            try:
                result = simulator.optimize_team_strategy(
                    team_drivers, opponents, objective
                )

                st.session_state['team_opt_result'] = result

            except Exception as e:
                st.error(f"Optimization error: {str(e)}")
                return

    # Display results
    if 'team_opt_result' in st.session_state:
        result = st.session_state['team_opt_result']

        st.markdown("---")
        st.header("Optimal Team Strategy")

        # Recommendation
        st.success(result['recommendation'])

        # Strategy details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Optimal Pit Strategies")
            strategy_data = []
            for driver_id, strategy in result['optimal_strategies'].items():
                strategy_data.append({
                    'Driver': driver_id,
                    'Pit Lap': strategy['pit_laps'][0]
                })
            st.table(pd.DataFrame(strategy_data))

        with col2:
            st.subheader("Expected Results")
            st.metric("Team Score", f"{result['team_score']:.1f}")
            st.metric("Team Positions", ", ".join(map(str, result['team_positions'])))

        # Full race results
        if result['expected_result']:
            st.subheader("Full Race Results with Optimal Strategy")

            results_df = pd.DataFrame(result['expected_result']['final_results'])
            results_df['Team Car'] = results_df['driver_id'].apply(lambda x: '✓' if x.startswith('T') else '')

            st.dataframe(
                results_df[['position', 'driver_name', 'Team Car', 'gap_to_leader', 'pit_stops']],
                hide_index=True,
                use_container_width=True
            )


def show_whatif_scenarios():
    """Display custom what-if scenario builder."""
    st.header("What-If Scenario Builder")

    st.markdown("""
    Build custom scenarios to answer questions like:
    - What if everyone had the same pace?
    - What if tire degradation was 50% higher?
    - What if pit stops were 5 seconds faster?
    """)

    # Base scenario configuration
    st.subheader("Base Scenario")

    scenario_type = st.selectbox(
        "Select Scenario Template",
        ["Equal Pace Battle", "High Degradation Race", "Fast Pit Stops", "Custom"]
    )

    if scenario_type == "Equal Pace Battle":
        st.info("All drivers have identical pace - winner determined purely by strategy")
        base_pace = 95.0
        pace_variation = 0.0
        tire_deg = 0.05
        pit_time = 25.0

    elif scenario_type == "High Degradation Race":
        st.info("Tires degrade quickly - multiple pit stops may be optimal")
        base_pace = 95.0
        pace_variation = 0.2
        tire_deg = 0.12
        pit_time = 25.0

    elif scenario_type == "Fast Pit Stops":
        st.info("Quick pit stops make aggressive strategies more viable")
        base_pace = 95.0
        pace_variation = 0.2
        tire_deg = 0.05
        pit_time = 18.0

    else:  # Custom
        col1, col2 = st.columns(2)

        with col1:
            base_pace = st.number_input("Base Pace (seconds)", value=95.0)
            pace_variation = st.slider("Pace Variation Between Drivers", 0.0, 1.0, 0.2)

        with col2:
            tire_deg = st.slider("Tire Degradation Rate", 0.01, 0.20, 0.05)
            pit_time = st.number_input("Pit Stop Time", value=25.0)

    # Number of drivers
    num_drivers = st.selectbox("Number of Drivers", [3, 5, 8, 10], index=1)

    # Strategy configuration
    st.subheader("Strategy Configuration")

    strategy_mode = st.radio(
        "How should drivers pit?",
        ["All different laps", "Same lap (chaos!)", "Custom per driver"]
    )

    if st.button("🔧 Run What-If Scenario", type="primary", use_container_width=True):
        with st.spinner("Running scenario..."):
            # Build drivers
            drivers_data = {}
            strategies = {}

            for i in range(num_drivers):
                driver_id = f"D{i+1}"
                drivers_data[driver_id] = {
                    'name': f"Driver {i+1}",
                    'base_lap_time': base_pace + (i * pace_variation),
                    'tire_deg_rate': tire_deg,
                    'consistency': 0.10
                }

                # Determine strategy
                if strategy_mode == "All different laps":
                    pit_lap = 8 + i * 2
                elif strategy_mode == "Same lap (chaos!)":
                    pit_lap = 13
                else:
                    pit_lap = 10 + i

                strategies[driver_id] = {'pit_laps': [min(pit_lap, 22)]}

            # Run simulation
            simulator = MultiDriverRaceSimulator(race_length=25, pit_loss_time=pit_time)

            try:
                result = simulator.simulate_race(drivers_data, strategies)
                st.session_state['whatif_result'] = result

            except Exception as e:
                st.error(f"Scenario error: {str(e)}")
                return

    # Display results
    if 'whatif_result' in st.session_state:
        result = st.session_state['whatif_result']

        st.markdown("---")
        st.header("Scenario Results")

        # Final results
        results_df = pd.DataFrame(result['final_results'])

        st.dataframe(
            results_df[['position', 'driver_name', 'gap_to_leader', 'pit_stops', 'avg_lap_time']],
            hide_index=True,
            use_container_width=True
        )

        # Position chart
        fig = create_position_chart(result['lap_by_lap'])
        st.plotly_chart(fig, use_container_width=True)


# Helper functions for visualizations

def create_position_chart(lap_by_lap: List[Dict]) -> go.Figure:
    """Create position changes over time chart."""
    fig = go.Figure()

    # Extract driver IDs
    driver_ids = list(lap_by_lap[0]['positions'].keys())

    # Create trace for each driver
    for driver_id in driver_ids:
        laps = []
        positions = []

        for lap_state in lap_by_lap:
            laps.append(lap_state['lap'])
            positions.append(lap_state['positions'][driver_id]['position'])

        fig.add_trace(go.Scatter(
            x=laps,
            y=positions,
            mode='lines+markers',
            name=driver_id,
            line=dict(width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title="Position Changes Over Time",
        xaxis_title="Lap Number",
        yaxis_title="Position",
        yaxis=dict(autorange='reversed'),  # P1 at top
        hovermode='x unified',
        height=400
    )

    return fig


def create_gap_chart(lap_by_lap: List[Dict]) -> go.Figure:
    """Create gap to leader evolution chart."""
    fig = go.Figure()

    # Extract driver IDs
    driver_ids = list(lap_by_lap[0]['positions'].keys())

    # Create trace for each driver
    for driver_id in driver_ids:
        laps = []
        gaps = []

        for lap_state in lap_by_lap:
            laps.append(lap_state['lap'])
            gaps.append(lap_state['positions'][driver_id]['gap_to_leader'])

        # Skip leader (gap = 0)
        if max(gaps) > 0:
            fig.add_trace(go.Scatter(
                x=laps,
                y=gaps,
                mode='lines',
                name=driver_id,
                line=dict(width=2)
            ))

    fig.update_layout(
        title="Gap to Leader Evolution",
        xaxis_title="Lap Number",
        yaxis_title="Gap (seconds)",
        hovermode='x unified',
        height=400
    )

    return fig


def create_pit_timeline(result: Dict, strategies: Dict) -> go.Figure:
    """Create pit stop timeline visualization."""
    fig = go.Figure()

    driver_ids = list(strategies.keys())

    for i, driver_id in enumerate(driver_ids):
        pit_laps = strategies[driver_id]['pit_laps']

        for pit_lap in pit_laps:
            fig.add_trace(go.Scatter(
                x=[pit_lap],
                y=[i],
                mode='markers',
                marker=dict(size=15, symbol='square'),
                name=f"{driver_id} Pit",
                showlegend=False,
                hovertext=f"{driver_id} pitted on lap {pit_lap}"
            ))

    fig.update_layout(
        title="Pit Stop Timeline",
        xaxis_title="Lap Number",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(driver_ids))),
            ticktext=driver_ids
        ),
        height=300
    )

    return fig


def prepare_lap_by_lap_csv(lap_by_lap: List[Dict]) -> pd.DataFrame:
    """Prepare lap-by-lap data for CSV export."""
    rows = []

    for lap_state in lap_by_lap:
        lap = lap_state['lap']

        for driver_id, driver_data in lap_state['positions'].items():
            rows.append({
                'Lap': lap,
                'Driver': driver_id,
                'Position': driver_data['position'],
                'Cumulative_Time': driver_data['cumulative_time'],
                'Last_Lap_Time': driver_data['last_lap_time'],
                'Tire_Age': driver_data['tire_age'],
                'Gap_To_Leader': driver_data['gap_to_leader']
            })

    return pd.DataFrame(rows)

# Main entry point for Streamlit multi-page app
def main():
    """Main entry point for standalone page execution"""
    import sys
    import os
    
    # Try to get data from session state (set by app.py)
    if 'race_data' in st.session_state:
        data = st.session_state['race_data']
        track = st.session_state.get('track', 'barber')
        race_num = st.session_state.get('race_num', 1)
        show_race_simulator(data, track, race_num)
    else:
        # Fallback: load data directly (for direct navigation)
        try:
            @st.cache_data
            def load_race_data_local(track="barber", race_num=1):
                """Load race data from CSV files"""
                try:
                    from pathlib import Path
                    base_path = Path(__file__).parent.parent.parent / "Data"
                    track_map = {
                        "barber": "barber",
                        "cota": "COTA",
                        "sonoma": "Sonoma",
                        "indianapolis": "indianapolis",
                        "road-america": "road-america/Road America",
                        "sebring": "sebring/Sebring"
                    }
                    track_folder = track_map.get(track.lower(), "barber")
                    
                    if track.lower() in ["barber", "cota", "sonoma"]:
                        if track.lower() == "barber":
                            race_folder = base_path / track_folder
                        else:
                            race_folder = base_path / track_folder / f"Race {race_num}"
                    else:
                        race_folder = base_path / track_folder / f"Race {race_num}"
                    
                    data = {}
                    results_files = list(race_folder.glob("03_*Results*.CSV")) + list(race_folder.glob("03_*Results*.csv"))
                    if results_files:
                        data['results'] = pd.read_csv(results_files[0], delimiter=';')
                    
                    section_files = list(race_folder.glob("23_*Sections*.CSV")) + list(race_folder.glob("23_*Sections*.csv"))
                    if section_files:
                        data['sections'] = pd.read_csv(section_files[0], delimiter=';')
                    
                    lap_files = list(race_folder.glob("*lap_time*.csv"))
                    if lap_files:
                        data['lap_times'] = pd.read_csv(lap_files[0])
                    
                    best_lap_files = list(race_folder.glob("99_*Best*.CSV")) + list(race_folder.glob("99_*Best*.csv"))
                    if best_lap_files:
                        data['best_laps'] = pd.read_csv(best_lap_files[0], delimiter=';')
                    
                    weather_files = list(race_folder.glob("26_*Weather*.CSV")) + list(race_folder.glob("26_*Weather*.csv"))
                    if weather_files:
                        data['weather'] = pd.read_csv(weather_files[0], delimiter=';')
                    
                    return data
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    return {}
            
            # Sidebar for track/race selection
            st.sidebar.subheader("Race Selection")
            track = st.sidebar.selectbox(
                "Select Track",
                ["barber", "cota", "sonoma", "indianapolis", "road-america", "sebring"],
                format_func=lambda x: x.replace("-", " ").title(),
                key="simulator_track"
            )
            
            race_num = st.sidebar.selectbox(
                "Select Race",
                [1, 2],
                key="simulator_race"
            )
            
            with st.spinner("Loading race data..."):
                data = load_race_data_local(track, race_num)
            
            if data:
                show_race_simulator(data, track, race_num)
            else:
                st.error("Failed to load race data. Please check the data directory.")
        except Exception as e:
            st.error(f"Error loading page: {str(e)}")
            st.exception(e)

# Run main if this is executed as a script (for Streamlit multi-page)
if __name__ == "__main__":
    main()
