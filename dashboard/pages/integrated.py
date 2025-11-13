"""
Integrated Insights Page
Combined tactical and strategic recommendations with what-if scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_integrated_insights(data, track, race_num):
    """
    Display integrated insights combining tactical and strategic analysis

    Args:
        data: Dictionary containing race dataframes
        track: Track name
        race_num: Race number
    """
    st.title(f"🔗 Integrated Insights: {track.replace('-', ' ').title()} - Race {race_num}")

    try:
        if 'sections' not in data or data['sections'].empty:
            st.warning("No data available for integrated analysis")
            return

        sections_df = data['sections']
        results_df = data.get('results', pd.DataFrame())

        # Driver selection
        st.header("Driver Selection")

        if 'DRIVER_NUMBER' in sections_df.columns:
            available_drivers = sorted(sections_df['DRIVER_NUMBER'].unique())
            selected_driver = st.selectbox(
                "Select Driver",
                available_drivers,
                format_func=lambda x: f"Car #{int(x)}" if pd.notna(x) else "Unknown"
            )

            st.markdown("---")

            # Filter data for selected driver
            driver_data = sections_df[sections_df['DRIVER_NUMBER'] == selected_driver].copy()

            if driver_data.empty:
                st.warning(f"No data available for driver #{selected_driver}")
                return

            # Convert lap times
            def time_to_seconds(time_str):
                try:
                    if ':' in str(time_str):
                        parts = str(time_str).split(':')
                        if len(parts) == 2:
                            mins, secs = parts
                            return float(mins) * 60 + float(secs)
                    return float(time_str)
                except:
                    return None

            if 'LAP_TIME' in driver_data.columns:
                driver_data['lap_seconds'] = driver_data['LAP_TIME'].apply(time_to_seconds)
                driver_data = driver_data.dropna(subset=['lap_seconds'])

            # Combined Recommendations
            st.header("Combined Recommendations")

            recommendations = []

            # Tactical recommendations
            if 'lap_seconds' in driver_data.columns:
                best_lap = driver_data['lap_seconds'].min()
                avg_lap = driver_data['lap_seconds'].mean()
                std_lap = driver_data['lap_seconds'].std()

                # Section analysis
                section_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
                available_sections = [col for col in section_cols if col in driver_data.columns]

                if available_sections:
                    for col in available_sections:
                        section_name = col.replace('_SECONDS', '')
                        optimal = driver_data[col].min()
                        average = driver_data[col].mean()
                        gap = average - optimal
                        gap_pct = (gap / optimal) * 100

                        if gap_pct > 3:  # More than 3% off optimal
                            recommendations.append({
                                'Module': 'Tactical',
                                'Category': 'Section Performance',
                                'Priority': 'High',
                                'Recommendation': f'Improve {section_name} by {gap_pct:.1f}% - Worth {gap:.3f}s/lap',
                                'Impact': 'Position'
                            })

                # Consistency
                consistency_pct = (std_lap / avg_lap) * 100
                if consistency_pct > 2:
                    recommendations.append({
                        'Module': 'Tactical',
                        'Category': 'Consistency',
                        'Priority': 'Medium',
                        'Recommendation': f'Reduce lap time variation from {consistency_pct:.2f}% to <2%',
                        'Impact': 'Position & Time'
                    })

            # Strategic recommendations
            if 'LAP_NUMBER' in driver_data.columns:
                total_laps = driver_data['LAP_NUMBER'].max()

                # Detect pit stops
                if 'lap_seconds' in driver_data.columns:
                    median_lap = driver_data['lap_seconds'].median()
                    driver_data['is_pit_lap'] = driver_data['lap_seconds'] > median_lap * 1.5
                    pit_laps = driver_data[driver_data['is_pit_lap']]

                    # Pit strategy recommendation
                    optimal_start = int(total_laps * 0.33)
                    optimal_end = int(total_laps * 0.67)

                    if len(pit_laps) > 0:
                        actual_pit_laps = pit_laps['LAP_NUMBER'].tolist()
                        in_window = sum(1 for lap in actual_pit_laps if optimal_start <= lap <= optimal_end)

                        if in_window < len(pit_laps):
                            recommendations.append({
                                'Module': 'Strategic',
                                'Category': 'Pit Timing',
                                'Priority': 'High',
                                'Recommendation': f'Move pit stop to laps {optimal_start}-{optimal_end} for optimal tire life',
                                'Impact': 'Position & Time'
                            })

                    # Tire degradation
                    racing_laps = driver_data[~driver_data['is_pit_lap']] if 'is_pit_lap' in driver_data.columns else driver_data

                    if len(racing_laps) > 5:
                        from scipy import stats
                        slope, _, _, _, _ = stats.linregress(racing_laps['LAP_NUMBER'], racing_laps['lap_seconds'])

                        if slope > 0.05:
                            recommendations.append({
                                'Module': 'Strategic',
                                'Category': 'Tire Management',
                                'Priority': 'High',
                                'Recommendation': f'High degradation rate ({slope:.4f}s/lap) - adjust driving style or pit earlier',
                                'Impact': 'Time'
                            })

            # Display recommendations table
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                st.dataframe(rec_df, hide_index=True, use_container_width=True, height=300)
            else:
                st.info("No specific recommendations at this time - performance is optimal")

            st.markdown("---")

            # What-If Scenario Simulator
            st.header("What-If Scenario Simulator")

            st.subheader("Simulate Performance Improvements")

            col1, col2 = st.columns(2)

            with col1:
                # Lap time improvement slider
                lap_improvement = st.slider(
                    "Lap Time Improvement (seconds)",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.5,
                    step=0.1,
                    help="Simulate reducing your average lap time"
                )

            with col2:
                # Consistency improvement slider
                consistency_improvement = st.slider(
                    "Consistency Improvement (%)",
                    min_value=0,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Simulate reducing lap time variation"
                )

            # Calculate current performance
            if 'lap_seconds' in driver_data.columns and 'LAP_NUMBER' in driver_data.columns:
                current_avg = driver_data['lap_seconds'].mean()
                current_std = driver_data['lap_seconds'].std()
                current_best = driver_data['lap_seconds'].min()
                total_laps = driver_data['LAP_NUMBER'].max()

                # Calculate improved performance
                improved_avg = current_avg - lap_improvement
                improved_std = current_std * (1 - consistency_improvement / 100)

                # Calculate total time saved
                time_saved = lap_improvement * total_laps

                # Display comparison
                st.subheader("Performance Comparison")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Average Lap Time",
                        f"{improved_avg:.3f}s",
                        delta=f"-{lap_improvement:.3f}s",
                        delta_color="normal"
                    )

                with col2:
                    st.metric(
                        "Lap Time Std Dev",
                        f"{improved_std:.3f}s",
                        delta=f"-{current_std - improved_std:.3f}s",
                        delta_color="normal"
                    )

                with col3:
                    st.metric(
                        "Total Time Saved",
                        f"{time_saved:.1f}s",
                        delta=f"{time_saved:.1f}s",
                        delta_color="normal"
                    )

                # Visualize improvement
                st.subheader("Simulated Lap Times")

                # Generate simulated lap times
                np.random.seed(42)
                simulated_laps = np.random.normal(improved_avg, improved_std, int(total_laps))
                simulated_laps = np.clip(simulated_laps, improved_avg - 2*improved_std, improved_avg + 2*improved_std)

                fig = go.Figure()

                # Current actual laps
                fig.add_trace(go.Scatter(
                    x=driver_data['LAP_NUMBER'],
                    y=driver_data['lap_seconds'],
                    mode='lines+markers',
                    name='Current Performance',
                    line=dict(color='lightblue', width=2),
                    marker=dict(size=4)
                ))

                # Simulated improved laps
                fig.add_trace(go.Scatter(
                    x=list(range(1, int(total_laps) + 1)),
                    y=simulated_laps,
                    mode='lines+markers',
                    name='Simulated Improvement',
                    line=dict(color='green', width=2),
                    marker=dict(size=4)
                ))

                fig.update_layout(
                    xaxis_title="Lap Number",
                    yaxis_title="Lap Time (seconds)",
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Cross-Module Impact Visualization
            st.header("Cross-Module Impact Analysis")

            st.subheader("How Improvements Affect Overall Performance")

            # Create impact matrix
            if 'lap_seconds' in driver_data.columns:
                impacts = []

                # Section improvements (Tactical)
                section_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
                available_sections = [col for col in section_cols if col in driver_data.columns]

                for col in available_sections:
                    section_name = col.replace('_SECONDS', '')
                    optimal = driver_data[col].min()
                    average = driver_data[col].mean()
                    gap = average - optimal

                    # Calculate impact on position (simplified)
                    time_gain_per_lap = gap
                    total_time_gain = time_gain_per_lap * total_laps

                    impacts.append({
                        'Improvement Area': f'{section_name} Optimization',
                        'Module': 'Tactical',
                        'Time Gain/Lap': f'{time_gain_per_lap:.3f}s',
                        'Total Race Gain': f'{total_time_gain:.1f}s',
                        'Difficulty': 'Medium'
                    })

                # Consistency improvement (Tactical)
                if current_std > 0:
                    potential_gain = current_std * 0.5  # Assume we can reduce variation by half
                    impacts.append({
                        'Improvement Area': 'Consistency Improvement',
                        'Module': 'Tactical',
                        'Time Gain/Lap': f'{potential_gain:.3f}s',
                        'Total Race Gain': f'{potential_gain * total_laps:.1f}s',
                        'Difficulty': 'Low'
                    })

                # Pit strategy (Strategic)
                if 'is_pit_lap' in driver_data.columns:
                    pit_laps = driver_data[driver_data['is_pit_lap']]
                    if len(pit_laps) > 0:
                        # Estimate time saved with optimal pit timing
                        potential_pit_save = 2.0  # seconds per pit stop

                        impacts.append({
                            'Improvement Area': 'Optimal Pit Timing',
                            'Module': 'Strategic',
                            'Time Gain/Lap': 'N/A',
                            'Total Race Gain': f'{potential_pit_save * len(pit_laps):.1f}s',
                            'Difficulty': 'Low'
                        })

                # Tire management (Strategic)
                racing_laps = driver_data[~driver_data['is_pit_lap']] if 'is_pit_lap' in driver_data.columns else driver_data
                if len(racing_laps) > 5:
                    from scipy import stats
                    slope, _, _, _, _ = stats.linregress(racing_laps['LAP_NUMBER'], racing_laps['lap_seconds'])

                    if slope > 0.02:
                        # Potential to reduce degradation
                        degradation_reduction = slope * 0.3  # 30% reduction
                        avg_stint_length = len(racing_laps)
                        total_save = degradation_reduction * (avg_stint_length / 2) * avg_stint_length

                        impacts.append({
                            'Improvement Area': 'Tire Management',
                            'Module': 'Strategic',
                            'Time Gain/Lap': f'{degradation_reduction:.4f}s/lap reduction',
                            'Total Race Gain': f'{total_save:.1f}s',
                            'Difficulty': 'Medium'
                        })

                if impacts:
                    impact_df = pd.DataFrame(impacts)
                    st.dataframe(impact_df, hide_index=True, use_container_width=True)

                    # Visualize total potential gains
                    st.subheader("Potential Performance Gains")

                    # Extract numeric values for plotting
                    gains = []
                    labels = []
                    for impact in impacts:
                        try:
                            gain = float(impact['Total Race Gain'].replace('s', ''))
                            gains.append(gain)
                            labels.append(impact['Improvement Area'])
                        except:
                            pass

                    if gains:
                        fig = go.Figure(go.Bar(
                            x=labels,
                            y=gains,
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                            text=[f'{g:.1f}s' for g in gains],
                            textposition='auto'
                        ))

                        fig.update_layout(
                            xaxis_title="Improvement Area",
                            yaxis_title="Total Time Gain (seconds)",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate cumulative impact
                        total_potential = sum(gains)
                        st.success(f"**Total Potential Time Gain: {total_potential:.1f} seconds**")

            st.markdown("---")

            # Projected Position Changes
            st.header("Projected Position Changes")

            if not results_df.empty and 'lap_seconds' in driver_data.columns:
                st.subheader("Position Impact Simulation")

                # Get current position
                if 'NUMBER' in results_df.columns and 'POSITION' in results_df.columns:
                    current_position_row = results_df[results_df['NUMBER'] == selected_driver]

                    if not current_position_row.empty:
                        current_position = current_position_row.iloc[0]['POSITION']

                        # Simulate position changes with improvements
                        improvement_scenarios = [0, 0.5, 1.0, 1.5, 2.0]
                        projected_positions = []

                        for improvement in improvement_scenarios:
                            time_saved = improvement * total_laps

                            # Simple position estimation based on gap to next position
                            # This is a simplified model
                            positions_gained = int(time_saved / 5)  # Assume 5 seconds per position on average

                            new_position = max(1, current_position - positions_gained)
                            projected_positions.append(new_position)

                        # Create visualization
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=improvement_scenarios,
                            y=projected_positions,
                            mode='lines+markers',
                            name='Projected Position',
                            line=dict(color='green', width=3),
                            marker=dict(size=10)
                        ))

                        fig.add_hline(
                            y=current_position,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Current Position"
                        )

                        fig.update_layout(
                            xaxis_title="Lap Time Improvement (seconds)",
                            yaxis_title="Final Position",
                            yaxis_autorange="reversed",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Display position change table
                        position_data = []
                        for improvement, position in zip(improvement_scenarios, projected_positions):
                            position_data.append({
                                'Improvement (s/lap)': improvement,
                                'Total Time Saved (s)': improvement * total_laps,
                                'Projected Position': int(position),
                                'Positions Gained': int(current_position - position)
                            })

                        position_df = pd.DataFrame(position_data)
                        st.dataframe(position_df, hide_index=True, use_container_width=True)

                    else:
                        st.info("Driver not found in results")

        else:
            st.warning("No driver data available")

    except Exception as e:
        st.error(f"Error displaying integrated insights: {str(e)}")
        st.exception(e)
