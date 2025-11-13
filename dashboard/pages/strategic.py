"""
Strategic Analysis Page
Pit strategy, tire degradation, and race strategy optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def show_strategic_analysis(data, track, race_num):
    """
    Display strategic analysis for race planning

    Args:
        data: Dictionary containing race dataframes
        track: Track name
        race_num: Race number
    """
    st.title(f"⚙️ Strategic Analysis: {track.replace('-', ' ').title()} - Race {race_num}")

    try:
        if 'sections' not in data or data['sections'].empty:
            st.warning("No section analysis data available for strategic analysis")
            return

        sections_df = data['sections']

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

            # Convert lap times to seconds
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

            if driver_data.empty or 'LAP_NUMBER' not in driver_data.columns:
                st.warning("Insufficient data for strategic analysis")
                return

            # Pit Stop Detection
            st.header("Pit Stop Analysis")

            # Detect pit stops based on large lap time increases
            if 'lap_seconds' in driver_data.columns:
                median_lap = driver_data['lap_seconds'].median()
                pit_threshold = median_lap * 1.5  # Laps 50% longer than median

                driver_data['is_pit_lap'] = driver_data['lap_seconds'] > pit_threshold

                pit_laps = driver_data[driver_data['is_pit_lap']].copy()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Detected Pit Stops", len(pit_laps))

                with col2:
                    if len(pit_laps) > 0:
                        avg_pit_time = pit_laps['lap_seconds'].mean() - median_lap
                        st.metric("Avg Pit Loss", f"{avg_pit_time:.1f}s")
                    else:
                        st.metric("Avg Pit Loss", "N/A")

                # Pit stop timeline
                if len(pit_laps) > 0:
                    st.subheader("Pit Stop Timeline")

                    # Create timeline visualization
                    fig = go.Figure()

                    # All laps
                    fig.add_trace(go.Scatter(
                        x=driver_data['LAP_NUMBER'],
                        y=driver_data['lap_seconds'],
                        mode='lines+markers',
                        name='Lap Times',
                        line=dict(color='lightblue', width=2),
                        marker=dict(size=4)
                    ))

                    # Pit stops
                    fig.add_trace(go.Scatter(
                        x=pit_laps['LAP_NUMBER'],
                        y=pit_laps['lap_seconds'],
                        mode='markers',
                        name='Pit Stops',
                        marker=dict(size=15, color='red', symbol='diamond')
                    ))

                    # Add pit stop annotations
                    for idx, row in pit_laps.iterrows():
                        fig.add_annotation(
                            x=row['LAP_NUMBER'],
                            y=row['lap_seconds'],
                            text=f"Pit: Lap {int(row['LAP_NUMBER'])}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor='red',
                            ax=0,
                            ay=-40
                        )

                    fig.update_layout(
                        xaxis_title="Lap Number",
                        yaxis_title="Lap Time (seconds)",
                        height=400,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Pit stop details table
                    if not pit_laps.empty:
                        pit_display = pit_laps[['LAP_NUMBER', 'LAP_TIME', 'lap_seconds']].copy()
                        pit_display.columns = ['Lap Number', 'Lap Time', 'Time (seconds)']
                        st.dataframe(pit_display, hide_index=True, use_container_width=True)

            st.markdown("---")

            # Tire Degradation Analysis
            st.header("Tire Degradation Analysis")

            if 'lap_seconds' in driver_data.columns and 'LAP_NUMBER' in driver_data.columns:
                # Remove pit laps for degradation analysis
                racing_laps = driver_data[~driver_data['is_pit_lap']].copy() if 'is_pit_lap' in driver_data.columns else driver_data.copy()

                if len(racing_laps) > 3:
                    st.subheader("Lap Time vs Lap Number (Tire Degradation)")

                    # Split into stints (between pit stops)
                    if 'is_pit_lap' in driver_data.columns and len(pit_laps) > 0:
                        pit_lap_numbers = pit_laps['LAP_NUMBER'].tolist()
                        stint_breaks = [0] + pit_lap_numbers + [driver_data['LAP_NUMBER'].max() + 1]

                        fig = go.Figure()

                        # Plot each stint separately
                        for i in range(len(stint_breaks) - 1):
                            stint_start = stint_breaks[i]
                            stint_end = stint_breaks[i + 1]

                            stint_data = racing_laps[
                                (racing_laps['LAP_NUMBER'] > stint_start) &
                                (racing_laps['LAP_NUMBER'] < stint_end)
                            ]

                            if len(stint_data) > 0:
                                # Calculate trend line
                                if len(stint_data) > 2:
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        stint_data['LAP_NUMBER'],
                                        stint_data['lap_seconds']
                                    )

                                    trend_line = slope * stint_data['LAP_NUMBER'] + intercept

                                    # Stint laps
                                    fig.add_trace(go.Scatter(
                                        x=stint_data['LAP_NUMBER'],
                                        y=stint_data['lap_seconds'],
                                        mode='markers',
                                        name=f'Stint {i+1}',
                                        marker=dict(size=8)
                                    ))

                                    # Trend line
                                    fig.add_trace(go.Scatter(
                                        x=stint_data['LAP_NUMBER'],
                                        y=trend_line,
                                        mode='lines',
                                        name=f'Stint {i+1} Trend',
                                        line=dict(dash='dash'),
                                        showlegend=False
                                    ))

                        fig.update_layout(
                            xaxis_title="Lap Number",
                            yaxis_title="Lap Time (seconds)",
                            height=450,
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        # Single stint - no pit stops
                        fig = go.Figure()

                        # All racing laps
                        fig.add_trace(go.Scatter(
                            x=racing_laps['LAP_NUMBER'],
                            y=racing_laps['lap_seconds'],
                            mode='markers',
                            name='Lap Times',
                            marker=dict(size=8, color='blue')
                        ))

                        # Calculate trend line
                        if len(racing_laps) > 2:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                racing_laps['LAP_NUMBER'],
                                racing_laps['lap_seconds']
                            )

                            trend_line = slope * racing_laps['LAP_NUMBER'] + intercept

                            fig.add_trace(go.Scatter(
                                x=racing_laps['LAP_NUMBER'],
                                y=trend_line,
                                mode='lines',
                                name='Degradation Trend',
                                line=dict(color='red', dash='dash', width=3)
                            ))

                            # Display degradation rate
                            st.info(f"**Tire Degradation Rate:** {slope:.4f} seconds/lap")

                        fig.update_layout(
                            xaxis_title="Lap Number",
                            yaxis_title="Lap Time (seconds)",
                            height=450,
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Degradation statistics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        first_5_avg = racing_laps.head(5)['lap_seconds'].mean() if len(racing_laps) >= 5 else racing_laps['lap_seconds'].mean()
                        st.metric("Avg First 5 Laps", f"{first_5_avg:.3f}s")

                    with col2:
                        last_5_avg = racing_laps.tail(5)['lap_seconds'].mean() if len(racing_laps) >= 5 else racing_laps['lap_seconds'].mean()
                        st.metric("Avg Last 5 Laps", f"{last_5_avg:.3f}s")

                    with col3:
                        degradation = last_5_avg - first_5_avg
                        st.metric("Overall Degradation", f"{degradation:.3f}s", delta=f"{degradation:.3f}s", delta_color="inverse")

            st.markdown("---")

            # Optimal Pit Window
            st.header("Optimal Pit Window Analysis")

            if 'lap_seconds' in driver_data.columns and len(driver_data) > 10:
                st.subheader("Pit Window Recommendation")

                total_laps = driver_data['LAP_NUMBER'].max()

                # Calculate optimal pit window (simplified model)
                # Typically 1/3 to 2/3 through the race
                optimal_start = int(total_laps * 0.33)
                optimal_end = int(total_laps * 0.67)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Race Laps", int(total_laps))

                with col2:
                    st.metric("Optimal Window Start", f"Lap {optimal_start}")

                with col3:
                    st.metric("Optimal Window End", f"Lap {optimal_end}")

                # Visualize pit window
                fig = go.Figure()

                # Lap times
                fig.add_trace(go.Scatter(
                    x=driver_data['LAP_NUMBER'],
                    y=driver_data['lap_seconds'],
                    mode='lines',
                    name='Lap Times',
                    line=dict(color='blue', width=2)
                ))

                # Optimal pit window (shaded region)
                fig.add_vrect(
                    x0=optimal_start,
                    x1=optimal_end,
                    fillcolor="green",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text="Optimal Pit Window",
                    annotation_position="top left"
                )

                # Mark actual pit stops if any
                if len(pit_laps) > 0:
                    for idx, row in pit_laps.iterrows():
                        fig.add_vline(
                            x=row['LAP_NUMBER'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Pit",
                            annotation_position="top"
                        )

                fig.update_layout(
                    xaxis_title="Lap Number",
                    yaxis_title="Lap Time (seconds)",
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Pit Strategy Comparison
            st.header("Pit Strategy Comparison")

            st.subheader("Actual vs Optimal Strategy")

            # Create comparison table
            comparison_data = []

            # Actual strategy
            if len(pit_laps) > 0:
                actual_pit_laps = pit_laps['LAP_NUMBER'].tolist()
                actual_strategy = f"{len(pit_laps)} stop(s) at lap(s): {', '.join(map(str, [int(x) for x in actual_pit_laps]))}"
                actual_in_window = sum(1 for lap in actual_pit_laps if optimal_start <= lap <= optimal_end)
            else:
                actual_strategy = "0 stops (full distance on one set)"
                actual_in_window = 0

            comparison_data.append({
                'Strategy': 'Actual',
                'Pit Stops': len(pit_laps),
                'Timing': actual_strategy,
                'In Optimal Window': actual_in_window,
                'Rating': '✓ Good' if len(pit_laps) == 0 or actual_in_window > 0 else '✗ Suboptimal'
            })

            # Recommended optimal strategy
            if total_laps > 20:
                optimal_lap = int((optimal_start + optimal_end) / 2)
                optimal_strategy = f"1 stop at lap {optimal_lap}"
                comparison_data.append({
                    'Strategy': 'Recommended',
                    'Pit Stops': 1,
                    'Timing': optimal_strategy,
                    'In Optimal Window': 1,
                    'Rating': 'Optimal'
                })
            else:
                comparison_data.append({
                    'Strategy': 'Recommended',
                    'Pit Stops': 0,
                    'Timing': 'No stop recommended (short race)',
                    'In Optimal Window': 'N/A',
                    'Rating': 'Optimal'
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)

            # Strategic insights
            st.subheader("Strategic Insights")

            insights = []

            # Pit timing insight
            if len(pit_laps) > 0:
                if actual_in_window == len(pit_laps):
                    insights.append("✓ All pit stops were within the optimal window")
                elif actual_in_window > 0:
                    insights.append(f"⚠ {actual_in_window}/{len(pit_laps)} pit stops were in optimal window")
                else:
                    insights.append("✗ Pit stops were outside optimal window - consider adjusting timing")

            # Degradation insight
            if 'lap_seconds' in driver_data.columns:
                racing_laps = driver_data[~driver_data['is_pit_lap']] if 'is_pit_lap' in driver_data.columns else driver_data

                if len(racing_laps) > 5:
                    slope, _, _, _, _ = stats.linregress(racing_laps['LAP_NUMBER'], racing_laps['lap_seconds'])

                    if slope > 0.05:
                        insights.append(f"⚠ High tire degradation rate ({slope:.4f}s/lap) - consider earlier pit stop")
                    elif slope < -0.05:
                        insights.append(f"⚠ Improving pace over race ({slope:.4f}s/lap) - track evolution or driver improvement")
                    else:
                        insights.append(f"✓ Consistent pace throughout stint ({slope:.4f}s/lap)")

            # Stint length insight
            if len(pit_laps) > 0:
                avg_stint_length = total_laps / (len(pit_laps) + 1)
                insights.append(f"Average stint length: {avg_stint_length:.1f} laps")

            # Display insights
            for insight in insights:
                st.markdown(f"- {insight}")

        else:
            st.warning("No driver data available")

    except Exception as e:
        st.error(f"Error displaying strategic analysis: {str(e)}")
        st.exception(e)
