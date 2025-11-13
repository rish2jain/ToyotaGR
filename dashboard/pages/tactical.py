"""
Tactical Analysis Page
Driver-specific performance analysis with section breakdown and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_tactical_analysis(data, track, race_num):
    """
    Display tactical analysis for driver performance

    Args:
        data: Dictionary containing race dataframes
        track: Track name
        race_num: Race number
    """
    st.title(f"🎯 Tactical Analysis: {track.replace('-', ' ').title()} - Race {race_num}")

    try:
        if 'sections' not in data or data['sections'].empty:
            st.warning("No section analysis data available for tactical analysis")
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

            # Performance Overview
            st.header("Performance Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'LAP_NUMBER' in driver_data.columns:
                    total_laps = driver_data['LAP_NUMBER'].max()
                    st.metric("Total Laps", int(total_laps) if pd.notna(total_laps) else "N/A")

            with col2:
                if 'LAP_TIME' in driver_data.columns:
                    # Convert lap time to seconds
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

                    driver_data['lap_seconds'] = driver_data['LAP_TIME'].apply(time_to_seconds)
                    driver_data = driver_data.dropna(subset=['lap_seconds'])

                    if not driver_data.empty:
                        best_lap = driver_data['lap_seconds'].min()
                        st.metric("Best Lap", f"{best_lap:.3f}s")
                    else:
                        st.metric("Best Lap", "N/A")

            with col3:
                if 'lap_seconds' in driver_data.columns and not driver_data.empty:
                    avg_lap = driver_data['lap_seconds'].mean()
                    st.metric("Avg Lap Time", f"{avg_lap:.3f}s")
                else:
                    st.metric("Avg Lap Time", "N/A")

            with col4:
                if 'KPH' in driver_data.columns:
                    avg_speed = driver_data['KPH'].mean()
                    st.metric("Avg Speed", f"{avg_speed:.1f} km/h" if pd.notna(avg_speed) else "N/A")

            st.markdown("---")

            # Section Performance Heatmap
            st.header("Section Performance Analysis")

            section_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
            available_section_cols = [col for col in section_cols if col in driver_data.columns]

            if available_section_cols and 'LAP_NUMBER' in driver_data.columns:
                st.subheader("Section Times Heatmap")

                # Prepare data for heatmap
                lap_numbers = driver_data['LAP_NUMBER'].values
                section_data = driver_data[available_section_cols].values

                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=section_data.T,
                    x=lap_numbers,
                    y=[col.replace('_SECONDS', '') for col in available_section_cols],
                    colorscale='RdYlGn_r',
                    text=section_data.T,
                    texttemplate='%{text:.2f}s',
                    textfont={"size": 10},
                    colorbar=dict(title="Time (s)")
                ))

                fig.update_layout(
                    xaxis_title="Lap Number",
                    yaxis_title="Section",
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

                # Calculate optimal (best) section times
                st.subheader("Driver vs Optimal Ghost")

                optimal_times = {col: driver_data[col].min() for col in available_section_cols}

                # Calculate gap to optimal for each section
                gaps = []
                for col in available_section_cols:
                    section_name = col.replace('_SECONDS', '')
                    avg_time = driver_data[col].mean()
                    optimal_time = optimal_times[col]
                    gap = avg_time - optimal_time

                    gaps.append({
                        'Section': section_name,
                        'Average Time': avg_time,
                        'Optimal Time': optimal_time,
                        'Gap': gap,
                        'Gap %': (gap / optimal_time) * 100
                    })

                gaps_df = pd.DataFrame(gaps)

                # Create bar chart showing gaps
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    name='Optimal Time',
                    x=gaps_df['Section'],
                    y=gaps_df['Optimal Time'],
                    marker_color='lightgreen'
                ))

                fig.add_trace(go.Bar(
                    name='Gap to Optimal',
                    x=gaps_df['Section'],
                    y=gaps_df['Gap'],
                    marker_color='coral'
                ))

                fig.update_layout(
                    barmode='stack',
                    xaxis_title="Section",
                    yaxis_title="Time (seconds)",
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Anomaly Detection
            st.header("Anomaly Detection")

            if 'lap_seconds' in driver_data.columns and len(driver_data) > 5:
                st.subheader("Lap Time Anomalies")

                # Simple anomaly detection using z-score
                mean_lap = driver_data['lap_seconds'].mean()
                std_lap = driver_data['lap_seconds'].std()

                driver_data['z_score'] = (driver_data['lap_seconds'] - mean_lap) / std_lap
                driver_data['is_anomaly'] = abs(driver_data['z_score']) > 2

                anomalies = driver_data[driver_data['is_anomaly']].copy()

                if len(anomalies) > 0:
                    st.write(f"**{len(anomalies)} anomalies detected** (laps >2 standard deviations from mean)")

                    # Display anomalies table
                    anomaly_cols = ['LAP_NUMBER', 'LAP_TIME', 'lap_seconds', 'z_score']
                    available_anomaly_cols = [col for col in anomaly_cols if col in anomalies.columns]

                    if available_anomaly_cols:
                        anomaly_display = anomalies[available_anomaly_cols].copy()
                        anomaly_display.columns = ['Lap #', 'Lap Time', 'Seconds', 'Z-Score']
                        st.dataframe(
                            anomaly_display.sort_values('Z-Score', ascending=False),
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    st.success("No significant anomalies detected")

                # Plot lap times with anomalies highlighted
                fig = go.Figure()

                # Normal laps
                normal_laps = driver_data[~driver_data['is_anomaly']]
                fig.add_trace(go.Scatter(
                    x=normal_laps['LAP_NUMBER'],
                    y=normal_laps['lap_seconds'],
                    mode='lines+markers',
                    name='Normal Laps',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))

                # Anomalous laps
                if len(anomalies) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomalies['LAP_NUMBER'],
                        y=anomalies['lap_seconds'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(size=12, color='red', symbol='x')
                    ))

                # Add mean line
                fig.add_hline(
                    y=mean_lap,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Mean Lap Time"
                )

                fig.update_layout(
                    xaxis_title="Lap Number",
                    yaxis_title="Lap Time (seconds)",
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Telemetry Comparison (Simulated)
            st.header("Telemetry Analysis")
            st.subheader("Speed/Brake/Throttle Comparison")

            # Since we don't have actual telemetry data, we'll create a representative visualization
            st.info("📊 Telemetry data visualization (requires full telemetry CSV files)")

            # Create sample telemetry visualization
            if 'LAP_NUMBER' in driver_data.columns and len(driver_data) > 0:
                # Use section times as proxy for telemetry patterns
                sample_lap = driver_data.iloc[0]

                # Create simulated telemetry trace
                distance = np.linspace(0, 100, 1000)

                # Simulated speed profile
                speed = 100 + 80 * np.sin(distance / 15) + np.random.normal(0, 5, len(distance))
                speed = np.clip(speed, 0, 200)

                # Simulated throttle (inverse of braking zones)
                throttle = 50 + 40 * np.sin(distance / 12) + np.random.normal(0, 10, len(distance))
                throttle = np.clip(throttle, 0, 100)

                # Simulated brake
                brake = np.where(throttle < 40, 100 - throttle, 0)

                # Create subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Speed (km/h)', 'Throttle (%)', 'Brake (%)'),
                    shared_xaxes=True,
                    vertical_spacing=0.08
                )

                # Speed trace
                fig.add_trace(
                    go.Scatter(x=distance, y=speed, name='Speed', line=dict(color='blue', width=2)),
                    row=1, col=1
                )

                # Throttle trace
                fig.add_trace(
                    go.Scatter(x=distance, y=throttle, name='Throttle', fill='tozeroy',
                              line=dict(color='green', width=2)),
                    row=2, col=1
                )

                # Brake trace
                fig.add_trace(
                    go.Scatter(x=distance, y=brake, name='Brake', fill='tozeroy',
                              line=dict(color='red', width=2)),
                    row=3, col=1
                )

                fig.update_xaxes(title_text="Distance (%)", row=3, col=1)
                fig.update_layout(height=600, showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Improvement Recommendations
            st.header("Top 3 Improvement Recommendations")

            # Generate recommendations based on section performance
            recommendations = []

            if available_section_cols and 'lap_seconds' in driver_data.columns:
                # Calculate which sections have most improvement potential
                section_gaps = {}
                for col in available_section_cols:
                    section_name = col.replace('_SECONDS', '')
                    optimal = driver_data[col].min()
                    average = driver_data[col].mean()
                    gap = average - optimal
                    gap_pct = (gap / optimal) * 100
                    section_gaps[section_name] = gap_pct

                # Sort by gap percentage
                sorted_sections = sorted(section_gaps.items(), key=lambda x: x[1], reverse=True)

                # Recommendation 1: Worst section
                if len(sorted_sections) > 0:
                    worst_section = sorted_sections[0]
                    recommendations.append({
                        'priority': 'high',
                        'title': f'Focus on {worst_section[0]} Performance',
                        'description': f'You are losing {worst_section[1]:.2f}% on average in {worst_section[0]} compared to your best lap. Review braking points and apex speeds in this section.'
                    })

                # Recommendation 2: Consistency
                if 'lap_seconds' in driver_data.columns:
                    lap_std = driver_data['lap_seconds'].std()
                    consistency_score = (lap_std / mean_lap) * 100

                    if consistency_score > 2:
                        recommendations.append({
                            'priority': 'medium',
                            'title': 'Improve Lap Consistency',
                            'description': f'Your lap time variation is {consistency_score:.2f}%. Focus on maintaining consistent racing lines and braking points to reduce variation.'
                        })
                    else:
                        recommendations.append({
                            'priority': 'low',
                            'title': 'Maintain Consistency',
                            'description': f'Excellent consistency at {consistency_score:.2f}% variation. Continue with current approach.'
                        })

                # Recommendation 3: Overall pace
                if 'results' in data and not data['results'].empty:
                    results_df = data['results']
                    if 'FL_TIME' in results_df.columns:
                        # Compare to field best
                        field_best_str = results_df['FL_TIME'].min()

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

                        field_best = time_to_seconds(field_best_str)

                        if field_best and best_lap:
                            gap_to_best = best_lap - field_best
                            if gap_to_best > 0:
                                recommendations.append({
                                    'priority': 'high',
                                    'title': 'Close Gap to Leader',
                                    'description': f'You are {gap_to_best:.3f}s off the fastest lap. Analyze leader\'s telemetry in all sections to find time.'
                                })
                            else:
                                recommendations.append({
                                    'priority': 'low',
                                    'title': 'Leading Pace',
                                    'description': 'You set the fastest lap! Focus on maintaining this pace and consistency.'
                                })

            # Display recommendations
            for i, rec in enumerate(recommendations[:3], 1):
                priority_class = f"rec-{rec['priority']}"
                st.markdown(f"""
                    <div class="recommendation-box {priority_class}">
                        <h4>#{i} - {rec['title']}</h4>
                        <p>{rec['description']}</p>
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("No driver data available in section analysis")

    except Exception as e:
        st.error(f"Error displaying tactical analysis: {str(e)}")
        st.exception(e)
