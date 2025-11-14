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
        if 'sections' not in data or data['sections'] is None or data['sections'].empty:
            st.warning("No section analysis data available for strategic analysis")
            return

        sections_df = data['sections'].copy()

        # Driver selection
        st.header("Driver Selection")

        # Handle column name with or without leading space - strip whitespace from column names
        sections_df.columns = sections_df.columns.str.strip()
        driver_col = 'DRIVER_NUMBER'
        
        if driver_col in sections_df.columns:
            available_drivers = sorted(sections_df[driver_col].unique())
            selected_driver = st.selectbox(
                "Select Driver",
                available_drivers,
                format_func=lambda x: f"Car #{int(x)}" if pd.notna(x) else "Unknown"
            )

            st.markdown("---")

            # Filter data for selected driver
            driver_data = sections_df[sections_df[driver_col] == selected_driver].copy()

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

                    # Weather-Adjusted Tire Degradation
                    if 'weather' in data and data['weather'] is not None and not data['weather'].empty:
                        st.subheader("Weather-Adjusted Tire Degradation")

                        try:
                            # Import weather adjuster
                            import sys
                            from pathlib import Path
                            project_root = Path(__file__).parent.parent.parent
                            sys.path.insert(0, str(project_root))

                            from src.integration.weather_adjuster import WeatherAdjuster

                            weather_df = data['weather']
                            adjuster = WeatherAdjuster()
                            conditions = adjuster.get_current_conditions(weather_df)

                            if conditions:
                                # Calculate degradation rate from actual data
                                if len(racing_laps) > 2:
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        racing_laps['LAP_NUMBER'],
                                        racing_laps['lap_seconds']
                                    )
                                    base_deg_rate = max(slope, 0.01)  # Ensure positive

                                    # Get weather-adjusted degradation
                                    adjusted_deg_rate, deg_explanation = adjuster.adjust_tire_degradation(
                                        base_deg_rate, conditions
                                    )

                                    # Create comparison chart
                                    fig = go.Figure()

                                    # Original degradation trend
                                    original_trend = intercept + slope * racing_laps['LAP_NUMBER']

                                    fig.add_trace(go.Scatter(
                                        x=racing_laps['LAP_NUMBER'],
                                        y=original_trend,
                                        mode='lines',
                                        name='Baseline Degradation',
                                        line=dict(color='blue', width=3, dash='dash')
                                    ))

                                    # Weather-adjusted degradation trend
                                    adjusted_trend = intercept + adjusted_deg_rate * (racing_laps['LAP_NUMBER'] - racing_laps['LAP_NUMBER'].min())

                                    fig.add_trace(go.Scatter(
                                        x=racing_laps['LAP_NUMBER'],
                                        y=adjusted_trend,
                                        mode='lines',
                                        name='Weather-Adjusted Degradation',
                                        line=dict(color='red', width=3)
                                    ))

                                    # Actual lap times
                                    fig.add_trace(go.Scatter(
                                        x=racing_laps['LAP_NUMBER'],
                                        y=racing_laps['lap_seconds'],
                                        mode='markers',
                                        name='Actual Lap Times',
                                        marker=dict(size=6, color='lightblue')
                                    ))

                                    fig.update_layout(
                                        xaxis_title="Lap Number",
                                        yaxis_title="Lap Time (seconds)",
                                        height=400,
                                        hovermode='x unified',
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Display comparison metrics
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric(
                                            "Baseline Deg Rate",
                                            f"{base_deg_rate:.4f}s/lap"
                                        )

                                    with col2:
                                        deg_change = ((adjusted_deg_rate / base_deg_rate) - 1) * 100
                                        st.metric(
                                            "Weather-Adjusted Rate",
                                            f"{adjusted_deg_rate:.4f}s/lap",
                                            delta=f"{deg_change:+.1f}%",
                                            delta_color="inverse"
                                        )

                                    with col3:
                                        # Calculate impact on pit window
                                        total_laps = driver_data['LAP_NUMBER'].max()
                                        laps_remaining = total_laps * 0.5  # Assume mid-race

                                        time_diff = (adjusted_deg_rate - base_deg_rate) * laps_remaining
                                        st.metric(
                                            "Impact on Remaining Stint",
                                            f"{time_diff:+.2f}s"
                                        )

                                    # Display weather impact explanation
                                    st.info(f"**Weather Impact:** {deg_explanation}")

                                    # Strategic recommendation based on weather
                                    if adjusted_deg_rate > base_deg_rate * 1.1:
                                        st.warning(
                                            "⚠ **Recommendation:** Weather conditions increase tire degradation. "
                                            "Consider earlier pit stop or more conservative pace to manage tire life."
                                        )
                                    elif adjusted_deg_rate < base_deg_rate * 0.95:
                                        st.success(
                                            "✓ **Recommendation:** Weather conditions reduce tire degradation. "
                                            "Can extend stint or push harder with reduced tire wear."
                                        )

                        except Exception as e:
                            # Silently fail if weather adjustment not available
                            pass

            st.markdown("---")

            # Optimal Pit Window with Bayesian Uncertainty
            st.header("Optimal Pit Window Analysis with Bayesian Uncertainty")

            if 'lap_seconds' in driver_data.columns and len(driver_data) > 10:
                st.subheader("Pit Window Recommendation")

                total_laps = driver_data['LAP_NUMBER'].max()

                # Import strategy optimizer
                try:
                    import sys
                    sys.path.append('/home/user/ToyotaGR')
                    from src.strategic.strategy_optimizer import PitStrategyOptimizer

                    # Initialize optimizer
                    optimizer = PitStrategyOptimizer(
                        pit_loss_seconds=25.0,
                        simulation_iterations=100,
                        uncertainty_model='bayesian'
                    )

                    # Build simple tire model from data
                    racing_laps = driver_data[~driver_data['is_pit_lap']] if 'is_pit_lap' in driver_data.columns else driver_data

                    if len(racing_laps) > 5:
                        slope, intercept, _, _, _ = stats.linregress(
                            racing_laps['LAP_NUMBER'],
                            racing_laps['lap_seconds']
                        )

                        tire_model = {
                            'baseline_lap_time': float(intercept),
                            'degradation_rate': float(slope),
                            'model_type': 'linear'
                        }
                    else:
                        # Fallback model
                        tire_model = {
                            'baseline_lap_time': racing_laps['lap_seconds'].mean(),
                            'degradation_rate': 0.05,
                            'model_type': 'linear'
                        }

                    # Calculate Bayesian optimal pit window
                    with st.spinner("Calculating optimal pit window with uncertainty quantification..."):
                        bayesian_results = optimizer.calculate_optimal_pit_window_with_uncertainty(
                            driver_data,
                            tire_model,
                            race_length=int(total_laps)
                        )

                    # Display results with confidence intervals
                    st.subheader("Bayesian Pit Strategy Recommendation")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Race Laps", int(total_laps))

                    with col2:
                        st.metric(
                            "Optimal Pit Lap",
                            f"Lap {bayesian_results['optimal_lap']}",
                            help=f"Posterior mean: {bayesian_results['posterior_mean']:.1f}"
                        )

                    with col3:
                        uncertainty_pct = bayesian_results['uncertainty'] * 100
                        st.metric(
                            "Uncertainty",
                            f"{uncertainty_pct:.1f}%",
                            help=f"Relative uncertainty (std/mean)"
                        )

                    with col4:
                        risk_level = bayesian_results['risk_assessment']['risk_level']
                        risk_color = {
                            'LOW': '🟢',
                            'MODERATE': '🟡',
                            'ELEVATED': '🟠',
                            'HIGH': '🔴'
                        }.get(risk_level, '⚪')
                        st.metric(
                            "Risk Level",
                            f"{risk_color} {risk_level}"
                        )

                    # Confidence interval display
                    st.subheader("Confidence Intervals")

                    # Confidence level slider
                    confidence_level = st.select_slider(
                        "Select Confidence Level",
                        options=[80, 90, 95],
                        value=90,
                        help="Adjust the confidence level for the credible interval"
                    )

                    # Get appropriate interval
                    if confidence_level == 95:
                        interval = bayesian_results['confidence_95']
                        interval_name = "95%"
                    elif confidence_level == 90:
                        interval = bayesian_results['confidence_90']
                        interval_name = "90%"
                    else:
                        interval = bayesian_results['confidence_80']
                        interval_name = "80%"

                    st.info(
                        f"**{interval_name} Confidence:** Pit between laps **{interval[0]}** and **{interval[1]}**\n\n"
                        f"This means we are {interval_name} confident the optimal pit lap falls within this range."
                    )

                    # All confidence intervals display
                    intervals_df = pd.DataFrame({
                        'Confidence Level': ['80%', '90%', '95%'],
                        'Lower Bound': [
                            bayesian_results['confidence_80'][0],
                            bayesian_results['confidence_90'][0],
                            bayesian_results['confidence_95'][0]
                        ],
                        'Upper Bound': [
                            bayesian_results['confidence_80'][1],
                            bayesian_results['confidence_90'][1],
                            bayesian_results['confidence_95'][1]
                        ],
                        'Window Size': [
                            bayesian_results['confidence_80'][1] - bayesian_results['confidence_80'][0],
                            bayesian_results['confidence_90'][1] - bayesian_results['confidence_90'][0],
                            bayesian_results['confidence_95'][1] - bayesian_results['confidence_95'][0]
                        ]
                    })
                    st.dataframe(intervals_df, hide_index=True, use_container_width=True)

                    # Posterior distribution visualization
                    st.subheader("Posterior Distribution (Pit Lap Probability)")

                    # Get visualization data
                    viz_data = optimizer.visualize_posterior_distribution(bayesian_results)

                    # Create violin plot
                    fig_violin = go.Figure()

                    # Add violin plot
                    fig_violin.add_trace(go.Violin(
                        y=bayesian_results['posterior_samples'],
                        name='Posterior Distribution',
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor='lightblue',
                        opacity=0.6,
                        x0='Optimal Pit Lap'
                    ))

                    # Add confidence interval markers
                    for conf_level, interval_data in [
                        ('95%', bayesian_results['confidence_95']),
                        ('90%', bayesian_results['confidence_90']),
                        ('80%', bayesian_results['confidence_80'])
                    ]:
                        fig_violin.add_hline(
                            y=interval_data[0],
                            line_dash="dash",
                            line_color="red",
                            opacity=0.5,
                            annotation_text=f"{conf_level} Lower"
                        )
                        fig_violin.add_hline(
                            y=interval_data[1],
                            line_dash="dash",
                            line_color="red",
                            opacity=0.5,
                            annotation_text=f"{conf_level} Upper"
                        )

                    fig_violin.update_layout(
                        yaxis_title="Pit Lap Number",
                        height=500,
                        showlegend=True
                    )

                    st.plotly_chart(fig_violin, use_container_width=True)

                    # PDF curve visualization
                    fig_pdf = go.Figure()

                    # Add PDF curve
                    fig_pdf.add_trace(go.Scatter(
                        x=viz_data['pdf']['x'],
                        y=viz_data['pdf']['y'],
                        mode='lines',
                        name='Probability Density',
                        fill='tozeroy',
                        line=dict(color='blue', width=3)
                    ))

                    # Add optimal lap marker
                    optimal_y = stats.norm.pdf(
                        bayesian_results['optimal_lap'],
                        bayesian_results['posterior_mean'],
                        bayesian_results['posterior_std']
                    )
                    fig_pdf.add_trace(go.Scatter(
                        x=[bayesian_results['optimal_lap']],
                        y=[optimal_y],
                        mode='markers',
                        name='Optimal Lap',
                        marker=dict(size=15, color='red', symbol='star')
                    ))

                    # Shade confidence intervals
                    conf_x = np.array(viz_data['pdf']['x'])
                    conf_y = np.array(viz_data['pdf']['y'])

                    # Shade selected confidence interval
                    mask = (conf_x >= interval[0]) & (conf_x <= interval[1])
                    fig_pdf.add_trace(go.Scatter(
                        x=conf_x[mask],
                        y=conf_y[mask],
                        fill='tozeroy',
                        mode='none',
                        name=f'{interval_name} Interval',
                        fillcolor='rgba(255, 0, 0, 0.2)'
                    ))

                    fig_pdf.update_layout(
                        xaxis_title="Pit Lap Number",
                        yaxis_title="Probability Density",
                        height=400,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_pdf, use_container_width=True)

                    # Risk Assessment
                    st.subheader("Risk Assessment")

                    risk_info = bayesian_results['risk_assessment']

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        **Risk Level:** {risk_color} **{risk_info['risk_level']}**

                        **Explanation:** {risk_info['explanation']}

                        **Strategy Note:** {risk_info['strategy_note']}
                        """)

                    with col2:
                        st.markdown(f"""
                        **Statistical Details:**
                        - Posterior Std Dev: {risk_info['posterior_std']:.2f} laps
                        - Relative Uncertainty: {risk_info['relative_uncertainty']*100:.1f}%
                        - Time Spread: {risk_info['time_spread_seconds']:.2f} seconds
                        """)

                    # Simulation results comparison
                    st.subheader("Simulation Results by Pit Lap")

                    sim_df = pd.DataFrame([
                        {
                            'Pit Lap': lap,
                            'Mean Time (s)': data['mean'],
                            'Std Dev (s)': data['std']
                        }
                        for lap, data in bayesian_results['simulation_results'].items()
                    ])

                    fig_sim = go.Figure()

                    # Mean times with error bars
                    fig_sim.add_trace(go.Scatter(
                        x=sim_df['Pit Lap'],
                        y=sim_df['Mean Time (s)'],
                        error_y=dict(
                            type='data',
                            array=sim_df['Std Dev (s)'],
                            visible=True
                        ),
                        mode='markers+lines',
                        name='Expected Race Time',
                        marker=dict(size=8)
                    ))

                    # Highlight optimal lap
                    optimal_row = sim_df[sim_df['Pit Lap'] == bayesian_results['optimal_lap']]
                    if not optimal_row.empty:
                        fig_sim.add_trace(go.Scatter(
                            x=optimal_row['Pit Lap'],
                            y=optimal_row['Mean Time (s)'],
                            mode='markers',
                            name='Optimal',
                            marker=dict(size=15, color='red', symbol='star')
                        ))

                    # Shade confidence interval
                    fig_sim.add_vrect(
                        x0=interval[0],
                        x1=interval[1],
                        fillcolor="green",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        annotation_text=f"{interval_name} Confidence"
                    )

                    fig_sim.update_layout(
                        xaxis_title="Pit Lap Number",
                        yaxis_title="Expected Total Race Time (seconds)",
                        height=400,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_sim, use_container_width=True)

                except ImportError as e:
                    st.warning(f"Bayesian analysis not available: {str(e)}")
                    st.info("Falling back to simplified pit window calculation...")

                    # Fallback to simple calculation
                    optimal_start = int(total_laps * 0.33)
                    optimal_end = int(total_laps * 0.67)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Race Laps", int(total_laps))

                    with col2:
                        st.metric("Optimal Window Start", f"Lap {optimal_start}")

                    with col3:
                        st.metric("Optimal Window End", f"Lap {optimal_end}")
                except Exception as e:
                    st.error(f"Error in Bayesian analysis: {str(e)}")
                    st.exception(e)

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

# Main entry point for Streamlit multi-page app
def main():
    """Main entry point for standalone page execution"""
    import sys
    import os
    from pathlib import Path
    
    # Try to get data from session state (set by app.py)
    if 'race_data' in st.session_state:
        data = st.session_state['race_data']
        track = st.session_state.get('track', 'barber')
        race_num = st.session_state.get('race_num', 1)
        show_strategic_analysis(data, track, race_num)
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
                key="strategic_track"
            )
            
            race_num = st.sidebar.selectbox(
                "Select Race",
                [1, 2],
                key="strategic_race"
            )
            
            with st.spinner("Loading race data..."):
                data = load_race_data_local(track, race_num)
            
            if data:
                show_strategic_analysis(data, track, race_num)
            else:
                st.error("Failed to load race data. Please check the data directory.")
        except Exception as e:
            st.error(f"Error loading page: {str(e)}")
            st.exception(e)

# Run main if this is executed as a script (for Streamlit multi-page)
if __name__ == "__main__":
    main()
