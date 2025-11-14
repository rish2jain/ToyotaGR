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
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.integration.causal_analysis import (
        CausalStrategyAnalyzer,
        prepare_race_data_for_causal_analysis,
        DOWHY_AVAILABLE
    )
except ImportError:
    DOWHY_AVAILABLE = False

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
        if 'sections' not in data or data['sections'] is None or data['sections'].empty:
            st.warning("No data available for integrated analysis")
            return

        sections_df = data['sections'].copy()
        results_df = data.get('results', pd.DataFrame())

        # Driver selection (shared across tabs)
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

            # Create tabs for different analysis types
            tab1, tab2, tab3 = st.tabs([
                "📊 Recommendations & What-If",
                "🔬 Causal Analysis",
                "🎯 Cross-Module Impact"
            ])

            # TAB 1: Combined Recommendations and What-If Simulator
            with tab1:
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

            # TAB 2: Causal Analysis
            with tab2:
                st.header("🔬 Causal Inference Analysis")

                if not DOWHY_AVAILABLE:
                    st.error("""
                    **Causal Analysis Unavailable**

                    The DoWhy library is required for causal inference analysis.
                    Install it with: `pip install dowhy`
                    """)
                else:
                    st.markdown("""
                    **Causal inference** goes beyond correlation to establish proper cause-and-effect
                    relationships. This allows us to answer questions like:
                    - "What if the driver improved Section 3 by 0.5 seconds?"
                    - "Does tire age causally affect lap time, or is it just correlated?"
                    - "What's the causal effect of pit timing on final position?"
                    """)

                    st.markdown("---")

                    # Analysis type selection
                    analysis_type = st.selectbox(
                        "Select Analysis Type",
                        [
                            "Section Improvement Effect",
                            "Pit Strategy Effect",
                            "Custom Causal Analysis",
                            "View Causal Graph"
                        ]
                    )

                    # Prepare data for causal analysis
                    try:
                        causal_data = prepare_race_data_for_causal_analysis(
                            driver_data,
                            results_df if not results_df.empty else None
                        )

                        if len(causal_data) < 20:
                            st.warning(f"⚠️ Limited data: Only {len(causal_data)} observations. "
                                      "Causal analysis requires at least 20 for reliable results.")

                        # Initialize analyzer
                        analyzer = CausalStrategyAnalyzer(min_data_points=15)

                        if analysis_type == "Section Improvement Effect":
                            st.subheader("Section Improvement Causal Effect")

                            st.markdown("""
                            This analysis estimates the **causal effect** of improving a specific
                            section on lap time, controlling for confounding factors like tire age
                            and fuel load.
                            """)

                            # Section selection
                            col1, col2 = st.columns(2)

                            with col1:
                                section_id = st.selectbox(
                                    "Select Section",
                                    [1, 2, 3],
                                    format_func=lambda x: f"Section {x}"
                                )

                            with col2:
                                improvement = st.slider(
                                    "Improvement (seconds)",
                                    min_value=0.0,
                                    max_value=2.0,
                                    value=0.5,
                                    step=0.1,
                                    help="How much faster would the driver be in this section?"
                                )

                            if st.button("Estimate Causal Effect", type="primary"):
                                with st.spinner("Running causal analysis..."):
                                    try:
                                        # Analyze section improvement effect
                                        effect = analyzer.analyze_section_improvement_effect(
                                            causal_data,
                                            section_id=section_id,
                                            outcome='lap_time'
                                        )

                                        # Display results
                                        st.success("✅ Causal Effect Estimated")

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.metric(
                                                "Effect Size",
                                                f"{effect.effect_size:.4f}s/s",
                                                help="Change in lap time per 1 second improvement in section"
                                            )

                                        with col2:
                                            ci_range = effect.confidence_interval[1] - effect.confidence_interval[0]
                                            st.metric(
                                                "95% Confidence",
                                                f"±{ci_range/2:.4f}s",
                                                help="Uncertainty in the effect estimate"
                                            )

                                        with col3:
                                            sig_label = "Significant" if effect.p_value < 0.05 else "Not Significant"
                                            st.metric(
                                                "Statistical Significance",
                                                sig_label,
                                                help=f"p-value: {effect.p_value:.4f}"
                                            )

                                        # Interpretation
                                        st.subheader("Interpretation")
                                        st.info(effect.interpretation)

                                        # Counterfactual prediction
                                        st.subheader("Counterfactual Prediction")

                                        current_section_time = causal_data[f'section_{section_id}_time'].mean()
                                        improved_section_time = current_section_time - improvement

                                        counterfactual = analyzer.estimate_counterfactual(
                                            data=causal_data,
                                            treatment=f'section_{section_id}_time',
                                            outcome='lap_time',
                                            intervention_value=improved_section_time
                                        )

                                        st.markdown(f"""
                                        **Scenario:** Improve Section {section_id} by {improvement:.2f} seconds

                                        - Current avg Section {section_id} time: **{current_section_time:.3f}s**
                                        - Improved Section {section_id} time: **{improved_section_time:.3f}s**
                                        - Current avg lap time: **{counterfactual.original_outcome:.3f}s**
                                        - **Predicted** avg lap time: **{counterfactual.counterfactual_outcome:.3f}s**
                                        - **Expected improvement:** {counterfactual.effect_size:.3f}s per lap

                                        {counterfactual.practical_interpretation}
                                        """)

                                        # Robustness check
                                        st.subheader("Robustness Analysis")

                                        robustness_label = "HIGH" if effect.robustness_score >= 0.75 else \
                                                          "MODERATE" if effect.robustness_score >= 0.5 else "LOW"

                                        st.metric(
                                            "Robustness Score",
                                            f"{effect.robustness_score:.2f}",
                                            help="Confidence in causal estimate after sensitivity tests"
                                        )

                                        if effect.robustness_score >= 0.75:
                                            st.success("✅ Effect is robust to sensitivity tests")
                                        elif effect.robustness_score >= 0.5:
                                            st.warning("⚠️ Moderate sensitivity to unmeasured confounding")
                                        else:
                                            st.error("❌ Effect may be due to unmeasured confounding")

                                    except Exception as e:
                                        st.error(f"Error in causal analysis: {str(e)}")
                                        st.exception(e)

                        elif analysis_type == "Pit Strategy Effect":
                            st.subheader("Pit Strategy Causal Effect")

                            st.markdown("""
                            This analysis estimates the **causal effect** of pit timing on race
                            position, controlling for factors like starting position and pace.
                            """)

                            # Check if we have pit and position data
                            if 'pit_lap' not in causal_data.columns or 'final_position' not in causal_data.columns:
                                st.warning("⚠️ Insufficient data for pit strategy analysis. "
                                          "Need pit lap and final position information.")
                            else:
                                if st.button("Analyze Pit Strategy Effect", type="primary"):
                                    with st.spinner("Running causal analysis..."):
                                        try:
                                            effect = analyzer.analyze_pit_strategy_effect(
                                                causal_data,
                                                outcome='final_position'
                                            )

                                            st.success("✅ Causal Effect Estimated")

                                            # Display results
                                            col1, col2 = st.columns(2)

                                            with col1:
                                                st.metric(
                                                    "Effect Size",
                                                    f"{effect.effect_size:.4f} positions/lap",
                                                    help="Change in final position per lap delay in pit timing"
                                                )

                                            with col2:
                                                sig_label = "Significant" if effect.p_value < 0.05 else "Not Significant"
                                                st.metric(
                                                    "Statistical Significance",
                                                    sig_label,
                                                    help=f"p-value: {effect.p_value:.4f}"
                                                )

                                            st.info(effect.interpretation)

                                        except Exception as e:
                                            st.error(f"Error in pit strategy analysis: {str(e)}")
                                            st.exception(e)

                        elif analysis_type == "Custom Causal Analysis":
                            st.subheader("Custom Causal Analysis")

                            st.markdown("Build your own causal analysis by selecting variables.")

                            # Variable selection
                            available_vars = [col for col in causal_data.columns
                                            if col not in ['driver_number', 'lap_number']]

                            col1, col2 = st.columns(2)

                            with col1:
                                treatment_var = st.selectbox(
                                    "Treatment Variable",
                                    available_vars,
                                    help="The variable you want to intervene on"
                                )

                            with col2:
                                outcome_var = st.selectbox(
                                    "Outcome Variable",
                                    [v for v in available_vars if v != treatment_var],
                                    help="The variable you want to measure"
                                )

                            # Common causes
                            potential_confounders = [v for v in available_vars
                                                    if v not in [treatment_var, outcome_var]]

                            common_causes = st.multiselect(
                                "Control Variables (Confounders)",
                                potential_confounders,
                                help="Variables that might affect both treatment and outcome"
                            )

                            if st.button("Run Causal Analysis", type="primary"):
                                with st.spinner("Analyzing..."):
                                    try:
                                        effect = analyzer.identify_causal_effect(
                                            data=causal_data,
                                            treatment=treatment_var,
                                            outcome=outcome_var,
                                            common_causes=common_causes if common_causes else None
                                        )

                                        st.success("✅ Analysis Complete")

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.metric("Effect Size", f"{effect.effect_size:.4f}")

                                        with col2:
                                            st.metric(
                                                "Confidence Interval",
                                                f"[{effect.confidence_interval[0]:.4f}, "
                                                f"{effect.confidence_interval[1]:.4f}]"
                                            )

                                        with col3:
                                            st.metric("P-Value", f"{effect.p_value:.4f}")

                                        st.info(effect.interpretation)

                                        # Robustness
                                        st.metric("Robustness Score", f"{effect.robustness_score:.2f}")

                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                                        st.exception(e)

                        elif analysis_type == "View Causal Graph":
                            st.subheader("Causal Graph Visualization")

                            st.markdown("""
                            This directed acyclic graph (DAG) shows the assumed causal relationships
                            between racing performance variables.
                            """)

                            if st.button("Generate Causal Graph"):
                                with st.spinner("Building causal graph..."):
                                    try:
                                        # Build graph
                                        analyzer.build_causal_graph(causal_data)

                                        # Visualize
                                        fig = analyzer.visualize_causal_graph()

                                        st.pyplot(fig)

                                        # Show graph info
                                        st.subheader("Graph Structure")

                                        col1, col2 = st.columns(2)

                                        with col1:
                                            st.metric("Nodes", analyzer.causal_graph['metadata']['num_nodes'])

                                        with col2:
                                            st.metric("Edges", analyzer.causal_graph['metadata']['num_edges'])

                                        with st.expander("View Edge List"):
                                            edges_df = pd.DataFrame(
                                                analyzer.causal_graph['edges'],
                                                columns=['From', 'To']
                                            )
                                            st.dataframe(edges_df, hide_index=True)

                                    except Exception as e:
                                        st.error(f"Error generating graph: {str(e)}")
                                        st.exception(e)

                    except Exception as e:
                        st.error(f"Error preparing data for causal analysis: {str(e)}")
                        st.exception(e)

            # TAB 3: Cross-Module Impact (move existing content here)
            with tab3:
                st.header("Cross-Module Impact Analysis")
                st.subheader("How Improvements Affect Overall Performance")

                # This would ideally be the cross-module impact content
                # For now, show a summary
                st.info("Cross-module impact analysis shows how tactical improvements "
                       "(section times, consistency) affect strategic outcomes "
                       "(pit timing, final position).")

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
                        total_laps = driver_data['LAP_NUMBER'].max() if 'LAP_NUMBER' in driver_data.columns else 25
                        total_time_gain = time_gain_per_lap * total_laps

                        impacts.append({
                            'Improvement Area': f'{section_name} Optimization',
                            'Module': 'Tactical',
                            'Time Gain/Lap': f'{time_gain_per_lap:.3f}s',
                            'Total Race Gain': f'{total_time_gain:.1f}s',
                            'Difficulty': 'Medium'
                        })

                    if impacts:
                        impact_df = pd.DataFrame(impacts)
                        st.dataframe(impact_df, hide_index=True, use_container_width=True)

                        # Visualize
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
                                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                text=[f'{g:.1f}s' for g in gains],
                                textposition='auto'
                            ))

                            fig.update_layout(
                                xaxis_title="Improvement Area",
                                yaxis_title="Total Time Gain (seconds)",
                                height=400
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            total_potential = sum(gains)
                            st.success(f"**Total Potential Time Gain: {total_potential:.1f} seconds**")

        else:
            st.warning("No driver data available")

    except Exception as e:
        st.error(f"Error displaying integrated insights: {str(e)}")
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
        show_integrated_insights(data, track, race_num)
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
                key="integrated_track"
            )
            
            race_num = st.sidebar.selectbox(
                "Select Race",
                [1, 2],
                key="integrated_race"
            )
            
            with st.spinner("Loading race data..."):
                data = load_race_data_local(track, race_num)
            
            if data:
                show_integrated_insights(data, track, race_num)
            else:
                st.error("Failed to load race data. Please check the data directory.")
        except Exception as e:
            st.error(f"Error loading page: {str(e)}")
            st.exception(e)

# Run main if this is executed as a script (for Streamlit multi-page)
if __name__ == "__main__":
    main()
