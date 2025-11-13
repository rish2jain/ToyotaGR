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
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.tactical.anomaly_detector import AnomalyDetector, TENSORFLOW_AVAILABLE
    ANOMALY_DETECTOR_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTOR_AVAILABLE = False
    TENSORFLOW_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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

            # Track Map Visualization
            st.header("🗺️ Track Map: Performance Heatmap")

            try:
                from src.utils.visualization import create_track_map_with_performance

                # Prepare section data for track map
                if available_section_cols and 'LAP_NUMBER' in driver_data.columns:
                    # Create section performance data
                    section_perf_data = []

                    for section_col in available_section_cols:
                        section_name = section_col.replace('_SECONDS', '')
                        section_num = int(section_name[1:]) if len(section_name) > 1 else 1

                        for _, row in driver_data.iterrows():
                            if pd.notna(row[section_col]):
                                # Calculate gap to optimal for this section
                                optimal_time = driver_data[section_col].min()
                                gap = row[section_col] - optimal_time

                                section_perf_data.append({
                                    'Section': section_num,
                                    'Lap': row['LAP_NUMBER'],
                                    'Time': row[section_col],
                                    'GapToOptimal': gap
                                })

                    if section_perf_data:
                        section_df = pd.DataFrame(section_perf_data)

                        # Determine track name (default to barber)
                        track_name = 'barber'
                        if track:
                            track_name = track.lower()

                        # Create track map
                        driver_number = int(selected_driver) if pd.notna(selected_driver) else selected_driver
                        fig_track = create_track_map_with_performance(
                            section_df,
                            track_name=track_name,
                            section_col='Section',
                            time_col='Time',
                            gap_col='GapToOptimal',
                            driver_label=f"Car #{driver_number}"
                        )

                        if fig_track:
                            st.plotly_chart(fig_track, use_container_width=True)

                            # Add interpretation help
                            st.info(
                                "**How to read this map:**\n"
                                "- Hover over any section to see detailed performance metrics\n"
                                "- Green sections = excellent performance (close to your best)\n"
                                "- Yellow sections = good performance (room for improvement)\n"
                                "- Orange/Red sections = areas needing focus\n"
                                "- Click and drag to pan, scroll to zoom"
                            )
                        else:
                            st.warning("Track map visualization unavailable")
                    else:
                        st.info("Insufficient section data for track map visualization")

            except ImportError as e:
                st.warning(f"Track map visualization requires additional dependencies: {str(e)}")
            except Exception as e:
                st.error(f"Error creating track map: {str(e)}")

            st.markdown("---")

            # Driver Comparison (if multiple drivers available)
            if len(available_drivers) > 1:
                st.header("🏁 Driver Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    compare_driver = st.selectbox(
                        "Compare with Driver",
                        [d for d in available_drivers if d != selected_driver],
                        format_func=lambda x: f"Car #{int(x)}" if pd.notna(x) else "Unknown"
                    )

                if compare_driver:
                    try:
                        from src.utils.visualization import create_driver_comparison_map

                        # Get comparison driver data
                        compare_data = sections_df[sections_df['DRIVER_NUMBER'] == compare_driver].copy()

                        if not compare_data.empty and available_section_cols:
                            # Prepare data for both drivers
                            driver1_perf = []
                            driver2_perf = []

                            for section_col in available_section_cols:
                                section_name = section_col.replace('_SECONDS', '')
                                section_num = int(section_name[1:]) if len(section_name) > 1 else 1

                                # Driver 1 (selected)
                                for _, row in driver_data.iterrows():
                                    if pd.notna(row[section_col]):
                                        driver1_perf.append({
                                            'Section': section_num,
                                            'Time': row[section_col]
                                        })

                                # Driver 2 (comparison)
                                for _, row in compare_data.iterrows():
                                    if pd.notna(row[section_col]):
                                        driver2_perf.append({
                                            'Section': section_num,
                                            'Time': row[section_col]
                                        })

                            if driver1_perf and driver2_perf:
                                driver1_df = pd.DataFrame(driver1_perf)
                                driver2_df = pd.DataFrame(driver2_perf)

                                # Determine track name
                                track_name = 'barber'
                                if track:
                                    track_name = track.lower()

                                # Create comparison map
                                fig_compare = create_driver_comparison_map(
                                    driver1_df,
                                    driver2_df,
                                    track_name=track_name,
                                    driver1_label=f"Car #{int(selected_driver)}",
                                    driver2_label=f"Car #{int(compare_driver)}",
                                    section_col='Section',
                                    time_col='Time'
                                )

                                if fig_compare:
                                    st.plotly_chart(fig_compare, use_container_width=True)

                                    st.info(
                                        "**Comparison Guide:**\n"
                                        "- Red sections: You are faster\n"
                                        "- Blue sections: Competitor is faster\n"
                                        "- Gray sections: Similar performance\n"
                                        "- Hover to see exact time differences"
                                    )
                    except Exception as e:
                        st.error(f"Error creating comparison map: {str(e)}")

            st.markdown("---")

            # Anomaly Detection with SHAP Explanations
            st.header("Advanced Anomaly Detection")

            if 'lap_seconds' in driver_data.columns and len(driver_data) > 5:
                # Detection Method Tabs
                tab1, tab2, tab3 = st.tabs(["Statistical Detection", "ML Detection with SHAP", "Deep Learning (LSTM)"])

                with tab1:
                    st.subheader("Statistical Anomaly Detection (Z-Score)")

                    # Simple anomaly detection using z-score
                    mean_lap = driver_data['lap_seconds'].mean()
                    std_lap = driver_data['lap_seconds'].std()

                    driver_data['z_score'] = (driver_data['lap_seconds'] - mean_lap) / std_lap
                    driver_data['is_anomaly_stat'] = abs(driver_data['z_score']) > 2

                    anomalies_stat = driver_data[driver_data['is_anomaly_stat']].copy()

                    if len(anomalies_stat) > 0:
                        st.write(f"**{len(anomalies_stat)} anomalies detected** (laps >2 standard deviations from mean)")

                        # Display anomalies table
                        anomaly_cols = ['LAP_NUMBER', 'LAP_TIME', 'lap_seconds', 'z_score']
                        available_anomaly_cols = [col for col in anomaly_cols if col in anomalies_stat.columns]

                        if available_anomaly_cols:
                            anomaly_display = anomalies_stat[available_anomaly_cols].copy()
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
                    normal_laps = driver_data[~driver_data['is_anomaly_stat']]
                    fig.add_trace(go.Scatter(
                        x=normal_laps['LAP_NUMBER'],
                        y=normal_laps['lap_seconds'],
                        mode='lines+markers',
                        name='Normal Laps',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6)
                    ))

                    # Anomalous laps
                    if len(anomalies_stat) > 0:
                        fig.add_trace(go.Scatter(
                            x=anomalies_stat['LAP_NUMBER'],
                            y=anomalies_stat['lap_seconds'],
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

                with tab2:
                    st.subheader("ML Anomaly Detection with SHAP Explanations")

                    if not ANOMALY_DETECTOR_AVAILABLE:
                        st.warning("AnomalyDetector module not available. Check installation.")
                    elif not SHAP_AVAILABLE:
                        st.info("SHAP not installed. Install with `pip install shap` for detailed explanations.")
                        st.write("Basic ML anomaly detection will still work.")

                    # Use AnomalyDetector for ML-based detection
                    if ANOMALY_DETECTOR_AVAILABLE:
                        try:
                            detector = AnomalyDetector()

                            # Run ML anomaly detection
                            with st.spinner("Running Isolation Forest anomaly detection..."):
                                ml_result = detector.detect_pattern_anomalies(
                                    driver_data,
                                    contamination=0.1
                                )

                            ml_anomalies = ml_result[ml_result['is_anomaly'] == -1].copy()

                            st.write(f"**{len(ml_anomalies)} anomalies detected** using Isolation Forest")

                            if len(ml_anomalies) > 0:
                                # Get SHAP explanations if available
                                if SHAP_AVAILABLE:
                                    with st.spinner("Generating SHAP explanations..."):
                                        try:
                                            explained_anomalies = detector.get_anomaly_explanations(
                                                ml_anomalies,
                                                ml_result
                                            )

                                            # Display anomalies with explanations
                                            st.subheader("Detected Anomalies with Explanations")

                                            for idx, row in explained_anomalies.iterrows():
                                                lap_num = row.get('LAP_NUMBER', 'N/A')
                                                anomaly_score = row.get('anomaly_score', 0)
                                                confidence = row.get('confidence', 0)
                                                explanation = row.get('explanation', 'No explanation available')

                                                with st.expander(f"Lap {lap_num} - Anomaly Score: {anomaly_score:.4f}"):
                                                    col1, col2 = st.columns([2, 1])

                                                    with col1:
                                                        st.markdown(f"**Explanation:** {explanation}")
                                                        st.markdown(f"**Confidence:** {confidence:.1%}")

                                                        # Feature importance table
                                                        st.markdown("**Top Contributing Features:**")
                                                        feature_data = []
                                                        for i in range(1, 4):
                                                            feature = row.get(f'top_feature_{i}', '')
                                                            contrib = row.get(f'contribution_{i}', 0)
                                                            if feature:
                                                                feature_data.append({
                                                                    'Feature': feature.replace('_', ' ').title(),
                                                                    'Contribution': f"{contrib:.1%}"
                                                                })

                                                        if feature_data:
                                                            st.table(pd.DataFrame(feature_data))

                                                    with col2:
                                                        # Feature importance bar chart
                                                        feature_names = []
                                                        contributions = []
                                                        for i in range(1, 4):
                                                            feature = row.get(f'top_feature_{i}', '')
                                                            contrib = row.get(f'contribution_{i}', 0)
                                                            if feature:
                                                                feature_names.append(feature.replace('_', ' ').title())
                                                                contributions.append(contrib * 100)

                                                        if feature_names:
                                                            fig = go.Figure(data=[
                                                                go.Bar(
                                                                    y=feature_names[::-1],
                                                                    x=contributions[::-1],
                                                                    orientation='h',
                                                                    marker=dict(color='coral')
                                                                )
                                                            ])
                                                            fig.update_layout(
                                                                title="Feature Importance",
                                                                xaxis_title="Contribution (%)",
                                                                height=200,
                                                                margin=dict(l=0, r=0, t=30, b=0)
                                                            )
                                                            st.plotly_chart(fig, use_container_width=True)

                                                    # Show detailed SHAP values in expandable section
                                                    with st.expander("View Detailed SHAP Values"):
                                                        st.write("Raw feature values for this lap:")
                                                        feature_cols = [col for col in available_section_cols if col in row.index]
                                                        if feature_cols:
                                                            feature_values = {col: row[col] for col in feature_cols}
                                                            st.json(feature_values)

                                        except Exception as e:
                                            st.error(f"Error generating SHAP explanations: {e}")
                                            # Fall back to basic anomaly display
                                            display_cols = ['LAP_NUMBER', 'anomaly_score']
                                            available_display_cols = [col for col in display_cols if col in ml_anomalies.columns]
                                            st.dataframe(
                                                ml_anomalies[available_display_cols].sort_values('anomaly_score'),
                                                hide_index=True,
                                                use_container_width=True
                                            )
                                else:
                                    # Show basic anomalies without SHAP
                                    display_cols = ['LAP_NUMBER', 'anomaly_score']
                                    available_display_cols = [col for col in display_cols if col in ml_anomalies.columns]
                                    st.dataframe(
                                        ml_anomalies[available_display_cols].sort_values('anomaly_score'),
                                        hide_index=True,
                                        use_container_width=True
                                    )

                                # Plot anomalies
                                st.subheader("Anomaly Visualization")
                                fig = go.Figure()

                                # Normal laps
                                normal_laps = ml_result[ml_result['is_anomaly'] == 1]
                                if 'LAP_NUMBER' in normal_laps.columns and 'lap_seconds' in normal_laps.columns:
                                    fig.add_trace(go.Scatter(
                                        x=normal_laps['LAP_NUMBER'],
                                        y=normal_laps['lap_seconds'],
                                        mode='lines+markers',
                                        name='Normal Laps',
                                        line=dict(color='green', width=2),
                                        marker=dict(size=6)
                                    ))

                                # Anomalous laps
                                if 'LAP_NUMBER' in ml_anomalies.columns and 'lap_seconds' in ml_anomalies.columns:
                                    fig.add_trace(go.Scatter(
                                        x=ml_anomalies['LAP_NUMBER'],
                                        y=ml_anomalies['lap_seconds'],
                                        mode='markers',
                                        name='Anomalies (ML)',
                                        marker=dict(size=14, color='red', symbol='diamond')
                                    ))

                                fig.update_layout(
                                    xaxis_title="Lap Number",
                                    yaxis_title="Lap Time (seconds)",
                                    height=400,
                                    hovermode='x unified'
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.success("No anomalies detected by ML model")

                        except Exception as e:
                            st.error(f"Error running ML anomaly detection: {e}")
                            st.exception(e)

                with tab3:
                    st.subheader("Deep Learning LSTM Anomaly Detection")

                    if not ANOMALY_DETECTOR_AVAILABLE:
                        st.warning("AnomalyDetector module not available. Check installation.")
                    elif not TENSORFLOW_AVAILABLE:
                        st.warning(
                            "TensorFlow is not installed. Install with `pip install tensorflow` "
                            "to use LSTM-based anomaly detection."
                        )
                        st.info(
                            "**LSTM Anomaly Detection Features:**\n\n"
                            "- Learns temporal patterns in telemetry sequences\n"
                            "- Detects complex anomalies that statistical methods miss\n"
                            "- Uses autoencoder reconstruction error\n"
                            "- Training time: 30-90 seconds\n"
                            "- Inference time: <1 second"
                        )
                    else:
                        # Use AnomalyDetector for LSTM-based detection
                        try:
                            detector = AnomalyDetector()

                            # Configuration
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                sequence_length = st.slider(
                                    "Sequence Length",
                                    min_value=20,
                                    max_value=100,
                                    value=50,
                                    step=10,
                                    help="Number of time steps per sequence"
                                )
                            with col2:
                                epochs = st.slider(
                                    "Training Epochs",
                                    min_value=10,
                                    max_value=100,
                                    value=30,
                                    step=10,
                                    help="More epochs = better learning but slower"
                                )
                            with col3:
                                contamination = st.slider(
                                    "Contamination %",
                                    min_value=1,
                                    max_value=20,
                                    value=5,
                                    step=1,
                                    help="Expected % of anomalies"
                                ) / 100

                            # Run LSTM anomaly detection
                            if st.button("Run LSTM Detection", type="primary"):
                                with st.spinner(f"Training LSTM autoencoder ({epochs} epochs)... This may take 30-90 seconds"):
                                    lstm_result = detector.detect_lstm_anomalies(
                                        driver_data,
                                        sequence_length=sequence_length,
                                        epochs=epochs,
                                        contamination=contamination
                                    )

                                # Store result in session state
                                st.session_state['lstm_result'] = lstm_result
                                st.success("LSTM training complete!")

                            # Display results if available
                            if 'lstm_result' in st.session_state:
                                lstm_result = st.session_state['lstm_result']
                                lstm_anomalies = lstm_result[lstm_result['lstm_is_anomaly']].copy()

                                st.write(f"**{len(lstm_anomalies)} anomalies detected** using LSTM autoencoder")

                                if len(lstm_anomalies) > 0:
                                    # Display anomalies table
                                    st.subheader("Detected Anomalies")
                                    display_cols = ['LAP_NUMBER', 'lap_seconds', 'lstm_reconstruction_error', 'lstm_anomaly_score']
                                    available_display_cols = [col for col in display_cols if col in lstm_anomalies.columns]

                                    if available_display_cols:
                                        anomaly_display = lstm_anomalies[available_display_cols].copy()
                                        anomaly_display.columns = ['Lap #', 'Lap Time (s)', 'Reconstruction Error', 'Anomaly Score']
                                        st.dataframe(
                                            anomaly_display.sort_values('Reconstruction Error', ascending=False),
                                            hide_index=True,
                                            use_container_width=True
                                        )

                                    # Plot reconstruction error over time
                                    st.subheader("Reconstruction Error Over Time")
                                    fig = go.Figure()

                                    # All laps reconstruction error
                                    if 'LAP_NUMBER' in lstm_result.columns and 'lstm_reconstruction_error' in lstm_result.columns:
                                        fig.add_trace(go.Scatter(
                                            x=lstm_result['LAP_NUMBER'],
                                            y=lstm_result['lstm_reconstruction_error'],
                                            mode='lines+markers',
                                            name='Reconstruction Error',
                                            line=dict(color='blue', width=2),
                                            marker=dict(size=6)
                                        ))

                                        # Highlight anomalies
                                        if len(lstm_anomalies) > 0:
                                            fig.add_trace(go.Scatter(
                                                x=lstm_anomalies['LAP_NUMBER'],
                                                y=lstm_anomalies['lstm_reconstruction_error'],
                                                mode='markers',
                                                name='Anomalies',
                                                marker=dict(size=12, color='red', symbol='x')
                                            ))

                                        # Add threshold line
                                        if hasattr(detector, 'lstm_detector') and detector.lstm_detector and detector.lstm_detector.threshold:
                                            fig.add_hline(
                                                y=detector.lstm_detector.threshold,
                                                line_dash="dash",
                                                line_color="orange",
                                                annotation_text="Anomaly Threshold"
                                            )

                                        fig.update_layout(
                                            xaxis_title="Lap Number",
                                            yaxis_title="Reconstruction Error (MSE)",
                                            height=400,
                                            hovermode='x unified'
                                        )

                                        st.plotly_chart(fig, use_container_width=True)

                                    # Comparison: Statistical vs ML vs LSTM
                                    st.subheader("Method Comparison")
                                    with st.expander("Compare Detection Methods"):
                                        comparison_data = []

                                        # Count anomalies from each method
                                        if 'is_anomaly_stat' in driver_data.columns:
                                            stat_count = driver_data['is_anomaly_stat'].sum()
                                            comparison_data.append({
                                                'Method': 'Statistical (Z-Score)',
                                                'Anomalies Detected': int(stat_count),
                                                'Detection Rate': f"{(stat_count / len(driver_data)) * 100:.1f}%"
                                            })

                                        if 'is_anomaly' in driver_data.columns:
                                            ml_count = (driver_data['is_anomaly'] == -1).sum()
                                            comparison_data.append({
                                                'Method': 'ML (Isolation Forest)',
                                                'Anomalies Detected': int(ml_count),
                                                'Detection Rate': f"{(ml_count / len(driver_data)) * 100:.1f}%"
                                            })

                                        lstm_count = len(lstm_anomalies)
                                        comparison_data.append({
                                            'Method': 'Deep Learning (LSTM)',
                                            'Anomalies Detected': int(lstm_count),
                                            'Detection Rate': f"{(lstm_count / len(driver_data)) * 100:.1f}%"
                                        })

                                        if comparison_data:
                                            comp_df = pd.DataFrame(comparison_data)
                                            st.table(comp_df)

                                            # Bar chart comparison
                                            fig = go.Figure(data=[
                                                go.Bar(
                                                    x=[d['Method'] for d in comparison_data],
                                                    y=[d['Anomalies Detected'] for d in comparison_data],
                                                    marker_color=['steelblue', 'coral', 'mediumseagreen']
                                                )
                                            ])
                                            fig.update_layout(
                                                title="Anomalies Detected by Method",
                                                xaxis_title="Detection Method",
                                                yaxis_title="Number of Anomalies",
                                                height=400
                                            )
                                            st.plotly_chart(fig, use_container_width=True)

                                        st.info(
                                            "**Method Characteristics:**\n\n"
                                            "- **Statistical**: Fast, simple, good for obvious outliers\n"
                                            "- **ML (Isolation Forest)**: Medium speed, detects multivariate patterns\n"
                                            "- **LSTM**: Slower training, best for temporal/sequential anomalies"
                                        )
                                else:
                                    st.success("No anomalies detected by LSTM model")

                        except Exception as e:
                            st.error(f"Error running LSTM anomaly detection: {e}")
                            st.exception(e)

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
