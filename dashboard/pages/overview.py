"""
Race Overview Page
Displays race summary, leaderboard, and key statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_race_overview(data, track, race_num):
    """
    Display race overview with summary statistics and leaderboard

    Args:
        data: Dictionary containing race dataframes
        track: Track name
        race_num: Race number
    """
    st.title(f"🏁 Race Overview: {track.replace('-', ' ').title()} - Race {race_num}")

    try:
        # Race Summary Section
        st.header("Race Summary")

        if 'results' in data and not data['results'].empty:
            results_df = data['results']

            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_drivers = len(results_df)
                st.metric("Total Drivers", total_drivers, delta=None)

            with col2:
                if 'LAPS' in results_df.columns:
                    max_laps = results_df['LAPS'].max()
                    st.metric("Total Laps", int(max_laps) if pd.notna(max_laps) else "N/A")
                else:
                    st.metric("Total Laps", "N/A")

            with col3:
                if 'FL_KPH' in results_df.columns:
                    fastest_speed = results_df['FL_KPH'].max()
                    st.metric("Top Speed", f"{fastest_speed:.1f} km/h" if pd.notna(fastest_speed) else "N/A")
                else:
                    st.metric("Top Speed", "N/A")

            with col4:
                if 'FL_TIME' in results_df.columns:
                    # Convert FL_TIME to seconds for averaging
                    fastest_lap = results_df['FL_TIME'].min()
                    st.metric("Fastest Lap", fastest_lap if pd.notna(fastest_lap) else "N/A")
                else:
                    st.metric("Fastest Lap", "N/A")

            st.markdown("---")

            # Leaderboard Section
            st.header("Final Standings")

            # Prepare leaderboard data
            leaderboard_columns = ['POSITION', 'NUMBER', 'STATUS', 'LAPS', 'TOTAL_TIME', 'GAP_FIRST', 'FL_TIME', 'FL_KPH']
            available_columns = [col for col in leaderboard_columns if col in results_df.columns]

            if available_columns:
                leaderboard_df = results_df[available_columns].copy()

                # Rename columns for better display
                column_rename = {
                    'POSITION': 'Pos',
                    'NUMBER': 'Car #',
                    'STATUS': 'Status',
                    'LAPS': 'Laps',
                    'TOTAL_TIME': 'Total Time',
                    'GAP_FIRST': 'Gap',
                    'FL_TIME': 'Fastest Lap',
                    'FL_KPH': 'Speed (km/h)'
                }
                leaderboard_df.rename(columns={k: v for k, v in column_rename.items() if k in leaderboard_df.columns}, inplace=True)

                # Style the dataframe
                st.dataframe(
                    leaderboard_df,
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("Leaderboard data not available")

            st.markdown("---")

            # Race Statistics Section
            st.header("Race Statistics")

            col1, col2 = st.columns(2)

            with col1:
                # Fastest lap distribution
                if 'FL_TIME' in results_df.columns and 'NUMBER' in results_df.columns:
                    st.subheader("Fastest Lap Times")

                    # Convert lap times to seconds for plotting
                    lap_data = results_df[['NUMBER', 'FL_TIME']].dropna()

                    if not lap_data.empty:
                        # Convert time string to seconds
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

                        lap_data['seconds'] = lap_data['FL_TIME'].apply(time_to_seconds)
                        lap_data = lap_data.dropna()

                        if not lap_data.empty:
                            fig = px.bar(
                                lap_data.sort_values('seconds'),
                                x='NUMBER',
                                y='seconds',
                                labels={'seconds': 'Lap Time (s)', 'NUMBER': 'Car Number'},
                                color='seconds',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(
                                showlegend=False,
                                height=350,
                                xaxis_title="Car Number",
                                yaxis_title="Fastest Lap (seconds)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Unable to parse lap time data")
                    else:
                        st.info("No fastest lap data available")

            with col2:
                # Completion status pie chart
                if 'STATUS' in results_df.columns:
                    st.subheader("Race Completion Status")

                    status_counts = results_df['STATUS'].value_counts()

                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Status data not available")

            # Section performance analysis
            if 'sections' in data and not data['sections'].empty:
                st.markdown("---")
                st.header("Section Performance Analysis")

                sections_df = data['sections']

                # Get top 5 drivers
                if 'DRIVER_NUMBER' in sections_df.columns:
                    top_drivers = results_df.head(5)['NUMBER'].tolist() if 'NUMBER' in results_df.columns else []

                    if top_drivers:
                        # Filter for top 5 drivers
                        top_sections = sections_df[sections_df['DRIVER_NUMBER'].isin(top_drivers)]

                        # Get section columns (S1, S2, S3)
                        section_cols = [col for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'] if col in top_sections.columns]

                        if section_cols:
                            st.subheader("Top 5 Drivers - Average Section Times")

                            # Calculate average section times
                            avg_sections = top_sections.groupby('DRIVER_NUMBER')[section_cols].mean().reset_index()

                            # Melt for plotting
                            avg_sections_melted = avg_sections.melt(
                                id_vars='DRIVER_NUMBER',
                                value_vars=section_cols,
                                var_name='Section',
                                value_name='Time (s)'
                            )

                            # Clean section names
                            avg_sections_melted['Section'] = avg_sections_melted['Section'].str.replace('_SECONDS', '')

                            fig = px.bar(
                                avg_sections_melted,
                                x='DRIVER_NUMBER',
                                y='Time (s)',
                                color='Section',
                                barmode='group',
                                labels={'DRIVER_NUMBER': 'Driver Number'},
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

            # Weather conditions
            if 'weather' in data and not data['weather'].empty:
                st.markdown("---")
                st.header("Weather Conditions")

                weather_df = data['weather']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if 'AIR_TEMP' in weather_df.columns:
                        avg_temp = weather_df['AIR_TEMP'].mean()
                        st.metric("Avg Air Temp", f"{avg_temp:.1f}°C" if pd.notna(avg_temp) else "N/A")

                with col2:
                    if 'TRACK_TEMP' in weather_df.columns:
                        avg_track_temp = weather_df['TRACK_TEMP'].mean()
                        st.metric("Avg Track Temp", f"{avg_track_temp:.1f}°C" if pd.notna(avg_track_temp) else "N/A")

                with col3:
                    if 'HUMIDITY' in weather_df.columns:
                        avg_humidity = weather_df['HUMIDITY'].mean()
                        st.metric("Avg Humidity", f"{avg_humidity:.1f}%" if pd.notna(avg_humidity) else "N/A")

                with col4:
                    if 'WIND_SPEED' in weather_df.columns:
                        avg_wind = weather_df['WIND_SPEED'].mean()
                        st.metric("Avg Wind Speed", f"{avg_wind:.1f} km/h" if pd.notna(avg_wind) else "N/A")

        else:
            st.warning("No race results data available")

    except Exception as e:
        st.error(f"Error displaying race overview: {str(e)}")
        st.exception(e)
