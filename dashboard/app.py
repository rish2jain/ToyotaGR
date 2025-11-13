"""
RaceIQ Pro - Streamlit Dashboard
Main entry point for the racing intelligence dashboard
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import page modules
from pages import overview, tactical, strategic, integrated, race_simulator

# Configure the page
st.set_page_config(
    page_title="RaceIQ Pro",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .recommendation-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .rec-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .rec-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .rec-low {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_race_data(track="barber", race_num=1):
    """
    Load race data from CSV files with caching

    Args:
        track: Track name (barber, COTA, sonoma, etc.)
        race_num: Race number (1 or 2)

    Returns:
        dict: Dictionary containing all loaded dataframes
    """
    try:
        # Determine the correct path based on track
        base_path = Path(__file__).parent.parent / "Data"

        # Map track names to folder names
        track_map = {
            "barber": "barber",
            "cota": "COTA",
            "sonoma": "Sonoma",
            "indianapolis": "indianapolis",
            "road-america": "road-america/Road America",
            "sebring": "sebring/Sebring"
        }

        track_folder = track_map.get(track.lower(), "barber")

        # Adjust for different folder structures
        if track.lower() in ["barber", "cota", "sonoma"]:
            if track.lower() == "barber":
                race_folder = base_path / track_folder
            else:
                race_folder = base_path / track_folder / f"Race {race_num}"
        else:
            race_folder = base_path / track_folder / f"Race {race_num}"

        data = {}

        # Load provisional results
        results_files = list(race_folder.glob("03_*Results*.CSV")) + list(race_folder.glob("03_*Results*.csv"))
        if results_files:
            data['results'] = pd.read_csv(results_files[0], delimiter=';')

        # Load section analysis
        section_files = list(race_folder.glob("23_*Sections*.CSV")) + list(race_folder.glob("23_*Sections*.csv"))
        if section_files:
            data['sections'] = pd.read_csv(section_files[0], delimiter=';')

        # Load lap times
        lap_files = list(race_folder.glob("*lap_time*.csv"))
        if lap_files:
            data['lap_times'] = pd.read_csv(lap_files[0])

        # Load best laps
        best_lap_files = list(race_folder.glob("99_*Best*.CSV")) + list(race_folder.glob("99_*Best*.csv"))
        if best_lap_files:
            data['best_laps'] = pd.read_csv(best_lap_files[0], delimiter=';')

        # Load weather data
        weather_files = list(race_folder.glob("26_*Weather*.CSV")) + list(race_folder.glob("26_*Weather*.csv"))
        if weather_files:
            data['weather'] = pd.read_csv(weather_files[0], delimiter=';')

        return data

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

def main():
    """Main application logic"""

    # Sidebar navigation
    st.sidebar.title("🏁 RaceIQ Pro")
    st.sidebar.markdown("---")

    # Track and race selection
    st.sidebar.subheader("Race Selection")
    track = st.sidebar.selectbox(
        "Select Track",
        ["barber", "cota", "sonoma", "indianapolis", "road-america", "sebring"],
        format_func=lambda x: x.replace("-", " ").title()
    )

    race_num = st.sidebar.selectbox("Select Race", [1, 2])

    st.sidebar.markdown("---")

    # Page navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏁 Race Overview", "🎯 Tactical Analysis", "⚙️ Strategic Analysis", "🔗 Integrated Insights", "🏎️ Race Simulator"],
        label_visibility="collapsed"
    )

    # Load data
    with st.spinner("Loading race data..."):
        data = load_race_data(track, race_num)

    if not data:
        st.error("Failed to load race data. Please check the data directory.")
        return

    # Store data in session state for access across pages
    st.session_state['race_data'] = data
    st.session_state['track'] = track
    st.session_state['race_num'] = race_num

    # Route to appropriate page
    if "Overview" in page:
        overview.show_race_overview(data, track, race_num)
    elif "Tactical" in page:
        tactical.show_tactical_analysis(data, track, race_num)
    elif "Strategic" in page:
        strategic.show_strategic_analysis(data, track, race_num)
    elif "Integrated" in page:
        integrated.show_integrated_insights(data, track, race_num)
    elif "Simulator" in page:
        race_simulator.show_race_simulator(data, track, race_num)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**RaceIQ Pro** v1.0")
    st.sidebar.markdown("Toyota GR Cup Hackathon 2025")

if __name__ == "__main__":
    main()
