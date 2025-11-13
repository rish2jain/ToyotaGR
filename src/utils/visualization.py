"""
RaceIQ Pro - Visualization Utilities

This module provides reusable visualization functions for race data analysis.
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .constants import VIZ_CONFIG
from .track_layouts import get_track_layout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = VIZ_CONFIG["dpi"]


def plot_lap_times(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    time_col: str = "LAP_TIME_SECONDS",
    drivers: Optional[List] = None,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot lap times over the race for selected drivers.

    Args:
        df: DataFrame with lap data
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        time_col: Column name for lap time
        drivers: List of driver numbers to plot (None = all)
        figsize: Figure size tuple
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = VIZ_CONFIG["default_figsize"]

    fig, ax = plt.subplots(figsize=figsize)

    # Filter drivers if specified
    plot_df = df.copy()
    if drivers is not None:
        plot_df = plot_df[plot_df[driver_col].isin(drivers)]

    # Plot each driver
    for driver in plot_df[driver_col].unique():
        driver_data = plot_df[plot_df[driver_col] == driver].sort_values(lap_col)
        ax.plot(
            driver_data[lap_col],
            driver_data[time_col],
            marker="o",
            markersize=VIZ_CONFIG["marker_size"] - 2,
            linewidth=VIZ_CONFIG["line_width"],
            label=f"Driver {driver}",
            alpha=0.8
        )

    ax.set_xlabel("Lap Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Lap Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(title or "Lap Times Throughout Race", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sector_comparison(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    drivers: Optional[List] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot sector time comparison across drivers.

    Args:
        df: DataFrame with section analysis data
        driver_col: Column name for driver identifier
        drivers: List of driver numbers to plot (None = all)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    sector_cols = ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]

    if not all(col in df.columns for col in sector_cols):
        logger.error("Missing sector columns")
        return None

    if figsize is None:
        figsize = (14, 6)

    # Calculate average sector times per driver
    avg_sectors = df.groupby(driver_col)[sector_cols].mean().reset_index()

    if drivers is not None:
        avg_sectors = avg_sectors[avg_sectors[driver_col].isin(drivers)]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(avg_sectors))
    width = 0.25

    colors = [
        VIZ_CONFIG["color_scheme"]["primary"],
        VIZ_CONFIG["color_scheme"]["secondary"],
        VIZ_CONFIG["color_scheme"]["accent"]
    ]

    for i, sector in enumerate(sector_cols):
        offset = width * (i - 1)
        ax.bar(
            x + offset,
            avg_sectors[sector],
            width,
            label=sector.replace("_SECONDS", ""),
            color=colors[i],
            alpha=0.8
        )

    ax.set_xlabel("Driver Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Average Sector Times by Driver", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(avg_sectors[driver_col].astype(str))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_consistency_heatmap(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    time_col: str = "LAP_TIME_SECONDS",
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot heatmap of lap time consistency.

    Args:
        df: DataFrame with lap data
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        time_col: Column name for lap time
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = (12, 8)

    # Create pivot table
    pivot = df.pivot(index=driver_col, columns=lap_col, values=time_col)

    # Calculate delta from personal best for each driver
    pivot_delta = pivot.sub(pivot.min(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot_delta,
        cmap="RdYlGn_r",
        center=0,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Delta from Personal Best (seconds)"},
        ax=ax
    )

    ax.set_xlabel("Lap Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Driver Number", fontsize=12, fontweight="bold")
    ax.set_title("Lap Time Consistency (Delta from Personal Best)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_speed_trace(
    df: pd.DataFrame,
    driver: int,
    lap: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot speed trace for a specific driver and lap.

    Args:
        df: DataFrame with telemetry data
        driver: Driver number
        lap: Lap number (None = all laps)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = VIZ_CONFIG["default_figsize"]

    # Filter data
    plot_df = df[df["vehicle_number"] == driver].copy()
    if lap is not None:
        plot_df = plot_df[plot_df["lap"] == lap]

    if len(plot_df) == 0:
        logger.warning(f"No data found for driver {driver}, lap {lap}")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot speed over time
    if "timestamp" in plot_df.columns:
        plot_df = plot_df.sort_values("timestamp")
        x_data = range(len(plot_df))
        x_label = "Data Point"
    else:
        x_data = range(len(plot_df))
        x_label = "Index"

    ax.plot(
        x_data,
        plot_df["gps_speed"],
        color=VIZ_CONFIG["color_scheme"]["primary"],
        linewidth=VIZ_CONFIG["line_width"]
    )

    # Add horizontal line for average speed
    avg_speed = plot_df["gps_speed"].mean()
    ax.axhline(
        avg_speed,
        color=VIZ_CONFIG["color_scheme"]["secondary"],
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg_speed:.1f} km/h"
    )

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Speed (km/h)", fontsize=12, fontweight="bold")

    title = f"Speed Trace - Driver {driver}"
    if lap is not None:
        title += f" - Lap {lap}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_track_map(
    df: pd.DataFrame,
    driver: Optional[int] = None,
    color_by: str = "speed",
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot track map from GPS coordinates.

    Args:
        df: DataFrame with telemetry data
        driver: Driver number (None = all drivers)
        color_by: Variable to color the trace by ('speed', 'lap', etc.)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if not all(col in df.columns for col in ["gps_lat", "gps_long"]):
        logger.error("Missing GPS coordinate columns")
        return None

    if figsize is None:
        figsize = (10, 10)

    plot_df = df.copy()
    if driver is not None:
        plot_df = plot_df[plot_df["vehicle_number"] == driver]

    fig, ax = plt.subplots(figsize=figsize)

    # Determine color mapping
    if color_by == "speed" and "gps_speed" in plot_df.columns:
        color_data = plot_df["gps_speed"]
        cmap = "viridis"
        label = "Speed (km/h)"
    elif color_by == "lap" and "lap" in plot_df.columns:
        color_data = plot_df["lap"]
        cmap = "tab20"
        label = "Lap Number"
    else:
        color_data = range(len(plot_df))
        cmap = "viridis"
        label = "Sequence"

    scatter = ax.scatter(
        plot_df["gps_long"],
        plot_df["gps_lat"],
        c=color_data,
        cmap=cmap,
        s=5,
        alpha=0.6
    )

    plt.colorbar(scatter, ax=ax, label=label)

    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")

    title = "Track Map"
    if driver is not None:
        title += f" - Driver {driver}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_position_chart(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    position_col: str = "position",
    drivers: Optional[List] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot position changes throughout the race.

    Args:
        df: DataFrame with position data
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        position_col: Column name for position
        drivers: List of driver numbers to plot (None = all)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = VIZ_CONFIG["default_figsize"]

    plot_df = df.copy()
    if drivers is not None:
        plot_df = plot_df[plot_df[driver_col].isin(drivers)]

    fig, ax = plt.subplots(figsize=figsize)

    for driver in plot_df[driver_col].unique():
        driver_data = plot_df[plot_df[driver_col] == driver].sort_values(lap_col)
        ax.plot(
            driver_data[lap_col],
            driver_data[position_col],
            marker="o",
            markersize=VIZ_CONFIG["marker_size"] - 2,
            linewidth=VIZ_CONFIG["line_width"],
            label=f"Driver {driver}",
            alpha=0.8
        )

    ax.set_xlabel("Lap Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Position", fontsize=12, fontweight="bold")
    ax.set_title("Position Changes Throughout Race", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Lower position numbers at top
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set integer ticks for position
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    return fig


def plot_gap_evolution(
    df: pd.DataFrame,
    driver_col: str = "DRIVER_NUMBER",
    lap_col: str = "LAP_NUMBER",
    gap_col: str = "gap_to_leader",
    drivers: Optional[List] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot gap to leader evolution throughout the race.

    Args:
        df: DataFrame with gap data
        driver_col: Column name for driver identifier
        lap_col: Column name for lap number
        gap_col: Column name for gap
        drivers: List of driver numbers to plot (None = all)
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = VIZ_CONFIG["default_figsize"]

    plot_df = df.copy()
    if drivers is not None:
        plot_df = plot_df[plot_df[driver_col].isin(drivers)]

    fig, ax = plt.subplots(figsize=figsize)

    for driver in plot_df[driver_col].unique():
        driver_data = plot_df[plot_df[driver_col] == driver].sort_values(lap_col)
        ax.plot(
            driver_data[lap_col],
            driver_data[gap_col],
            marker="o",
            markersize=VIZ_CONFIG["marker_size"] - 2,
            linewidth=VIZ_CONFIG["line_width"],
            label=f"Driver {driver}",
            alpha=0.8
        )

    ax.set_xlabel("Lap Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gap to Leader (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Gap Evolution Throughout Race", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add horizontal line at 0
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300):
    """
    Save figure to file.

    Args:
        fig: matplotlib Figure object
        filepath: Path to save file
        dpi: Resolution in dots per inch
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info(f"Figure saved to {filepath}")


# ============================================================================
# Track Map Visualizations with Performance Overlay
# ============================================================================


def _map_performance_to_colors(section_gaps: pd.Series, colorscale: str = 'RdYlGn_r') -> List[str]:
    """
    Map performance gaps to colors

    Args:
        section_gaps: Series with gap values per section
        colorscale: Plotly colorscale name

    Returns:
        List of color strings
    """
    if len(section_gaps) == 0:
        return []

    # Normalize gaps to 0-1 range
    min_gap = section_gaps.min()
    max_gap = section_gaps.max()

    if max_gap == min_gap:
        # All gaps are the same
        normalized = [0.5] * len(section_gaps)
    else:
        normalized = (section_gaps - min_gap) / (max_gap - min_gap)

    # Map to colors (using simple RGB interpolation)
    colors = []
    for val in normalized:
        # Red (slow) to Yellow to Green (fast)
        # Reverse because smaller gap is better
        val = 1 - val

        if val < 0.5:
            # Red to Yellow
            r = 255
            g = int(255 * (val * 2))
            b = 0
        else:
            # Yellow to Green
            r = int(255 * (1 - (val - 0.5) * 2))
            g = 255
            b = 0

        colors.append(f'rgb({r},{g},{b})')

    return colors


def _get_performance_rating(gap: float, optimal: float = 0.0) -> str:
    """
    Get performance rating based on gap to optimal

    Args:
        gap: Gap to optimal in seconds
        optimal: Optimal time (default 0 for relative gaps)

    Returns:
        Performance rating string
    """
    if gap < 0.05:
        return "Excellent"
    elif gap < 0.15:
        return "Good"
    elif gap < 0.30:
        return "Average"
    else:
        return "Needs Improvement"


def create_track_map_with_performance(
    section_data: pd.DataFrame,
    track_name: str = 'barber',
    section_col: str = 'Section',
    time_col: str = 'Time',
    gap_col: str = 'GapToOptimal',
    driver_label: str = None
) -> 'go.Figure':
    """
    Create interactive track map with color-coded performance overlay

    Args:
        section_data: DataFrame with section performance data
        track_name: Track name ('barber', 'cota', 'sonoma', etc.)
        section_col: Column name for section identifier
        time_col: Column name for section time
        gap_col: Column name for gap to optimal
        driver_label: Optional driver label for title

    Returns:
        plotly.graph_objects.Figure with interactive track map
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is required for track map visualization")
        return None

    # Load track coordinates
    track_layout = get_track_layout(track_name)
    track_sections = track_layout['sections']
    track_info = track_layout['track_info']

    # Calculate performance gaps per section
    if gap_col in section_data.columns:
        section_gaps = section_data.groupby(section_col)[gap_col].mean()
    elif time_col in section_data.columns:
        # Calculate gaps from times
        section_times = section_data.groupby(section_col)[time_col].mean()
        optimal_times = section_data.groupby(section_col)[time_col].min()
        section_gaps = section_times - optimal_times
    else:
        logger.error(f"Neither {gap_col} nor {time_col} found in data")
        return None

    # Map data sections to track sections
    data_sections = sorted(section_data[section_col].unique())
    num_track_sections = len(track_sections)

    # Create color mapping
    colors = _map_performance_to_colors(section_gaps)

    # Create figure
    fig = go.Figure()

    # Add track sections with performance coloring
    for i, section in enumerate(track_sections):
        # Map data section to track section
        data_idx = int((i / num_track_sections) * len(data_sections)) if len(data_sections) > 0 else 0
        data_section = data_sections[data_idx] if data_idx < len(data_sections) else data_sections[-1]

        # Get gap for this section
        gap = section_gaps.get(data_section, 0.0)
        color = colors[data_idx] if data_idx < len(colors) else 'rgb(128,128,128)'

        # Create hover text
        hover_text = (
            f"<b>{section['name']}</b><br>"
            f"Type: {section['type'].title()}<br>"
            f"Gap: {gap:.3f}s<br>"
            f"Performance: {_get_performance_rating(gap)}<br>"
            f"{section['description']}"
        )

        # Add trace for this section
        fig.add_trace(go.Scatter(
            x=section['x'],
            y=section['y'],
            mode='lines',
            line=dict(
                color=color,
                width=12
            ),
            name=section['name'],
            showlegend=False,
            hovertemplate=hover_text + "<extra></extra>",
            hoverlabel=dict(
                bgcolor=color,
                font_size=12,
                font_color='white'
            )
        ))

    # Add start/finish line marker
    if len(track_sections) > 0:
        first_section = track_sections[0]
        fig.add_trace(go.Scatter(
            x=[first_section['x'][0]],
            y=[first_section['y'][0]],
            mode='markers+text',
            marker=dict(
                size=15,
                color='white',
                symbol='square',
                line=dict(color='black', width=2)
            ),
            text=['S/F'],
            textposition='top center',
            textfont=dict(size=14, color='black', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Update layout
    title_text = f"Track Map: Performance by Section"
    if driver_label:
        title_text += f" - {driver_label}"

    fig.update_layout(
        title={
            'text': f"{title_text}<br><sub>{track_info['name']} ({track_info['length']}, {track_info['turns']} turns)</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        showlegend=False,
        xaxis=dict(
            visible=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            visible=False
        ),
        hovermode='closest',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2b2b2b',
        width=900,
        height=700,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    # Add color legend annotation
    legend_text = (
        "<b>Performance Guide:</b><br>"
        "🟢 Green: Fast (< 0.05s gap)<br>"
        "🟡 Yellow: Good (0.05-0.15s gap)<br>"
        "🟠 Orange: Average (0.15-0.30s gap)<br>"
        "🔴 Red: Slow (> 0.30s gap)"
    )

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=legend_text,
        showarrow=False,
        align='left',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=11, color='black')
    )

    return fig


def create_driver_comparison_map(
    driver1_data: pd.DataFrame,
    driver2_data: pd.DataFrame,
    track_name: str = 'barber',
    driver1_label: str = "Driver 1",
    driver2_label: str = "Driver 2",
    section_col: str = 'Section',
    time_col: str = 'Time'
) -> 'go.Figure':
    """
    Create track map comparing two drivers' performance

    Args:
        driver1_data: DataFrame with first driver's section data
        driver2_data: DataFrame with second driver's section data
        track_name: Track name
        driver1_label: Label for first driver
        driver2_label: Label for second driver
        section_col: Column name for section identifier
        time_col: Column name for section time

    Returns:
        plotly.graph_objects.Figure with comparison
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is required for track map visualization")
        return None

    # Load track layout
    track_layout = get_track_layout(track_name)
    track_sections = track_layout['sections']
    track_info = track_layout['track_info']

    # Calculate section times for both drivers
    d1_times = driver1_data.groupby(section_col)[time_col].mean()
    d2_times = driver2_data.groupby(section_col)[time_col].mean()

    # Calculate who's faster in each section
    time_diffs = d1_times - d2_times  # Positive = driver 2 faster

    # Map data sections to track sections
    data_sections = sorted(set(driver1_data[section_col].unique()) | set(driver2_data[section_col].unique()))
    num_track_sections = len(track_sections)

    # Create figure
    fig = go.Figure()

    # Add track sections with comparison coloring
    for i, section in enumerate(track_sections):
        # Map to data section
        data_idx = int((i / num_track_sections) * len(data_sections)) if len(data_sections) > 0 else 0
        data_section = data_sections[data_idx] if data_idx < len(data_sections) else data_sections[-1]

        # Get time difference
        diff = time_diffs.get(data_section, 0.0)

        # Color based on who's faster
        if abs(diff) < 0.05:
            color = 'rgb(200,200,200)'  # Gray for equal
            faster = "Equal"
        elif diff < 0:
            # Driver 1 faster
            intensity = min(abs(diff) * 255 / 0.5, 255)
            color = f'rgb({int(intensity)},100,100)'
            faster = driver1_label
        else:
            # Driver 2 faster
            intensity = min(abs(diff) * 255 / 0.5, 255)
            color = f'rgb(100,100,{int(intensity)})'
            faster = driver2_label

        # Hover text
        d1_time = d1_times.get(data_section, 0.0)
        d2_time = d2_times.get(data_section, 0.0)

        hover_text = (
            f"<b>{section['name']}</b><br>"
            f"{driver1_label}: {d1_time:.3f}s<br>"
            f"{driver2_label}: {d2_time:.3f}s<br>"
            f"Difference: {abs(diff):.3f}s<br>"
            f"Faster: {faster}"
        )

        # Add trace
        fig.add_trace(go.Scatter(
            x=section['x'],
            y=section['y'],
            mode='lines',
            line=dict(color=color, width=12),
            name=section['name'],
            showlegend=False,
            hovertemplate=hover_text + "<extra></extra>"
        ))

    # Add start/finish marker
    if len(track_sections) > 0:
        first_section = track_sections[0]
        fig.add_trace(go.Scatter(
            x=[first_section['x'][0]],
            y=[first_section['y'][0]],
            mode='markers+text',
            marker=dict(
                size=15,
                color='white',
                symbol='square',
                line=dict(color='black', width=2)
            ),
            text=['S/F'],
            textposition='top center',
            textfont=dict(size=14, color='black', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': f"Driver Comparison: {driver1_label} vs {driver2_label}<br><sub>{track_info['name']}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        showlegend=False,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        hovermode='closest',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2b2b2b',
        width=900,
        height=700,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    # Add legend
    legend_text = (
        f"<b>Color Guide:</b><br>"
        f"🔴 Red: {driver1_label} faster<br>"
        f"🔵 Blue: {driver2_label} faster<br>"
        f"⚪ Gray: Equal performance"
    )

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=legend_text,
        showarrow=False,
        align='left',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=11, color='black')
    )

    return fig


# ============================================================================
# Racing Line Visualization Functions
# ============================================================================


def create_racing_line_comparison(
    driver1_line: Dict[str, Any],
    driver2_line: Dict[str, Any],
    track_layout: Dict[str, Any],
    driver1_label: str = "Driver 1",
    driver2_label: str = "Driver 2"
) -> 'go.Figure':
    """
    Create track map with two racing lines overlaid.

    Shows:
    - Driver A line (blue)
    - Driver B line (red)
    - Entry/Apex/Exit markers
    - Speed heatmap overlay
    - Difference annotations

    Args:
        driver1_line: First driver's reconstructed line
        driver2_line: Second driver's reconstructed line
        track_layout: Track layout dictionary from get_track_layout
        driver1_label: Label for first driver
        driver2_label: Label for second driver

    Returns:
        plotly.graph_objects.Figure with racing line comparison
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is required for racing line visualization")
        return None

    track_sections = track_layout['sections']
    track_info = track_layout['track_info']

    # Create figure
    fig = go.Figure()

    # Draw track outline (gray)
    for section in track_sections:
        fig.add_trace(go.Scatter(
            x=section['x'],
            y=section['y'],
            mode='lines',
            line=dict(color='rgba(128,128,128,0.3)', width=8),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Get trajectories
    traj1 = driver1_line['trajectory']
    traj2 = driver2_line['trajectory']

    # Map trajectory to track coordinates (approximate)
    track_x, track_y = _extract_track_coordinates(track_sections)

    # Interpolate trajectories onto track coordinates
    line1_x, line1_y = _map_trajectory_to_track(traj1, track_x, track_y)
    line2_x, line2_y = _map_trajectory_to_track(traj2, track_x, track_y)

    # Draw Driver 1 line (blue)
    fig.add_trace(go.Scatter(
        x=line1_x,
        y=line1_y,
        mode='lines',
        name=driver1_label,
        line=dict(color='blue', width=4),
        opacity=0.7,
        hovertemplate=f'<b>{driver1_label}</b><br>Distance: %{{customdata[0]:.1f}}%<br>Speed: %{{customdata[1]:.1f}} km/h<extra></extra>',
        customdata=np.column_stack([traj1['distance'].values[:len(line1_x)], traj1['speed'].values[:len(line1_x)]])
    ))

    # Draw Driver 2 line (red)
    fig.add_trace(go.Scatter(
        x=line2_x,
        y=line2_y,
        mode='lines',
        name=driver2_label,
        line=dict(color='red', width=4),
        opacity=0.7,
        hovertemplate=f'<b>{driver2_label}</b><br>Distance: %{{customdata[0]:.1f}}%<br>Speed: %{{customdata[1]:.1f}} km/h<extra></extra>',
        customdata=np.column_stack([traj2['distance'].values[:len(line2_x)], traj2['speed'].values[:len(line2_x)]])
    ))

    # Add corner markers for both drivers
    corners1 = driver1_line['corners']
    corners2 = driver2_line['corners']

    for corner in corners1:
        apex_idx = int(corner['apex'] * len(line1_x) / 100)
        if apex_idx < len(line1_x):
            fig.add_trace(go.Scatter(
                x=[line1_x[apex_idx]],
                y=[line1_y[apex_idx]],
                mode='markers',
                marker=dict(size=10, color='darkblue', symbol='circle', line=dict(width=2, color='white')),
                showlegend=False,
                hovertemplate=f'<b>{driver1_label} Apex</b><br>Corner {corner["corner_number"]}<br>Speed: {corner["apex_speed"]:.1f} km/h<extra></extra>'
            ))

    for corner in corners2:
        apex_idx = int(corner['apex'] * len(line2_x) / 100)
        if apex_idx < len(line2_x):
            fig.add_trace(go.Scatter(
                x=[line2_x[apex_idx]],
                y=[line2_y[apex_idx]],
                mode='markers',
                marker=dict(size=10, color='darkred', symbol='diamond', line=dict(width=2, color='white')),
                showlegend=False,
                hovertemplate=f'<b>{driver2_label} Apex</b><br>Corner {corner["corner_number"]}<br>Speed: {corner["apex_speed"]:.1f} km/h<extra></extra>'
            ))

    # Add start/finish line
    if track_sections:
        first_section = track_sections[0]
        fig.add_trace(go.Scatter(
            x=[first_section['x'][0]],
            y=[first_section['y'][0]],
            mode='markers+text',
            marker=dict(size=15, color='white', symbol='square', line=dict(color='black', width=2)),
            text=['S/F'],
            textposition='top center',
            textfont=dict(size=14, color='black', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': f"Racing Line Comparison: {driver1_label} vs {driver2_label}<br><sub>{track_info['name']}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        hovermode='closest',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2b2b2b',
        width=900,
        height=700,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    return fig


def create_corner_analysis(
    corner_data: Dict[str, Any],
    driver1_label: str = "Driver 1",
    driver2_label: str = "Driver 2",
    differences: List[Dict[str, Any]] = None
) -> 'go.Figure':
    """
    Detailed corner-by-corner comparison.

    Shows:
    - Brake point comparison
    - Minimum speed comparison
    - Apex location
    - Exit speed comparison

    Args:
        corner_data: Dictionary with corner comparison data
        driver1_label: Label for first driver
        driver2_label: Label for second driver
        differences: List of corner differences from compare_racing_lines

    Returns:
        plotly.graph_objects.Figure with corner analysis
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is required for corner analysis visualization")
        return None

    if not differences:
        logger.warning("No corner differences provided for analysis")
        return None

    # Create subplots for different metrics
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Apex Speed Comparison',
            'Brake Point Comparison',
            'Corner Radius Comparison',
            'Speed Delta by Corner'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )

    corner_numbers = [d['corner_number'] for d in differences]

    # Apex Speed Comparison
    apex_speeds_1 = [d[f'{driver1_label}_apex_speed'] for d in differences]
    apex_speeds_2 = [d[f'{driver2_label}_apex_speed'] for d in differences]

    fig.add_trace(
        go.Bar(name=driver1_label, x=corner_numbers, y=apex_speeds_1, marker_color='blue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name=driver2_label, x=corner_numbers, y=apex_speeds_2, marker_color='red'),
        row=1, col=1
    )

    # Brake Point Comparison
    brake_points_1 = [d[f'{driver1_label}_brake_point'] for d in differences]
    brake_points_2 = [d[f'{driver2_label}_brake_point'] for d in differences]

    fig.add_trace(
        go.Bar(name=driver1_label, x=corner_numbers, y=brake_points_1, marker_color='blue', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name=driver2_label, x=corner_numbers, y=brake_points_2, marker_color='red', showlegend=False),
        row=1, col=2
    )

    # Corner Radius Comparison
    radius_1 = [d[f'{driver1_label}_radius'] for d in differences]
    radius_2 = [d[f'{driver2_label}_radius'] for d in differences]

    fig.add_trace(
        go.Bar(name=driver1_label, x=corner_numbers, y=radius_1, marker_color='blue', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(name=driver2_label, x=corner_numbers, y=radius_2, marker_color='red', showlegend=False),
        row=2, col=1
    )

    # Speed Delta
    speed_deltas = [d['apex_speed_delta_kph'] for d in differences]
    colors = ['green' if delta > 0 else 'orange' for delta in speed_deltas]

    fig.add_trace(
        go.Bar(
            x=corner_numbers,
            y=speed_deltas,
            marker_color=colors,
            name='Speed Delta',
            showlegend=False
        ),
        row=2, col=2
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Corner Number", row=1, col=1)
    fig.update_xaxes(title_text="Corner Number", row=1, col=2)
    fig.update_xaxes(title_text="Corner Number", row=2, col=1)
    fig.update_xaxes(title_text="Corner Number", row=2, col=2)

    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Distance (%)", row=1, col=2)
    fig.update_yaxes(title_text="Radius (m)", row=2, col=1)
    fig.update_yaxes(title_text="Speed Delta (km/h)", row=2, col=2)

    fig.update_layout(
        title={'text': f"Corner-by-Corner Analysis: {driver1_label} vs {driver2_label}", 'x': 0.5, 'xanchor': 'center'},
        height=700,
        showlegend=True,
        legend=dict(x=0.85, y=0.98),
        plot_bgcolor='white'
    )

    return fig


def create_speed_trace_comparison(
    driver1_line: Dict[str, Any],
    driver2_line: Dict[str, Any],
    driver1_label: str = "Driver 1",
    driver2_label: str = "Driver 2",
    corner_number: Optional[int] = None
) -> 'go.Figure':
    """Create speed trace comparison through a corner or full lap."""
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is required for speed trace visualization")
        return None

    fig = go.Figure()

    traj1 = driver1_line['trajectory']
    traj2 = driver2_line['trajectory']

    title_suffix = " - Full Lap"
    if corner_number is not None:
        corners1 = driver1_line['corners']
        corners2 = driver2_line['corners']

        if corner_number <= len(corners1) and corner_number <= len(corners2):
            corner1 = corners1[corner_number - 1]
            corner2 = corners2[corner_number - 1]

            traj1 = traj1[(traj1['distance'] >= corner1['entry']) & (traj1['distance'] <= corner1['exit'])].copy()
            traj2 = traj2[(traj2['distance'] >= corner2['entry']) & (traj2['distance'] <= corner2['exit'])].copy()
            title_suffix = f" - Corner {corner_number}"

    fig.add_trace(go.Scatter(
        x=traj1['distance'], y=traj1['speed'], mode='lines', name=driver1_label,
        line=dict(color='blue', width=3),
        hovertemplate='<b>%{fullData.name}</b><br>Distance: %{x:.1f}%<br>Speed: %{y:.1f} km/h<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=traj2['distance'], y=traj2['speed'], mode='lines', name=driver2_label,
        line=dict(color='red', width=3),
        hovertemplate='<b>%{fullData.name}</b><br>Distance: %{x:.1f}%<br>Speed: %{y:.1f} km/h<extra></extra>'
    ))

    fig.update_layout(
        title={'text': f"Speed Trace Comparison: {driver1_label} vs {driver2_label}{title_suffix}", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Distance (%)", yaxis_title="Speed (km/h)",
        hovermode='x unified', height=500, plot_bgcolor='white', legend=dict(x=0.02, y=0.98)
    )

    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

    return fig


def _extract_track_coordinates(track_sections: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract x, y coordinates from track sections."""
    x_coords = []
    y_coords = []
    for section in track_sections:
        x_coords.extend(section['x'])
        y_coords.extend(section['y'])
    return np.array(x_coords), np.array(y_coords)


def _map_trajectory_to_track(
    trajectory: pd.DataFrame,
    track_x: np.ndarray,
    track_y: np.ndarray,
    lateral_offset_scale: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Map trajectory distance to track coordinates with lateral offset."""
    distance_normalized = trajectory['distance'].values / 100.0
    track_indices = (distance_normalized * (len(track_x) - 1)).astype(int)
    track_indices = np.clip(track_indices, 0, len(track_x) - 1)

    traj_x = track_x[track_indices]
    traj_y = track_y[track_indices]

    if 'lateral_offset' in trajectory.columns:
        lateral_offset = trajectory['lateral_offset'].values[:len(traj_x)]
        for i in range(len(traj_x)):
            if i < len(track_x) - 1:
                dx = track_x[min(track_indices[i] + 1, len(track_x) - 1)] - track_x[track_indices[i]]
                dy = track_y[min(track_indices[i] + 1, len(track_y) - 1)] - track_y[track_indices[i]]
                length = np.sqrt(dx**2 + dy**2) + 1e-6
                perp_x = -dy / length
                perp_y = dx / length
                offset = lateral_offset[i] * lateral_offset_scale
                traj_x[i] += perp_x * offset
                traj_y[i] += perp_y * offset

    return traj_x, traj_y
