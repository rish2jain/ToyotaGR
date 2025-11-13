"""
RaceIQ Pro - Visualization Utilities

This module provides reusable visualization functions for race data analysis.
"""

import logging
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import VIZ_CONFIG

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
