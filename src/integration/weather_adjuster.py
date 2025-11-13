"""
Weather Adjuster for RaceIQ Pro

This module adjusts racing performance estimates based on weather conditions,
including tire degradation, lap times, and risk assessment.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class WeatherConditions:
    """Current weather conditions"""
    temperature: float  # Air temperature in Celsius
    track_temp: float  # Track temperature in Celsius
    humidity: float  # Humidity percentage
    wind_speed: float  # Wind speed in km/h
    precipitation: float  # Rain/precipitation (0 = dry, 1 = wet)
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class WeatherImpact:
    """Calculated impact of weather on performance"""
    tire_degradation_multiplier: float  # Multiplier for tire deg rate (1.0 = no change)
    lap_time_modifier: float  # Multiplier for lap times (1.0 = no change)
    risky_sections: List[int]  # Section numbers with elevated risk
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    impact_description: str  # Human-readable description


class WeatherAdjuster:
    """
    Adjusts racing performance estimates based on weather conditions.

    This class provides methods to:
    - Adjust tire degradation rates based on temperature
    - Modify lap time estimates based on weather
    - Identify high-risk track sections
    - Generate weather-based strategic recommendations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the WeatherAdjuster.

        Args:
            config: Optional configuration dictionary with thresholds
        """
        self.config = config or {}

        # Temperature thresholds (Celsius)
        self.hot_temp_threshold = self.config.get('hot_temp_threshold', 30.0)
        self.cold_temp_threshold = self.config.get('cold_temp_threshold', 18.0)
        self.track_hot_threshold = self.config.get('track_hot_threshold', 40.0)
        self.track_cold_threshold = self.config.get('track_cold_threshold', 25.0)

        # Wind thresholds (km/h)
        self.high_wind_threshold = self.config.get('high_wind_threshold', 25.0)
        self.moderate_wind_threshold = self.config.get('moderate_wind_threshold', 15.0)

        # Humidity thresholds (%)
        self.high_humidity_threshold = self.config.get('high_humidity_threshold', 70.0)

    def get_current_conditions(
        self, weather_df: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> Optional[WeatherConditions]:
        """
        Get current or average weather conditions from weather data.

        Args:
            weather_df: Weather data DataFrame
            timestamp: Optional timestamp to get conditions for (uses latest if None)

        Returns:
            WeatherConditions object or None if data unavailable
        """
        if weather_df is None or weather_df.empty:
            return None

        try:
            if timestamp is not None and 'timestamp' in weather_df.columns:
                # Find closest weather record to timestamp
                weather_df['time_diff'] = abs(weather_df['timestamp'] - timestamp)
                row = weather_df.loc[weather_df['time_diff'].idxmin()]
            else:
                # Use most recent or average conditions
                row = weather_df.iloc[-1] if len(weather_df) > 0 else weather_df.mean()

            # Extract temperature (use estimated if actual not available)
            track_temp = row.get('TRACK_TEMP', 0)
            if pd.isna(track_temp) or track_temp == 0:
                track_temp = row.get('TRACK_TEMP_ESTIMATED', row.get('AIR_TEMP', 25) + 10)

            return WeatherConditions(
                temperature=float(row.get('AIR_TEMP', 25)),
                track_temp=float(track_temp),
                humidity=float(row.get('HUMIDITY', 50)),
                wind_speed=float(row.get('WIND_SPEED', 0)),
                precipitation=float(row.get('RAIN', 0)),
                timestamp=row.get('timestamp', None)
            )
        except Exception as e:
            print(f"Error extracting weather conditions: {e}")
            return None

    def adjust_tire_degradation(
        self, base_degradation: float, conditions: WeatherConditions
    ) -> Tuple[float, str]:
        """
        Adjust tire degradation rate based on temperature.

        Hot temps (>30°C air / >40°C track): Increase degradation by 10-20%
        Cold temps (<18°C air / <25°C track): Decrease degradation by 5-10%
        High humidity: Slight decrease in degradation (cooler track)

        Args:
            base_degradation: Base tire degradation rate (seconds/lap)
            conditions: Current weather conditions

        Returns:
            Tuple of (adjusted_degradation, explanation)
        """
        if conditions is None:
            return base_degradation, "No weather data available"

        multiplier = 1.0
        factors = []

        # Track temperature impact (primary factor)
        if conditions.track_temp > self.track_hot_threshold:
            # Hot track increases tire wear significantly
            temp_excess = conditions.track_temp - self.track_hot_threshold
            temp_factor = 1.0 + (0.01 * temp_excess)  # 1% increase per degree above threshold
            temp_factor = min(temp_factor, 1.20)  # Cap at 20% increase
            multiplier *= temp_factor
            factors.append(f"Hot track temp ({conditions.track_temp:.1f}°C): +{(temp_factor-1)*100:.0f}% degradation")

        elif conditions.track_temp < self.track_cold_threshold:
            # Cold track reduces tire wear but also grip
            temp_deficit = self.track_cold_threshold - conditions.track_temp
            temp_factor = 1.0 - (0.005 * temp_deficit)  # 0.5% decrease per degree below threshold
            temp_factor = max(temp_factor, 0.90)  # Cap at 10% decrease
            multiplier *= temp_factor
            factors.append(f"Cold track temp ({conditions.track_temp:.1f}°C): {(temp_factor-1)*100:.0f}% degradation")

        # Air temperature impact (secondary factor)
        if conditions.temperature > self.hot_temp_threshold:
            air_factor = 1.0 + ((conditions.temperature - self.hot_temp_threshold) * 0.005)
            air_factor = min(air_factor, 1.10)
            multiplier *= air_factor
            factors.append(f"Hot air temp ({conditions.temperature:.1f}°C): +{(air_factor-1)*100:.0f}% degradation")

        # Humidity impact
        if conditions.humidity > self.high_humidity_threshold:
            humidity_factor = 0.97  # High humidity cools track slightly
            multiplier *= humidity_factor
            factors.append(f"High humidity ({conditions.humidity:.0f}%): -3% degradation")

        # Rain impact
        if conditions.precipitation > 0:
            rain_factor = 0.85  # Wet conditions reduce tire temp and wear
            multiplier *= rain_factor
            factors.append(f"Wet conditions: -15% degradation (but slower lap times)")

        adjusted_degradation = base_degradation * multiplier

        explanation = " | ".join(factors) if factors else "Normal conditions"

        return adjusted_degradation, explanation

    def adjust_lap_times(
        self, base_time: float, conditions: WeatherConditions
    ) -> Tuple[float, str]:
        """
        Adjust expected lap times based on weather.

        Rain: +5-15% lap time
        High winds: +2-5% lap time
        Hot track: -1-3% lap time (better grip initially)
        Cold track: +2-4% lap time (less grip)

        Args:
            base_time: Base lap time in seconds
            conditions: Current weather conditions

        Returns:
            Tuple of (adjusted_time, explanation)
        """
        if conditions is None:
            return base_time, "No weather data available"

        modifier = 1.0
        factors = []

        # Precipitation impact (dominant factor)
        if conditions.precipitation > 0:
            # Wet conditions significantly slow lap times
            rain_modifier = 1.10  # 10% slower in wet conditions
            modifier *= rain_modifier
            factors.append(f"Wet track: +{(rain_modifier-1)*100:.0f}% lap time")
        else:
            # Dry conditions - temperature effects
            if conditions.track_temp > self.track_hot_threshold:
                # Hot track improves grip initially (but increases tire wear)
                temp_modifier = 0.98  # 2% faster with hot track
                modifier *= temp_modifier
                factors.append(f"Hot track ({conditions.track_temp:.1f}°C): {(temp_modifier-1)*100:.0f}% lap time")

            elif conditions.track_temp < self.track_cold_threshold:
                # Cold track reduces grip
                temp_deficit = self.track_cold_threshold - conditions.track_temp
                temp_modifier = 1.0 + (0.002 * temp_deficit)  # 0.2% slower per degree
                temp_modifier = min(temp_modifier, 1.04)  # Cap at 4% slower
                modifier *= temp_modifier
                factors.append(f"Cold track ({conditions.track_temp:.1f}°C): +{(temp_modifier-1)*100:.0f}% lap time")

        # Wind impact
        if conditions.wind_speed > self.high_wind_threshold:
            wind_modifier = 1.04  # 4% slower in high winds
            modifier *= wind_modifier
            factors.append(f"High winds ({conditions.wind_speed:.1f} km/h): +{(wind_modifier-1)*100:.0f}% lap time")
        elif conditions.wind_speed > self.moderate_wind_threshold:
            wind_modifier = 1.02  # 2% slower in moderate winds
            modifier *= wind_modifier
            factors.append(f"Moderate winds ({conditions.wind_speed:.1f} km/h): +{(wind_modifier-1)*100:.0f}% lap time")

        adjusted_time = base_time * modifier

        explanation = " | ".join(factors) if factors else "Normal conditions"

        return adjusted_time, explanation

    def identify_risky_sections(
        self, track_sections: List[Dict[str, Any]], conditions: WeatherConditions
    ) -> List[Dict[str, Any]]:
        """
        Flag sections that become risky in current weather.

        Rain + high-speed sections = HIGH RISK
        Wind + exposed corners = MEDIUM RISK
        Hot track + hard braking = MEDIUM RISK (brake fade)

        Args:
            track_sections: List of track sections with characteristics
            conditions: Current weather conditions

        Returns:
            List of sections with risk assessments
        """
        if conditions is None:
            return track_sections

        risky_sections = []

        for section in track_sections:
            section_num = section.get('section_number', 0)
            section_type = section.get('type', 'unknown')  # 'high_speed', 'braking', 'technical'
            avg_speed = section.get('avg_speed', 0)  # km/h

            risk_level = "LOW"
            risk_factors = []

            # Rain risks
            if conditions.precipitation > 0:
                if section_type == 'high_speed' or avg_speed > 180:
                    risk_level = "HIGH"
                    risk_factors.append("High-speed section in wet conditions")
                elif section_type == 'braking':
                    risk_level = "HIGH"
                    risk_factors.append("Heavy braking zone in wet conditions")
                else:
                    risk_level = "MEDIUM"
                    risk_factors.append("Wet conditions reduce grip")

            # Wind risks
            elif conditions.wind_speed > self.high_wind_threshold:
                if section_type == 'high_speed':
                    risk_level = "MEDIUM"
                    risk_factors.append(f"High winds ({conditions.wind_speed:.0f} km/h) affect stability")

            # Hot track risks
            elif conditions.track_temp > self.track_hot_threshold + 5:
                if section_type == 'braking':
                    risk_level = "MEDIUM"
                    risk_factors.append(f"Hot track ({conditions.track_temp:.0f}°C) increases brake fade risk")

            # Cold track risks
            elif conditions.track_temp < self.track_cold_threshold:
                if section_type == 'technical':
                    risk_level = "MEDIUM"
                    risk_factors.append(f"Cold track ({conditions.track_temp:.0f}°C) reduces grip in technical sections")

            risky_sections.append({
                'section_number': section_num,
                'section_type': section_type,
                'risk_level': risk_level,
                'risk_factors': risk_factors
            })

        return risky_sections

    def calculate_weather_impact(
        self, conditions: WeatherConditions, base_degradation: float = 0.05
    ) -> WeatherImpact:
        """
        Calculate overall weather impact on performance.

        Args:
            conditions: Current weather conditions
            base_degradation: Base tire degradation rate (seconds/lap)

        Returns:
            WeatherImpact object with all calculated impacts
        """
        if conditions is None:
            return WeatherImpact(
                tire_degradation_multiplier=1.0,
                lap_time_modifier=1.0,
                risky_sections=[],
                risk_level="LOW",
                impact_description="No weather data available"
            )

        # Calculate tire degradation impact
        adjusted_deg, deg_explanation = self.adjust_tire_degradation(base_degradation, conditions)
        deg_multiplier = adjusted_deg / base_degradation if base_degradation > 0 else 1.0

        # Calculate lap time impact
        base_lap = 100.0  # Use 100s as reference
        adjusted_lap, lap_explanation = self.adjust_lap_times(base_lap, conditions)
        lap_modifier = adjusted_lap / base_lap

        # Determine overall risk level
        if conditions.precipitation > 0:
            risk_level = "HIGH"
            impact_desc = f"WET CONDITIONS: {lap_explanation}"
        elif conditions.wind_speed > self.high_wind_threshold:
            risk_level = "MEDIUM"
            impact_desc = f"WINDY CONDITIONS: {lap_explanation}"
        elif conditions.track_temp > self.track_hot_threshold + 5:
            risk_level = "MEDIUM"
            impact_desc = f"HOT CONDITIONS: Track {conditions.track_temp:.1f}°C → {deg_explanation}"
        elif conditions.track_temp < self.track_cold_threshold:
            risk_level = "MEDIUM"
            impact_desc = f"COLD CONDITIONS: Track {conditions.track_temp:.1f}°C → {lap_explanation}"
        else:
            risk_level = "LOW"
            impact_desc = "IDEAL CONDITIONS: Minimal weather impact"

        # Identify risky sections (simplified - would need actual section data)
        risky_sections = []
        if conditions.precipitation > 0:
            risky_sections = [3, 7, 12]  # High-speed sections (would be dynamic in production)
        elif conditions.wind_speed > self.high_wind_threshold:
            risky_sections = [5, 9]  # Exposed sections

        return WeatherImpact(
            tire_degradation_multiplier=deg_multiplier,
            lap_time_modifier=lap_modifier,
            risky_sections=risky_sections,
            risk_level=risk_level,
            impact_description=impact_desc
        )

    def generate_weather_recommendations(
        self, weather_data: pd.DataFrame, race_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate weather-based strategic recommendations.

        Args:
            weather_data: Weather data DataFrame
            race_data: Optional race data for context

        Returns:
            Dictionary with recommendations:
            {
                'tire_adjustment': str (description of tire deg adjustment),
                'risky_sections': List[int] (section numbers),
                'lap_time_modifier': float (multiplier for lap times),
                'strategic_notes': List[str] (actionable recommendations),
                'conditions': WeatherConditions (current conditions),
                'impact': WeatherImpact (calculated impact)
            }
        """
        # Get current conditions
        conditions = self.get_current_conditions(weather_data)

        if conditions is None:
            return {
                'tire_adjustment': "Weather data unavailable",
                'risky_sections': [],
                'lap_time_modifier': 1.0,
                'strategic_notes': ["Weather data not available - using baseline strategy"],
                'conditions': None,
                'impact': None
            }

        # Calculate impact
        base_degradation = race_data.get('base_tire_degradation', 0.05) if race_data else 0.05
        impact = self.calculate_weather_impact(conditions, base_degradation)

        # Generate strategic notes
        strategic_notes = []

        # Tire strategy notes
        if impact.tire_degradation_multiplier > 1.10:
            strategic_notes.append(
                f"⚠ Tire degradation increased by {(impact.tire_degradation_multiplier-1)*100:.0f}% - "
                f"Consider earlier pit stop or more conservative pace"
            )
        elif impact.tire_degradation_multiplier < 0.95:
            strategic_notes.append(
                f"✓ Tire degradation reduced by {(1-impact.tire_degradation_multiplier)*100:.0f}% - "
                f"Can extend stint"
            )

        # Lap time notes
        if impact.lap_time_modifier > 1.05:
            strategic_notes.append(
                f"⚠ Lap times expected {(impact.lap_time_modifier-1)*100:.0f}% slower - "
                f"Adjust pit window calculations"
            )
        elif impact.lap_time_modifier < 0.98:
            strategic_notes.append(
                f"✓ Lap times expected {(1-impact.lap_time_modifier)*100:.0f}% faster - "
                f"Opportunity to push harder"
            )

        # Condition-specific notes
        if conditions.precipitation > 0:
            strategic_notes.append("⚠ WET CONDITIONS: Focus on consistency, avoid risks in high-speed sections")
            strategic_notes.append("Consider intermediate/wet tire strategy if rain increases")

        if conditions.wind_speed > self.high_wind_threshold:
            strategic_notes.append(
                f"⚠ HIGH WINDS ({conditions.wind_speed:.0f} km/h): "
                f"Expect reduced stability in exposed sections"
            )

        if conditions.track_temp > self.track_hot_threshold + 5:
            strategic_notes.append(
                f"⚠ HOT TRACK ({conditions.track_temp:.0f}°C): "
                f"Monitor tire pressures and brake temps closely"
            )

        if conditions.track_temp < self.track_cold_threshold:
            strategic_notes.append(
                f"⚠ COLD TRACK ({conditions.track_temp:.0f}°C): "
                f"Extended tire warm-up needed, careful on first few laps"
            )

        if not strategic_notes:
            strategic_notes.append("✓ IDEAL CONDITIONS: No major weather adjustments needed")

        # Build tire adjustment description
        tire_adjustment = (
            f"Tire degradation: {impact.tire_degradation_multiplier:.2f}x baseline "
            f"({'+' if impact.tire_degradation_multiplier >= 1 else ''}"
            f"{(impact.tire_degradation_multiplier-1)*100:.0f}%)"
        )

        return {
            'tire_adjustment': tire_adjustment,
            'risky_sections': impact.risky_sections,
            'lap_time_modifier': impact.lap_time_modifier,
            'strategic_notes': strategic_notes,
            'conditions': conditions,
            'impact': impact
        }
