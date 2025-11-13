# Barber Motorsports Park - Sample Data Files

This directory contains sample files from the Barber Motorsports Park race data to demonstrate the data structure and format. These are representative samples extracted from the full datasets.

## Sample Files

### Telemetry Data
- **R1_barber_telemetry_data_sample.csv**: Sample telemetry data (header + 10,000 rows)
  - Contains high-frequency telemetry measurements: speed, gear, throttle position, brake pressure
  - Original file size: ~1.5 GB
  - Sample size: ~10,000 rows (much smaller, suitable for Git)

### Lap Timing Data
- **R1_barber_lap_time_sample.csv**: Sample lap time data (header + 100 rows)
  - Total lap duration for each car/lap
- **R1_barber_lap_start_sample.csv**: Sample lap start timestamps (header + 100 rows)
  - Timestamp when each lap started
- **R1_barber_lap_end_sample.csv**: Sample lap end timestamps (header + 100 rows)
  - Timestamp when each lap finished

### Race Results
- **03_Provisional_Results_Race_1_sample.CSV**: Sample race results (header + 50 rows)
  - Pre-adjustment race results with positions, total laps, elapsed time, gaps

### Section Analysis
- **23_AnalysisEnduranceWithSections_Race_1_sample.CSV**: Sample section-by-section analysis (header + 50 rows)
  - Track divided into segments with timing breakdowns

## Data Structure

### Telemetry Data Columns (Typical)
- `Timestamp`: Time measurement
- `Speed`: Vehicle speed (mph or km/h)
- `Gear`: Current gear selection (1-6)
- `ath`: Throttle blade position (engine throttle opening)
- `aps`: Throttle pedal position (driver input)
- `pbrake_r`: Rear brake pressure
- `pbrake_f`: Front brake pressure
- Additional car-specific telemetry fields

### Usage

These sample files can be used to:
1. Understand the data structure and format
2. Test data loading and processing scripts
3. Develop analysis pipelines without needing the full large datasets
4. Share data structure examples with team members

## Full Datasets

The complete datasets (including full telemetry files) are available locally in the parent `Data/barber/` directory but are excluded from Git due to file size limitations (>100MB per file).

To work with the full datasets:
- Use the files directly from your local `Data/barber/` directory
- Or set up Git LFS for large file tracking
- Or use cloud storage for the full telemetry files

## Notes

- Sample files maintain the same column structure as the originals
- Row counts are reduced for manageability
- All samples include the header row for easy processing

