# Toyota GR Cup Hackathon Project

Repository for the Toyota GR Cup "Hack the Track" hackathon project.

## Project Overview

**RaceIQ Pro** - A complete racing intelligence platform combining tactical driver coaching with strategic race planning.

### Key Features

- **Tactical Analysis Module**: Section-by-section performance analysis, anomaly detection, and driver coaching
- **Strategic Analysis Module**: Tire degradation modeling, pit strategy optimization, and position impact analysis
- **Integration Engine**: Cross-module intelligence connecting tactical insights with strategic recommendations

## Repository Structure

```
ToyotaGR/
├── Data/                          # Race data files
│   ├── barber/                    # Barber Motorsports Park data
│   ├── COTA/                      # Circuit of the Americas data
│   ├── indianapolis/              # Indianapolis Motor Speedway data
│   ├── road-america/              # Road America data
│   ├── sebring/                   # Sebring International Raceway data
│   ├── Sonoma/                    # Sonoma Raceway data
│   └── virginia-international-raceway/  # VIR data
├── Research/                      # Research and analysis documents
│   ├── Enhanced Recommendation - Toyota GR Cup Hackathon.md
│   ├── Dataset Analysis_ What's Actually Possible with th.md
│   └── Other LLMs.md
├── skills/                        # Project skill documentation
└── RaceMind_Proposal.md          # Initial project proposal
```

## Data Files

### Included Files

- Race results (official and provisional)
- Lap timing data (start, end, duration)
- Section-by-section analysis
- Weather data
- Best lap analysis
- Circuit maps (PDF)

### Excluded Files (Large Telemetry Data)

The following large telemetry CSV files (>100MB each) are excluded from Git due to GitHub's file size limits:

- `*_telemetry_data.csv` files (ranging from 800MB to 3.4GB)
- These files contain high-frequency telemetry data (speed, throttle, brake pressure, gear)

**To include these files:**
1. Use Git LFS: `git lfs track "*.csv"` and commit the telemetry files
2. Or store them in cloud storage (AWS S3, Google Drive, etc.) and reference in documentation
3. Or compress/sample the data before committing

**Note**: The telemetry data is available locally in the `Data/` directories but is not tracked by Git.

## Key Documents

- **[Enhanced Recommendation](./Research/Enhanced%20Recommendation%20-%20Toyota%20GR%20Cup%20Hackathon.md)**: Comprehensive project plan with technical stack, implementation roadmap, and recommendations
- **[Dataset Analysis](./Research/Dataset%20Analysis_%20What's%20Actually%20Possible%20with%20th.md)**: Analysis of available data and feasibility assessment
- **[LLM Feedback](./Research/Other%20LLMs.md)**: Synthesis of feedback from multiple LLM perspectives

## Technical Stack

- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, SHAP, PyOD
- **Deep Learning**: TensorFlow/PyTorch (for LSTM anomaly detection)
- **Visualization**: Streamlit, Plotly, matplotlib
- **Bayesian Inference**: pymc3/arviz (for uncertainty quantification)
- **Optimization**: scipy.optimize, numpy (Monte Carlo simulation)

## Getting Started

1. Clone the repository
2. Install dependencies (see requirements.txt - to be added)
3. Explore the data in the `Data/` directory
4. Review the enhanced recommendation document for implementation details

## Project Status

🚧 **In Development** - Initial planning and research phase complete

## License

[Add license information]

## Contact

[Add contact information]

