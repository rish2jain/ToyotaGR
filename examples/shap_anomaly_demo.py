"""
SHAP Anomaly Detection Demo
Demonstrates SHAP explainability for anomaly detection in racing telemetry
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tactical.anomaly_detector import AnomalyDetector

# Check if SHAP is available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP is not installed. Install with: pip install shap")
    print("This demo will run but explanations will be limited.\n")


def load_sample_data():
    """
    Load sample racing telemetry data from the Data directory.
    Falls back to synthetic data if real data is not available.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'Data')

    # Try to load real data
    sample_files = [
        'barber-2023/barber-2023-race1/barber-2023-race1-sections.csv',
        'barber-2023/barber-2023-race2/barber-2023-race2-sections.csv'
    ]

    for file_path in sample_files:
        full_path = os.path.join(data_dir, file_path)
        if os.path.exists(full_path):
            print(f"Loading data from: {file_path}")
            df = pd.read_csv(full_path)
            print(f"Loaded {len(df)} rows with columns: {list(df.columns)}\n")
            return df

    # Generate synthetic data if no real data found
    print("No real data found. Generating synthetic telemetry data...\n")

    np.random.seed(42)
    n_laps = 50

    # Generate synthetic telemetry
    data = {
        'LAP_NUMBER': range(1, n_laps + 1),
        'DRIVER_NUMBER': [1] * n_laps,
        'S1_SECONDS': np.random.normal(25.5, 0.5, n_laps),
        'S2_SECONDS': np.random.normal(28.2, 0.6, n_laps),
        'S3_SECONDS': np.random.normal(32.1, 0.7, n_laps),
        'TOP_SPEED': np.random.normal(185, 5, n_laps),
        'KPH': np.random.normal(145, 8, n_laps),
    }

    df = pd.DataFrame(data)

    # Inject some anomalies
    anomaly_laps = [10, 25, 42]
    for lap in anomaly_laps:
        idx = lap - 1
        df.at[idx, 'S1_SECONDS'] += np.random.uniform(2, 4)  # Slow S1
        df.at[idx, 'TOP_SPEED'] -= np.random.uniform(10, 20)  # Lower top speed

    print(f"Generated {len(df)} synthetic laps with {len(anomaly_laps)} injected anomalies\n")
    return df


def run_statistical_anomaly_demo(detector, telemetry_df):
    """Demonstrate statistical anomaly detection (Tier 1)"""
    print("="*70)
    print("TIER 1: STATISTICAL ANOMALY DETECTION (Z-Score)")
    print("="*70)

    result_df = detector.detect_statistical_anomalies(
        telemetry_df,
        window=5,
        threshold=2.5
    )

    anomalies = result_df[result_df['anomaly_count'] > 0]
    print(f"Found {len(anomalies)} anomalies using statistical methods")

    if len(anomalies) > 0:
        print("\nTop 3 anomalies:")
        display_cols = ['LAP_NUMBER', 'anomaly_count']
        available_cols = [col for col in display_cols if col in anomalies.columns]
        print(anomalies[available_cols].head(3).to_string(index=False))

    print("\n")
    return result_df


def run_ml_anomaly_demo(detector, telemetry_df):
    """Demonstrate machine learning anomaly detection (Tier 2)"""
    print("="*70)
    print("TIER 2: ML ANOMALY DETECTION (Isolation Forest)")
    print("="*70)

    result_df = detector.detect_pattern_anomalies(
        telemetry_df,
        contamination=0.1  # Expect 10% anomalies
    )

    anomalies = result_df[result_df['is_anomaly'] == -1]
    print(f"Found {len(anomalies)} anomalies using Isolation Forest")
    print(f"Average anomaly score: {anomalies['anomaly_score'].mean():.4f}")

    if len(anomalies) > 0:
        print("\nTop 3 most anomalous laps:")
        display_cols = ['LAP_NUMBER', 'anomaly_score']
        available_cols = [col for col in display_cols if col in anomalies.columns]
        print(anomalies[available_cols].sort_values('anomaly_score').head(3).to_string(index=False))

    print("\n")
    return result_df, anomalies


def run_shap_explanation_demo(detector, anomalies_df, telemetry_df):
    """Demonstrate SHAP explanations for anomalies"""
    print("="*70)
    print("SHAP EXPLAINABILITY FOR ANOMALIES")
    print("="*70)

    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping explanation demo.")
        print("Install SHAP with: pip install shap\n")
        return

    if len(anomalies_df) == 0:
        print("No anomalies to explain.\n")
        return

    # Get explanations for all anomalies
    print("Generating SHAP explanations for all anomalies...")
    explained_df = detector.get_anomaly_explanations(anomalies_df, telemetry_df)

    print(f"Generated explanations for {len(explained_df)} anomalies\n")

    # Show detailed explanation for top 3 anomalies
    top_anomalies = explained_df.sort_values('anomaly_score').head(3)

    for i, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
        print(f"\n{'='*70}")
        print(f"ANOMALY #{i} - Lap {row.get('LAP_NUMBER', 'N/A')}")
        print(f"{'='*70}")
        print(f"Anomaly Score: {row['anomaly_score']:.4f}")
        print(f"Confidence: {row['confidence']:.2%}")
        print(f"\nExplanation: {row['explanation']}")

        print(f"\nTop Contributing Features:")
        for j in range(1, 4):
            feature = row.get(f'top_feature_{j}', '')
            contribution = row.get(f'contribution_{j}', 0)
            if feature:
                print(f"  {j}. {feature}: {contribution:.1%}")

    print("\n")
    return explained_df


def visualize_shap_importance(detector, anomaly_row):
    """Visualize SHAP feature importance for a single anomaly"""
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping visualization.")
        return

    print("="*70)
    print("SHAP FEATURE IMPORTANCE VISUALIZATION")
    print("="*70)

    try:
        # Generate explanation
        explanation = detector.explain_anomaly(anomaly_row)

        # Create bar chart of feature importance
        top_features = explanation['top_features'][:10]  # Top 10 features

        features = [f['feature'] for f in top_features]
        contributions = [f['contribution'] * 100 for f in top_features]
        colors = ['red' if f['direction'] == 'high' else 'blue' if f['direction'] == 'low' else 'gray'
                  for f in top_features]

        plt.figure(figsize=(10, 6))
        plt.barh(features, contributions, color=colors)
        plt.xlabel('Contribution (%)')
        plt.title('SHAP Feature Importance for Anomaly Detection')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), 'shap_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to: {output_path}")

        # Try to display if in interactive environment
        try:
            plt.show()
        except:
            pass

        print("\n")

    except Exception as e:
        print(f"Error generating visualization: {e}\n")


def main():
    """Main demonstration function"""
    print("\n" + "="*70)
    print("SHAP ANOMALY DETECTION DEMO - RaceIQ Pro")
    print("="*70 + "\n")

    # Load data
    telemetry_df = load_sample_data()

    # Initialize detector
    detector = AnomalyDetector()

    # Run statistical anomaly detection
    statistical_result = run_statistical_anomaly_demo(detector, telemetry_df)

    # Run ML anomaly detection
    ml_result, ml_anomalies = run_ml_anomaly_demo(detector, telemetry_df)

    # Generate SHAP explanations
    if len(ml_anomalies) > 0:
        explained_df = run_shap_explanation_demo(detector, ml_anomalies, ml_result)

        # Visualize feature importance for the most anomalous lap
        if explained_df is not None and len(explained_df) > 0:
            most_anomalous = ml_result.loc[explained_df['anomaly_score'].idxmin()]
            visualize_shap_importance(detector, most_anomalous)

    # Summary
    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Statistical methods detect outliers based on z-scores")
    print("2. ML methods (Isolation Forest) detect complex multivariate anomalies")
    print("3. SHAP explanations identify which features contributed to anomalies")
    print("4. Feature importance helps engineers understand what went wrong")
    print("\nNext Steps:")
    print("- Integrate SHAP explanations into the Streamlit dashboard")
    print("- Use explanations to provide actionable recommendations")
    print("- Build historical anomaly patterns database")
    print("\n")


if __name__ == "__main__":
    main()
