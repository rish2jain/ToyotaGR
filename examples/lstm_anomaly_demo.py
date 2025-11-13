"""
LSTM Anomaly Detection Demo

This script demonstrates how to use the LSTM-based anomaly detector
for racing telemetry data. It shows:

1. Loading telemetry data
2. Training the LSTM autoencoder
3. Detecting anomalies based on reconstruction error
4. Comparing with statistical methods
5. Visualizing results

Usage:
    python examples/lstm_anomaly_demo.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tactical.anomaly_detector import AnomalyDetector, LSTMAnomalyDetector, TENSORFLOW_AVAILABLE


def create_sample_telemetry_data(n_samples=500, n_anomalies=25):
    """
    Create sample telemetry data with known anomalies for testing.

    Args:
        n_samples: Total number of samples
        n_anomalies: Number of anomalous samples to inject

    Returns:
        DataFrame with telemetry features
    """
    print(f"Creating sample telemetry data: {n_samples} samples, {n_anomalies} anomalies")

    # Create time series data
    time = np.linspace(0, 100, n_samples)

    # Normal patterns (sinusoidal with noise)
    speed = 120 + 30 * np.sin(time / 5) + np.random.normal(0, 5, n_samples)
    throttle = 50 + 40 * np.sin(time / 4 + 1) + np.random.normal(0, 3, n_samples)
    brake = np.maximum(0, 30 - throttle / 2) + np.random.normal(0, 2, n_samples)
    steering = 10 * np.sin(time / 3) + np.random.normal(0, 2, n_samples)
    rpm = 5000 + 2000 * np.sin(time / 5) + np.random.normal(0, 100, n_samples)
    gear = np.clip(np.round(3 + np.sin(time / 8)), 2, 5)

    # Inject anomalies (sudden changes in patterns)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    for idx in anomaly_indices:
        speed[idx] += np.random.uniform(-40, 40)
        throttle[idx] += np.random.uniform(-30, 30)
        brake[idx] += np.random.uniform(0, 40)
        steering[idx] += np.random.uniform(-20, 20)
        rpm[idx] += np.random.uniform(-1000, 1000)

    # Create lap numbers (simulate 10 laps)
    lap_numbers = np.repeat(range(1, 11), n_samples // 10)[:n_samples]

    # Create lap times (with anomalies having higher lap times)
    base_lap_time = 85.5
    lap_times = np.full(n_samples, base_lap_time)
    lap_times[anomaly_indices] += np.random.uniform(2, 5, len(anomaly_indices))

    df = pd.DataFrame({
        'LAP_NUMBER': lap_numbers,
        'KPH': np.clip(speed, 0, 200),
        'aps': np.clip(throttle, 0, 100),
        'pbrake_f': np.clip(brake, 0, 100),
        'Steering_Angle': steering,
        'nmot': np.clip(rpm, 0, 8000),
        'gear': gear,
        'lap_seconds': lap_times,
        'is_true_anomaly': False  # Ground truth
    })

    df.loc[anomaly_indices, 'is_true_anomaly'] = True

    print(f"✓ Sample data created with {df['is_true_anomaly'].sum()} true anomalies")
    return df


def demo_lstm_detection(telemetry_data):
    """
    Demonstrate LSTM anomaly detection.

    Args:
        telemetry_data: DataFrame with telemetry data
    """
    print("\n" + "="*80)
    print("LSTM ANOMALY DETECTION DEMO")
    print("="*80)

    if not TENSORFLOW_AVAILABLE:
        print("\n❌ TensorFlow is not installed!")
        print("Install with: pip install tensorflow")
        return

    # Initialize detector
    print("\n1. Initializing LSTM Anomaly Detector...")
    lstm_detector = LSTMAnomalyDetector(sequence_length=50, verbose=1)
    print("   ✓ Detector initialized with sequence length = 50")

    # Run detection
    print("\n2. Training LSTM autoencoder and detecting anomalies...")
    print("   (This may take 30-90 seconds depending on your hardware)")

    try:
        result_df = lstm_detector.detect_pattern_anomalies(
            telemetry_data,
            epochs=30,  # Reduced for demo speed
            batch_size=32,
            contamination=0.05
        )

        print("   ✓ LSTM training complete!")

        # Analyze results
        lstm_anomalies = result_df[result_df['lstm_is_anomaly']]
        print(f"\n3. Results:")
        print(f"   - Total samples: {len(result_df)}")
        print(f"   - Anomalies detected: {len(lstm_anomalies)}")
        print(f"   - Detection rate: {len(lstm_anomalies) / len(result_df) * 100:.1f}%")

        # Calculate performance metrics if ground truth available
        if 'is_true_anomaly' in result_df.columns:
            true_anomalies = result_df['is_true_anomaly']
            detected_anomalies = result_df['lstm_is_anomaly']

            true_positives = ((true_anomalies == True) & (detected_anomalies == True)).sum()
            false_positives = ((true_anomalies == False) & (detected_anomalies == True)).sum()
            false_negatives = ((true_anomalies == True) & (detected_anomalies == False)).sum()

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n4. Performance Metrics:")
            print(f"   - Precision: {precision:.3f}")
            print(f"   - Recall: {recall:.3f}")
            print(f"   - F1 Score: {f1_score:.3f}")
            print(f"   - True Positives: {true_positives}")
            print(f"   - False Positives: {false_positives}")
            print(f"   - False Negatives: {false_negatives}")

        return result_df

    except Exception as e:
        print(f"\n❌ Error during LSTM detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_detection_methods(telemetry_data):
    """
    Compare LSTM with statistical anomaly detection.

    Args:
        telemetry_data: DataFrame with telemetry data
    """
    print("\n" + "="*80)
    print("COMPARISON: STATISTICAL vs LSTM DETECTION")
    print("="*80)

    # Initialize detector
    detector = AnomalyDetector()

    # Statistical detection
    print("\n1. Running Statistical Detection (Z-Score)...")
    stat_result = detector.detect_statistical_anomalies(
        telemetry_data,
        window=5,
        threshold=2.5
    )
    stat_anomalies = stat_result[stat_result['anomaly_count'] > 0]
    print(f"   ✓ Statistical method detected {len(stat_anomalies)} anomalies")

    # LSTM detection
    if TENSORFLOW_AVAILABLE:
        print("\n2. Running LSTM Detection...")
        lstm_result = detector.detect_lstm_anomalies(
            telemetry_data,
            sequence_length=50,
            epochs=30,
            contamination=0.05
        )
        lstm_anomalies = lstm_result[lstm_result['lstm_is_anomaly']]
        print(f"   ✓ LSTM method detected {len(lstm_anomalies)} anomalies")

        # Compare
        print("\n3. Comparison Summary:")
        print(f"   {'Method':<25} {'Anomalies':<15} {'Detection Rate':<15}")
        print(f"   {'-'*25} {'-'*15} {'-'*15}")
        print(f"   {'Statistical (Z-Score)':<25} {len(stat_anomalies):<15} {len(stat_anomalies)/len(telemetry_data)*100:.1f}%")
        print(f"   {'Deep Learning (LSTM)':<25} {len(lstm_anomalies):<15} {len(lstm_anomalies)/len(telemetry_data)*100:.1f}%")

        # Agreement between methods
        if 'anomaly_count' in stat_result.columns:
            stat_mask = stat_result['anomaly_count'] > 0
            lstm_mask = lstm_result['lstm_is_anomaly']
            agreement = (stat_mask == lstm_mask).sum()
            print(f"\n   Agreement between methods: {agreement / len(telemetry_data) * 100:.1f}%")

        return stat_result, lstm_result
    else:
        print("\n❌ TensorFlow not available, skipping LSTM comparison")
        return stat_result, None


def visualize_results(telemetry_data, lstm_result):
    """
    Create visualizations of LSTM detection results.

    Args:
        telemetry_data: Original telemetry data
        lstm_result: Results from LSTM detection
    """
    if lstm_result is None:
        print("\nSkipping visualization (no LSTM results)")
        return

    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Speed with anomalies highlighted
    ax = axes[0]
    ax.plot(range(len(telemetry_data)), telemetry_data['KPH'],
            label='Speed', alpha=0.7, linewidth=1)

    anomaly_indices = lstm_result[lstm_result['lstm_is_anomaly']].index
    ax.scatter(anomaly_indices, telemetry_data.loc[anomaly_indices, 'KPH'],
              color='red', s=50, label='LSTM Anomalies', zorder=5)

    if 'is_true_anomaly' in telemetry_data.columns:
        true_anomaly_indices = telemetry_data[telemetry_data['is_true_anomaly']].index
        ax.scatter(true_anomaly_indices, telemetry_data.loc[true_anomaly_indices, 'KPH'],
                  color='orange', s=30, marker='x', label='True Anomalies', zorder=4, alpha=0.6)

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed with LSTM-Detected Anomalies')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Reconstruction Error
    ax = axes[1]
    ax.plot(range(len(lstm_result)), lstm_result['lstm_reconstruction_error'],
           label='Reconstruction Error', color='blue', alpha=0.7)

    if hasattr(lstm_result, 'threshold'):
        ax.axhline(y=lstm_result.threshold, color='orange', linestyle='--',
                  label='Threshold', linewidth=2)

    ax.scatter(anomaly_indices, lstm_result.loc[anomaly_indices, 'lstm_reconstruction_error'],
              color='red', s=50, label='Anomalies', zorder=5)

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title('LSTM Reconstruction Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Anomaly Score Distribution
    ax = axes[2]
    normal_scores = lstm_result[~lstm_result['lstm_is_anomaly']]['lstm_anomaly_score']
    anomaly_scores = lstm_result[lstm_result['lstm_is_anomaly']]['lstm_anomaly_score']

    ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='green')
    ax.hist(anomaly_scores, bins=30, alpha=0.6, label='Anomalies', color='red')

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Anomaly Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = 'examples/lstm_anomaly_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        print("  (Close the plot window to continue)")


def main():
    """Main demo function."""
    print("\n" + "="*80)
    print("LSTM ANOMALY DETECTION FOR RACING TELEMETRY")
    print("RaceIQ Pro - Deep Learning Module Demo")
    print("="*80)

    # Check TensorFlow availability
    if not TENSORFLOW_AVAILABLE:
        print("\n❌ TensorFlow is not installed!")
        print("\nTo run this demo, install TensorFlow:")
        print("   pip install tensorflow")
        print("\nFor CPU-only installation:")
        print("   pip install tensorflow-cpu")
        return

    print("\n✓ TensorFlow is available")

    # Option 1: Use sample data
    print("\n" + "-"*80)
    print("OPTION 1: Using Synthetic Sample Data")
    print("-"*80)

    sample_data = create_sample_telemetry_data(n_samples=500, n_anomalies=25)

    # Run LSTM detection
    lstm_result = demo_lstm_detection(sample_data)

    if lstm_result is not None:
        # Compare methods
        stat_result, lstm_result = compare_detection_methods(sample_data)

        # Visualize
        visualize_results(sample_data, lstm_result)

    # Option 2: Use real data if available
    print("\n" + "-"*80)
    print("OPTION 2: Using Real Barber Telemetry Data (if available)")
    print("-"*80)

    real_data_path = 'Data/barber/Samples/R1_barber_telemetry_data_sample.csv'
    if os.path.exists(real_data_path):
        print(f"\n✓ Found real telemetry data at: {real_data_path}")
        print("   (Implementation note: Real data requires preprocessing)")
        print("   (See src/pipeline/data_loader.py for loading utilities)")
    else:
        print(f"\n⚠ Real telemetry data not found at: {real_data_path}")
        print("   Use synthetic data above for demonstration")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. LSTM autoencoders learn temporal patterns in telemetry sequences")
    print("2. Anomalies are detected by high reconstruction error")
    print("3. LSTM can detect complex patterns that statistical methods miss")
    print("4. Training takes 30-90 seconds, inference is fast (<1 second)")
    print("5. Best used for time-series data with temporal dependencies")
    print("\nFor integration into your application:")
    print("- See: src/tactical/anomaly_detector.py (LSTMAnomalyDetector class)")
    print("- Dashboard: dashboard/pages/tactical.py (Deep Learning tab)")
    print("- Docs: docs/LSTM_ANOMALY_DETECTION.md")


if __name__ == "__main__":
    main()
