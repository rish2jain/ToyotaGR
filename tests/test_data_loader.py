"""
Test suite for data_loader module
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.data_loader import DataLoader, load_data_for_track


class TestDataLoader:
    """Test cases for DataLoader class"""

    def test_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.base_path is not None

    def test_load_lap_time_sample(self):
        """Test loading lap time sample data"""
        loader = DataLoader()
        try:
            df = loader.load_lap_time_data()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "lap" in df.columns
            assert "vehicle_number" in df.columns
            print(f"✓ Loaded {len(df)} lap time records")
        except FileNotFoundError as e:
            print(f"⚠ Sample data files not available: {e}")
            raise

    def test_load_section_analysis_sample(self):
        """Test loading section analysis sample data"""
        loader = DataLoader()
        try:
            df = loader.load_section_analysis()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "DRIVER_NUMBER" in df.columns
            assert "LAP_NUMBER" in df.columns
            print(f"✓ Loaded {len(df)} section analysis records")
        except FileNotFoundError as e:
            print(f"⚠ Sample data files not available: {e}")
            raise

    def test_load_all_sample_data(self):
        """Test loading all sample data"""
        loader = DataLoader()
        data = loader.load_all_sample_data()
        assert isinstance(data, dict)
        print(f"✓ Loaded {len(data)} data files")
        for key, df in data.items():
            print(f"  - {key}: {len(df)} records")


if __name__ == "__main__":
    # Run tests manually
    test = TestDataLoader()
    print("Running DataLoader tests...\n")

    try:
        test.test_initialization()
        print("✓ Initialization test passed\n")
    except Exception as e:
        print(f"✗ Initialization test failed: {e}\n")

    try:
        test.test_load_lap_time_sample()
        print("✓ Lap time loading test passed\n")
    except Exception as e:
        print(f"✗ Lap time loading test failed: {e}\n")

    try:
        test.test_load_section_analysis_sample()
        print("✓ Section analysis loading test passed\n")
    except Exception as e:
        print(f"✗ Section analysis loading test failed: {e}\n")

    try:
        test.test_load_all_sample_data()
        print("\n✓ All data loading test passed")
    except Exception as e:
        print(f"\n✗ All data loading test failed: {e}")
