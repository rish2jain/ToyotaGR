"""
Verify RaceIQ Pro project structure and imports
"""

import sys
from pathlib import Path

print("=" * 80)
print("RaceIQ Pro - Project Structure Verification")
print("=" * 80)

# Check directory structure
print("\n1. Directory Structure:")
project_dirs = [
    "src/pipeline",
    "src/tactical",
    "src/strategic",
    "src/integration",
    "src/utils",
    "dashboard",
    "tests",
    "notebooks",
    "data/processed",
]

for dir_path in project_dirs:
    full_path = Path(dir_path)
    status = "✓" if full_path.exists() else "✗"
    print(f"   {status} {dir_path}")

# Check key files
print("\n2. Key Files:")
key_files = [
    "src/__init__.py",
    "src/pipeline/__init__.py",
    "src/pipeline/data_loader.py",
    "src/pipeline/validator.py",
    "src/pipeline/feature_engineer.py",
    "src/utils/__init__.py",
    "src/utils/constants.py",
    "src/utils/metrics.py",
    "src/utils/visualization.py",
    "src/tactical/__init__.py",
    "src/strategic/__init__.py",
    "src/integration/__init__.py",
    "requirements.txt",
    "tests/test_data_loader.py",
    "notebooks/01_data_exploration.ipynb",
]

for file_path in key_files:
    full_path = Path(file_path)
    status = "✓" if full_path.exists() else "✗"
    size = f"({full_path.stat().st_size} bytes)" if full_path.exists() else ""
    print(f"   {status} {file_path} {size}")

# Check sample data files
print("\n3. Sample Data Files:")
sample_dir = Path("Data/barber/Samples")
if sample_dir.exists():
    sample_files = list(sample_dir.glob("*.csv")) + list(sample_dir.glob("*.CSV"))
    for file in sorted(sample_files):
        size = file.stat().st_size
        print(f"   ✓ {file.name} ({size:,} bytes)")
else:
    print(f"   ✗ Sample data directory not found: {sample_dir}")

# Try importing modules (without pandas requirement)
print("\n4. Module Import Check:")

sys.path.insert(0, str(Path.cwd() / "src"))

try:
    from utils import constants
    print("   ✓ utils.constants imported successfully")
    print(f"     - PROJECT_ROOT: {constants.PROJECT_ROOT}")
    print(f"     - DATA_DIR: {constants.DATA_DIR}")
except Exception as e:
    print(f"   ✗ Failed to import utils.constants: {e}")

# Check tactical module placeholders
try:
    from tactical import OptimalGhostAnalyzer, AnomalyDetector, SectionAnalyzer
    print("   ✓ tactical module classes imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import tactical module: {e}")

# Check strategic module placeholders
try:
    from strategic import PitStopDetector, TireDegradationModel, PitStrategyOptimizer
    print("   ✓ strategic module classes imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import strategic module: {e}")

# Check integration module placeholders
try:
    from integration import IntegrationEngine, RecommendationBuilder
    print("   ✓ integration module classes imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import integration module: {e}")

print("\n" + "=" * 80)
print("Verification Complete!")
print("=" * 80)

print("\nNOTE: To run full tests with data loading, install dependencies:")
print("  pip install -r requirements.txt")
print("\nThen run:")
print("  python tests/test_data_loader.py")
print("  streamlit run dashboard/app.py")
