"""
Configuration file
"""

from pathlib import Path

# ============================================================================
# Directory Setup
# ============================================================================

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data file names
IST_DATA_FILE = 'IST_data_v25.csv'

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Analysis Parameters
# ============================================================================
RANDOM_SEED = 42



if __name__ == "__main__":
    # Test that all directories are created
    print("Project Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data:     {RAW_DATA_DIR}")
    print(f"Processed:    {PROCESSED_DATA_DIR}")
    print(f"Figures:      {FIGURES_DIR}")
    print(f"Tables:       {TABLES_DIR}")
    print("=" * 60)
    print("All directories created successfully!")