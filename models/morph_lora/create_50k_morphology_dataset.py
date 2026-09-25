import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("dataset/final/train.csv")
OUTPUT_FILE = Path("models/morph_lora/morphology_source_50k.csv")

TOTAL_SAMPLES = 50_000
RANDOM_SEED = 42

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("50K MORPHOLOGY EXPERIMENT - DATASET CREATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal training dataset: {len(df):,} rows")

# ============================================================
# REMOVE EXACT DUPLICATE PAIRS
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["sanskrit", "english"]
).reset_index(drop=True)

print(f"After duplicate removal: {len(df):,}")
print(f"Removed: {before - len(df):,}")

# ============================================================
# CHECK AVAILABLE DATA
# ============================================================

if len(df) < TOTAL_SAMPLES:
    raise ValueError(
        f"Dataset contains only {len(df):,} rows, "
        f"but {TOTAL_SAMPLES:,} are required."
    )

# ============================================================
# SAMPLE 50K
# ============================================================

sample = df.sample(
    n=TOTAL_SAMPLES,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

sample.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

# ============================================================
# VERIFICATION
# ============================================================

saved = pd.read_csv(OUTPUT_FILE)

print("\n" + "=" * 70)
print("50K DATASET CREATED")
print("=" * 70)

print(f"Rows: {len(saved):,}")
print(f"Columns: {list(saved.columns)}")
print(f"Duplicate pairs: {saved.duplicated(['sanskrit', 'english']).sum():,}")

print("\nFirst 5 records:")
print(saved.head())

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nDONE")