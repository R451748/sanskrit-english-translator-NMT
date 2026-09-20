import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "dataset/final/train.csv"
OUTPUT_FILE = "dataset/final/morphology_50k.csv"

SUBSET_SIZE = 50_000

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING MORPHOLOGY SUBSET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Full dataset size: {len(df):,}")

# ============================================================
# CHECK COLUMN
# ============================================================

if "sanskrit" not in df.columns:
    raise ValueError(
        "Column 'sanskrit' not found in dataset."
    )

# Remove missing Sanskrit sentences
df = df.dropna(subset=["sanskrit"]).copy()

# Remove empty sentences
df["sanskrit"] = df["sanskrit"].astype(str).str.strip()

df = df[df["sanskrit"] != ""]

# ============================================================
# CREATE SUBSET
# ============================================================

subset = df.head(SUBSET_SIZE).copy()

print(f"Morphology subset size: {len(subset):,}")

# ============================================================
# SAVE
# ============================================================

subset.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Saved to:")
print(OUTPUT_FILE)

print(f"Rows saved: {len(subset):,}")