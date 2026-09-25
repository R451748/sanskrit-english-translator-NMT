import json
import random
from pathlib import Path

INPUT_FILE = Path(
    "models/morph_lora/morphology_train_50k_filtered.json"
)

OUTPUT_DIR = Path(
    "models/morph_lora/morphology_50k_filtered_split"
)

TRAIN_FILE = OUTPUT_DIR / "train.json"
VAL_FILE = OUTPUT_DIR / "validation.json"

SEED = 42
TRAIN_RATIO = 0.90

random.seed(SEED)

# --------------------------------------------------
# Load
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("50K FILTERED MORPHOLOGY SPLIT")
print("=" * 70)

print(f"Total records: {len(data):,}")

# --------------------------------------------------
# Shuffle deterministically
# --------------------------------------------------

random.shuffle(data)

split_index = int(len(data) * TRAIN_RATIO)

train_data = data[:split_index]
val_data = data[split_index:]

# --------------------------------------------------
# Create output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Save
# --------------------------------------------------

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(
        train_data,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(VAL_FILE, "w", encoding="utf-8") as f:
    json.dump(
        val_data,
        f,
        ensure_ascii=False,
        indent=2
    )

# --------------------------------------------------
# Verify
# --------------------------------------------------

train_indices = {
    record.get("index")
    for record in train_data
}

val_indices = {
    record.get("index")
    for record in val_data
}

overlap = train_indices & val_indices

print()
print(f"Training records:   {len(train_data):,}")
print(f"Validation records: {len(val_data):,}")
print(f"Train/Val overlap:  {len(overlap)}")

print()
print("Files created:")
print(TRAIN_FILE)
print(VAL_FILE)

print("=" * 70)