import json
import random
from pathlib import Path


PROJECT_ROOT = Path(
    "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "morphology_train_5504.json"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "train.json"
)

VAL_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "validation.json"
)


SEED = 42
VAL_RATIO = 0.10


# ============================================================
# LOAD
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "Total records:",
    len(data)
)


# ============================================================
# SHUFFLE
# ============================================================

random.seed(SEED)

random.shuffle(data)


# ============================================================
# SPLIT
# ============================================================

val_size = int(
    len(data) * VAL_RATIO
)

val_data = data[:val_size]

train_data = data[val_size:]


# ============================================================
# SAVE
# ============================================================

with open(
    TRAIN_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        train_data,
        f,
        ensure_ascii=False,
        indent=2
    )


with open(
    VAL_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        val_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n================================")
print("Morphology split created")
print("================================")

print(
    "Training:",
    len(train_data)
)

print(
    "Validation:",
    len(val_data)
)

print(
    "Seed:",
    SEED
)

print(
    "Train file:",
    TRAIN_FILE
)

print(
    "Validation file:",
    VAL_FILE
)
