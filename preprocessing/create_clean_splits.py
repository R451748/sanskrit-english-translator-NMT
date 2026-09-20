import pandas as pd
import os

INPUT_DIR = "dataset/processed"
OUTPUT_DIR = "dataset/final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(INPUT_DIR, "train.csv")
VAL_FILE = os.path.join(INPUT_DIR, "validation.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test.csv")

print("=" * 70)
print("CREATING LEAKAGE-FREE FINAL DATASET")
print("=" * 70)

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print("\nOriginal cleaned sizes:")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

# ------------------------------------------------------------
# Remove any duplicate rows again for safety
# ------------------------------------------------------------

train = train.drop_duplicates(
    subset=["sanskrit", "english"]
).reset_index(drop=True)

validation = validation.drop_duplicates(
    subset=["sanskrit", "english"]
).reset_index(drop=True)

test = test.drop_duplicates(
    subset=["sanskrit", "english"]
).reset_index(drop=True)

# ------------------------------------------------------------
# Sanskrit sentence sets
# ------------------------------------------------------------

train_sanskrit = set(train["sanskrit"])

# ------------------------------------------------------------
# Remove Validation sentences already present in Train
# ------------------------------------------------------------

validation_before = len(validation)

validation = validation[
    ~validation["sanskrit"].isin(train_sanskrit)
].reset_index(drop=True)

validation_removed = validation_before - len(validation)

# ------------------------------------------------------------
# Remove Test sentences present in Train
# ------------------------------------------------------------

test_before = len(test)

test = test[
    ~test["sanskrit"].isin(train_sanskrit)
].reset_index(drop=True)

test_removed_train = test_before - len(test)

# ------------------------------------------------------------
# Remove Test sentences present in Validation
# ------------------------------------------------------------

validation_sanskrit = set(validation["sanskrit"])

test_before_val = len(test)

test = test[
    ~test["sanskrit"].isin(validation_sanskrit)
].reset_index(drop=True)

test_removed_validation = test_before_val - len(test)

# ------------------------------------------------------------
# Save final datasets
# ------------------------------------------------------------

train.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False,
    encoding="utf-8"
)

validation.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False,
    encoding="utf-8"
)

test.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False,
    encoding="utf-8"
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\nRemoved from Validation:", validation_removed)
print("Removed from Test due to Train:", test_removed_train)
print("Removed from Test due to Validation:", test_removed_validation)

print("\nFinal dataset sizes:")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

print("\nTotal:", len(train) + len(validation) + len(test))

print("\nSaved:")
print("dataset/final/train.csv")
print("dataset/final/validation.csv")
print("dataset/final/test.csv")

print("\n" + "=" * 70)
print("FINAL DATASET CREATED")
print("=" * 70)