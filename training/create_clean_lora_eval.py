import pandas as pd
import os

PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset/final/validation.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset/final/lora_clean_eval.csv"
)

# ------------------------------------------------------------
# Load full validation set
# ------------------------------------------------------------

df = pd.read_csv(VAL_FILE)

print("Full validation size:", len(df))

# ------------------------------------------------------------
# Recreate the exact 2,500-row validation selection
# used by the 25K LoRA experiment
# ------------------------------------------------------------

used_validation = df.sample(
    n=2500,
    random_state=42
)

used_indices = set(
    used_validation.index
)

# ------------------------------------------------------------
# Remove those rows
# ------------------------------------------------------------

clean_eval = df[
    ~df.index.isin(used_indices)
].copy()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

clean_eval.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Clean evaluation size:", len(clean_eval))

print("Saved:")
print(OUTPUT_FILE)

