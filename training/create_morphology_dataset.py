import json
import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "dataset/final/train.csv"

MORPHOLOGY_FILE = (
    "morphology/sentence_morphology_features_2720.json"
)

OUTPUT_FILE = (
    "dataset/processed/morphology_train_2720.json"
)

# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("CREATING MORPHOLOGY TRAINING DATASET")
print("=" * 70)

print("\nLoading MITRA training data...")

df = pd.read_csv(TRAIN_FILE)

df["sanskrit"] = (
    df["sanskrit"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["english"] = (
    df["english"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(
    f"Training rows available: {len(df):,}"
)

print("\nLoading morphology features...")

with open(
    MORPHOLOGY_FILE,
    "r",
    encoding="utf-8"
) as f:
    morphology_data = json.load(f)

tag_to_id = morphology_data["tag_to_id"]
morphology_records = morphology_data["records"]

print(
    f"Morphology records: "
    f"{len(morphology_records):,}"
)

# ============================================================
# CREATE ENGLISH LOOKUP
# ============================================================

english_lookup = {}

for _, row in df.iterrows():

    sanskrit = row["sanskrit"]

    if sanskrit:
        english_lookup[sanskrit] = row["english"]

# ============================================================
# COMBINE
# ============================================================

output = []

matched = 0
missing_english = 0

for record in morphology_records:

    sanskrit = (
        record.get("sanskrit", "")
        .strip()
    )

    if not sanskrit:
        continue

    english = english_lookup.get(
        sanskrit
    )

    if english is None:
        missing_english += 1
        continue

    output.append({

        "index": record.get("index"),

        "sanskrit": sanskrit,

        "english": english,

        "surfaces": record.get(
            "surfaces",
            []
        ),

        "lemmas": record.get(
            "lemmas",
            []
        ),

        "morphology_tag_ids": record.get(
            "morphology_tag_ids",
            []
        )
    })

    matched += 1

# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "tag_to_id": tag_to_id,
            "records": output
        },
        f,
        ensure_ascii=False,
        indent=2
    )

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("DATASET CREATION COMPLETE")
print("=" * 70)

print(
    f"Morphology records : "
    f"{len(morphology_records):,}"
)

print(
    f"Matched records    : "
    f"{matched:,}"
)

print(
    f"Missing English    : "
    f"{missing_english:,}"
)

print(
    f"Tag vocabulary     : "
    f"{len(tag_to_id):,}"
)

print(
    f"\nSaved to:"
)

print(OUTPUT_FILE)

print("=" * 70)

# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\nFirst 5 examples:")
print("=" * 70)

for record in output[:5]:

    print("\nSanskrit:")
    print(record["sanskrit"])

    print("\nEnglish:")
    print(record["english"])

    print("\nMorphology IDs:")
    print(record["morphology_tag_ids"])

    print("-" * 70)