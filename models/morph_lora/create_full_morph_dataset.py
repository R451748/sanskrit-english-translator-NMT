import json
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(
    "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "final"
    / "train.csv"
)

MORPH_FILE = (
    PROJECT_ROOT
    / "morphology"
    / "sentence_morphology_5504.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "morphology_train_5504.json"
)


print("Loading original training data...")

df = pd.read_csv(
    TRAIN_FILE
)

print(
    "Training rows:",
    len(df)
)


# Sanskrit → English lookup
translation_lookup = {}

for _, row in df.iterrows():

    sanskrit = str(
        row["sanskrit"]
    ).strip()

    english = str(
        row["english"]
    ).strip()

    if sanskrit:
        translation_lookup[sanskrit] = english


print(
    "Unique Sanskrit sentences:",
    len(translation_lookup)
)


# Load morphology
print("\nLoading morphology...")

with open(
    MORPH_FILE,
    "r",
    encoding="utf-8"
) as f:

    morphology = json.load(f)


print(
    "Morphology records:",
    len(morphology)
)


# Combine
records = []

missing = 0

for morph in morphology:

    sanskrit = str(
        morph["sanskrit"]
    ).strip()

    english = translation_lookup.get(
        sanskrit
    )

    if english is None:

        missing += 1

        continue

    records.append({

        "index": morph["index"],

        "sanskrit": sanskrit,

        "english": english,

        "morphology_tags":
            morph["morphology_tags"],

        "morphology_tag_ids":
            morph["morphology_tag_ids"],

        "morphology_tag_count":
            morph["morphology_tag_count"],

        "surfaces":
            morph["surfaces"],

        "lemmas":
            morph["lemmas"]
    })


# Save
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n====================================")
print("Morphology dataset created")
print("====================================")

print(
    "Morphology records:",
    len(morphology)
)

print(
    "Matched:",
    len(records)
)

print(
    "Missing:",
    missing
)

print(
    "Output:",
    OUTPUT_FILE
)


if records:

    print("\nFirst record:")

    print(
        json.dumps(
            records[0],
            ensure_ascii=False,
            indent=2
        )
    )