import json
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "final"
    / "lora_train_25k.csv"
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


# ============================================================
# LOAD TRANSLATION DATA
# ============================================================

print("Loading translation dataset...")

df = pd.read_csv(
    TRAIN_FILE
)

print(
    f"Translation rows: {len(df)}"
)


# ============================================================
# CREATE SANSKRIT LOOKUP
# ============================================================

sanskrit_lookup = {}

for _, row in df.iterrows():

    sanskrit = str(
        row["sanskrit"]
    ).strip()

    english = str(
        row["english"]
    ).strip()

    if sanskrit:

        sanskrit_lookup[sanskrit] = english


print(
    f"Unique Sanskrit sentences: "
    f"{len(sanskrit_lookup)}"
)


# ============================================================
# LOAD MORPHOLOGY
# ============================================================

print("\nLoading morphology...")

with open(
    MORPH_FILE,
    "r",
    encoding="utf-8"
) as f:

    morphology = json.load(f)

print(
    f"Morphology records: {len(morphology)}"
)


# ============================================================
# MATCH USING SANSKRIT TEXT
# ============================================================

records = []

matched = 0
not_found = 0


for record in morphology:

    morphology_sanskrit = str(
        record["sanskrit"]
    ).strip()

    # --------------------------------------------------------
    # Find corresponding English translation
    # --------------------------------------------------------

    if morphology_sanskrit not in sanskrit_lookup:

        not_found += 1

        continue

    english = sanskrit_lookup[
        morphology_sanskrit
    ]

    records.append({

        "index":
            record["index"],

        "sanskrit":
            morphology_sanskrit,

        "english":
            english,

        "morphology_tags":
            record["morphology_tags"],

        "morphology_tag_ids":
            record["morphology_tag_ids"],

        "morphology_tag_count":
            record["morphology_tag_count"],

        "surfaces":
            record["surfaces"],

        "lemmas":
            record["lemmas"]
    })

    matched += 1


# ============================================================
# SAVE
# ============================================================

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


# ============================================================
# SUMMARY
# ============================================================

print("\n====================================")
print("Dataset creation complete")
print("====================================")

print(
    "Morphology records:",
    len(morphology)
)

print(
    "Matched:",
    matched
)

print(
    "Not found:",
    not_found
)

print(
    "Output records:",
    len(records)
)

print(
    "Output:",
    OUTPUT_FILE
)


# ============================================================
# FIRST RECORD
# ============================================================

if records:

    print("\nFirst matched record:")

    print(
        json.dumps(
            records[0],
            ensure_ascii=False,
            indent=2
        )
    )