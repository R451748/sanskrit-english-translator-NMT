import json
import glob
import os
import re
from collections import Counter


INPUT_DIR = "morphology/batches"
OUTPUT_FILE = "morphology/clean_morphology_6336.json"


# ------------------------------------------------------------
# Load all batch files
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "batch_*.json")
    )
)

records = []

for file in files:

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    records.extend(data)


print("=" * 70)
print("CLEANING EXISTING MORPHOLOGY DATA")
print("=" * 70)

print("Batch files :", len(files))
print("Raw records :", len(records))


# ------------------------------------------------------------
# Keep only records with valid dataset index
# ------------------------------------------------------------

indexed = []

for record in records:

    if "index" not in record:
        continue

    if "sanskrit" not in record:
        continue

    if "morphology" not in record:
        continue

    index = record["index"]

    if not isinstance(index, int):
        try:
            index = int(index)
        except:
            continue

    morphology = str(
        record["morphology"]
    ).strip()

    sanskrit = str(
        record["sanskrit"]
    ).strip()

    if not sanskrit:
        continue

    if not morphology:
        continue

    indexed.append(
        {
            "index": index,
            "sanskrit": sanskrit,
            "morphology": morphology
        }
    )


print(
    "Records with valid index:",
    len(indexed)
)


# ------------------------------------------------------------
# Remove duplicate indexes
# ------------------------------------------------------------

unique = {}

for record in indexed:

    index = record["index"]

    if index not in unique:

        unique[index] = record


records = list(
    unique.values()
)

records.sort(
    key=lambda x: x["index"]
)


print(
    "Unique indexed records:",
    len(records)
)


# ------------------------------------------------------------
# Basic morphology validation
# ------------------------------------------------------------

valid_records = []
suspicious_records = []

for record in records:

    morphology = record["morphology"]

    tokens = morphology.split()

    if len(tokens) == 0:

        continue

    valid_tokens = 0
    suspicious_tokens = 0

    for token in tokens:

        # Expected basic form:
        #
        # surface_lemma_tag
        #
        # Example:
        # rāmaḥ_rāma_SNM

        parts = token.split("_")

        if len(parts) >= 3:

            surface = parts[0]
            lemma = parts[1]
            tag = "_".join(parts[2:])

            if (
                surface
                and lemma
                and tag
            ):

                valid_tokens += 1

            else:

                suspicious_tokens += 1

        else:

            suspicious_tokens += 1


    # Accept if at least one valid morphology token exists.
    if valid_tokens > 0:

        record["valid_token_count"] = valid_tokens
        record["suspicious_token_count"] = suspicious_tokens

        if suspicious_tokens > 0:

            suspicious_records.append(record)

        else:

            valid_records.append(record)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print()
print("=" * 70)
print("QUALITY RESULTS")
print("=" * 70)

print(
    "Clean records:",
    len(valid_records)
)

print(
    "Records containing suspicious tokens:",
    len(suspicious_records)
)

print(
    "Total discarded:",
    len(records)
    - len(valid_records)
    - len(suspicious_records)
)


# ------------------------------------------------------------
# Token statistics
# ------------------------------------------------------------

all_tags = Counter()

total_tokens = 0
valid_tokens = 0
suspicious_tokens = 0

for record in records:

    for token in record["morphology"].split():

        total_tokens += 1

        parts = token.split("_")

        if len(parts) >= 3:

            tag = "_".join(
                parts[2:]
            )

            if tag:

                all_tags[tag] += 1
                valid_tokens += 1

            else:

                suspicious_tokens += 1

        else:

            suspicious_tokens += 1


print()
print(
    "Total morphology tokens:",
    total_tokens
)

print(
    "Valid morphology tokens:",
    valid_tokens
)

print(
    "Suspicious morphology tokens:",
    suspicious_tokens
)

print(
    "Unique morphology tags:",
    len(all_tags)
)


# ------------------------------------------------------------
# Save clean dataset
# ------------------------------------------------------------

output_records = []

for record in valid_records:

    output_records.append(
        {
            "index": record["index"],
            "sanskrit": record["sanskrit"],
            "morphology": record["morphology"]
        }
    )


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_records,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Records:",
    len(output_records)
)

print("=" * 70)