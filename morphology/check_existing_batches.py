import json
import glob
import re
from collections import Counter


BATCH_DIR = "morphology/batches"


files = sorted(
    glob.glob(
        f"{BATCH_DIR}/batch_*.json"
    )
)


all_records = []

for file in files:

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    all_records.extend(data)


print("=" * 70)
print("MORPHOLOGY BATCH QUALITY CHECK")
print("=" * 70)

print(
    "Batch files:",
    len(files)
)

print(
    "Total records:",
    len(all_records)
)


# ============================================================
# INDEX ANALYSIS
# ============================================================

indices = []

missing_index = 0

for record in all_records:

    if "index" in record:

        indices.append(
            int(record["index"])
        )

    else:

        missing_index += 1


print()
print("Records with index:", len(indices))
print("Records without index:", missing_index)

if indices:

    print(
        "Minimum index:",
        min(indices)
    )

    print(
        "Maximum index:",
        max(indices)
    )

    print(
        "Unique indexes:",
        len(set(indices))
    )


# ============================================================
# DUPLICATE INDEXES
# ============================================================

counter = Counter(indices)

duplicate_indices = [
    index
    for index, count in counter.items()
    if count > 1
]

print(
    "Duplicate indexes:",
    len(duplicate_indices)
)


# ============================================================
# EMPTY MORPHOLOGY
# ============================================================

empty_morphology = []

for record in all_records:

    morphology = str(
        record.get("morphology", "")
    ).strip()

    if not morphology:

        empty_morphology.append(
            record.get("index")
        )


print(
    "Empty morphology:",
    len(empty_morphology)
)


# ============================================================
# POSSIBLE TRUNCATION
# ============================================================

possible_truncated = []

for record in all_records:

    morphology = str(
        record.get("morphology", "")
    ).strip()

    if not morphology:
        continue

    tokens = morphology.split()

    if not tokens:
        continue

    last_token = tokens[-1]

    # A morphology token normally has:
    # surface_lemma_tag
    #
    # If the final token contains only one underscore
    # or appears to end in an incomplete tag, flag it.

    parts = last_token.split("_")

    suspicious = False

    if len(parts) < 3:

        suspicious = True

    elif parts[-1] == "":

        suspicious = True

    # Very short final morphology tag
    elif len(parts[-1]) <= 1:

        suspicious = True

    if suspicious:

        possible_truncated.append(
            record.get("index")
        )


print(
    "Possible truncated/malformed:",
    len(possible_truncated)
)


# ============================================================
# MORPHOLOGY LENGTH
# ============================================================

lengths = []

for record in all_records:

    morphology = str(
        record.get("morphology", "")
    ).strip()

    if morphology:

        lengths.append(
            len(morphology.split())
        )


if lengths:

    print()
    print(
        "Average morphology tokens:",
        round(
            sum(lengths) / len(lengths),
            2
        )
    )

    print(
        "Maximum morphology tokens:",
        max(lengths)
    )

    print(
        "Minimum morphology tokens:",
        min(lengths)
    )


# ============================================================
# SHOW EXAMPLES OF SUSPICIOUS OUTPUT
# ============================================================

print()
print("=" * 70)
print("SUSPICIOUS EXAMPLES")
print("=" * 70)

shown = 0

for record in all_records:

    morphology = str(
        record.get("morphology", "")
    ).strip()

    if not morphology:
        continue

    tokens = morphology.split()

    if not tokens:
        continue

    last_token = tokens[-1]

    parts = last_token.split("_")

    suspicious = (
        len(parts) < 3
        or parts[-1] == ""
        or len(parts[-1]) <= 1
    )

    if suspicious:

        print()
        print(
            "Index:",
            record.get("index")
        )

        print(
            "Sanskrit:",
            record.get("sanskrit")
        )

        print(
            "Morphology:",
            morphology
        )

        shown += 1

        if shown >= 10:

            break


print()
print("=" * 70)
print("QUALITY CHECK COMPLETE")
print("=" * 70)