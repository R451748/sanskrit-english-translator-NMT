import json
import glob
import os
import re
from collections import Counter


INPUT_DIR = "morphology/batches"
OUTPUT_FILE = "morphology/morphology_analysis_5504.json"


# ============================================================
# LOAD RECORDS
# ============================================================

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

    for record in data:

        # Only records that have dataset index
        if "index" not in record:
            continue

        if "sanskrit" not in record:
            continue

        if "morphology" not in record:
            continue

        records.append(record)


# ============================================================
# REMOVE DUPLICATE INDEXES
# ============================================================

unique = {}

for record in records:

    index = int(record["index"])

    if index not in unique:

        unique[index] = record


records = sorted(
    unique.values(),
    key=lambda x: int(x["index"])
)


print("=" * 70)
print("MORPHOLOGY ANALYSIS")
print("=" * 70)

print("Batch files:", len(files))
print("Indexed records:", len(records))


# ============================================================
# BASIC STATISTICS
# ============================================================

total_tokens = 0
nonempty_records = 0
empty_records = 0

token_counter = Counter()

for record in records:

    morphology = str(
        record["morphology"]
    ).strip()

    if not morphology:

        empty_records += 1
        continue

    nonempty_records += 1

    tokens = morphology.split()

    total_tokens += len(tokens)

    for token in tokens:

        token_counter[token] += 1


print()
print("Non-empty morphology:", nonempty_records)
print("Empty morphology:", empty_records)
print("Total morphology tokens:", total_tokens)
print("Unique output tokens:", len(token_counter))


# ============================================================
# UNDERSCORE ANALYSIS
# ============================================================

pattern_3 = 0
pattern_2 = 0
pattern_1 = 0
no_underscore = 0

for token in token_counter:

    underscore_count = token.count("_")

    if underscore_count >= 2:

        pattern_3 += token_counter[token]

    elif underscore_count == 1:

        pattern_2 += token_counter[token]

    elif underscore_count == 0:

        no_underscore += token_counter[token]


print()
print("=" * 70)
print("TOKEN FORMAT ANALYSIS")
print("=" * 70)

print(
    "Tokens with >=2 underscores:",
    pattern_3
)

print(
    "Tokens with 1 underscore:",
    pattern_2
)

print(
    "Tokens without underscore:",
    no_underscore
)


# ============================================================
# COMMON MORPHOLOGY OUTPUTS
# ============================================================

print()
print("=" * 70)
print("MOST COMMON OUTPUT TOKENS")
print("=" * 70)

for token, count in token_counter.most_common(50):

    print(
        f"{count:5d}  {token}"
    )


# ============================================================
# RECORD-LEVEL CLASSIFICATION
# ============================================================

good_records = []
review_records = []
empty_records_list = []

for record in records:

    morphology = str(
        record["morphology"]
    ).strip()

    if not morphology:

        empty_records_list.append(record)
        continue

    tokens = morphology.split()

    has_structured = False
    has_unstructured = False

    for token in tokens:

        if token.count("_") >= 2:

            has_structured = True

        else:

            has_unstructured = True


    # We do NOT discard records containing
    # unusual tokens.
    #
    # They are retained for research review.

    if has_structured and not has_unstructured:

        good_records.append(record)

    else:

        review_records.append(record)


print()
print("=" * 70)
print("RECORD CLASSIFICATION")
print("=" * 70)

print(
    "Structured records:",
    len(good_records)
)

print(
    "Records needing review:",
    len(review_records)
)

print(
    "Empty records:",
    len(empty_records_list)
)


# ============================================================
# SAVE ALL INDEXED RECORDS
# ============================================================

output = {
    "statistics": {
        "batch_files": len(files),
        "indexed_records": len(records),
        "nonempty_records": nonempty_records,
        "empty_records": empty_records,
        "total_tokens": total_tokens,
        "unique_output_tokens": len(token_counter),
        "structured_records": len(good_records),
        "review_records": len(review_records)
    },

    "records": records
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
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
    "Records saved:",
    len(records)
)

print("=" * 70)