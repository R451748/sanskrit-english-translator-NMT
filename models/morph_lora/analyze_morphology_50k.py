import json
import re
from collections import Counter
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "models/morph_lora/morphology_train_50k.json"
)

OUTPUT_FILE = Path(
    "models/morph_lora/morphology_50k_analysis.json"
)

# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("50K MORPHOLOGY DATASET ANALYSIS")
print("=" * 70)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"\nTotal records: {len(records):,}")

# ============================================================
# BASIC STATISTICS
# ============================================================

empty_morphology = 0
empty_sanskrit = 0
empty_english = 0

tag_counts = []
morphology_token_counts = []

unique_morphology_tokens = Counter()
unique_tags = Counter()

malformed_records = []
truncated_like_records = []

# Common signs of incomplete generation
truncated_patterns = [
    r"\.\.\.$",
    r"…$",
    r"\.\.\.$",
]

# ============================================================
# ANALYZE EACH RECORD
# ============================================================

for record in records:

    idx = record.get("index")
    sanskrit = str(record.get("sanskrit", "")).strip()
    english = str(record.get("english", "")).strip()
    morphology = str(record.get("morphology", "")).strip()

    if not sanskrit:
        empty_sanskrit += 1

    if not english:
        empty_english += 1

    if not morphology:
        empty_morphology += 1
        malformed_records.append(idx)
        continue

    # --------------------------------------------------------
    # Morphology output format:
    #
    # word_lemma_TAG
    #
    # Example:
    # rāmaḥ_rāma_SNM
    # --------------------------------------------------------

    tokens = morphology.split()

    morphology_token_counts.append(len(tokens))

    for token in tokens:

        unique_morphology_tokens[token] += 1

        # Try to extract final morphology tag
        parts = token.rsplit("_", 1)

        if len(parts) == 2:
            word_part, tag = parts

            if tag:
                unique_tags[tag] += 1

                tag_counts.append(tag)

        else:
            malformed_records.append(idx)

    # --------------------------------------------------------
    # Check for suspiciously short output
    # --------------------------------------------------------

    sanskrit_words = len(
        sanskrit.split()
    )

    morph_words = len(tokens)

    # If morphology contains very few tokens compared
    # with the Sanskrit sentence, flag it for inspection.
    if sanskrit_words >= 10 and morph_words <= 3:
        truncated_like_records.append({
            "index": idx,
            "sanskrit_words": sanskrit_words,
            "morphology_tokens": morph_words,
            "sanskrit": sanskrit,
            "morphology": morphology
        })

    # Check obvious truncation patterns
    for pattern in truncated_patterns:

        if re.search(pattern, morphology):
            truncated_like_records.append({
                "index": idx,
                "reason": "possible_truncation",
                "sanskrit": sanskrit,
                "morphology": morphology
            })
            break


# ============================================================
# STATISTICS
# ============================================================

if morphology_token_counts:

    avg_tokens = (
        sum(morphology_token_counts)
        / len(morphology_token_counts)
    )

    min_tokens = min(
        morphology_token_counts
    )

    max_tokens = max(
        morphology_token_counts
    )

    median_tokens = sorted(
        morphology_token_counts
    )[len(morphology_token_counts) // 2]

else:

    avg_tokens = 0
    min_tokens = 0
    max_tokens = 0
    median_tokens = 0


# ============================================================
# COVERAGE
# ============================================================

valid_morphology = (
    len(records) - empty_morphology
)

coverage = (
    valid_morphology / len(records) * 100
    if records
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BASIC QUALITY")
print("=" * 70)

print(
    f"Empty Sanskrit:              {empty_sanskrit:,}"
)

print(
    f"Empty English:               {empty_english:,}"
)

print(
    f"Empty morphology:            {empty_morphology:,}"
)

print(
    f"Morphology coverage:         {coverage:.2f}%"
)

print(
    f"Malformed records:           {len(set(malformed_records)):,}"
)

print(
    f"Possible truncated/sparse:   "
    f"{len(truncated_like_records):,}"
)


print("\n" + "=" * 70)
print("MORPHOLOGY TOKEN STATISTICS")
print("=" * 70)

print(
    f"Average morphology tokens:   {avg_tokens:.2f}"
)

print(
    f"Median morphology tokens:    {median_tokens}"
)

print(
    f"Minimum morphology tokens:   {min_tokens}"
)

print(
    f"Maximum morphology tokens:   {max_tokens}"
)

print(
    f"Unique morphology tokens:    "
    f"{len(unique_morphology_tokens):,}"
)

print(
    f"Unique morphology tags:      "
    f"{len(unique_tags):,}"
)


# ============================================================
# MOST COMMON TAGS
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 MORPHOLOGY TAGS")
print("=" * 70)

for tag, count in unique_tags.most_common(30):

    print(
        f"{tag:25s} {count:,}"
    )


# ============================================================
# MORPHOLOGY TOKEN LENGTH DISTRIBUTION
# ============================================================

distribution = Counter(
    morphology_token_counts
)

print("\n" + "=" * 70)
print("MORPHOLOGY TOKENS / SENTENCE")
print("=" * 70)

for count in sorted(distribution)[:30]:

    print(
        f"{count:3d} tokens : "
        f"{distribution[count]:,} sentences"
    )


# ============================================================
# SAMPLE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE MORPHOLOGY OUTPUTS")
print("=" * 70)

for record in records[:10]:

    print("\nIndex:", record["index"])

    print(
        "Sanskrit:",
        record["sanskrit"][:300]
    )

    print(
        "Morphology:",
        record["morphology"][:500]
    )


# ============================================================
# SUSPICIOUS EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SUSPICIOUS / SPARSE EXAMPLES")
print("=" * 70)

for item in truncated_like_records[:20]:

    print("\nIndex:", item["index"])

    print(
        "Sanskrit:",
        item["sanskrit"][:400]
    )

    print(
        "Morphology:",
        item["morphology"][:500]
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

analysis = {
    "total_records": len(records),

    "empty_sanskrit": empty_sanskrit,
    "empty_english": empty_english,
    "empty_morphology": empty_morphology,

    "morphology_coverage_percent": coverage,

    "malformed_records": len(
        set(malformed_records)
    ),

    "possible_sparse_or_truncated_records": len(
        truncated_like_records
    ),

    "average_morphology_tokens": avg_tokens,
    "median_morphology_tokens": median_tokens,
    "minimum_morphology_tokens": min_tokens,
    "maximum_morphology_tokens": max_tokens,

    "unique_morphology_tokens": len(
        unique_morphology_tokens
    ),

    "unique_morphology_tags": len(
        unique_tags
    ),

    "top_morphology_tags": dict(
        unique_tags.most_common(50)
    ),

    "token_count_distribution": {
        str(k): v
        for k, v in sorted(
            distribution.items()
        )
    }
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        analysis,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_FILE}"
)