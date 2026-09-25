import json
import re
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "models/morph_lora/morphology_train_50k.json"
)

OUTPUT_FILE = Path(
    "models/morph_lora/morphology_50k_validation.json"
)

# Coverage thresholds
GOOD_THRESHOLD = 0.80
ACCEPTABLE_THRESHOLD = 0.60

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("50K MORPHOLOGY COVERAGE VALIDATION")
print("=" * 70)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"\nTotal records: {len(records):,}")


# ============================================================
# HELPERS
# ============================================================

def normalize_word(word):
    """
    Normalize a Sanskrit word for simple comparison.
    """
    word = word.strip()
    word = re.sub(r"[|।॥,;:!?()\[\]{}]", "", word)
    return word


def extract_morphology_word(token):
    """
    Extract the surface-word part from:

        word_lemma_TAG

    Example:

        rāmaḥ_rāma_SNM

    -> rāmaḥ
    """

    if "_" not in token:
        return None

    return token.split("_")[0]


def is_tagged_token(token):
    """
    Check whether the token appears to contain:

        word_lemma_TAG

    or at least:

        word_
    """

    if "_" not in token:
        return False

    parts = token.split("_")

    # Need at least word + something
    return len(parts) >= 2 and parts[0].strip() != ""


# ============================================================
# STORAGE
# ============================================================

results = []

category_counts = Counter()

coverage_values = []

empty_records = []
poor_records = []
acceptable_records = []
good_records = []

# ============================================================
# PROCESS
# ============================================================

for record in records:

    index = record["index"]

    sanskrit = str(
        record.get("sanskrit", "")
    ).strip()

    morphology = str(
        record.get("morphology", "")
    ).strip()

    # --------------------------------------------------------
    # Sanskrit words
    # --------------------------------------------------------

    sanskrit_tokens = [
        normalize_word(x)
        for x in sanskrit.split()
        if normalize_word(x)
    ]

    source_word_count = len(
        sanskrit_tokens
    )

    # --------------------------------------------------------
    # Morphology tokens
    # --------------------------------------------------------

    morph_tokens = [
        x.strip()
        for x in morphology.split()
        if x.strip()
    ]

    valid_morph_tokens = [
        x
        for x in morph_tokens
        if is_tagged_token(x)
    ]

    morph_word_count = len(
        valid_morph_tokens
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if source_word_count == 0:

        coverage = 0.0

    else:

        coverage = min(
            morph_word_count / source_word_count,
            1.0
        )

    coverage_values.append(
        coverage
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    if morph_word_count == 0:

        category = "empty"

        empty_records.append(record)

    elif coverage < ACCEPTABLE_THRESHOLD:

        category = "poor"

        poor_records.append({
            **record,
            "source_word_count": source_word_count,
            "morphology_word_count": morph_word_count,
            "coverage": coverage
        })

    elif coverage < GOOD_THRESHOLD:

        category = "acceptable"

        acceptable_records.append({
            **record,
            "source_word_count": source_word_count,
            "morphology_word_count": morph_word_count,
            "coverage": coverage
        })

    else:

        category = "good"

        good_records.append({
            **record,
            "source_word_count": source_word_count,
            "morphology_word_count": morph_word_count,
            "coverage": coverage
        })

    category_counts[category] += 1

    results.append({
        "index": index,
        "source_word_count": source_word_count,
        "morphology_word_count": morph_word_count,
        "coverage": round(coverage, 4),
        "category": category
    })


# ============================================================
# STATISTICS
# ============================================================

average_coverage = (
    sum(coverage_values)
    / len(coverage_values)
    if coverage_values
    else 0
)

sorted_coverage = sorted(
    coverage_values
)

median_coverage = (
    sorted_coverage[len(sorted_coverage) // 2]
    if sorted_coverage
    else 0
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE SUMMARY")
print("=" * 70)

print(
    f"Good (>= 80%):             "
    f"{category_counts['good']:,}"
)

print(
    f"Acceptable (60-79%):       "
    f"{category_counts['acceptable']:,}"
)

print(
    f"Poor (< 60%):              "
    f"{category_counts['poor']:,}"
)

print(
    f"Empty:                      "
    f"{category_counts['empty']:,}"
)

print(
    f"\nAverage coverage:          "
    f"{average_coverage * 100:.2f}%"
)

print(
    f"Median coverage:           "
    f"{median_coverage * 100:.2f}%"
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE DISTRIBUTION")
print("=" * 70)

ranges = [
    ("0%", 0.0, 0.0),
    ("1-19%", 0.0001, 0.20),
    ("20-39%", 0.20, 0.40),
    ("40-59%", 0.40, 0.60),
    ("60-79%", 0.60, 0.80),
    ("80-99%", 0.80, 1.00),
    ("100%", 1.00, 1.01),
]

for name, low, high in ranges:

    count = sum(
        1
        for value in coverage_values
        if low <= value < high
    )

    print(
        f"{name:10s}: {count:,}"
    )


# ============================================================
# LONG SENTENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LONG SENTENCE COVERAGE")
print("=" * 70)

long_ranges = [
    ("1-10 words", 1, 10),
    ("11-20 words", 11, 20),
    ("21-30 words", 21, 30),
    ("31-40 words", 31, 40),
    ("41+ words", 41, 10_000),
]

for name, low, high in long_ranges:

    subset = [
        r
        for r in results
        if low <= r["source_word_count"] <= high
    ]

    if subset:

        avg = (
            sum(r["coverage"] for r in subset)
            / len(subset)
        )

        good = sum(
            1
            for r in subset
            if r["coverage"] >= GOOD_THRESHOLD
        )

        print(
            f"{name:15s}: "
            f"{len(subset):6,} sentences | "
            f"avg coverage "
            f"{avg * 100:6.2f}% | "
            f">=80% "
            f"{good:6,}"
        )


# ============================================================
# SHOW POOR EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("POOR COVERAGE EXAMPLES")
print("=" * 70)

for record in poor_records[:20]:

    print(
        f"\nIndex: {record['index']}"
    )

    print(
        f"Source words: "
        f"{record['source_word_count']}"
    )

    print(
        f"Morphology tokens: "
        f"{record['morphology_word_count']}"
    )

    print(
        f"Coverage: "
        f"{record['coverage'] * 100:.2f}%"
    )

    print(
        "Sanskrit:",
        record["sanskrit"][:400]
    )

    print(
        "Morphology:",
        record["morphology"][:500]
    )


# ============================================================
# SAVE DETAILED VALIDATION
# ============================================================

validation = {
    "total_records": len(records),

    "good_records": category_counts["good"],
    "acceptable_records": category_counts["acceptable"],
    "poor_records": category_counts["poor"],
    "empty_records": category_counts["empty"],

    "average_coverage": average_coverage,
    "median_coverage": median_coverage,

    "records": results
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        validation,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_FILE}"
)