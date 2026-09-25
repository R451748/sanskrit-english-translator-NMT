import json
import re
from pathlib import Path

INPUT_FILE = Path(
    "models/morph_lora/morphology_train_50k.json"
)

OUTPUT_FILE = Path(
    "models/morph_lora/morphology_50k_alignment.json"
)


def normalize_word(word):
    word = word.strip()

    word = re.sub(
        r"""[|।॥,;:!?()\[\]{}"'“”‘’]""",
        "",
        word
    )

    return word.strip()


def get_morph_surface(token):
    """
    Example:
        rāmaḥ_rāma_SNM
        -> rāmaḥ
    """

    if "_" not in token:
        return ""

    return token.split("_", 1)[0].strip()


print("=" * 70)
print("50K MORPHOLOGY WORD ALIGNMENT CHECK")
print("=" * 70)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"\nTotal records: {len(records):,}")


good_records = []
partial_records = []
poor_records = []
empty_records = []

coverage_values = []


for record in records:

    index = record["index"]

    sanskrit = str(
        record.get("sanskrit", "")
    ).strip()

    morphology = str(
        record.get("morphology", "")
    ).strip()

    source_words = [
        normalize_word(x)
        for x in sanskrit.split()
        if normalize_word(x)
    ]

    morph_tokens = [
        x.strip()
        for x in morphology.split()
        if x.strip()
    ]

    morph_words = []

    for token in morph_tokens:

        surface = get_morph_surface(token)

        if surface:
            morph_words.append(
                normalize_word(surface)
            )

    source_count = len(source_words)
    morph_count = len(morph_words)

    matched = 0
    source_position = 0

    for morph_word in morph_words:

        if source_position >= source_count:
            break

        # Exact sequential match
        if morph_word == source_words[source_position]:

            matched += 1
            source_position += 1
            continue

        # Allow morphology to skip up to 3 source words
        found = False

        for offset in range(1, 4):

            candidate_position = (
                source_position + offset
            )

            if candidate_position >= source_count:
                break

            if (
                morph_word
                == source_words[candidate_position]
            ):

                matched += 1

                source_position = (
                    candidate_position + 1
                )

                found = True
                break

        if not found:
            continue

    if source_count == 0:
        coverage = 0.0
    else:
        coverage = matched / source_count

    coverage_values.append(coverage)

    result = {
        **record,
        "source_word_count": source_count,
        "morphology_word_count": morph_count,
        "matched_words": matched,
        "alignment_coverage": coverage
    }

    if morph_count == 0:

        empty_records.append(result)

    elif coverage >= 0.80:

        good_records.append(result)

    elif coverage >= 0.60:

        partial_records.append(result)

    else:

        poor_records.append(result)


# ============================================================
# SUMMARY
# ============================================================

total = len(records)

print("\n" + "=" * 70)
print("ALIGNMENT SUMMARY")
print("=" * 70)

print(
    f"Good alignment (>=80%):       "
    f"{len(good_records):,}"
)

print(
    f"Partial alignment (60-79%):   "
    f"{len(partial_records):,}"
)

print(
    f"Poor alignment (<60%):        "
    f"{len(poor_records):,}"
)

print(
    f"Empty morphology:             "
    f"{len(empty_records):,}"
)

usable_60 = (
    len(good_records)
    + len(partial_records)
)

print(
    f"\nUsable >=60%:                 "
    f"{usable_60:,}"
)

print(
    f"Usable >=80%:                 "
    f"{len(good_records):,}"
)


# ============================================================
# COVERAGE STATISTICS
# ============================================================

if coverage_values:

    average = (
        sum(coverage_values)
        / len(coverage_values)
    )

    sorted_values = sorted(coverage_values)

    median = sorted_values[
        len(sorted_values) // 2
    ]

else:

    average = 0
    median = 0


print(
    f"\nAverage word alignment:       "
    f"{average * 100:.2f}%"
)

print(
    f"Median word alignment:        "
    f"{median * 100:.2f}%"
)


# ============================================================
# LONG SENTENCES
# ============================================================

print("\n" + "=" * 70)
print("LONG SENTENCE ALIGNMENT")
print("=" * 70)

all_results = (
    good_records
    + partial_records
    + poor_records
    + empty_records
)

ranges = [
    ("1-10 words", 1, 10),
    ("11-20 words", 11, 20),
    ("21-30 words", 21, 30),
    ("31-40 words", 31, 40),
    ("41+ words", 41, 10000),
]

for name, low, high in ranges:

    subset = [
        r
        for r in all_results
        if low <= r["source_word_count"] <= high
    ]

    if not subset:
        continue

    avg = (
        sum(
            r["alignment_coverage"]
            for r in subset
        )
        / len(subset)
    )

    good = sum(
        1
        for r in subset
        if r["alignment_coverage"] >= 0.80
    )

    usable = sum(
        1
        for r in subset
        if r["alignment_coverage"] >= 0.60
    )

    print(
        f"{name:15s}: "
        f"{len(subset):6,} | "
        f"avg {avg * 100:6.2f}% | "
        f">=80% {good:6,} | "
        f">=60% {usable:6,}"
    )


# ============================================================
# POOR EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("POOR ALIGNMENT EXAMPLES")
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
        f"Morphology words: "
        f"{record['morphology_word_count']}"
    )

    print(
        f"Matched words: "
        f"{record['matched_words']}"
    )

    print(
        f"Alignment: "
        f"{record['alignment_coverage'] * 100:.2f}%"
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
# SAVE
# ============================================================

output = {
    "total_records": total,

    "good_alignment": len(good_records),
    "partial_alignment": len(partial_records),
    "poor_alignment": len(poor_records),
    "empty_morphology": len(empty_records),

    "usable_60_percent": usable_60,
    "usable_80_percent": len(good_records),

    "average_alignment": average,
    "median_alignment": median,

    "good_records": good_records,
    "partial_records": partial_records,
    "poor_records": poor_records,
    "empty_records": empty_records
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


print("\n" + "=" * 70)
print("ALIGNMENT CHECK COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_FILE}"
)