import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "morphology"
    / "test_morphology_merged_20.json"
)


# ============================================================
# LOAD
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


print("=" * 75)
print("MERGED MORPHOLOGY QUALITY ANALYSIS")
print("=" * 75)

print(
    "Sentences:",
    len(data)
)


# ============================================================
# HELPERS
# ============================================================

def is_valid_morphology(item):
    """
    Basic validation of:
        surface_lemma_tag
    """

    surface = item.get("surface", "")
    lemma = item.get("lemma", "")
    tag = item.get("tag", "")

    return bool(
        surface
        and lemma
        and tag
    )


def is_verb_tag(tag):
    """
    Sanskrit5 morphology tags beginning with:

    SPr = present
    SPs = past
    SFu = future
    SPr2Im = imperative
    SPr3O = optative
    """

    return (
        tag.startswith("SPr")
        or tag.startswith("SPs")
        or tag.startswith("SFu")
    )


def get_tense(tag):

    if tag.startswith("SPr"):
        return "PRESENT"

    if tag.startswith("SPs"):
        return "PAST"

    if tag.startswith("SFu"):
        return "FUTURE"

    return None


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total_words = 0
total_morphology = 0
valid_records = 0
invalid_records = 0

sentences_100 = 0
sentences_90 = 0
sentences_below_90 = 0

all_tags = Counter()

tense_counter = Counter()

missing_sentences = []


# ============================================================
# SENTENCE ANALYSIS
# ============================================================

for number, record in enumerate(
    data,
    start=1
):

    sentence = record["sanskrit"]

    original_words = sentence.split()

    morphology = record.get(
        "morphology",
        []
    )

    word_count = len(
        original_words
    )

    morph_count = len(
        morphology
    )

    total_words += word_count
    total_morphology += morph_count

    coverage = 0.0

    if word_count > 0:
        coverage = (
            morph_count
            /
            word_count
            *
            100
        )

    # --------------------------------------------------------
    # Coverage categories
    # --------------------------------------------------------

    if coverage >= 100:
        sentences_100 += 1

    elif coverage >= 90:
        sentences_90 += 1

    else:
        sentences_below_90 += 1

    # --------------------------------------------------------
    # Validate morphology
    # --------------------------------------------------------

    invalid = []

    for item in morphology:

        if is_valid_morphology(item):

            valid_records += 1

            tag = item["tag"]

            all_tags[tag] += 1

            tense = get_tense(tag)

            if tense:
                tense_counter[tense] += 1

        else:

            invalid_records += 1
            invalid.append(item)

    # --------------------------------------------------------
    # Missing estimate
    # --------------------------------------------------------

    missing = max(
        word_count - morph_count,
        0
    )

    if missing > 0:

        missing_sentences.append(
            {
                "number": number,
                "coverage": coverage,
                "missing": missing,
                "sentence": sentence
            }
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "-" * 75)

    print(
        f"Sentence {number}"
    )

    print(
        "Words:",
        word_count
    )

    print(
        "Morphology:",
        morph_count
    )

    print(
        "Coverage:",
        f"{coverage:.2f}%"
    )

    # --------------------------------------------------------
    # Verb information
    # --------------------------------------------------------

    verbs = []

    for item in morphology:

        tag = item.get(
            "tag",
            ""
        )

        if is_verb_tag(tag):

            verbs.append(
                (
                    item.get("surface"),
                    item.get("lemma"),
                    tag
                )
            )

    if verbs:

        print("Verbs:")

        for surface, lemma, tag in verbs:

            tense = get_tense(tag)

            print(
                f"  {surface} → "
                f"{lemma} → "
                f"{tag}"
                + (
                    f" ({tense})"
                    if tense
                    else ""
                )
            )

    else:

        print(
            "Verbs: NONE DETECTED"
        )


# ============================================================
# OVERALL COVERAGE
# ============================================================

overall_coverage = 0.0

if total_words > 0:

    overall_coverage = (
        total_morphology
        /
        total_words
        *
        100
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("OVERALL QUALITY SUMMARY")
print("=" * 75)

print(
    "Total sentences:",
    len(data)
)

print(
    "Total Sanskrit words:",
    total_words
)

print(
    "Total morphology records:",
    total_morphology
)

print(
    "Estimated coverage:",
    f"{overall_coverage:.2f}%"
)

print(
    "100% coverage:",
    sentences_100
)

print(
    "90–99.99% coverage:",
    sentences_90
)

print(
    "Below 90%:",
    sentences_below_90
)

print(
    "Valid morphology records:",
    valid_records
)

print(
    "Invalid morphology records:",
    invalid_records
)


# ============================================================
# TENSE SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("VERB / TENSE INFORMATION")
print("=" * 75)

print(
    "Present:",
    tense_counter["PRESENT"]
)

print(
    "Past:",
    tense_counter["PAST"]
)

print(
    "Future:",
    tense_counter["FUTURE"]
)


# ============================================================
# TOP MORPHOLOGY TAGS
# ============================================================

print("\n")
print("=" * 75)
print("TOP MORPHOLOGY TAGS")
print("=" * 75)

for tag, count in all_tags.most_common(25):

    print(
        f"{tag:<15} {count}"
    )


# ============================================================
# LOW COVERAGE SENTENCES
# ============================================================

print("\n")
print("=" * 75)
print("SENTENCES BELOW 90% COVERAGE")
print("=" * 75)

if not missing_sentences:

    print(
        "None"
    )

else:

    for item in missing_sentences:

        print(
            f"\nSentence {item['number']}"
        )

        print(
            f"Coverage: "
            f"{item['coverage']:.2f}%"
        )

        print(
            f"Estimated missing: "
            f"{item['missing']}"
        )

        print(
            item["sentence"]
        )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n")
print("=" * 75)
print("INTERPRETATION")
print("=" * 75)

if overall_coverage >= 90:

    print(
        "GOOD: Estimated morphology coverage is above 90%."
    )

else:

    print(
        "NEEDS IMPROVEMENT: Coverage is below 90%."
    )

print(
    "\nNote: Coverage is an estimate based on"
    " morphology-record count versus whitespace-token count."
)

print(
    "It is NOT a true word-level alignment score."
)

print(
    "Use this result for preliminary quality analysis,"
    " not as a final research metric."
)