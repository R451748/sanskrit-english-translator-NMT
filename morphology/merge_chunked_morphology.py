import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "morphology"
    / "test_morphology_chunked_20.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "morphology"
    / "test_morphology_merged_20.json"
)


# ============================================================
# MORPHOLOGY PARSER
# ============================================================

def parse_morphology(text):
    """
    Convert morphology output into structured records.

    Example:
    ramaḥ_rāma_SNM
    """

    results = []

    if not text:
        return results

    # Split by whitespace
    tokens = text.strip().split()

    for token in tokens:

        # Ignore malformed tokens
        if "_" not in token:
            continue

        parts = token.split("_")

        if len(parts) < 3:
            continue

        surface = parts[0]
        lemma = parts[1]
        tag = parts[-1]

        if not surface:
            continue

        results.append(
            {
                "surface": surface,
                "lemma": lemma,
                "tag": tag,
                "raw": token
            }
        )

    return results


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def is_duplicate(previous, current):
    """
    Detect duplicate morphology caused by overlapping chunks.

    Primary comparison:
        surface

    Secondary comparison:
        surface + lemma + tag
    """

    if not previous:
        return False

    if previous["surface"] == current["surface"]:
        return True

    return (
        previous["surface"] == current["surface"]
        and
        previous["lemma"] == current["lemma"]
        and
        previous["tag"] == current["tag"]
    )


# ============================================================
# MERGE ONE SENTENCE
# ============================================================

def merge_sentence(sentence_record):

    original_sentence = sentence_record["sanskrit"]

    chunks = sentence_record["chunks"]

    merged = []

    for chunk in chunks:

        morphology_text = chunk.get(
            "morphology",
            ""
        )

        parsed = parse_morphology(
            morphology_text
        )

        for item in parsed:

            # ------------------------------------------------
            # Remove duplicate overlap
            # ------------------------------------------------

            duplicate = False

            # Check only recent items because
            # overlap occurs between neighboring chunks.
            recent_items = merged[-5:]

            for previous in recent_items:

                if is_duplicate(
                    previous,
                    item
                ):
                    duplicate = True
                    break

            if duplicate:
                continue

            merged.append(item)

    # ========================================================
    # STATISTICS
    # ========================================================

    original_words = original_sentence.split()

    original_word_count = len(
        original_words
    )

    morphology_count = len(
        merged
    )

    # Surface forms recovered
    recovered_surfaces = [
        item["surface"]
        for item in merged
    ]

    recovered_count = len(
        recovered_surfaces
    )

    coverage = 0.0

    if original_word_count > 0:
        coverage = (
            recovered_count
            /
            original_word_count
            *
            100
        )

    return {
        "test_row": sentence_record["test_row"],
        "sanskrit": original_sentence,
        "original_word_count": original_word_count,
        "morphology_token_count": morphology_count,
        "coverage_estimate_percent": round(
            coverage,
            2
        ),
        "morphology": merged
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("MERGING CHUNKED MORPHOLOGY")
print("=" * 70)

print("\nInput:")
print(INPUT_FILE)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


print(
    "\nInput sentences:",
    len(data)
)


# ============================================================
# PROCESS
# ============================================================

merged_results = []

total_original_words = 0
total_morphology_tokens = 0


for number, record in enumerate(
    data,
    start=1
):

    result = merge_sentence(
        record
    )

    merged_results.append(
        result
    )

    total_original_words += (
        result["original_word_count"]
    )

    total_morphology_tokens += (
        result["morphology_token_count"]
    )

    print(
        f"\n[{number}/{len(data)}]"
    )

    print(
        "Original words:",
        result["original_word_count"]
    )

    print(
        "Morphology tokens:",
        result["morphology_token_count"]
    )

    print(
        "Coverage estimate:",
        result["coverage_estimate_percent"],
        "%"
    )


# ============================================================
# OVERALL COVERAGE
# ============================================================

overall_coverage = 0.0

if total_original_words > 0:

    overall_coverage = (
        total_morphology_tokens
        /
        total_original_words
        *
        100
    )


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
        merged_results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MERGE COMPLETED")
print("=" * 70)

print(
    "Sentences:",
    len(merged_results)
)

print(
    "Original words:",
    total_original_words
)

print(
    "Merged morphology tokens:",
    total_morphology_tokens
)

print(
    "Coverage estimate:",
    round(
        overall_coverage,
        2
    ),
    "%"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)