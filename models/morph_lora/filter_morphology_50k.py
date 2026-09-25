import json
import re
from pathlib import Path

INPUT_FILE = Path("models/morph_lora/morphology_train_50k.json")
OUTPUT_FILE = Path("models/morph_lora/morphology_train_50k_filtered.json")
REJECTED_FILE = Path("models/morph_lora/morphology_train_50k_rejected.json")

MIN_COVERAGE = 0.60
MIN_MORPH_TOKENS = 2


def source_words(text):
    return text.strip().split()


def morphology_tokens(text):
    if not text:
        return []

    # Morphology is normally separated by whitespace
    return text.strip().split()


def calculate_coverage(source, morphology):
    src_words = source_words(source)
    morph_words = morphology_tokens(morphology)

    if not src_words:
        return 0.0

    return len(morph_words) / len(src_words)


def looks_valid_token(token):
    """
    Basic sanity check.

    We don't require exact surface-word matching because
    morphology can legitimately normalize a word to its lemma.
    """

    if not token:
        return False

    # Expected morphology format usually contains underscores.
    if "_" not in token:
        return False

    parts = token.split("_")

    # At least surface + lemma + some morphological information
    if len(parts) < 3:
        return False

    surface = parts[0].strip()
    lemma = parts[1].strip()

    if not surface or not lemma:
        return False

    return True


def morphology_quality(morphology):
    tokens = morphology_tokens(morphology)

    if not tokens:
        return False, 0

    valid = sum(looks_valid_token(t) for t in tokens)

    return valid > 0, valid


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


accepted = []
rejected = []

stats = {
    "total": len(data),
    "accepted": 0,
    "rejected_empty": 0,
    "rejected_low_coverage": 0,
    "rejected_invalid": 0,
    "rejected_too_few_tokens": 0,
}


for record in data:

    source = record.get("sanskrit", "").strip()
    morphology = record.get("morphology", "").strip()

    src_count = len(source_words(source))
    morph_count = len(morphology_tokens(morphology))

    reason = None

    # --------------------------------------------------
    # 1. Empty source / morphology
    # --------------------------------------------------

    if not source or not morphology:
        reason = "empty_morphology"
        stats["rejected_empty"] += 1

    # --------------------------------------------------
    # 2. Too few morphology tokens
    # --------------------------------------------------

    elif morph_count < MIN_MORPH_TOKENS:
        reason = "too_few_morphology_tokens"
        stats["rejected_too_few_tokens"] += 1

    # --------------------------------------------------
    # 3. Coverage
    # --------------------------------------------------

    else:
        coverage = calculate_coverage(source, morphology)

        if coverage < MIN_COVERAGE:
            reason = f"low_coverage_{coverage:.3f}"
            stats["rejected_low_coverage"] += 1

    # --------------------------------------------------
    # 4. Token validity
    # --------------------------------------------------

    if reason is None:
        valid, valid_count = morphology_quality(morphology)

        if not valid:
            reason = "invalid_morphology"
            stats["rejected_invalid"] += 1

    # --------------------------------------------------
    # Accept / reject
    # --------------------------------------------------

    if reason is None:

        record_copy = dict(record)

        coverage = calculate_coverage(source, morphology)

        record_copy["morphology_coverage"] = round(
            coverage, 4
        )

        record_copy["morphology_token_count"] = morph_count

        accepted.append(record_copy)

    else:

        rejected_record = dict(record)
        rejected_record["rejection_reason"] = reason

        rejected.append(rejected_record)


stats["accepted"] = len(accepted)


# ------------------------------------------------------
# Save filtered dataset
# ------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        accepted,
        f,
        ensure_ascii=False,
        indent=2
    )


# ------------------------------------------------------
# Save rejected dataset separately
# ------------------------------------------------------

with open(REJECTED_FILE, "w", encoding="utf-8") as f:
    json.dump(
        rejected,
        f,
        ensure_ascii=False,
        indent=2
    )


# ------------------------------------------------------
# Report
# ------------------------------------------------------

print()
print("=" * 70)
print("50K MORPHOLOGY QUALITY FILTER")
print("=" * 70)

print(f"Input records:              {stats['total']:,}")
print(f"Accepted records:           {stats['accepted']:,}")
print(f"Rejected records:           {len(rejected):,}")

print()
print("Rejection breakdown:")
print(f"  Empty morphology:         {stats['rejected_empty']:,}")
print(f"  Too few tokens:           {stats['rejected_too_few_tokens']:,}")
print(f"  Low coverage (<60%):     {stats['rejected_low_coverage']:,}")
print(f"  Invalid morphology:      {stats['rejected_invalid']:,}")

print()
print(f"Minimum coverage:           {MIN_COVERAGE:.0%}")
print(f"Minimum morphology tokens:  {MIN_MORPH_TOKENS}")

print()
print(f"Filtered dataset:")
print(OUTPUT_FILE)

print()
print(f"Rejected records:")
print(REJECTED_FILE)

print("=" * 70)