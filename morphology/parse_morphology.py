import json
import re
from collections import Counter

INPUT_FILE = "morphology/morphology_2720.json"
OUTPUT_FILE = "morphology/parsed_morphology_2720.json"

# Known morphology-tag pattern used by Sanskrit5.
# Examples:
# SNM, SANe, SPr3In, SPs3In, SVM, Co, Cp, U, etc.
TAG_PATTERN = re.compile(
    r"^(?:"
    r"S[A-Za-z0-9]+|"
    r"P[A-Za-z0-9]+|"
    r"C[A-Za-z0-9]+|"
    r"D[A-Za-z0-9]+|"
    r"F[A-Za-z0-9]+|"
    r"U"
    r")$"
)


def parse_morphology(text):
    """
    Convert morphology output into structured tokens.

    Expected normal form:
        surface_lemma_tag

    Example:
        rāmaḥ_rāma_SNM
    """

    if not text:
        return [], []

    # Normalize whitespace
    text = " ".join(str(text).split())

    # Remove common punctuation-only tokens
    raw_tokens = text.split()

    parsed = []
    malformed = []

    for token in raw_tokens:

        # Ignore punctuation
        if token in {"/", "//", ".", ",", ";", "।", "॥"}:
            continue

        parts = token.split("_")

        # We need at least:
        # surface + lemma + tag
        if len(parts) < 3:
            malformed.append(token)
            continue

        # The last component is normally the morphology tag.
        tag = parts[-1]

        # Everything before the final two components is preserved
        # as surface/lemma where possible.
        surface = parts[0]
        lemma = parts[1]

        # Empty fields are invalid
        if not surface or not lemma or not tag:
            malformed.append(token)
            continue

        parsed.append({
            "surface": surface,
            "lemma": lemma,
            "tag": tag
        })

    return parsed, malformed


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SANSKRIT MORPHOLOGY PARSER")
print("=" * 70)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print("Input records:", len(records))

# ============================================================
# PROCESS
# ============================================================

parsed_records = []

tag_counter = Counter()

total_tokens = 0
total_parsed = 0
total_malformed = 0

for record in records:

    sanskrit = record.get("sanskrit", "")
    morphology = record.get("morphology", "")

    parsed_tokens, malformed_tokens = parse_morphology(
        morphology
    )

    for item in parsed_tokens:
        tag_counter[item["tag"]] += 1

    total_tokens += (
        len(parsed_tokens) +
        len(malformed_tokens)
    )

    total_parsed += len(parsed_tokens)
    total_malformed += len(malformed_tokens)

    parsed_records.append({
        "index": record.get("index"),
        "sanskrit": sanskrit,
        "morphology": morphology,
        "tokens": parsed_tokens,
        "malformed": malformed_tokens
    })

# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        parsed_records,
        f,
        ensure_ascii=False,
        indent=2
    )

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("PARSING COMPLETE")
print("=" * 70)

print(f"Records              : {len(parsed_records):,}")
print(f"Total tokens         : {total_tokens:,}")
print(f"Parsed tokens        : {total_parsed:,}")
print(f"Malformed tokens     : {total_malformed:,}")

if total_tokens > 0:
    print(
        f"Parse success        : "
        f"{total_parsed / total_tokens * 100:.2f}%"
    )

print("\nTop morphology tags:")

for tag, count in tag_counter.most_common(30):
    print(f"{tag:12s} {count:,}")

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 70)