import json
import re
from collections import Counter

INPUT_FILE = "morphology/morphology_2720.json"
OUTPUT_FILE = "morphology/parsed_morphology_improved_2720.json"

# ------------------------------------------------------------
# Known Sanskrit5 morphology tag patterns
# ------------------------------------------------------------

TAG_RE = re.compile(
    r"""
    (?:
        S(?:NM|NF|NN|LM|LF|LN|AM|AF|AN|IM|IF|IN|DM|DF|DN|
          BM|BF|BN|VM|VF|VN|GM|GF|GN|LM|LF|LN|
          ANe|AF|AM|INe|IF|IM|SPr\d+[A-Za-z]+|SPs\d+[A-Za-z]+|
          PN[A-Za-z]+|P[A-Za-z]+)
        |
        P[A-Za-z0-9]+
        |
        C[A-Za-z0-9]+
        |
        D[A-Za-z0-9]+
        |
        F[A-Za-z0-9]+
        |
        U
    )
    """,
    re.VERBOSE
)

# More general fallback:
GENERAL_TAG_RE = re.compile(
    r"(?:S|P|C|D|F|U)[A-Za-z0-9]+"
)


def clean_text(text):
    """Normalize whitespace and remove obvious punctuation."""

    if not text:
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    return " ".join(text.split())


def is_tag(text):
    """Check whether a string looks like a morphology tag."""

    if not text:
        return False

    if text == "U":
        return True

    return bool(
        GENERAL_TAG_RE.fullmatch(text)
    )


def recover_from_token(token):
    """
    Try to recover one or more morphology analyses
    from a possibly concatenated token.

    Normal:
        ramaḥ_rāma_SNM

    Possible concatenated:
        SVMtvam_tvad_SN
    """

    results = []

    token = token.strip()

    if not token:
        return results

    # --------------------------------------------------------
    # Case 1: clean token
    # --------------------------------------------------------

    parts = token.split("_")

    if len(parts) >= 3:

        surface = parts[0]
        lemma = parts[1]
        tag = parts[2]

        if surface and lemma and is_tag(tag):

            results.append({
                "surface": surface,
                "lemma": lemma,
                "tag": tag,
                "recovered": False
            })

            # Handle remaining pieces if they form another token
            remaining = "_".join(parts[3:])

            if remaining:
                results.extend(
                    recover_from_token(remaining)
                )

            return results

    # --------------------------------------------------------
    # Case 2: Find tag followed by another Sanskrit token
    #
    # Example:
    #
    # SVMtvam_tvad_SN
    #
    # --------------------------------------------------------

    match = GENERAL_TAG_RE.search(token)

    if match:

        tag_start = match.start()
        tag_end = match.end()

        # Prefix before the tag
        prefix = token[:tag_start]

        # Rest after the tag
        suffix = token[tag_end:]

        # We need at least some content before tag
        if prefix:

            # Prefix may contain surface/lemma
            prefix_parts = prefix.split("_")

            if len(prefix_parts) >= 2:

                surface = prefix_parts[0]
                lemma = prefix_parts[1]

                if surface and lemma:

                    results.append({
                        "surface": surface,
                        "lemma": lemma,
                        "tag": match.group(),
                        "recovered": True
                    })

                    if suffix:

                        results.extend(
                            recover_from_token(
                                suffix
                            )
                        )

                    return results

    return results


def parse_morphology(text):

    text = clean_text(text)

    if not text:
        return [], []

    # Remove slash punctuation
    text = re.sub(
        r"[\/]+",
        " ",
        text
    )

    raw_tokens = text.split()

    parsed = []
    malformed = []

    for token in raw_tokens:

        recovered = recover_from_token(token)

        if recovered:

            parsed.extend(recovered)

        else:

            malformed.append(token)

    return parsed, malformed


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("IMPROVED SANSKRIT MORPHOLOGY PARSER")
print("=" * 70)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    records = json.load(f)

print(
    f"Input records: {len(records):,}"
)

# ============================================================
# PROCESS
# ============================================================

output_records = []

tag_counter = Counter()

total_tokens = 0
parsed_tokens = 0
recovered_tokens = 0
malformed_tokens = 0

for record in records:

    morphology = record.get(
        "morphology",
        ""
    )

    tokens, malformed = parse_morphology(
        morphology
    )

    for item in tokens:

        tag_counter[
            item["tag"]
        ] += 1

        if item["recovered"]:
            recovered_tokens += 1

    total_tokens += (
        len(tokens)
        + len(malformed)
    )

    parsed_tokens += len(tokens)
    malformed_tokens += len(malformed)

    output_records.append({

        "index": record.get("index"),

        "sanskrit": record.get(
            "sanskrit",
            ""
        ),

        "morphology": morphology,

        "tokens": tokens,

        "malformed": malformed
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
        output_records,
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

print(
    f"Records              : "
    f"{len(output_records):,}"
)

print(
    f"Total tokens         : "
    f"{total_tokens:,}"
)

print(
    f"Parsed tokens        : "
    f"{parsed_tokens:,}"
)

print(
    f"Recovered tokens     : "
    f"{recovered_tokens:,}"
)

print(
    f"Malformed tokens     : "
    f"{malformed_tokens:,}"
)

if total_tokens:

    print(
        f"Parse success        : "
        f"{parsed_tokens / total_tokens * 100:.2f}%"
    )

print("\nTop morphology tags:")

for tag, count in tag_counter.most_common(30):

    print(
        f"{tag:12s} {count:,}"
    )

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 70)