import json
import os
import re
from collections import Counter


INPUT_FILE = "morphology/morphology_analysis_5504.json"

OUTPUT_FILE = "morphology/sentence_morphology_5504.json"

VOCAB_FILE = "morphology/morphology_tag_vocab.json"


# ============================================================
# LOAD DATA
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


records = data["records"]

print("=" * 70)
print("CREATING SENTENCE-LEVEL MORPHOLOGY FEATURES")
print("=" * 70)

print(
    "Input records:",
    len(records)
)


# ============================================================
# EXTRACT MORPHOLOGY TAGS
# ============================================================

processed = []

tag_counter = Counter()

total_tokens = 0
tokens_with_tags = 0
tokens_without_tags = 0


for record in records:

    index = int(record["index"])

    sanskrit = str(
        record["sanskrit"]
    ).strip()

    morphology = str(
        record["morphology"]
    ).strip()

    morphology_tokens = morphology.split()

    tags = []
    surfaces = []
    lemmas = []

    for token in morphology_tokens:

        total_tokens += 1

        # ----------------------------------------------------
        # Find underscore-separated morphology structure
        #
        # surface_lemma_tag
        # ----------------------------------------------------

        parts = token.split("_")

        if len(parts) >= 3:

            surface = parts[0]

            lemma = parts[1]

            tag = "_".join(
                parts[2:]
            )

            # Clean whitespace
            surface = surface.strip()
            lemma = lemma.strip()
            tag = tag.strip()

            surfaces.append(surface)
            lemmas.append(lemma)

            # Empty tag such as:
            #
            # ca_ca_
            #
            # is ignored.
            if tag:

                tags.append(tag)

                tag_counter[tag] += 1

                tokens_with_tags += 1

            else:

                tokens_without_tags += 1

        else:

            # Examples:
            #
            # /
            # //
            #
            # or malformed fragments.
            tokens_without_tags += 1


    # --------------------------------------------------------
    # Keep sentence if it contains at least one tag
    # --------------------------------------------------------

    if len(tags) == 0:

        continue


    processed.append(
        {
            "index": index,

            "sanskrit": sanskrit,

            "morphology_tags": tags,

            "morphology_tag_count": len(tags),

            "surfaces": surfaces,

            "lemmas": lemmas
        }
    )


# ============================================================
# BUILD VOCABULARY
# ============================================================

# PAD = 0
# UNK = 1

tag_vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}


# Most frequent tags first
for tag, count in tag_counter.most_common():

    if tag not in tag_vocab:

        tag_vocab[tag] = len(tag_vocab)


# ============================================================
# ADD INTEGER TAG IDS
# ============================================================

for record in processed:

    tag_ids = []

    for tag in record["morphology_tags"]:

        tag_ids.append(
            tag_vocab.get(
                tag,
                tag_vocab["<UNK>"]
            )
        )

    record["morphology_tag_ids"] = tag_ids


# ============================================================
# STATISTICS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    "Input records:",
    len(records)
)

print(
    "Usable sentences:",
    len(processed)
)

print(
    "Total morphology tokens:",
    total_tokens
)

print(
    "Tokens with tags:",
    tokens_with_tags
)

print(
    "Tokens without tags:",
    tokens_without_tags
)

print(
    "Unique morphology tags:",
    len(tag_vocab)
)


if processed:

    tag_counts = [
        len(record["morphology_tags"])
        for record in processed
    ]

    print(
        "Average tags/sentence:",
        round(
            sum(tag_counts) / len(tag_counts),
            2
        )
    )

    print(
        "Maximum tags/sentence:",
        max(tag_counts)
    )

    print(
        "Minimum tags/sentence:",
        min(tag_counts)
    )


# ============================================================
# TOP TAGS
# ============================================================

print()
print("=" * 70)
print("TOP MORPHOLOGY TAGS")
print("=" * 70)

for tag, count in tag_counter.most_common(30):

    print(
        f"{tag:15s} {count:6d}"
    )


# ============================================================
# SAVE FEATURES
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        processed,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE VOCABULARY
# ============================================================

with open(
    VOCAB_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        tag_vocab,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "Features:",
    OUTPUT_FILE
)

print(
    "Vocabulary:",
    VOCAB_FILE
)

print("=" * 70)