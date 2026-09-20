import json
import os
from collections import Counter

INPUT_FILE = "morphology/parsed_morphology_2720.json"
OUTPUT_FILE = "morphology/sentence_morphology_features_2720.json"


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


print("=" * 70)
print("CREATING SENTENCE-LEVEL MORPHOLOGY FEATURES")
print("=" * 70)

print("Input records:", len(data))


# ============================================================
# GLOBAL TAG VOCABULARY
# ============================================================

tag_counter = Counter()

for record in data:

    for token in record.get("tokens", []):

        tag = token.get("tag")

        if tag:
            tag_counter[tag] += 1


tag_to_id = {
    "<PAD>": 0,
    "<UNK>": 1
}

for tag in sorted(tag_counter):

    if tag not in tag_to_id:
        tag_to_id[tag] = len(tag_to_id)


# ============================================================
# CREATE SENTENCE FEATURES
# ============================================================

output = []

for record in data:

    tokens = record.get("tokens", [])

    # --------------------------------------------------------
    # Morphology tag sequence
    # --------------------------------------------------------

    tag_ids = []

    # --------------------------------------------------------
    # Lemma sequence
    # --------------------------------------------------------

    lemmas = []

    # --------------------------------------------------------
    # Surface sequence
    # --------------------------------------------------------

    surfaces = []

    for token in tokens:

        surface = token.get(
            "surface",
            ""
        )

        lemma = token.get(
            "lemma",
            ""
        )

        tag = token.get(
            "tag",
            ""
        )

        if not tag:
            continue

        surfaces.append(surface)

        lemmas.append(lemma)

        tag_ids.append(
            tag_to_id.get(
                tag,
                tag_to_id["<UNK>"]
            )
        )

    # --------------------------------------------------------
    # Frequency representation
    # --------------------------------------------------------

    tag_frequency = Counter(tag_ids)

    frequency_vector = [
        0
    ] * len(tag_to_id)

    for tag_id, count in tag_frequency.items():

        if tag_id < len(frequency_vector):

            frequency_vector[tag_id] = count

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.append({

        "index": record.get(
            "index"
        ),

        "sanskrit": record.get(
            "sanskrit",
            ""
        ),

        "surfaces": surfaces,

        "lemmas": lemmas,

        "morphology_tag_ids": tag_ids,

        "morphology_frequency": frequency_vector
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
        {
            "tag_to_id": tag_to_id,
            "records": output
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

total_features = sum(
    len(record["morphology_tag_ids"])
    for record in output
)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Sentences          : {len(output):,}"
)

print(
    f"Morphology tags    : "
    f"{total_features:,}"
)

print(
    f"Tag vocabulary     : "
    f"{len(tag_to_id):,}"
)

print(
    f"\nSaved to:"
)

print(OUTPUT_FILE)

print("=" * 70)