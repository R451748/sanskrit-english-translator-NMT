import json
import os
from collections import Counter

INPUT_FILE = "morphology/parsed_morphology_2720.json"

OUTPUT_FILE = "morphology/morphology_features_2720.json"
VOCAB_FILE = "morphology/morphology_tag_vocab.json"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ============================================================
# LOAD
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("CREATING MORPHOLOGY FEATURES")
print("=" * 70)

print("Input sentences:", len(data))


# ============================================================
# COLLECT TAGS
# ============================================================

counter = Counter()

for record in data:

    for token in record.get("tokens", []):

        tag = token.get("tag")

        if tag:
            counter[tag] += 1


# ============================================================
# CREATE VOCABULARY
# ============================================================

tag_to_id = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1
}

for tag in sorted(counter.keys()):

    if tag not in tag_to_id:

        tag_to_id[tag] = len(tag_to_id)


id_to_tag = {
    str(v): k
    for k, v in tag_to_id.items()
}


print("Unique tags:", len(counter))
print("Vocabulary size:", len(tag_to_id))


# ============================================================
# CONVERT SENTENCES
# ============================================================

feature_records = []

for record in data:

    morphology_ids = []

    for token in record.get("tokens", []):

        tag = token.get("tag")

        if tag:

            morphology_ids.append(
                tag_to_id.get(
                    tag,
                    tag_to_id[UNK_TOKEN]
                )
            )

    feature_records.append({

        "index": record.get("index"),

        "sanskrit": record.get(
            "sanskrit",
            ""
        ),

        "morphology_ids": morphology_ids,

        "morphology_tags": [
            id_to_tag[str(i)]
            for i in morphology_ids
        ]
    })


# ============================================================
# SAVE FEATURES
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_records,
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
        {
            "tag_to_id": tag_to_id,
            "id_to_tag": id_to_tag,
            "frequencies": dict(counter)
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

total_tokens = sum(
    len(x["morphology_ids"])
    for x in feature_records
)

print("\n" + "=" * 70)
print("FEATURE CREATION COMPLETE")
print("=" * 70)

print(
    f"Sentences          : {len(feature_records):,}"
)

print(
    f"Morphology tokens  : {total_tokens:,}"
)

print(
    f"Tag vocabulary     : {len(tag_to_id):,}"
)

print(
    f"\nFeatures saved to:"
)

print(OUTPUT_FILE)

print(
    f"\nVocabulary saved to:"
)

print(VOCAB_FILE)

print("=" * 70)