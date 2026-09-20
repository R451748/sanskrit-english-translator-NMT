import json
import pandas as pd


TRAIN_FILE = "dataset/final/train.csv"
MORPH_FILE = "morphology/sentence_morphology_5504.json"


print("=" * 70)
print("MORPHOLOGY / TRAINING DATA ALIGNMENT CHECK")
print("=" * 70)


# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------

df = pd.read_csv(
    TRAIN_FILE
)

print(
    "Training rows:",
    len(df)
)


# ------------------------------------------------------------
# Load morphology
# ------------------------------------------------------------

with open(
    MORPH_FILE,
    "r",
    encoding="utf-8"
) as f:

    morphology = json.load(f)


print(
    "Morphology records:",
    len(morphology)
)


# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

matched = 0
mismatched = 0

examples = []


for record in morphology:

    index = int(
        record["index"]
    )

    morphology_text = str(
        record["sanskrit"]
    ).strip()

    dataset_text = str(
        df.iloc[index]["sanskrit"]
    ).strip()


    if morphology_text == dataset_text:

        matched += 1

    else:

        mismatched += 1

        if len(examples) < 10:

            examples.append(
                {
                    "index": index,
                    "morphology": morphology_text,
                    "dataset": dataset_text
                }
            )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Matched:",
    matched
)

print(
    "Mismatched:",
    mismatched
)

print(
    "Match percentage:",
    round(
        matched / len(morphology) * 100,
        2
    )
)


# ------------------------------------------------------------
# Show mismatches
# ------------------------------------------------------------

if examples:

    print()
    print("=" * 70)
    print("MISMATCH EXAMPLES")
    print("=" * 70)

    for item in examples:

        print()
        print(
            "Index:",
            item["index"]
        )

        print(
            "Morphology:",
            item["morphology"]
        )

        print(
            "Dataset:",
            item["dataset"]
        )


print()
print("=" * 70)
print("ALIGNMENT CHECK COMPLETE")
print("=" * 70)