import json
from collections import Counter

INPUT_FILE = "morphology/parsed_morphology_2720.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

tag_counter = Counter()
sentence_tags = []

for record in data:

    tags = []

    for token in record.get("tokens", []):

        tag = token.get("tag")

        if tag:
            tag_counter[tag] += 1
            tags.append(tag)

    sentence_tags.append(tags)

print("=" * 70)
print("MORPHOLOGY TAG ANALYSIS")
print("=" * 70)

print("Sentences:", len(data))
print("Unique tags:", len(tag_counter))
print("Total valid tags:", sum(tag_counter.values()))

print("\nTop 50 morphology tags:")
print("-" * 50)

for tag, count in tag_counter.most_common(50):
    print(f"{tag:15s} {count:6d}")

print("\n" + "=" * 70)

# ------------------------------------------------------------
# Coverage
# ------------------------------------------------------------

sentences_with_tags = sum(
    1 for tags in sentence_tags if tags
)

print(
    f"Sentences containing morphology: "
    f"{sentences_with_tags}/{len(data)} "
    f"({sentences_with_tags / len(data) * 100:.2f}%)"
)

# ------------------------------------------------------------
# Tag frequency groups
# ------------------------------------------------------------

print("\nTag frequency distribution:")

freq_groups = {
    "1-5": 0,
    "6-10": 0,
    "11-50": 0,
    "51-100": 0,
    "101-500": 0,
    "500+": 0
}

for count in tag_counter.values():

    if count <= 5:
        freq_groups["1-5"] += 1
    elif count <= 10:
        freq_groups["6-10"] += 1
    elif count <= 50:
        freq_groups["11-50"] += 1
    elif count <= 100:
        freq_groups["51-100"] += 1
    elif count <= 500:
        freq_groups["101-500"] += 1
    else:
        freq_groups["500+"] += 1

for group, count in freq_groups.items():
    print(f"{group:10s}: {count}")

print("=" * 70)