import json
from collections import Counter

INPUT_FILE = "morphology/sentence_morphology_5504.json"


print("=" * 70)
print("SENTENCE-LEVEL MORPHOLOGY INSPECTION")
print("=" * 70)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nTotal records: {len(data)}")

# --------------------------------------------------
# Basic statistics
# --------------------------------------------------

tag_counts = Counter()
tag_count_per_sentence = []

for record in data:
    tags = record.get("morphology_tags", [])

    tag_counts.update(tags)
    tag_count_per_sentence.append(len(tags))

print(f"Total morphology tags: {sum(tag_counts.values())}")
print(f"Unique morphology tags: {len(tag_counts)}")

print(
    f"Average tags per sentence: "
    f"{sum(tag_count_per_sentence) / len(tag_count_per_sentence):.2f}"
)

print(f"Minimum tags in sentence: {min(tag_count_per_sentence)}")
print(f"Maximum tags in sentence: {max(tag_count_per_sentence)}")

# --------------------------------------------------
# Index range
# --------------------------------------------------

indices = [record["index"] for record in data]

print("\nIndex information:")
print(f"Minimum index: {min(indices)}")
print(f"Maximum index: {max(indices)}")
print(f"Unique indices: {len(set(indices))}")

# --------------------------------------------------
# Most frequent tags
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 MORPHOLOGY TAGS")
print("=" * 70)

for tag, count in tag_counts.most_common(30):
    print(f"{tag:15} {count}")

# --------------------------------------------------
# Examples
# --------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE RECORDS")
print("=" * 70)

for i, record in enumerate(data[:10], start=1):
    print(f"\nExample {i}")
    print("-" * 50)

    print("Index:")
    print(record.get("index"))

    print("Sanskrit:")
    print(record.get("sanskrit"))

    print("Surfaces:")
    print(record.get("surfaces"))

    print("Lemmas:")
    print(record.get("lemmas"))

    print("Morphology tags:")
    print(record.get("morphology_tags"))

    print("Tag IDs:")
    print(record.get("morphology_tag_ids"))

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)