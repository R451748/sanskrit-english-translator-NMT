import json

FILE = "morphology/sentence_morphology_features_2720.json"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

tag_to_id = data["tag_to_id"]
records = data["records"]

print("=" * 70)
print("SENTENCE MORPHOLOGY FEATURE INSPECTION")
print("=" * 70)

print("Tag vocabulary:", len(tag_to_id))
print("Records:", len(records))

print("\nFirst 5 records:")
print("=" * 70)

for i, record in enumerate(records[:5], start=1):

    print(f"\nRecord {i}")

    print("Sanskrit:")
    print(record["sanskrit"])

    print("\nSurfaces:")
    print(record["surfaces"])

    print("\nLemmas:")
    print(record["lemmas"])

    print("\nMorphology IDs:")
    print(record["morphology_tag_ids"])

    print("\nNumber of morphology features:")
    print(len(record["morphology_tag_ids"]))

    print("-" * 70)

print("\nSample tag vocabulary:")

for tag, idx in list(tag_to_id.items())[:30]:
    print(f"{idx:4d} -> {tag}")

print("=" * 70)