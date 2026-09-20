import json

FILE = "morphology/morphology_2720.json"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("MORPHOLOGY DATASET INSPECTION")
print("=" * 70)

print("Total records:", len(data))

print("\nFirst 10 records:\n")

for i, record in enumerate(data[:10], start=1):

    print("-" * 70)

    print(f"Record {i}")
    print("Index      :", record.get("index"))
    print("Sanskrit   :", record.get("sanskrit"))
    print("Morphology :", record.get("morphology"))

print("\n" + "=" * 70)