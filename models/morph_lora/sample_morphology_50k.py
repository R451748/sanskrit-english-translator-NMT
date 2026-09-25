import json
import random
from pathlib import Path

INPUT_FILE = Path("models/morph_lora/morphology_train_50k.json")
OUTPUT_FILE = Path("models/morph_lora/morphology_50k_qualitative_sample.json")

SEED = 42
random.seed(SEED)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

def word_count(text):
    return len(text.split())

groups = {
    "short_1_10": [],
    "medium_11_20": [],
    "long_21_30": [],
    "very_long_31_40": [],
    "very_long_41_plus": [],
}

for record in data:
    source = record.get("sanskrit", "").strip()
    count = word_count(source)

    if 1 <= count <= 10:
        groups["short_1_10"].append(record)
    elif 11 <= count <= 20:
        groups["medium_11_20"].append(record)
    elif 21 <= count <= 30:
        groups["long_21_30"].append(record)
    elif 31 <= count <= 40:
        groups["very_long_31_40"].append(record)
    elif count >= 41:
        groups["very_long_41_plus"].append(record)

sample = {}

for group_name, records in groups.items():
    sample[group_name] = random.sample(
        records,
        min(8, len(records))
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

print("=" * 70)
print("50K MORPHOLOGY QUALITATIVE SAMPLE")
print("=" * 70)

for group_name, records in sample.items():
    print(f"\n{group_name}: {len(records)} examples")

    for i, record in enumerate(records, 1):
        print(f"\n--- Example {i} ---")
        print("Sanskrit:")
        print(record.get("sanskrit", ""))

        print("\nMorphology:")
        print(record.get("morphology", ""))

        print("\nEnglish:")
        print(record.get("english", ""))

print("\n" + "=" * 70)
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)