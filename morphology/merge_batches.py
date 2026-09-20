import json
import glob
import os

INPUT_DIR = "morphology/batches"
OUTPUT_FILE = "morphology/morphology_2720.json"

files = sorted(glob.glob(
    os.path.join(INPUT_DIR, "batch_*.json")
))

all_records = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        records = json.load(f)

    all_records.extend(records)

print("=" * 60)
print("MORPHOLOGY BATCH MERGE")
print("=" * 60)

print(f"Batch files : {len(files)}")
print(f"Records     : {len(all_records)}")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        all_records,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nSaved to: {OUTPUT_FILE}")
print("=" * 60)