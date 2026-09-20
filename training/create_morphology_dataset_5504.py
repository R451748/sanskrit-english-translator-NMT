import json
import numpy as np
from pathlib import Path


# ============================================================
# FILES
# ============================================================

MORPHOLOGY_FILE = "morphology/sentence_morphology_5504.json"

SOURCE_FILE = "dataset/processed/sequences/train_source.npy"
TARGET_FILE = "dataset/processed/sequences/train_target.npy"
OUTPUT_FILE = "dataset/processed/morphology_train_5504.json"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("CREATING 5,504-EXAMPLE MORPHOLOGY NMT DATASET")
print("=" * 70)

with open(MORPHOLOGY_FILE, "r", encoding="utf-8") as f:
    morphology_data = json.load(f)

source = np.load(SOURCE_FILE)
target = np.load(TARGET_FILE)

print(f"\nNMT source shape: {source.shape}")
print(f"NMT target shape: {target.shape}")
print(f"Morphology records: {len(morphology_data)}")


# ============================================================
# CREATE INDEX LOOKUP
# ============================================================

morphology_by_index = {}

for record in morphology_data:
    idx = int(record["index"])
    morphology_by_index[idx] = record


# ============================================================
# BUILD DATASET
# ============================================================

dataset = []

missing = 0

for idx in sorted(morphology_by_index.keys()):

    record = morphology_by_index[idx]

    if idx >= len(source):
        print(f"WARNING: index {idx} outside NMT dataset")
        missing += 1
        continue

    dataset.append({
        "index": idx,
        "sanskrit": record["sanskrit"],
        "source_tokens": source[idx].tolist(),
        "target_tokens": target[idx].tolist(),
        "morphology_tags": record["morphology_tags"],
        "morphology_tag_ids": record["morphology_tag_ids"]
    })


# ============================================================
# SAVE
# ============================================================

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET CREATED")
print("=" * 70)

print(f"Records created: {len(dataset)}")
print(f"Missing records: {missing}")

if dataset:
    print(f"First index: {dataset[0]['index']}")
    print(f"Last index: {dataset[-1]['index']}")

    print("\nFirst record:")
    print("Index:", dataset[0]["index"])
    print("Sanskrit:", dataset[0]["sanskrit"])
    print("Morphology:", dataset[0]["morphology_tags"])
    print("Morphology IDs:", dataset[0]["morphology_tag_ids"])
    print("Source length:", len(dataset[0]["source_tokens"]))
    print("Target length:", len(dataset[0]["target_tokens"]))

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)