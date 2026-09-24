import json
import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# CONFIG
# ============================================================

TEST_FILE = "dataset/final/test.csv"

OUTPUT_FILE = "models/morph_lora/test_morphology_5056.json"

MODEL_NAME = "chronbmm/sanskrit5-multitask"

BATCH_SIZE = 32
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("TEST-SET MORPHOLOGY GENERATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

df = pd.read_csv(TEST_FILE)

print(f"Test sentences: {len(df)}")
print(f"Columns: {list(df.columns)}")

if "sanskrit" not in df.columns:
    raise ValueError("Column 'sanskrit' not found in test.csv")

# ============================================================
# LOAD LOCAL MORPHOLOGY MODEL
# ============================================================

print("\nLoading local morphology model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    local_files_only=False
)

model.to(DEVICE)
model.eval()

print("Morphology model loaded successfully.")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_morphology(sentences):

    prompts = [
        "SLM: " + str(sentence)
        for sentence in sentences
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False
        )

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return decoded


# ============================================================
# PROCESS TEST SET
# ============================================================

results = []

total = len(df)

print("\nStarting morphology generation...")
print(f"Total sentences: {total}")
print(f"Batch size: {BATCH_SIZE}")
print()

for start in range(0, total, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total)

    batch_df = df.iloc[start:end]

    sentences = batch_df["sanskrit"].astype(str).tolist()

    try:

        morphology_outputs = generate_morphology(sentences)

    except RuntimeError as e:

        if "out of memory" in str(e).lower():

            print("\nCUDA OUT OF MEMORY.")
            print("Reduce BATCH_SIZE from 32 to 16 or 8.")
            raise

        raise

    for i, (idx, sanskrit, morphology) in enumerate(
        zip(
            batch_df.index.tolist(),
            sentences,
            morphology_outputs
        )
    ):

        results.append(
            {
                "test_index": int(idx),
                "sanskrit": sanskrit,
                "morphology": morphology
            }
        )

    processed = end

    percent = processed / total * 100

    print(
        f"[{processed:5d}/{total}] "
        f"{percent:6.2f}%"
    )

    # Save intermediate checkpoint every batch
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(f"Generated records: {len(results)}")

if len(results) != total:
    raise RuntimeError(
        f"Expected {total} morphology records "
        f"but generated {len(results)}"
    )

# Check indices
indices = [r["test_index"] for r in results]

if len(set(indices)) != total:
    raise RuntimeError(
        "Duplicate test indices detected."
    )

# Check Sanskrit alignment
for record in results:

    idx = record["test_index"]

    original = str(
        df.iloc[idx]["sanskrit"]
    )

    if record["sanskrit"] != original:

        raise RuntimeError(
            f"Sanskrit mismatch at test index {idx}"
        )

print("Index alignment: PASS")
print("Sanskrit alignment: PASS")

# ============================================================
# FINAL SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nOutput:")
print(OUTPUT_FILE)

print("\nFirst 3 examples:")

for record in results[:3]:

    print("\nSanskrit:")
    print(record["sanskrit"])

    print("\nMorphology:")
    print(record["morphology"])

print("\n" + "=" * 70)
print("TEST MORPHOLOGY GENERATION COMPLETE")
print("=" * 70)


