import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("models/morph_lora/morphology_source_50k.csv")

OUTPUT_DIR = Path("models/morph_lora/morphology_50k_batches")
FINAL_OUTPUT = Path("models/morph_lora/morphology_train_50k.json")

MODEL_NAME = "chronbmm/sanskrit5-multitask"

BATCH_SIZE = 8
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 256

# Morphology task used in the previous experiment
TASK_PREFIX = "SLM"

SAVE_EVERY = 100


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("50K LOCAL SANSKRIT MORPHOLOGY GENERATION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"\nInput records: {len(df):,}")

if "sanskrit" not in df.columns:
    raise ValueError("Column 'sanskrit' not found.")

# Keep only required column
df = df[["sanskrit", "english"]].copy()

df = df.reset_index(drop=True)

print("Dataset loaded successfully.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading morphology model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.to(device)
model.eval()

print("Morphology model loaded.")


# ============================================================
# RESUME SUPPORT
# ============================================================

existing_files = sorted(
    OUTPUT_DIR.glob("batch_*.json")
)

processed_indices = set()

for file in existing_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            if "index" in record:
                processed_indices.add(record["index"])

    except Exception as e:
        print(f"Warning: could not read {file}: {e}")


print(
    f"\nAlready processed: "
    f"{len(processed_indices):,} / {len(df):,}"
)


# ============================================================
# GENERATION
# ============================================================

def generate_morphology(texts):

    prompts = [
        f"{TASK_PREFIX}: {text}"
        for text in texts
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    inputs = {
        key: value.to(device)
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
# PROCESS
# ============================================================

start_time = time.time()

pending_indices = [
    i for i in range(len(df))
    if i not in processed_indices
]

print(
    f"Remaining records: "
    f"{len(pending_indices):,}"
)

batch_number = (
    len(existing_files) + 1
)

for start in range(
    0,
    len(pending_indices),
    BATCH_SIZE
):

    batch_indices = pending_indices[
        start:start + BATCH_SIZE
    ]

    texts = [
        str(df.iloc[i]["sanskrit"])
        for i in batch_indices
    ]

    try:

        morphology_outputs = generate_morphology(
            texts
        )

        records = []

        for idx, sanskrit, morphology in zip(
            batch_indices,
            texts,
            morphology_outputs
        ):

            records.append({
                "index": int(idx),
                "sanskrit": sanskrit,
                "english": str(df.iloc[idx]["english"]),
                "morphology": morphology
            })

        batch_file = (
            OUTPUT_DIR /
            f"batch_{batch_number:05d}.json"
        )

        with open(
            batch_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                records,
                f,
                ensure_ascii=False,
                indent=2
            )

        processed_indices.update(
            batch_indices
        )

        if (
            start // BATCH_SIZE + 1
        ) % SAVE_EVERY == 0:

            elapsed = time.time() - start_time

            print(
                f"\nProcessed "
                f"{len(processed_indices):,}/"
                f"{len(df):,} "
                f"({len(processed_indices) / len(df) * 100:.2f}%)"
            )

            print(
                f"Elapsed: "
                f"{elapsed / 3600:.2f} hours"
            )

            if torch.cuda.is_available():

                allocated = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                reserved = (
                    torch.cuda.memory_reserved()
                    / 1024**3
                )

                print(
                    f"GPU memory: "
                    f"{allocated:.2f} GB allocated / "
                    f"{reserved:.2f} GB reserved"
                )

        batch_number += 1

    except RuntimeError as e:

        print(
            f"\nRuntime error at indices "
            f"{batch_indices}: {e}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise

    except Exception as e:

        print(
            f"\nError at indices "
            f"{batch_indices}: {e}"
        )

        raise


# ============================================================
# COMBINE BATCHES
# ============================================================

print("\nCombining morphology batches...")

all_records = []

for batch_file in sorted(
    OUTPUT_DIR.glob("batch_*.json")
):

    with open(
        batch_file,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    all_records.extend(records)


# Sort by original dataset index
all_records.sort(
    key=lambda x: x["index"]
)


# Remove accidental duplicate indices
unique_records = {}

for record in all_records:

    unique_records[record["index"]] = record


all_records = [
    unique_records[i]
    for i in sorted(unique_records)
]


# ============================================================
# VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print(
    f"Expected records: {len(df):,}"
)

print(
    f"Generated records: {len(all_records):,}"
)

expected_indices = set(range(len(df)))
actual_indices = {
    record["index"]
    for record in all_records
}

missing = expected_indices - actual_indices
extra = actual_indices - expected_indices

print(
    f"Missing indices: {len(missing):,}"
)

print(
    f"Unexpected indices: {len(extra):,}"
)


# Sanskrit alignment
alignment_errors = 0

for record in all_records:

    idx = record["index"]

    expected = str(
        df.iloc[idx]["sanskrit"]
    )

    if record["sanskrit"] != expected:
        alignment_errors += 1


print(
    f"Sanskrit alignment errors: "
    f"{alignment_errors:,}"
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

if (
    len(all_records) == len(df)
    and len(missing) == 0
    and len(extra) == 0
    and alignment_errors == 0
):

    with open(
        FINAL_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("50K MORPHOLOGY DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Records: {len(all_records):,}"
    )

    print(
        f"Saved to: {FINAL_OUTPUT}"
    )

else:

    print(
        "\nDataset is incomplete or alignment failed."
    )

    print(
        "Final JSON was NOT created."
    )


print("\nDONE")