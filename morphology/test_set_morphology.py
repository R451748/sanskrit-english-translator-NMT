import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "final"
    / "test.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "morphology"
    / "test_morphology_20.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "chronbmm/sanskrit5-multitask"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIG
# ============================================================

NUM_SENTENCES = 20
MAX_NEW_TOKENS = 128
NUM_BEAMS = 2


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TEST-SET MORPHOLOGY TEST")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATA
# ============================================================

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}"
    )

df = pd.read_csv(TEST_FILE)

print("\nTest dataset:")
print("Total sentences:", len(df))

if "sanskrit" not in df.columns:
    raise ValueError(
        "Column 'sanskrit' not found in test.csv"
    )


# Take only first 20 for testing
test_df = df.head(NUM_SENTENCES).copy()

print(
    "Processing:",
    len(test_df),
    "sentences"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")

print("\nLoading morphology model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print("Model loaded.")


# ============================================================
# PROCESS
# ============================================================

results = []

print("\n" + "=" * 70)
print("PROCESSING")
print("=" * 70)

for count, (index, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    sentence = str(row["sanskrit"]).strip()

    print(
        f"\n[{count}/{len(test_df)}]"
    )

    print("Sanskrit:", sentence)

    prompt = "SLM " + sentence

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS
        )

    morphology = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("Morphology:", morphology)

    results.append(
        {
            "test_row": int(index),
            "sanskrit": sentence,
            "morphology": morphology
        }
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

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
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)

print(
    "Records saved:",
    len(results)
)

print(
    "Output:",
    OUTPUT_FILE
)