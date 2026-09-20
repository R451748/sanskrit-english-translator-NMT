import json
import math
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "morphology"
    / "test_chunks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "chronbmm/sanskrit5-multitask"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 20

CHUNK_SIZE = 8
OVERLAP = 2

MAX_NEW_TOKENS = 128
NUM_BEAMS = 2


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    sentence,
    chunk_size=8,
    overlap=2
):

    words = sentence.split()

    if len(words) <= chunk_size:
        return [words]

    chunks = []

    step = chunk_size - overlap

    for start in range(
        0,
        len(words),
        step
    ):

        chunk = words[
            start:start + chunk_size
        ]

        if not chunk:
            break

        chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks


# ============================================================
# MORPHOLOGY
# ============================================================

@torch.no_grad()
def analyze_chunk(
    chunk,
    tokenizer,
    model
):

    sentence = " ".join(chunk)

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

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result


# ============================================================
# PROCESS ONE SENTENCE
# ============================================================

def process_sentence(
    index,
    sentence,
    tokenizer,
    model
):

    chunks = create_chunks(
        sentence,
        CHUNK_SIZE,
        OVERLAP
    )

    chunk_results = []

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        morphology = analyze_chunk(
            chunk,
            tokenizer,
            model
        )

        chunk_results.append(
            {
                "chunk_number": chunk_number,
                "words": chunk,
                "morphology": morphology
            }
        )

    return {
        "test_row": int(index),
        "sanskrit": sentence,
        "chunks": chunk_results
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("RESUMABLE TEST-SET MORPHOLOGY PROCESSOR")
print("=" * 75)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


df = pd.read_csv(
    TEST_FILE
)

total_sentences = len(df)

print(
    "Total test sentences:",
    total_sentences
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
# BATCH INFORMATION
# ============================================================

num_batches = math.ceil(
    total_sentences / BATCH_SIZE
)

print(
    "\nBatch size:",
    BATCH_SIZE
)

print(
    "Total batches:",
    num_batches
)


# ============================================================
# PROCESS
# ============================================================

for batch_number in range(
    num_batches
):

    output_file = (
        OUTPUT_DIR
        / f"batch_{batch_number + 1:04d}.json"
    )

    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    if output_file.exists():

        print(
            f"\n[{batch_number + 1}/{num_batches}] "
            "Already completed — skipping."
        )

        continue

    start = (
        batch_number
        * BATCH_SIZE
    )

    end = min(
        start + BATCH_SIZE,
        total_sentences
    )

    batch_df = df.iloc[
        start:end
    ]

    print("\n" + "=" * 75)

    print(
        f"Batch {batch_number + 1}/{num_batches}"
    )

    print(
        f"Rows {start} - {end - 1}"
    )

    print("=" * 75)

    batch_results = []

    for local_number, (
        index,
        row
    ) in enumerate(
        batch_df.iterrows(),
        start=1
    ):

        sentence = str(
            row["sanskrit"]
        ).strip()

        print(
            f"[{local_number}/{len(batch_df)}] "
            f"Test row {index}"
        )

        result = process_sentence(
            index,
            sentence,
            tokenizer,
            model
        )

        batch_results.append(
            result
        )

    # --------------------------------------------------------
    # SAVE IMMEDIATELY
    # --------------------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            batch_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nSaved:",
        output_file
    )

    # --------------------------------------------------------
    # GPU MEMORY CLEANUP
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# FINAL
# ============================================================

completed_files = list(
    OUTPUT_DIR.glob(
        "batch_*.json"
    )
)

print("\n" + "=" * 75)

print(
    "PROCESSING FINISHED"
)

print(
    "Batch files available:",
    len(completed_files)
)

print(
    "Output directory:",
    OUTPUT_DIR
)

print("=" * 75)