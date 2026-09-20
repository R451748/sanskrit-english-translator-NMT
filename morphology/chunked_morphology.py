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
    / "test_morphology_chunked_20.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "chronbmm/sanskrit5-multitask"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SENTENCES = 20

# Number of words per morphology chunk
CHUNK_SIZE = 8

# Overlap between consecutive chunks
OVERLAP = 2

MAX_NEW_TOKENS = 128
NUM_BEAMS = 2


# ============================================================
# CHUNK FUNCTION
# ============================================================

def create_chunks(sentence, chunk_size=8, overlap=2):
    """
    Split a Sanskrit sentence into overlapping word chunks.
    """

    words = sentence.split()

    if len(words) <= chunk_size:
        return [words]

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk = words[start:start + chunk_size]

        if not chunk:
            break

        chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks


# ============================================================
# MORPHOLOGY FUNCTION
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
# START
# ============================================================

print("=" * 70)
print("CHUNKED MORPHOLOGY TEST")
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

df = pd.read_csv(TEST_FILE)

test_df = df.head(NUM_SENTENCES).copy()

print(
    "\nProcessing:",
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

    sentence = str(
        row["sanskrit"]
    ).strip()

    print("\n" + "-" * 70)

    print(
        f"[{count}/{len(test_df)}]"
    )

    print("Original:")
    print(sentence)

    chunks = create_chunks(
        sentence,
        CHUNK_SIZE,
        OVERLAP
    )

    print(
        "\nNumber of chunks:",
        len(chunks)
    )

    chunk_results = []

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_text = " ".join(chunk)

        print(
            f"\nChunk {chunk_number}:"
        )

        print(chunk_text)

        morphology = analyze_chunk(
            chunk,
            tokenizer,
            model
        )

        print("Morphology:")
        print(morphology)

        chunk_results.append(
            {
                "chunk_number": chunk_number,
                "words": chunk,
                "morphology": morphology
            }
        )

    results.append(
        {
            "test_row": int(index),
            "sanskrit": sentence,
            "chunks": chunk_results
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


print("\n" + "=" * 70)
print("CHUNKED MORPHOLOGY TEST COMPLETED")
print("=" * 70)

print(
    "Records:",
    len(results)
)

print(
    "Output:",
    OUTPUT_FILE
)