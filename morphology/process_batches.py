import os
import json
import time
import math
import gc

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "chronbmm/sanskrit5-multitask"

INPUT_FILE = "dataset/final/train.csv"
OUTPUT_DIR = "morphology/batches"

# IMPORTANT:
# Keep 64 because your existing batches use 64 sentences.
OUTPUT_BATCH_SIZE = 64

# Actual GPU inference batch.
# If CUDA OOM occurs, change this to 64.
INFERENCE_BATCH_SIZE = 96

MAX_INPUT_LENGTH = 384

# 128 is safer than 96 for long Sanskrit sentences.
MAX_NEW_TOKENS = 128

TASK_PREFIX = "SLM"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GPU SETUP
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("CUDA available")
    print("GPU:", torch.cuda.get_device_name(0))

    torch.set_float32_matmul_precision("high")

else:

    DEVICE = torch.device("cpu")

    print("WARNING: CUDA is not available")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

total_sentences = len(df)

print()
print("=" * 70)
print("SANSKRIT MORPHOLOGY EXTRACTION")
print("=" * 70)

print("Input file           :", INPUT_FILE)
print("Total sentences      :", total_sentences)
print("Output batch size    :", OUTPUT_BATCH_SIZE)
print("Inference batch size :", INFERENCE_BATCH_SIZE)
print("Max input length     :", MAX_INPUT_LENGTH)
print("Max new tokens       :", MAX_NEW_TOKENS)
print("=" * 70)
print()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

if DEVICE.type == "cuda":

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    )

else:

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )


model.to(DEVICE)
model.eval()

if hasattr(model.config, "use_cache"):
    model.config.use_cache = True

print("Model loaded successfully.")
print()


# ============================================================
# FIND COMPLETED BATCHES
# ============================================================

total_batches = math.ceil(
    total_sentences / OUTPUT_BATCH_SIZE
)

completed_batches = set()

for filename in os.listdir(OUTPUT_DIR):

    if not filename.startswith("batch_"):
        continue

    if not filename.endswith(".json"):
        continue

    try:

        batch_number = int(
            filename.replace("batch_", "")
                    .replace(".json", "")
        )

        completed_batches.add(batch_number)

    except ValueError:

        continue


print("Total batches     :", total_batches)
print("Completed batches :", len(completed_batches))

if completed_batches:

    print(
        "Highest completed :",
        max(completed_batches)
    )

print()


# ============================================================
# FIND MISSING BATCHES
# ============================================================

missing_batches = [
    batch_number
    for batch_number in range(1, total_batches + 1)
    if batch_number not in completed_batches
]

print("Remaining batches :", len(missing_batches))
print()


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def process_inference_batch(texts):

    prompts = [
        TASK_PREFIX + " " + str(text)
        for text in texts
    ]

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.inference_mode():

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
            use_cache=True
        )

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return decoded, outputs


# ============================================================
# MAIN PROCESSING
# ============================================================

start_all = time.time()

processed_this_run = 0
possible_truncated = 0

i = 0

while i < len(missing_batches):

    # --------------------------------------------------------
    # Collect consecutive logical batches
    # --------------------------------------------------------

    batch_ranges = []
    sentences_to_process = []

    while (
        i < len(missing_batches)
        and len(sentences_to_process) < INFERENCE_BATCH_SIZE
    ):

        batch_number = missing_batches[i]

        start_index = (
            (batch_number - 1)
            * OUTPUT_BATCH_SIZE
        )

        end_index = min(
            start_index + OUTPUT_BATCH_SIZE,
            total_sentences
        )

        batch_df = df.iloc[
            start_index:end_index
        ]

        batch_texts = (
            batch_df["sanskrit"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # Don't exceed GPU inference batch size.
        if (
            len(sentences_to_process) > 0
            and
            len(sentences_to_process)
            + len(batch_texts)
            > INFERENCE_BATCH_SIZE
        ):

            break

        sentences_to_process.extend(
            batch_texts
        )

        batch_ranges.append(
            (
                batch_number,
                start_index,
                end_index,
                len(batch_texts)
            )
        )

        i += 1


    # --------------------------------------------------------
    # GPU inference
    # --------------------------------------------------------

    inference_start = time.time()

    try:

        decoded, raw_outputs = process_inference_batch(
            sentences_to_process
        )

    except RuntimeError as e:

        if "out of memory" in str(e).lower():

            print()
            print("=" * 70)
            print("CUDA OUT OF MEMORY")
            print("=" * 70)
            print(
                "Current inference batch:",
                INFERENCE_BATCH_SIZE
            )
            print()
            print(
                "Change:"
            )
            print(
                "INFERENCE_BATCH_SIZE = 96"
            )
            print(
                "to:"
            )
            print(
                "INFERENCE_BATCH_SIZE = 64"
            )
            print("=" * 70)

            torch.cuda.empty_cache()
            gc.collect()

            raise

        raise


    inference_time = (
        time.time() - inference_start
    )

    inference_speed = (
        len(sentences_to_process)
        / inference_time
    )


    # --------------------------------------------------------
    # Check possible truncation
    # --------------------------------------------------------

    truncated_here = 0

    for output in raw_outputs:

        if output.shape[0] >= MAX_NEW_TOKENS:

            truncated_here += 1

    possible_truncated += truncated_here


    # --------------------------------------------------------
    # Save individual logical batches
    # --------------------------------------------------------

    offset = 0

    for (
        batch_number,
        start_index,
        end_index,
        batch_count
    ) in batch_ranges:

        batch_outputs = decoded[
            offset:
            offset + batch_count
        ]

        records = []

        for j in range(batch_count):

            row_index = start_index + j

            records.append(
                {
                    "index": int(row_index),

                    "sanskrit": str(
                        df.iloc[row_index]["sanskrit"]
                    ),

                    "morphology": batch_outputs[j]
                }
            )

        output_file = os.path.join(
            OUTPUT_DIR,
            f"batch_{batch_number:05d}.json"
        )

        temporary_file = (
            output_file + ".tmp"
        )

        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                records,
                f,
                ensure_ascii=False,
                indent=2
            )

        os.replace(
            temporary_file,
            output_file
        )

        offset += batch_count


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    processed_this_run += len(
        sentences_to_process
    )

    existing_sentences = (
        len(completed_batches)
        * OUTPUT_BATCH_SIZE
    )

    current_processed = min(
        total_sentences,
        existing_sentences
        + processed_this_run
    )

    progress = (
        current_processed
        / total_sentences
    ) * 100


    total_elapsed = (
        time.time() - start_all
    )

    if processed_this_run > 0:

        overall_speed = (
            processed_this_run
            / total_elapsed
        )

    else:

        overall_speed = 0


    remaining_sentences = (
        total_sentences
        - current_processed
    )

    if overall_speed > 0:

        eta_seconds = (
            remaining_sentences
            / overall_speed
        )

    else:

        eta_seconds = 0


    # --------------------------------------------------------
    # Print status
    # --------------------------------------------------------

    print()

    print(
        "Batches          : "
        f"{batch_ranges[0][0]}-"
        f"{batch_ranges[-1][0]}"
    )

    print(
        "Progress         : "
        f"{progress:.2f}%"
    )

    print(
        "Sentences        : "
        f"{current_processed:,}/"
        f"{total_sentences:,}"
    )

    print(
        "Inference speed  : "
        f"{inference_speed:.2f} sentences/sec"
    )

    print(
        "Run speed        : "
        f"{overall_speed:.2f} sentences/sec"
    )

    print(
        "Possible trunc.  : "
        f"{possible_truncated:,}"
    )

    print(
        "Estimated left   : "
        f"{eta_seconds / 3600:.1f} hours"
    )


    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if DEVICE.type == "cuda":

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )

        print(
            "GPU memory       : "
            f"{allocated:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved"
        )


    print(
        "Saved            : "
        f"batch_{batch_ranges[0][0]:05d}.json"
        " ... "
        f"batch_{batch_ranges[-1][0]:05d}.json"
    )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del decoded
    del raw_outputs

    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("MORPHOLOGY EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Total sentences        :",
    total_sentences
)

print(
    "Total batches           :",
    total_batches
)

print(
    "Possible truncated      :",
    possible_truncated
)

print(
    "Output directory        :",
    OUTPUT_DIR
)

print("=" * 70)