import pandas as pd
import sentencepiece as spm
import numpy as np
import os

# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "dataset/final/train.csv"
VAL_FILE = "dataset/final/validation.csv"
TEST_FILE = "dataset/final/test.csv"

TOKENIZER_DIR = "dataset/processed/tokenizers"
OUTPUT_DIR = "dataset/processed/sequences"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SANSKRIT_MODEL = os.path.join(
    TOKENIZER_DIR,
    "sanskrit.model"
)

ENGLISH_MODEL = os.path.join(
    TOKENIZER_DIR,
    "english.model"
)

# Maximum sequence lengths
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 256


# ============================================================
# LOAD TOKENIZERS
# ============================================================

print("=" * 70)
print("LOADING SENTENCEPIECE TOKENIZERS")
print("=" * 70)

src_tokenizer = spm.SentencePieceProcessor(
    model_file=SANSKRIT_MODEL
)

tgt_tokenizer = spm.SentencePieceProcessor(
    model_file=ENGLISH_MODEL
)

print("Sanskrit vocabulary:",
      src_tokenizer.get_piece_size())

print("English vocabulary:",
      tgt_tokenizer.get_piece_size())


# ============================================================
# ENCODE SENTENCE
# ============================================================

def encode_source(text):

    ids = src_tokenizer.encode(
        str(text),
        out_type=int
    )

    ids = ids[:MAX_SOURCE_LENGTH - 1]

    ids.append(src_tokenizer.eos_id())

    ids += [
        src_tokenizer.pad_id()
    ] * (MAX_SOURCE_LENGTH - len(ids))

    return ids


def encode_target(text):

    ids = tgt_tokenizer.encode(
        str(text),
        out_type=int
    )

    ids = ids[:MAX_TARGET_LENGTH - 2]

    ids = (
        [tgt_tokenizer.bos_id()]
        + ids
        + [tgt_tokenizer.eos_id()]
    )

    ids += [
        tgt_tokenizer.pad_id()
    ] * (MAX_TARGET_LENGTH - len(ids))

    return ids


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(input_file, output_name):

    print("\n" + "-" * 70)
    print("Processing:", input_file)
    print("-" * 70)

    df = pd.read_csv(input_file)

    print("Rows:", len(df))

    source_sequences = []
    target_sequences = []

    for i, row in enumerate(
        zip(df["sanskrit"], df["english"])
    ):

        source_sequences.append(
            encode_source(row[0])
        )

        target_sequences.append(
            encode_target(row[1])
        )

        if (i + 1) % 50000 == 0:
            print(
                f"Processed: {i + 1:,}"
            )

    source_array = np.array(
        source_sequences,
        dtype=np.int32
    )

    target_array = np.array(
        target_sequences,
        dtype=np.int32
    )

    source_output = os.path.join(
        OUTPUT_DIR,
        f"{output_name}_source.npy"
    )

    target_output = os.path.join(
        OUTPUT_DIR,
        f"{output_name}_target.npy"
    )

    np.save(
        source_output,
        source_array
    )

    np.save(
        target_output,
        target_array
    )

    print("\nSaved:")
    print(source_output)
    print(target_output)

    print("Source shape:", source_array.shape)
    print("Target shape:", target_array.shape)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset(
        TRAIN_FILE,
        "train"
    )

    process_dataset(
        VAL_FILE,
        "validation"
    )

    process_dataset(
        TEST_FILE,
        "test"
    )

    print("\n" + "=" * 70)
    print("SEQUENCE PREPARATION COMPLETE")
    print("=" * 70)