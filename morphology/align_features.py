import json
import os
import pandas as pd
import sentencepiece as spm

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_MORPHOLOGY = "morphology/morphology_features_2720.json"

INPUT_DATASET = "dataset/final/train.csv"

TOKENIZER_MODEL = (
    "dataset/processed/tokenizers/sanskrit.model"
)

OUTPUT_FILE = (
    "morphology/aligned_morphology_2720.json"
)

PAD_ID = 0
UNK_ID = 1

# ============================================================
# LOAD MORPHOLOGY FEATURES
# ============================================================

print("=" * 70)
print("MORPHOLOGY ↔ SENTENCEPIECE ALIGNMENT")
print("=" * 70)

with open(
    INPUT_MORPHOLOGY,
    "r",
    encoding="utf-8"
) as f:
    morphology_data = json.load(f)

print(
    f"Morphology records: "
    f"{len(morphology_data):,}"
)

# ============================================================
# LOAD SENTENCEPIECE
# ============================================================

print("\nLoading SentencePiece tokenizer...")

sp = spm.SentencePieceProcessor()

if not sp.load(TOKENIZER_MODEL):
    raise RuntimeError(
        f"Could not load tokenizer:\n{TOKENIZER_MODEL}"
    )

print(
    f"Vocabulary size: {sp.get_piece_size():,}"
)

# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_DATASET)

df["sanskrit"] = (
    df["sanskrit"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Create sentence → row mapping.
# This allows us to retrieve the exact Sanskrit sentence.
sentence_map = {}

for _, row in df.iterrows():

    sentence = row["sanskrit"]

    if sentence:
        sentence_map[sentence] = row.to_dict()

print(
    f"Training sentences available: "
    f"{len(sentence_map):,}"
)

# ============================================================
# ALIGNMENT FUNCTION
# ============================================================

def align_sentence(sentence, morphology_tags):
    """
    Align word-level morphology tags with
    SentencePiece subword tokens.

    Strategy:
        1. Split Sanskrit into whitespace words.
        2. Tokenize each word independently.
        3. Assign that word's morphology tag
           to every SentencePiece sub-token.

    Example:

        rāmaḥ → [▁rāma, ḥ]
        tag   → [SNM, SNM]
    """

    words = sentence.split()

    # --------------------------------------------------------
    # Build morphology tags by position
    # --------------------------------------------------------

    # morphology_tags contains only successfully parsed
    # morphology tokens, so it may be shorter than the
    # original Sanskrit word list.

    aligned_tokens = []
    aligned_ids = []

    morph_index = 0

    for word in words:

        # Remove obvious punctuation around the word
        clean_word = word.strip()

        if not clean_word:
            continue

        # ----------------------------------------------------
        # Tokenize the individual word
        # ----------------------------------------------------

        pieces = sp.encode_as_pieces(
            clean_word
        )

        if not pieces:
            continue

        # ----------------------------------------------------
        # Find morphology tag
        # ----------------------------------------------------

        if morph_index < len(morphology_tags):

            tag_id = morphology_tags[
                morph_index
            ]

        else:

            tag_id = UNK_ID

        # ----------------------------------------------------
        # Assign same morphology ID to
        # every subword of the word
        # ----------------------------------------------------

        for piece in pieces:

            aligned_tokens.append(piece)
            aligned_ids.append(tag_id)

        morph_index += 1

    return aligned_tokens, aligned_ids


# ============================================================
# PROCESS
# ============================================================

output = []

total_tokens = 0
unknown_tags = 0
empty_alignments = 0

for count, record in enumerate(
    morphology_data,
    start=1
):

    sentence = record.get(
        "sanskrit",
        ""
    )

    morphology_ids = record.get(
        "morphology_ids",
        []
    )

    morphology_tags = record.get(
        "morphology_tags",
        []
    )

    if not sentence:
        continue

    # --------------------------------------------------------
    # Align
    # --------------------------------------------------------

    tokens, aligned_ids = align_sentence(
        sentence,
        morphology_ids
    )

    if not tokens:
        empty_alignments += 1
        continue

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    output.append({

        "index": record.get(
            "index"
        ),

        "sanskrit": sentence,

        "source_tokens": tokens,

        "morphology_ids": aligned_ids,

        "morphology_tags": [
            str(x)
            for x in aligned_ids
        ]
    })

    total_tokens += len(tokens)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if count % 500 == 0:

        print(
            f"Processed: "
            f"{count:,}/"
            f"{len(morphology_data):,}"
        )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALIGNMENT COMPLETE")
print("=" * 70)

print(
    f"Input records       : "
    f"{len(morphology_data):,}"
)

print(
    f"Aligned records     : "
    f"{len(output):,}"
)

print(
    f"Source tokens       : "
    f"{total_tokens:,}"
)

print(
    f"Empty alignments    : "
    f"{empty_alignments:,}"
)

print(
    f"\nSaved to:"
)

print(OUTPUT_FILE)

print("=" * 70)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\nFirst 5 aligned examples:")
print("=" * 70)

for record in output[:5]:

    print("\nSanskrit:")
    print(record["sanskrit"])

    print("\nSentencePiece:")
    print(record["source_tokens"])

    print("\nMorphology IDs:")
    print(record["morphology_ids"])

    print("\n" + "-" * 70)