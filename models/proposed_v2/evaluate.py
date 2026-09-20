import sys
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU, CHRF

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.proposed.dataset import create_dataset
from models.proposed_v2.model import (
    MorphologyConditionedTransformer
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "proposed_v2"
    / "checkpoints"
    / "best_model.pt"
)

BATCH_SIZE = 8

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000
MORPH_VOCAB_SIZE = 228

BOS_ID = 2
EOS_ID = 3
PAD_ID = 0

BEAM_SIZE = 4
MAX_LENGTH = 128

REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM = 3

LENGTH_PENALTY = 0.7


# ============================================================
# Load tokenizers
# ============================================================

import sentencepiece as spm

src_tokenizer = spm.SentencePieceProcessor()

src_tokenizer.load(
    str(
        PROJECT_ROOT
        / "dataset"
        / "processed"
        / "tokenizers"
        / "sanskrit.model"
    )
)

tgt_tokenizer = spm.SentencePieceProcessor()

tgt_tokenizer.load(
    str(
        PROJECT_ROOT
        / "dataset"
        / "processed"
        / "tokenizers"
        / "english.model"
    )
)


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("V2 MORPHOLOGY-CONDITIONED TRANSFORMER EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

dataset = create_dataset()

print("Dataset size:", len(dataset))


# ============================================================
# Validation split
# Same seed and split as training
# ============================================================

from torch.utils.data import random_split

train_size = int(
    0.90 * len(dataset)
)

val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Validation samples:", len(val_dataset))


# ============================================================
# Model
# ============================================================

model = MorphologyConditionedTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    morphology_vocab_size=MORPH_VOCAB_SIZE,

    d_model=256,
    nhead=4,

    num_encoder_layers=3,
    num_decoder_layers=3,

    dim_feedforward=1024,
    dropout=0.1,

    src_pad_idx=0,
    tgt_pad_idx=0,
    morph_pad_idx=0,

    max_src_len=128,
    max_morph_len=64,
    max_tgt_len=256
).to(DEVICE)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Checkpoint validation loss:",
    checkpoint["val_loss"]
)


# ============================================================
# No-repeat ngram
# ============================================================

def get_banned_tokens(
    sequence,
    n
):

    if len(sequence) < n - 1:
        return set()

    prefix = tuple(
        sequence[-(n - 1):]
    )

    banned = set()

    for i in range(
        len(sequence) - n + 1
    ):

        ngram = tuple(
            sequence[i:i + n]
        )

        if ngram[:-1] == prefix:
            banned.add(
                ngram[-1]
            )

    return banned


# ============================================================
# Repetition penalty
# ============================================================

def apply_repetition_penalty(
    logits,
    sequence,
    penalty
):

    if penalty == 1.0:
        return logits

    for token in set(sequence):

        if logits[token] < 0:
            logits[token] *= penalty
        else:
            logits[token] /= penalty

    return logits


# ============================================================
# Beam search
# ============================================================

@torch.no_grad()
def beam_search(
    src,
    morphology
):

    beams = [
        (
            [BOS_ID],
            0.0
        )
    ]

    for _ in range(MAX_LENGTH):

        candidates = []

        for sequence, score in beams:

            if sequence[-1] == EOS_ID:

                candidates.append(
                    (
                        sequence,
                        score
                    )
                )

                continue

            decoder_input = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=DEVICE
            )

            output = model(
                src,
                morphology,
                decoder_input
            )

            logits = output[
                0,
                -1
            ]

            logits = apply_repetition_penalty(
                logits,
                sequence,
                REPETITION_PENALTY
            )

            banned = get_banned_tokens(
                sequence,
                NO_REPEAT_NGRAM
            )

            if banned:

                for token in banned:
                    logits[token] = -float("inf")

            log_probs = F.log_softmax(
                logits,
                dim=-1
            )

            top_values, top_indices = torch.topk(
                log_probs,
                BEAM_SIZE
            )

            for value, token in zip(
                top_values.tolist(),
                top_indices.tolist()
            ):

                new_sequence = (
                    sequence
                    + [token]
                )

                new_score = (
                    score
                    + value
                )

                candidates.append(
                    (
                        new_sequence,
                        new_score
                    )
                )

        def normalized_score(item):

            sequence, score = item

            length = max(
                1,
                len(sequence) - 1
            )

            return (
                score
                / (length ** LENGTH_PENALTY)
            )

        candidates.sort(
            key=normalized_score,
            reverse=True
        )

        beams = candidates[
            :BEAM_SIZE
        ]

        if all(
            seq[-1] == EOS_ID
            for seq, _ in beams
        ):
            break

    return beams[0][0]


# ============================================================
# Decode
# ============================================================

def decode_target(ids):

    ids = [
        int(x)
        for x in ids
        if int(x) not in (
            PAD_ID,
            BOS_ID
        )
    ]

    if EOS_ID in ids:
        ids = ids[
            :ids.index(EOS_ID)
        ]

    return tgt_tokenizer.decode(
        ids
    )


# ============================================================
# Repetition measurement
# ============================================================

def repetition_rate(text):

    tokens = text.lower().split()

    if len(tokens) < 2:
        return 0.0

    repeated = 0

    for i in range(
        1,
        len(tokens)
    ):

        if tokens[i] == tokens[i - 1]:
            repeated += 1

    return (
        repeated
        / (len(tokens) - 1)
    )


# ============================================================
# Evaluation
# ============================================================

predictions = []
references = []

total_repetition = 0.0

print("\nRunning beam-search evaluation...")

for index in range(
    len(val_dataset)
):

    sample = val_dataset[index]

    src = sample["source"].unsqueeze(0).to(
        DEVICE
    )

    morphology = sample[
        "morphology"
    ].unsqueeze(0).to(
        DEVICE
    )

    target = sample[
        "target"
    ]

    prediction_ids = beam_search(
        src,
        morphology
    )

    prediction = decode_target(
        prediction_ids
    )

    reference = decode_target(
        target
    )

    predictions.append(
        prediction
    )

    references.append(
        reference
    )

    total_repetition += (
        repetition_rate(
            prediction
        )
    )

    if (index + 1) % 25 == 0:

        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(val_dataset)}"
        )


# ============================================================
# Metrics
# ============================================================

bleu = BLEU()

chrf = CHRF()

bleu_score = bleu.corpus_score(
    predictions,
    [references]
).score

chrf_score = chrf.corpus_score(
    predictions,
    [references]
).score

average_repetition = (
    total_repetition
    / len(predictions)
    * 100
)


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 70)
print("V2 EVALUATION RESULTS")
print("=" * 70)

print(
    f"Samples: {len(predictions)}"
)

print(
    f"BLEU: {bleu_score:.4f}"
)

print(
    f"chrF: {chrf_score:.4f}"
)

print(
    f"Repetition rate: "
    f"{average_repetition:.2f}%"
)


# ============================================================
# Samples
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE TRANSLATIONS")
print("=" * 70)

for i in range(
    min(10, len(predictions))
):

    print(
        f"\n[{i + 1}]"
    )

    print(
        "Reference:",
        references[i]
    )

    print(
        "Prediction:",
        predictions[i]
    )


# ============================================================
# Save results
# ============================================================

results_dir = (
    PROJECT_ROOT
    / "models"
    / "proposed_v2"
    / "results"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

results_file = (
    results_dir
    / "v2_evaluation_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V2 Morphology-Conditioned Transformer\n"
    )

    f.write(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}\n"
    )

    f.write(
        f"Validation loss: "
        f"{checkpoint['val_loss']}\n"
    )

    f.write(
        f"Samples: "
        f"{len(predictions)}\n"
    )

    f.write(
        f"BLEU: "
        f"{bleu_score:.4f}\n"
    )

    f.write(
        f"chrF: "
        f"{chrf_score:.4f}\n"
    )

    f.write(
        f"Repetition rate: "
        f"{average_repetition:.2f}%\n\n"
    )

    for i in range(
        len(predictions)
    ):

        f.write(
            f"[{i + 1}]\n"
        )

        f.write(
            f"Reference: "
            f"{references[i]}\n"
        )

        f.write(
            f"Prediction: "
            f"{predictions[i]}\n\n"
        )


print(
    "\nResults saved to:"
)

print(results_file)