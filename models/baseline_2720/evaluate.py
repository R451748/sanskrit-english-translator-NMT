import os
import time
import sys

import torch
import sentencepiece as spm

from sacrebleu.metrics import BLEU, CHRF


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from models.transformer.model import (
    SanskritEnglishTransformer
)

from models.proposed.dataset import (
    create_dataset
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    "models/baseline_2720/checkpoints/"
    "best_model.pt"
)

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

MAX_LENGTH = 128

BEAM_SIZE = 4

REPETITION_PENALTY = 1.2

NO_REPEAT_NGRAM = 3

LENGTH_PENALTY = 0.7

OUTPUT_FILE = (
    "models/baseline_2720/"
    "baseline_beam_results.txt"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("CONTROLLED BASELINE TRANSFORMER")
print("BEAM SEARCH + REPETITION CONTROL")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# ENGLISH TOKENIZER
# ============================================================

tokenizer = spm.SentencePieceProcessor()

tokenizer.load(
    "dataset/processed/tokenizers/english.model"
)


# ============================================================
# DATASET
# ============================================================

dataset = create_dataset()

generator = torch.Generator().manual_seed(42)

train_size = int(
    0.90 * len(dataset)
)

validation_size = (
    len(dataset) - train_size
)

_, validation_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, validation_size],
    generator=generator
)

print(
    f"\nEvaluation sentences: "
    f"{len(validation_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

model = SanskritEnglishTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,

    d_model=256,
    nhead=4,

    num_encoder_layers=3,
    num_decoder_layers=3,

    dim_feedforward=1024,

    dropout=0.1,

    src_pad_idx=PAD_ID,
    tgt_pad_idx=PAD_ID,

    max_src_len=128,
    max_tgt_len=256
).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nCheckpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Validation loss:",
    checkpoint["validation_loss"]
)


# ============================================================
# NO-REPEAT N-GRAM
# ============================================================

def has_repeated_ngram(
    sequence,
    next_token,
    n
):

    if len(sequence) < n - 1:
        return False

    candidate = (
        sequence[-(n - 1):]
        + [next_token]
    )

    for i in range(
        len(sequence) - n + 1
    ):

        previous = sequence[
            i:i + n
        ]

        if previous == candidate:
            return True

    return False


# ============================================================
# BEAM SEARCH
# ============================================================

def beam_search(src):

    beams = [
        (
            [BOS_ID],
            0.0
        )
    ]

    for step in range(
        MAX_LENGTH
    ):

        candidates = []

        for tokens, score in beams:

            if tokens[-1] == EOS_ID:

                candidates.append(
                    (
                        tokens,
                        score
                    )
                )

                continue

            decoder_input = torch.tensor(
                [tokens],
                dtype=torch.long,
                device=device
            )

            with torch.no_grad():

                output = model(
                    src,
                    decoder_input
                )

            logits = output[
                0,
                -1,
                :
            ]

            # ------------------------------------------------
            # Repetition penalty
            # ------------------------------------------------

            for token_id in set(tokens):

                if logits[token_id] > 0:

                    logits[token_id] /= (
                        REPETITION_PENALTY
                    )

                else:

                    logits[token_id] *= (
                        REPETITION_PENALTY
                    )

            # Never generate PAD

            logits[PAD_ID] = -float("inf")

            log_probs = torch.log_softmax(
                logits,
                dim=-1
            )

            top_values, top_indices = torch.topk(
                log_probs,
                BEAM_SIZE
            )

            for log_prob, token_id in zip(
                top_values.tolist(),
                top_indices.tolist()
            ):

                if has_repeated_ngram(
                    tokens,
                    token_id,
                    NO_REPEAT_NGRAM
                ):

                    continue

                new_tokens = (
                    tokens
                    + [token_id]
                )

                new_score = (
                    score
                    + log_prob
                )

                candidates.append(
                    (
                        new_tokens,
                        new_score
                    )
                )

        if not candidates:
            break

        # ----------------------------------------------------
        # Length-normalized ranking
        # ----------------------------------------------------

        def beam_score(item):

            tokens, score = item

            length = max(
                1,
                len(tokens) - 1
            )

            return (
                score
                / (
                    length
                    ** LENGTH_PENALTY
                )
            )

        candidates.sort(
            key=beam_score,
            reverse=True
        )

        beams = candidates[
            :BEAM_SIZE
        ]

        if all(
            tokens[-1] == EOS_ID
            for tokens, _ in beams
        ):

            break

    best_tokens, _ = max(
        beams,
        key=lambda item: (
            item[1]
            / (
                max(
                    1,
                    len(item[0]) - 1
                )
                ** LENGTH_PENALTY
            )
        )
    )

    return best_tokens


# ============================================================
# CLEAN IDS
# ============================================================

def clean_ids(ids):

    result = []

    for token_id in ids:

        if token_id == BOS_ID:
            continue

        if token_id == PAD_ID:
            continue

        if token_id == EOS_ID:
            break

        result.append(token_id)

    return result


# ============================================================
# REPETITION METRIC
# ============================================================

def repetition_rate(words):

    if len(words) < 4:
        return 0.0

    seen = set()

    repeated = 0

    total = 0

    for i in range(
        len(words) - 2
    ):

        trigram = tuple(
            words[i:i + 3]
        )

        total += 1

        if trigram in seen:

            repeated += 1

        seen.add(trigram)

    if total == 0:
        return 0.0

    return repeated / total


# ============================================================
# EVALUATION
# ============================================================

references = []

hypotheses = []

repetition_scores = []

examples = []

start_time = time.time()

print(
    "\nGenerating baseline translations..."
)


for index in range(
    len(validation_dataset)
):

    item = validation_dataset[index]

    src = (
        item["source"]
        .unsqueeze(0)
        .to(device)
    )

    target = item["target"]

    generated = beam_search(
        src
    )

    generated_ids = clean_ids(
        generated
    )

    target_ids = clean_ids(
        target.tolist()
    )

    hypothesis = tokenizer.decode(
        generated_ids
    )

    reference = tokenizer.decode(
        target_ids
    )

    hypotheses.append(
        hypothesis
    )

    references.append(
        reference
    )

    repetition_scores.append(
        repetition_rate(
            hypothesis.split()
        )
    )

    if len(examples) < 10:

        examples.append(
            (
                reference,
                hypothesis
            )
        )

    if (
        (index + 1) % 25 == 0
        or index + 1 == len(validation_dataset)
    ):

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            (index + 1)
            / elapsed
        )

        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(validation_dataset)} "
            f"| "
            f"{speed:.2f} sentences/sec"
        )


# ============================================================
# METRICS
# ============================================================

bleu = BLEU()

chrf = CHRF()

bleu_score = bleu.corpus_score(
    hypotheses,
    [references]
)

chrf_score = chrf.corpus_score(
    hypotheses,
    [references]
)

average_repetition = (
    sum(repetition_scores)
    / len(repetition_scores)
)

elapsed = (
    time.time()
    - start_time
)

speed = (
    len(validation_dataset)
    / elapsed
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)

print("BASELINE EVALUATION COMPLETE")

print("=" * 70)

print(
    f"\nSentences: "
    f"{len(validation_dataset)}"
)

print(
    f"BLEU: "
    f"{bleu_score.score:.4f}"
)

print(
    f"chrF: "
    f"{chrf_score.score:.4f}"
)

print(
    f"Repetition rate: "
    f"{average_repetition * 100:.2f}%"
)

print(
    f"Evaluation time: "
    f"{elapsed / 60:.2f} minutes"
)

print(
    f"Speed: "
    f"{speed:.2f} sentences/sec"
)


# ============================================================
# SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TRANSLATIONS")
print("=" * 70)

for i, (
    reference,
    hypothesis
) in enumerate(
    examples,
    start=1
):

    print(
        f"\nExample {i}"
    )

    print(
        "Reference:"
    )

    print(reference)

    print(
        "\nGenerated:"
    )

    print(hypothesis)

    print("-" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CONTROLLED BASELINE TRANSFORMER\n"
    )

    f.write(
        "BEAM SEARCH + REPETITION CONTROL\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}\n"
    )

    f.write(
        f"Validation loss: "
        f"{checkpoint['validation_loss']}\n"
    )

    f.write(
        f"Sentences: "
        f"{len(validation_dataset)}\n"
    )

    f.write(
        f"BLEU: "
        f"{bleu_score.score:.4f}\n"
    )

    f.write(
        f"chrF: "
        f"{chrf_score.score:.4f}\n"
    )

    f.write(
        f"Repetition rate: "
        f"{average_repetition * 100:.2f}%\n"
    )

    f.write(
        f"Evaluation time: "
        f"{elapsed / 60:.2f} minutes\n"
    )

    f.write(
        f"Speed: "
        f"{speed:.2f} sentences/sec\n\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "SAMPLE TRANSLATIONS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for i, (
        reference,
        hypothesis
    ) in enumerate(
        examples,
        start=1
    ):

        f.write(
            f"Example {i}\n\n"
        )

        f.write(
            "Reference:\n"
            + reference
            + "\n\n"
        )

        f.write(
            "Generated:\n"
            + hypothesis
            + "\n\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

print(
    "\nResults saved to:"
)

print(OUTPUT_FILE)

print("=" * 70)