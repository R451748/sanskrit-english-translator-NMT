import torch
import sentencepiece as spm

from model import SanskritEnglishTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3

MAX_SRC_LEN = 128
MAX_TGT_LEN = 256

CHECKPOINT = (
    "models/transformer/checkpoints_dynamic/best_model.pt"
)

SRC_TOKENIZER = (
    "dataset/processed/tokenizers/sanskrit.model"
)

TGT_TOKENIZER = (
    "dataset/processed/tokenizers/english.model"
)

# Beam search
BEAM_SIZE = 4

# Prevent excessive repetition
NO_REPEAT_NGRAM_SIZE = 3

# Small penalty against repeatedly selecting same tokens
REPETITION_PENALTY = 1.2


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("TRANSFORMER BEAM SEARCH INFERENCE")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# LOAD TOKENIZERS
# ============================================================

print("Loading tokenizers...")

src_sp = spm.SentencePieceProcessor()
src_sp.load(SRC_TOKENIZER)

tgt_sp = spm.SentencePieceProcessor()
tgt_sp.load(TGT_TOKENIZER)

print("Tokenizers loaded.")
print()


# ============================================================
# CREATE MODEL
# ============================================================

print("Creating model...")

model = SanskritEnglishTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=256,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1,
    src_pad_idx=PAD_IDX,
    tgt_pad_idx=PAD_IDX,
    max_src_len=MAX_SRC_LEN,
    max_tgt_len=MAX_TGT_LEN
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'N/A')}"
)

print(
    f"Validation loss: "
    f"{checkpoint.get('val_loss', 'N/A')}"
)

print()


# ============================================================
# SOURCE ENCODING
# ============================================================

def encode_source(text):

    ids = src_sp.encode(
        text,
        out_type=int
    )

    # Reserve space for EOS
    ids = ids[
        :MAX_SRC_LEN - 1
    ]

    ids.append(EOS_IDX)

    return ids


# ============================================================
# TARGET DECODING
# ============================================================

def decode_target(ids):

    tokens = []

    for token in ids:

        token = int(token)

        if token == BOS_IDX:
            continue

        if token == EOS_IDX:
            break

        if token == PAD_IDX:
            break

        tokens.append(token)

    if not tokens:
        return ""

    return tgt_sp.decode(
        tokens
    )


# ============================================================
# N-GRAM REPETITION CHECK
# ============================================================

def creates_repeated_ngram(
    sequence,
    next_token,
    n
):

    if n <= 0:
        return False

    if len(sequence) < n - 1:
        return False

    candidate = (
        sequence
        + [next_token]
    )

    new_ngram = tuple(
        candidate[-n:]
    )

    existing = set()

    for i in range(
        len(candidate) - n
    ):

        ngram = tuple(
            candidate[
                i:i + n
            ]
        )

        existing.add(
            ngram
        )

    return new_ngram in existing


# ============================================================
# REPETITION PENALTY
# ============================================================

def apply_repetition_penalty(
    logits,
    sequence,
    penalty
):

    if penalty <= 1.0:
        return logits

    for token in set(sequence):

        if token == BOS_IDX:
            continue

        if token == EOS_IDX:
            continue

        value = logits[token]

        if value > 0:
            logits[token] = (
                value / penalty
            )
        else:
            logits[token] = (
                value * penalty
            )

    return logits


# ============================================================
# BEAM SEARCH
# ============================================================

def beam_search(
    src_text,
    beam_size=BEAM_SIZE
):

    src_ids = encode_source(
        src_text
    )

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE
    )

    beams = [
        (
            [BOS_IDX],
            0.0
        )
    ]

    with torch.inference_mode():

        for step in range(
            MAX_TGT_LEN - 1
        ):

            candidates = []

            for sequence, score in beams:

                # ------------------------------------------------
                # Already finished
                # ------------------------------------------------

                if sequence[-1] == EOS_IDX:

                    candidates.append(
                        (
                            sequence,
                            score
                        )
                    )

                    continue

                generated = torch.tensor(
                    [sequence],
                    dtype=torch.long,
                    device=DEVICE
                )

                # ------------------------------------------------
                # Model prediction
                # ------------------------------------------------

                if DEVICE.type == "cuda":

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16
                    ):

                        output = model(
                            src,
                            generated
                        )

                else:

                    output = model(
                        src,
                        generated
                    )

                logits = output[
                    0,
                    -1,
                    :
                ].float()

                # ------------------------------------------------
                # Repetition penalty
                # ------------------------------------------------

                logits = apply_repetition_penalty(
                    logits,
                    sequence,
                    REPETITION_PENALTY
                )

                # ------------------------------------------------
                # Log probabilities
                # ------------------------------------------------

                log_probs = torch.log_softmax(
                    logits,
                    dim=-1
                )

                # ------------------------------------------------
                # Candidate tokens
                # ------------------------------------------------

                top_values, top_indices = torch.topk(
                    log_probs,
                    beam_size * 2
                )

                for value, token_id in zip(
                    top_values,
                    top_indices
                ):

                    token_id = int(
                        token_id.item()
                    )

                    # --------------------------------------------
                    # Avoid repeated n-grams
                    # --------------------------------------------

                    if (
                        token_id != EOS_IDX
                        and creates_repeated_ngram(
                            sequence,
                            token_id,
                            NO_REPEAT_NGRAM_SIZE
                        )
                    ):
                        continue

                    new_sequence = (
                        sequence
                        + [token_id]
                    )

                    new_score = (
                        score
                        + float(
                            value.item()
                        )
                    )

                    candidates.append(
                        (
                            new_sequence,
                            new_score
                        )
                    )

            if not candidates:
                break

            # ----------------------------------------------------
            # Length-normalized beam ranking
            # ----------------------------------------------------

            def beam_score(item):

                sequence, score = item

                length = max(
                    1,
                    len(sequence) - 1
                )

                return (
                    score
                    / (length ** 0.7)
                )

            candidates.sort(
                key=beam_score,
                reverse=True
            )

            beams = candidates[
                :beam_size
            ]

            # ----------------------------------------------------
            # Stop when all beams finished
            # ----------------------------------------------------

            if all(
                sequence[-1] == EOS_IDX
                for sequence, _ in beams
            ):
                break

    best_sequence = max(
        beams,
        key=beam_score
    )[0]

    return decode_target(
        best_sequence
    )


# ============================================================
# TEST SENTENCES
# ============================================================

test_sentences = [

    "ramaḥ vanam gacchati",

    "śrī bhagavān uvāca",

    "arjunaḥ uvāca",

    "mayi eva manaḥ ādhatsva",

    "bhaktyā tu ananyayā śakyaḥ aham",

]


# ============================================================
# RUN TEST
# ============================================================

print("=" * 70)
print("BEAM SEARCH TRANSLATION TEST")
print("=" * 70)

for i, sentence in enumerate(
    test_sentences,
    start=1
):

    print()
    print(
        f"Example {i}"
    )

    print(
        f"Sanskrit : {sentence}"
    )

    result = beam_search(
        sentence
    )

    print(
        f"English  : {result}"
    )


print()
print("=" * 70)
print("INFERENCE COMPLETED")
print("=" * 70)