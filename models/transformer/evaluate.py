import torch
import numpy as np
import sentencepiece as spm
from sacrebleu.metrics import BLEU, CHRF

from model import SanskritEnglishTransformer


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_SAMPLES = 50

SRC_FILE = "dataset/processed/sequences/test_source.npy"
TGT_FILE = "dataset/processed/sequences/test_target.npy"

SRC_TOKENIZER = (
    "dataset/processed/tokenizers/sanskrit.model"
)

TGT_TOKENIZER = (
    "dataset/processed/tokenizers/english.model"
)

CHECKPOINT = (
    "models/transformer/checkpoints_dynamic/best_model.pt"
)

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3

MAX_TGT_LEN = 256


# ============================================================
# LOAD TOKENIZERS
# ============================================================

src_sp = spm.SentencePieceProcessor()
src_sp.load(SRC_TOKENIZER)

tgt_sp = spm.SentencePieceProcessor()
tgt_sp.load(TGT_TOKENIZER)


# ============================================================
# MODEL
# ============================================================

model = SanskritEnglishTransformer(
    src_vocab_size=16000,
    tgt_vocab_size=16000,
    d_model=256,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1,
    src_pad_idx=PAD_IDX,
    tgt_pad_idx=PAD_IDX,
    max_src_len=128,
    max_tgt_len=256
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("=" * 70)
print("DECODING COMPARISON")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
print(f"Validation loss: {checkpoint.get('val_loss')}")
print(f"Evaluation samples: {NUM_SAMPLES}")
print()


# ============================================================
# LOAD TEST DATA
# ============================================================

test_src = np.load(SRC_FILE)
test_tgt = np.load(TGT_FILE)

test_src = test_src[:NUM_SAMPLES]
test_tgt = test_tgt[:NUM_SAMPLES]


# ============================================================
# DECODE TARGET
# ============================================================

def decode_target(ids):

    tokens = []

    for token in ids:

        token = int(token)

        if token in [PAD_IDX, BOS_IDX]:
            continue

        if token == EOS_IDX:
            break

        tokens.append(token)

    return tgt_sp.decode(tokens)


# ============================================================
# GREEDY DECODING
# ============================================================

def greedy_translate(src):

    src = src.unsqueeze(0)

    generated = torch.tensor(
        [[BOS_IDX]],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.inference_mode():

        for _ in range(MAX_TGT_LEN - 1):

            output = model(
                src,
                generated
            )

            next_token = output[
                0, -1
            ].argmax().item()

            generated = torch.cat(
                [
                    generated,
                    torch.tensor(
                        [[next_token]],
                        dtype=torch.long,
                        device=DEVICE
                    )
                ],
                dim=1
            )

            if next_token == EOS_IDX:
                break

    return decode_target(
        generated[0].cpu().numpy()
    )


# ============================================================
# REPETITION CHECK
# ============================================================

def creates_repeated_ngram(
    sequence,
    token,
    n=3
):

    if len(sequence) < n - 1:
        return False

    candidate = sequence + [token]

    new_ngram = tuple(
        candidate[-n:]
    )

    for i in range(
        len(candidate) - n
    ):

        old_ngram = tuple(
            candidate[i:i+n]
        )

        if old_ngram == new_ngram:
            return True

    return False


# ============================================================
# BEAM SEARCH
# ============================================================

def beam_translate(
    src,
    beam_size=4,
    repetition_penalty=1.2
):

    src = src.unsqueeze(0)

    beams = [
        ([BOS_IDX], 0.0)
    ]

    with torch.inference_mode():

        for _ in range(MAX_TGT_LEN - 1):

            candidates = []

            for sequence, score in beams:

                if sequence[-1] == EOS_IDX:

                    candidates.append(
                        (sequence, score)
                    )

                    continue

                generated = torch.tensor(
                    [sequence],
                    dtype=torch.long,
                    device=DEVICE
                )

                output = model(
                    src,
                    generated
                )

                logits = output[
                    0, -1
                ].float()

                # Repetition penalty
                for token in set(sequence):

                    if token in [
                        BOS_IDX,
                        EOS_IDX
                    ]:
                        continue

                    if logits[token] > 0:
                        logits[token] /= repetition_penalty
                    else:
                        logits[token] *= repetition_penalty

                log_probs = torch.log_softmax(
                    logits,
                    dim=-1
                )

                values, indices = torch.topk(
                    log_probs,
                    beam_size * 2
                )

                for value, index in zip(
                    values,
                    indices
                ):

                    token = int(
                        index.item()
                    )

                    if (
                        token != EOS_IDX
                        and creates_repeated_ngram(
                            sequence,
                            token,
                            3
                        )
                    ):
                        continue

                    new_sequence = (
                        sequence + [token]
                    )

                    new_score = (
                        score +
                        float(value.item())
                    )

                    candidates.append(
                        (
                            new_sequence,
                            new_score
                        )
                    )

            if not candidates:
                break

            def score_beam(item):

                sequence, score = item

                length = max(
                    1,
                    len(sequence) - 1
                )

                return (
                    score /
                    (length ** 0.7)
                )

            candidates.sort(
                key=score_beam,
                reverse=True
            )

            beams = candidates[
                :beam_size
            ]

            if all(
                seq[-1] == EOS_IDX
                for seq, _ in beams
            ):
                break

    best = max(
        beams,
        key=score_beam
    )

    return decode_target(
        best[0]
    )


# ============================================================
# COLLECT RESULTS
# ============================================================

references = []
greedy_predictions = []
beam_predictions = []


print("Translating...")
print()


for i in range(NUM_SAMPLES):

    reference = decode_target(
        test_tgt[i]
    )

    greedy = greedy_translate(
        torch.tensor(
            test_src[i],
            dtype=torch.long,
            device=DEVICE
        )
    )

    beam = beam_translate(
        torch.tensor(
            test_src[i],
            dtype=torch.long,
            device=DEVICE
        )
    )

    references.append(reference)
    greedy_predictions.append(greedy)
    beam_predictions.append(beam)

    print(
        f"[{i+1}/{NUM_SAMPLES}]"
    )


# ============================================================
# METRICS
# ============================================================

bleu = BLEU()
chrf = CHRF()


greedy_bleu = bleu.corpus_score(
    greedy_predictions,
    [references]
)

greedy_chrf = chrf.corpus_score(
    greedy_predictions,
    [references]
)


beam_bleu = bleu.corpus_score(
    beam_predictions,
    [references]
)

beam_chrf = chrf.corpus_score(
    beam_predictions,
    [references]
)


# ============================================================
# REPETITION RATE
# ============================================================

def repetition_rate(sentences):

    repeated = 0

    for sentence in sentences:

        words = sentence.lower().split()

        found = False

        for i in range(
            len(words) - 2
        ):

            trigram = (
                words[i],
                words[i+1],
                words[i+2]
            )

            for j in range(
                i + 1,
                len(words) - 2
            ):

                if (
                    trigram ==
                    (
                        words[j],
                        words[j+1],
                        words[j+2]
                    )
                ):

                    found = True
                    break

            if found:
                break

        if found:
            repeated += 1

    return (
        repeated /
        len(sentences)
    ) * 100


greedy_rep = repetition_rate(
    greedy_predictions
)

beam_rep = repetition_rate(
    beam_predictions
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print()
print("GREEDY DECODING")
print(
    f"BLEU : {greedy_bleu.score:.4f}"
)
print(
    f"chrF : {greedy_chrf.score:.4f}"
)
print(
    f"Repetition rate : {greedy_rep:.2f}%"
)

print()
print("BEAM SEARCH + REPETITION CONTROL")
print(
    f"BLEU : {beam_bleu.score:.4f}"
)
print(
    f"chrF : {beam_chrf.score:.4f}"
)
print(
    f"Repetition rate : {beam_rep:.2f}%"
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print()
print("=" * 70)
print("SAMPLE COMPARISON")
print("=" * 70)

for i in range(
    min(10, NUM_SAMPLES)
):

    print()
    print(
        f"Example {i+1}"
    )

    print(
        "Reference:"
    )
    print(
        references[i]
    )

    print(
        "Greedy:"
    )
    print(
        greedy_predictions[i]
    )

    print(
        "Beam:"
    )
    print(
        beam_predictions[i]
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)