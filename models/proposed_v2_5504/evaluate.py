import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import random
import numpy as np
import torch
import torch.nn as nn
import sentencepiece as spm

from torch.utils.data import random_split

from models.proposed_v2_5504.dataset import (
    MorphologyTranslationDataset,
    create_dataloader
)

from models.proposed_v2_5504.model import (
    MorphologyConditionedTransformer
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 8
MORPH_VOCAB_SIZE = 392

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "proposed_v2_5504"
    / "checkpoints"
    / "best_model.pt"
)

ENGLISH_TOKENIZER = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "tokenizers"
    / "english.model"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "models"
    / "proposed_v2_5504"
    / "results"
    / "v2_5504_evaluation_results.txt"
)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("V2-5504 MODEL EVALUATION")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECKPOINT
# ============================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

print("\nCheckpoint:")
print(CHECKPOINT)

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Checkpoint validation loss:",
    checkpoint.get("val_loss")
)


# ============================================================
# DATASET
# ============================================================

dataset = MorphologyTranslationDataset()

total_size = len(dataset)

train_size = int(
    total_size * 0.90
)

val_size = total_size - train_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("\nDataset:")
print("Total:", total_size)
print("Validation:", len(val_dataset))


val_loader = create_dataloader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = MorphologyConditionedTransformer(
    src_vocab_size=16000,
    tgt_vocab_size=16000,
    morph_vocab_size=MORPH_VOCAB_SIZE,
    d_model=256,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# ENGLISH TOKENIZER
# ============================================================

if not ENGLISH_TOKENIZER.exists():
    raise FileNotFoundError(
        f"English tokenizer not found:\n"
        f"{ENGLISH_TOKENIZER}"
    )

english_sp = spm.SentencePieceProcessor()

english_sp.load(
    str(ENGLISH_TOKENIZER)
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# ============================================================
# GREEDY TRANSLATION
# ============================================================

@torch.no_grad()
def translate(
    source,
    morphology,
    max_length=256
):

    batch_size = source.size(0)

    # --------------------------------------------------------
    # Special tokens
    #
    # PAD = 0
    # UNK = 1
    # BOS = 2
    # EOS = 3
    # --------------------------------------------------------

    generated = torch.full(
        (batch_size, 1),
        2,
        dtype=torch.long,
        device=device
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device
    )

    for _ in range(max_length - 1):

        # ----------------------------------------------------
        # IMPORTANT
        #
        # The trained model predicts:
        #
        # target[:, 1:]
        #
        # from the decoder input.
        #
        # With only [BOS], the model can return zero positions.
        #
        # Add temporary PAD so that the model has one decoder
        # prediction position.
        # ----------------------------------------------------

        decoder_input = torch.cat(
            [
                generated,
                torch.zeros(
                    (batch_size, 1),
                    dtype=torch.long,
                    device=device
                )
            ],
            dim=1
        )

        output = model(
            source,
            morphology,
            decoder_input
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if output.size(1) == 0:
            break

        # ----------------------------------------------------
        # Predict next token
        # ----------------------------------------------------

        next_token = output[:, -1, :].argmax(
            dim=-1
        )

        # ----------------------------------------------------
        # Keep EOS for sentences already finished
        # ----------------------------------------------------

        eos_tensor = torch.full_like(
            next_token,
            3
        )

        next_token = torch.where(
            finished,
            eos_tensor,
            next_token
        )

        # ----------------------------------------------------
        # Append prediction
        # ----------------------------------------------------

        generated = torch.cat(
            [
                generated,
                next_token.unsqueeze(1)
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Update finished status
        # ----------------------------------------------------

        finished |= (
            next_token == 3
        )

        # ----------------------------------------------------
        # Stop if every sentence reached EOS
        # ----------------------------------------------------

        if finished.all():
            break

    return generated


# ============================================================
# DECODE
# ============================================================

def decode_tokens(tokens):

    tokens = tokens.tolist()

    # Remove PAD
    tokens = [
        token
        for token in tokens
        if token != 0
    ]

    # Remove BOS
    if tokens and tokens[0] == 2:
        tokens = tokens[1:]

    # Remove everything after EOS
    if 3 in tokens:
        tokens = tokens[:tokens.index(3)]

    if not tokens:
        return ""

    return english_sp.decode(tokens)


# ============================================================
# REPETITION CHECK
# ============================================================

def has_repetition(text):

    words = text.lower().split()

    if len(words) < 3:
        return False

    for i in range(len(words) - 2):

        if (
            words[i]
            == words[i + 1]
            == words[i + 2]
        ):
            return True

    return False


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION EVALUATION")
print("=" * 70)

total_loss = 0.0

predictions = []
references = []

repetition_count = 0
total_sentences = 0

examples = []


with torch.no_grad():

    for batch_number, batch in enumerate(
        val_loader,
        start=1
    ):

        source = batch["source"].to(device)

        target = batch["target"].to(device)

        morphology = batch["morphology"].to(device)

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------

        output = model(
            source,
            morphology,
            target
        )

        target_output = target[:, 1:]

        loss = criterion(
            output.reshape(
                -1,
                output.size(-1)
            ),
            target_output.reshape(-1)
        )

        total_loss += loss.item()

        # ----------------------------------------------------
        # Translation
        # ----------------------------------------------------

        generated = translate(
            source,
            morphology
        )

        for i in range(
            generated.size(0)
        ):

            prediction = decode_tokens(
                generated[i]
            )

            reference = decode_tokens(
                target[i]
            )

            predictions.append(
                prediction
            )

            references.append(
                reference
            )

            total_sentences += 1

            if has_repetition(prediction):
                repetition_count += 1

            if len(examples) < 10:

                examples.append({
                    "index": batch["indices"][i].item(),
                    "prediction": prediction,
                    "reference": reference
                })

        if batch_number % 10 == 0:

            print(
                f"Processed batches: "
                f"{batch_number}/{len(val_loader)}"
            )


# ============================================================
# LOSS
# ============================================================

average_loss = (
    total_loss / len(val_loader)
)

repetition_rate = (
    repetition_count
    / total_sentences
    * 100
)


# ============================================================
# BLEU / CHRF
# ============================================================

bleu_score = None
chrf_score = None

try:

    import sacrebleu

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references]
    )

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references]
    )

    bleu_score = bleu.score
    chrf_score = chrf.score

except ImportError:

    print(
        "\nWARNING: sacrebleu is not installed."
    )

    print(
        "Install using:"
    )

    print(
        "pip install sacrebleu"
    )

except Exception as e:

    print(
        "\nWARNING calculating BLEU/chrF:"
    )

    print(e)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V2-5504 EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nValidation sentences: "
    f"{total_sentences}"
)

print(
    f"Validation loss: "
    f"{average_loss:.4f}"
)

if bleu_score is not None:

    print(
        f"BLEU: "
        f"{bleu_score:.4f}"
    )

    print(
        f"chrF: "
        f"{chrf_score:.4f}"
    )

print(
    f"Repetition rate: "
    f"{repetition_rate:.2f}%"
)


# ============================================================
# EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("TRANSLATION EXAMPLES")
print("=" * 70)

for number, example in enumerate(
    examples,
    start=1
):

    print(
        f"\nExample {number}"
    )

    print(
        f"Index: {example['index']}"
    )

    print(
        "Prediction:",
        example["prediction"]
    )

    print(
        "Reference:",
        example["reference"]
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V2-5504 MODEL EVALUATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}\n"
    )

    f.write(
        f"Checkpoint validation loss: "
        f"{checkpoint.get('val_loss')}\n"
    )

    f.write(
        f"Validation sentences: "
        f"{total_sentences}\n"
    )

    f.write(
        f"Validation loss: "
        f"{average_loss:.4f}\n"
    )

    if bleu_score is not None:

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
        f"{repetition_rate:.2f}%\n"
    )

    f.write(
        "\n" + "=" * 70 + "\n"
    )

    f.write(
        "TRANSLATION EXAMPLES\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    for number, example in enumerate(
        examples,
        start=1
    ):

        f.write(
            f"\nExample {number}\n"
        )

        f.write(
            f"Index: {example['index']}\n"
        )

        f.write(
            f"Prediction: "
            f"{example['prediction']}\n"
        )

        f.write(
            f"Reference: "
            f"{example['reference']}\n"
        )


print("\nResults saved to:")
print(RESULT_FILE)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)