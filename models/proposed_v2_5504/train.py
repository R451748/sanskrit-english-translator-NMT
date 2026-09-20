import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import random_split

from dataset import (
    MorphologyTranslationDataset,
    create_dataloader
)

from model import MorphologyConditionedTransformer


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 8
EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 0.5

TRAIN_RATIO = 0.90

MORPH_VOCAB_SIZE = 392

CHECKPOINT_DIR = "models/proposed_v2_5504/checkpoints"

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pt"
)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest_model.pt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("V2-5504 MORPHOLOGY-CONDITIONED TRANSFORMER TRAINING")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# DATASET
# ============================================================

dataset = MorphologyTranslationDataset()

total_size = len(dataset)

train_size = int(
    total_size * TRAIN_RATIO
)

val_size = total_size - train_size


print("\nDataset:")
print("Total:", total_size)
print("Train:", train_size)
print("Validation:", val_size)


# ============================================================
# SAME DETERMINISTIC SPLIT
# ============================================================

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("\nSplit seed:", SEED)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = create_dataloader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTrain batches:", len(train_loader))
print("Validation batches:", len(val_loader))


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


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nModel parameters:",
    f"{total_params:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    for batch_idx, batch in enumerate(
        train_loader,
        start=1
    ):

        source = batch["source"].to(
            device,
            non_blocking=True
        )

        target = batch["target"].to(
            device,
            non_blocking=True
        )

        morphology = batch["morphology"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            source,
            morphology,
            target
        )

        # Target excludes BOS and aligns
        # with decoder output.
        target_output = target[:, 1:]

        loss = criterion(
            output.reshape(-1, output.size(-1)),
            target_output.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"    Batch "
                f"[{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0

    for batch in val_loader:

        source = batch["source"].to(
            device,
            non_blocking=True
        )

        target = batch["target"].to(
            device,
            non_blocking=True
        )

        morphology = batch["morphology"].to(
            device,
            non_blocking=True
        )

        output = model(
            source,
            morphology,
            target
        )

        target_output = target[:, 1:]

        loss = criterion(
            output.reshape(-1, output.size(-1)),
            target_output.reshape(-1)
        )

        total_loss += loss.item()

    return total_loss / len(val_loader)


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float("inf")

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    print("\n" + "=" * 70)
    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )
    print("=" * 70)

    train_loss = train_one_epoch()

    val_loss = validate()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']:.7f}"
    )

    # --------------------------------------------------------
    # Latest checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "morph_vocab_size": MORPH_VOCAB_SIZE,
            "seed": SEED
        },
        LATEST_CHECKPOINT
    )

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "morph_vocab_size": MORPH_VOCAB_SIZE,
                "seed": SEED
            },
            BEST_CHECKPOINT
        )

        print(
            "\n*** Best model saved ***"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    "\nBest checkpoint:"
)

print(
    BEST_CHECKPOINT
)

print("\nTraining history:")

for item in history:

    print(
        f"Epoch {item['epoch']:2d} | "
        f"Train: {item['train_loss']:.4f} | "
        f"Val: {item['val_loss']:.4f}"
    )

print("\n" + "=" * 70)