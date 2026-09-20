import os
import sys
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from pathlib import Path

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from models.proposed.dataset import (
    create_dataset,
    morphology_collate
)

from models.proposed_v2.model import (
    MorphologyConditionedTransformer
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 8
EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.5

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000
MORPH_VOCAB_SIZE = 228

D_MODEL = 256
NHEAD = 4

ENCODER_LAYERS = 3
DECODER_LAYERS = 3

FEEDFORWARD_DIM = 1024
DROPOUT = 0.1

SEED = 42

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "proposed_v2"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("MORPHOLOGY-CONDITIONED TRANSFORMER V2")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# ============================================================
# Dataset
# ============================================================

print("\nLoading dataset...")

dataset = create_dataset()

print(
    "Total samples:",
    len(dataset)
)


# ============================================================
# Train / validation split
# ============================================================

train_size = int(
    0.90 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=morphology_collate,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=morphology_collate,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# Model
# ============================================================

print("\nCreating V2 model...")

model = MorphologyConditionedTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    morphology_vocab_size=MORPH_VOCAB_SIZE,

    d_model=D_MODEL,
    nhead=NHEAD,

    num_encoder_layers=ENCODER_LAYERS,
    num_decoder_layers=DECODER_LAYERS,

    dim_feedforward=FEEDFORWARD_DIM,
    dropout=DROPOUT,

    src_pad_idx=0,
    tgt_pad_idx=0,
    morph_pad_idx=0,

    max_src_len=128,
    max_morph_len=64,
    max_tgt_len=256
).to(device)


# ============================================================
# Parameter count
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Training variables
# ============================================================

best_val_loss = float("inf")

best_epoch = 0


# ============================================================
# Training loop
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    print("\n")
    print("=" * 70)
    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    start_time = time.time()

    for batch_idx, batch in enumerate(
        train_loader,
        start=1
    ):

        src = batch["source"].to(
            device,
            non_blocking=True
        )

        tgt = batch["target"].to(
            device,
            non_blocking=True
        )

        morphology = batch[
            "morphology"
        ].to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Teacher forcing
        # ----------------------------------------------------

        decoder_input = tgt[:, :-1]

        expected_output = tgt[:, 1:]

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(
            src,
            morphology,
            decoder_input
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            output.reshape(
                -1,
                TGT_VOCAB_SIZE
            ),
            expected_output.reshape(
                -1
            )
        )

        # ----------------------------------------------------
        # NaN / Inf protection
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            print(
                "\nWARNING: Non-finite loss detected."
            )

            print(
                "Batch:",
                batch_idx
            )

            print(
                "Loss:",
                loss.item()
            )

            raise RuntimeError(
                "Training stopped because loss became NaN/Inf."
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()

        train_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if batch_idx % 50 == 0:

            avg_loss = (
                train_loss
                / batch_idx
            )

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} "
                f"Time: {elapsed / 60:.1f} min"
            )

    # --------------------------------------------------------
    # Average train loss
    # --------------------------------------------------------

    train_loss /= len(
        train_loader
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            src = batch["source"].to(
                device,
                non_blocking=True
            )

            tgt = batch["target"].to(
                device,
                non_blocking=True
            )

            morphology = batch[
                "morphology"
            ].to(
                device,
                non_blocking=True
            )

            decoder_input = tgt[:, :-1]

            expected_output = tgt[:, 1:]

            output = model(
                src,
                morphology,
                decoder_input
            )

            loss = criterion(
                output.reshape(
                    -1,
                    TGT_VOCAB_SIZE
                ),
                expected_output.reshape(
                    -1
                )
            )

            val_loss += loss.item()

    val_loss /= len(
        val_loader
    )

    epoch_time = (
        time.time()
        - start_time
    )

    # ========================================================
    # Results
    # ========================================================

    print("\n")
    print("-" * 70)

    print(
        f"Epoch {epoch}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time / 60:.2f} minutes"
    )

    print("-" * 70)

    # ========================================================
    # Save latest
    # ========================================================

    latest_path = (
        CHECKPOINT_DIR
        / "latest_model.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "train_loss":
                train_loss,
            "val_loss":
                val_loss,
            "config": {
                "src_vocab_size":
                    SRC_VOCAB_SIZE,

                "tgt_vocab_size":
                    TGT_VOCAB_SIZE,

                "morphology_vocab_size":
                    MORPH_VOCAB_SIZE,

                "d_model":
                    D_MODEL,

                "nhead":
                    NHEAD,

                "encoder_layers":
                    ENCODER_LAYERS,

                "decoder_layers":
                    DECODER_LAYERS,

                "feedforward_dim":
                    FEEDFORWARD_DIM,

                "dropout":
                    DROPOUT
            }
        },
        latest_path
    )

    # ========================================================
    # Save best
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        best_path = (
            CHECKPOINT_DIR
            / "best_model.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "train_loss":
                    train_loss,
                "val_loss":
                    val_loss,
                "config": {
                    "src_vocab_size":
                        SRC_VOCAB_SIZE,

                    "tgt_vocab_size":
                        TGT_VOCAB_SIZE,

                    "morphology_vocab_size":
                        MORPH_VOCAB_SIZE,

                    "d_model":
                        D_MODEL,

                    "nhead":
                        NHEAD,

                    "encoder_layers":
                        ENCODER_LAYERS,

                    "decoder_layers":
                        DECODER_LAYERS,

                    "feedforward_dim":
                        FEEDFORWARD_DIM,

                    "dropout":
                        DROPOUT
                }
            },
            best_path
        )

        print(
            f"\n✓ Best model saved."
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.4f}"
        )


# ============================================================
# Finished
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best Epoch:",
    best_epoch
)

print(
    "Best Validation Loss:",
    f"{best_val_loss:.4f}"
)

print(
    "Checkpoint:",
    CHECKPOINT_DIR
    / "best_model.pt"
)