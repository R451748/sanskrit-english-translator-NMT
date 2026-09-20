import os
import time
import math

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset import create_dataloader
from model import SanskritEnglishTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.5

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# IMPORTANT:
# This is a separate folder so your original checkpoint
# is NOT overwritten.
CHECKPOINT_DIR = "models/transformer/checkpoints_dynamic"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("TRANSFORMER BASELINE - DYNAMIC PADDING")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )

print()


# ============================================================
# DATA LOADERS
# ============================================================

print("Loading datasets...")

train_loader = create_dataloader(
    "train",
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = create_dataloader(
    "validation",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    f"Training batches: {len(train_loader)}"
)

print(
    f"Validation batches: {len(val_loader)}"
)

print()


# ============================================================
# MODEL
# ============================================================

print("Creating Transformer model...")
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
    max_src_len=128,
    max_tgt_len=256
).to(DEVICE)

# ============================================================
# MODEL PARAMETERS
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
    f"Total parameters: {total_params:,}"
)

print(
    f"Trainable parameters: {trainable_params:,}"
)

print()


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_loss = float("inf")


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    epoch_start = time.time()

    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 70)

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    total_train_loss = 0.0
    valid_train_batches = 0

    for batch_idx, (src, tgt) in enumerate(train_loader):

        # Move data to GPU
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # ----------------------------------------------------
        # Teacher Forcing
        # ----------------------------------------------------
        #
        # Target:
        #
        # [BOS, word1, word2, word3, EOS, PAD...]
        #
        # Decoder input:
        #
        # [BOS, word1, word2, word3, ...]
        #
        # Expected output:
        #
        # [word1, word2, word3, EOS, ...]
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
        # Forward pass
        # ----------------------------------------------------

        output = model(
            src,
            decoder_input
        )

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            output.reshape(
                -1,
                TGT_VOCAB_SIZE
            ),
            expected_output.reshape(-1)
        )

        # ----------------------------------------------------
        # Check for NaN / Inf
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            print()
            print(
                f"WARNING: Invalid loss detected "
                f"at batch {batch_idx + 1}"
            )

            print(
                f"Loss: {loss.item()}"
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            continue

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
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Track loss
        # ----------------------------------------------------

        total_train_loss += loss.item()

        valid_train_batches += 1

        # ----------------------------------------------------
        # Print progress every 100 batches
        # ----------------------------------------------------

        if (batch_idx + 1) % 100 == 0:

            average_loss = (
                total_train_loss
                / valid_train_batches
            )

            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"| Loss: {loss.item():.4f} "
                f"| Avg: {average_loss:.4f}"
            )

    # ========================================================
    # TRAINING LOSS
    # ========================================================

    if valid_train_batches > 0:

        train_loss = (
            total_train_loss
            / valid_train_batches
        )

    else:

        train_loss = float("nan")


    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("Running validation...")

    model.eval()

    total_val_loss = 0.0
    valid_val_batches = 0

    with torch.no_grad():

        for src, tgt in val_loader:

            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            # Teacher forcing
            decoder_input = tgt[:, :-1]

            expected_output = tgt[:, 1:]

            # Forward pass
            output = model(
                src,
                decoder_input
            )

            # Loss
            loss = criterion(
                output.reshape(
                    -1,
                    TGT_VOCAB_SIZE
                ),
                expected_output.reshape(-1)
            )

            if torch.isfinite(loss):

                total_val_loss += loss.item()

                valid_val_batches += 1


    # ========================================================
    # VALIDATION LOSS
    # ========================================================

    if valid_val_batches > 0:

        val_loss = (
            total_val_loss
            / valid_val_batches
        )

    else:

        val_loss = float("nan")


    # ========================================================
    # EPOCH INFORMATION
    # ========================================================

    epoch_time = time.time() - epoch_start

    print()
    print("-" * 70)

    print(
        f"Epoch: {epoch + 1}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Learning Rate: {LEARNING_RATE}"
    )

    print(
        f"Epoch Time: {epoch_time / 60:.2f} minutes"
    )

    print("-" * 70)


    # ========================================================
    # SAVE LATEST CHECKPOINT
    # ========================================================

    latest_path = os.path.join(
        CHECKPOINT_DIR,
        "latest_model.pt"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        latest_path
    )

    print(
        f"Latest checkpoint saved:"
    )

    print(
        latest_path
    )


    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    if (
        math.isfinite(val_loss)
        and val_loss < best_val_loss
    ):

        best_val_loss = val_loss

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            best_path
        )

        print(
            f"Best model saved:"
        )

        print(
            best_path
        )

    print()


# ============================================================
# TRAINING COMPLETED
# ============================================================

print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

if math.isfinite(best_val_loss):

    print(
        f"Best Validation Loss: {best_val_loss:.4f}"
    )

else:

    print(
        "Best Validation Loss: NaN"
    )

print(
    f"Checkpoint directory:"
)

print(
    CHECKPOINT_DIR
)

print("=" * 70)