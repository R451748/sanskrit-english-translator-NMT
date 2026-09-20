import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


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


# ============================================================
# PROJECT IMPORTS
# ============================================================

from models.transformer.model import SanskritEnglishTransformer

from models.proposed.dataset import (
    create_dataset,
    morphology_collate
)

# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8
EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.5

PAD_IDX = 0

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000

CHECKPOINT_DIR = (
    "models/baseline_2720/checkpoints"
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
print("2,720-SENTENCE EXPERIMENT")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

full_dataset = create_dataset()

print(
    f"Total records: {len(full_dataset):,}"
)


# ============================================================
# SAME 90/10 SPLIT AS PROPOSED MODEL
# ============================================================

generator = torch.Generator().manual_seed(42)

train_size = int(
    0.90 * len(full_dataset)
)

validation_size = (
    len(full_dataset) - train_size
)

train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, validation_size],
    generator=generator
)

print(
    f"\nTraining samples: "
    f"{len(train_dataset):,}"
)

print(
    f"Validation samples: "
    f"{len(validation_dataset):,}"
)


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=morphology_collate,
    num_workers=0,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=morphology_collate,
    num_workers=0,
    pin_memory=True
)

print(
    f"\nTraining batches: "
    f"{len(train_loader):,}"
)

print(
    f"Validation batches: "
    f"{len(validation_loader):,}"
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

    src_pad_idx=PAD_IDX,
    tgt_pad_idx=PAD_IDX,

    max_src_len=128,
    max_tgt_len=256
).to(device)


# ============================================================
# PARAMETER COUNT
# ============================================================

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nModel parameters: "
    f"{parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
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
# TRAINING
# ============================================================

best_validation_loss = float("inf")


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    total_train_loss = 0.0

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

        # ----------------------------------------------------
        # Teacher forcing
        # ----------------------------------------------------

        decoder_input = tgt[:, :-1]

        expected_output = tgt[:, 1:]

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            src,
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
            expected_output.reshape(-1)
        )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"NaN/Inf loss at "
                f"epoch {epoch}, "
                f"batch {batch_idx}"
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        total_train_loss += loss.item()

        if batch_idx % 50 == 0:

            average_loss = (
                total_train_loss
                / batch_idx
            )

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {average_loss:.4f}"
            )

    # ========================================================
    # TRAIN LOSS
    # ========================================================

    train_loss = (
        total_train_loss
        / len(train_loader)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_validation_loss = 0.0

    with torch.no_grad():

        for batch in validation_loader:

            src = batch["source"].to(
                device,
                non_blocking=True
            )

            tgt = batch["target"].to(
                device,
                non_blocking=True
            )

            decoder_input = tgt[:, :-1]

            expected_output = tgt[:, 1:]

            output = model(
                src,
                decoder_input
            )

            loss = criterion(
                output.reshape(
                    -1,
                    TGT_VOCAB_SIZE
                ),
                expected_output.reshape(-1)
            )

            total_validation_loss += (
                loss.item()
            )

    validation_loss = (
        total_validation_loss
        / len(validation_loader)
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Validation Loss:  {validation_loss:.4f}"
    )

    print(
        f"Learning Rate:    "
        f"{optimizer.param_groups[0]['lr']}"
    )

    print(
        f"Epoch Time:       "
        f"{epoch_time / 60:.2f} minutes"
    )

    print("=" * 70)

    # ========================================================
    # SAVE LATEST
    # ========================================================

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss
        },
        os.path.join(
            CHECKPOINT_DIR,
            "latest_model.pt"
        )
    )

    # ========================================================
    # SAVE BEST
    # ========================================================

    if validation_loss < best_validation_loss:

        best_validation_loss = (
            validation_loss
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

                "validation_loss":
                    validation_loss
            },
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pt"
            )
        )

        print(
            "\nNew best model saved."
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("BASELINE TRAINING COMPLETE")

print(
    f"Best Validation Loss: "
    f"{best_validation_loss:.4f}"
)

print(
    "Checkpoint:"
)

print(
    f"{CHECKPOINT_DIR}\\best_model.pt"
)

print("=" * 70)