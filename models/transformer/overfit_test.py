import time
import torch
import torch.nn as nn
from torch.optim import AdamW

from model import SanskritEnglishTransformer


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_SAMPLES = 32
BATCH_SIZE = 8
STEPS = 1500

LR = 1e-4

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3

SRC_FILE = (
    "dataset/processed/sequences/train_source.npy"
)

TGT_FILE = (
    "dataset/processed/sequences/train_target.npy"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TRANSFORMER OVERFIT SANITY TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# LOAD DATA
# ============================================================
# ============================================================
# LOAD DATA
# ============================================================

print("Loading 32 training examples...")

src_all = torch.from_numpy(
    __import__("numpy").load(SRC_FILE)
).long()

tgt_all = torch.from_numpy(
    __import__("numpy").load(TGT_FILE)
).long()

src = src_all[:NUM_SAMPLES]
tgt = tgt_all[:NUM_SAMPLES]

print(f"Source shape: {src.shape}")
print(f"Target shape: {tgt.shape}")

print(f"Source dtype: {src.dtype}")
print(f"Target dtype: {tgt.dtype}")

print()

# ============================================================
# MODEL
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
    max_src_len=128,
    max_tgt_len=256
).to(DEVICE)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0
)


# ============================================================
# TRAIN
# ============================================================

model.train()

start_time = time.time()

for step in range(1, STEPS + 1):

    # Cycle through the 32 examples
    start = ((step - 1) * BATCH_SIZE) % NUM_SAMPLES
    end = start + BATCH_SIZE

    if end <= NUM_SAMPLES:

        src_batch = src[start:end]
        tgt_batch = tgt[start:end]

    else:

        # Wrap around
        first = src[start:NUM_SAMPLES]
        second = src[:end - NUM_SAMPLES]

        src_batch = torch.cat(
            [first, second],
            dim=0
        )

        first = tgt[start:NUM_SAMPLES]
        second = tgt[:end - NUM_SAMPLES]

        tgt_batch = torch.cat(
            [first, second],
            dim=0
        )

    src_batch = src_batch.to(DEVICE)
    tgt_batch = tgt_batch.to(DEVICE)

    # Teacher forcing
    decoder_input = tgt_batch[:, :-1]
    expected_output = tgt_batch[:, 1:]

    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        src_batch,
        decoder_input
    )

    loss = criterion(
        output.reshape(
            -1,
            TGT_VOCAB_SIZE
        ),
        expected_output.reshape(-1)
    )

    if not torch.isfinite(loss):

        print(
            f"ERROR: Invalid loss at step {step}: "
            f"{loss.item()}"
        )

        break

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if step == 1 or step % 25 == 0:

        elapsed = time.time() - start_time

        print(
            f"Step [{step}/{STEPS}] "
            f"Loss: {loss.item():.4f} "
            f"Time: {elapsed / 60:.2f} min"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("OVERFIT TEST COMPLETED")
print("=" * 70)

print(
    f"Final Loss: {loss.item():.4f}"
)

print(
    f"Total Time: "
    f"{(time.time() - start_time) / 60:.2f} minutes"
)

print("=" * 70)