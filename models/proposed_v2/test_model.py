import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.proposed_v2.model import (
    MorphologyConditionedTransformer
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = MorphologyConditionedTransformer(
    src_vocab_size=16000,
    tgt_vocab_size=16000,
    morphology_vocab_size=228,
    d_model=256,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1
).to(device)


src = torch.randint(
    1,
    16000,
    (2, 20)
).to(device)

morphology = torch.randint(
    1,
    228,
    (2, 8)
).to(device)

tgt = torch.randint(
    1,
    16000,
    (2, 30)
).to(device)


with torch.no_grad():

    output = model(
        src,
        morphology,
        tgt
    )


print("Device:", device)
print("Source:", src.shape)
print("Morphology:", morphology.shape)
print("Target:", tgt.shape)
print("Output:", output.shape)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)