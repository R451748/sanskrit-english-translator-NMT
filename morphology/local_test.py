import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


MODEL_ID = "chronbmm/sanskrit5-multitask"

REVISION = "c0d2ada54f3d19903149425aa888a203601423f8"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("LOCAL SANSKRIT MORPHOLOGY TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=REVISION
)

print("Tokenizer loaded.")

print()
print("Loading morphology model...")

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    revision=REVISION
)

model = model.to(DEVICE)
model.eval()

print("Model loaded.")
print()


# ------------------------------------------------------------
# Test Sanskrit sentences
# ------------------------------------------------------------

sentences = [
    "śrī bhagavān uvāca",
    "rāmaḥ vanaṃ gacchati",
    "arjunaḥ uvāca",
    "kṛṣṇaḥ pāṇḍavān prati uvāca",
    "yena vijñānamātreṇa sarvajñatvaṃ prajāyate"
]


# ------------------------------------------------------------
# Run morphology
# ------------------------------------------------------------

for i, sentence in enumerate(
    sentences,
    start=1
):

    print("=" * 70)
    print(f"Example {i}")
    print(f"Input: {sentence}")
    print()

    # SLM =
    # Segmentation + Lemma + Morphosyntax
    prompt = "SLM " + sentence

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_length=512,
            num_beams=1
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print("Morphology output:")
    print(result)

print()
print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)