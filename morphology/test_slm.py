import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "chronbmm/sanskrit5-multitask"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("SANSKRIT5 SLM MORPHOLOGY TEST")
print("=" * 70)

print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

sentences = [
    "rāmaḥ vanaṃ gacchati",
    "śrī bhagavān uvāca",
    "arjunaḥ uvāca",
    "kṛṣṇaḥ pāṇḍavān prati uvāca",
    "yena vijñānamātreṇa sarvajñatvaṃ prajāyate"
]

for sentence in sentences:

    print("\n" + "-" * 70)
    print("INPUT:")
    print(sentence)

    # SLM = unsandhied + lemma + morphosyntax
    prompt = "SLM " + sentence

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("OUTPUT:")
    print(result)

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)