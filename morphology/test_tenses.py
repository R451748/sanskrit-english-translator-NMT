import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "chronbmm/sanskrit5-multitask"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SANSKRIT TENSE / MOOD TEST")
print("=" * 70)
print("Device:", DEVICE)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()


sentences = [
    ("Present", "rāmaḥ vanaṃ gacchati"),
    ("Past", "rāmaḥ vanaṃ agacchat"),
    ("Future", "rāmaḥ vanaṃ gamiṣyati"),
    ("Imperative", "rāma vanaṃ gaccha"),
    ("Optative", "rāmaḥ vanaṃ gacchet"),
    ("Perfect", "rāmaḥ vanaṃ jagāma"),
]


for name, sentence in sentences:

    print("\n" + "-" * 70)
    print("Expected:", name)
    print("Input   :", sentence)

    prompt = "SLM " + sentence

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=2
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("Output  :", result)


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)