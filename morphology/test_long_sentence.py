import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "chronbmm/sanskrit5-multitask"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sentence = (
    "rāmaḥ sītayā lakṣmaṇena ca saha "
    "vanaṃ gatvā tatra bahūni varṣāṇi nivasati"
)

print("=" * 70)
print("LONG SENTENCE MORPHOLOGY TEST")
print("=" * 70)

print("Device:", DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

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

print("\nInput:")
print(sentence)

print("\nMorphology output:")
print(result)

print("\n" + "=" * 70)