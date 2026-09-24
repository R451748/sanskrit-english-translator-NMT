import os
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from indic_transliteration import sanscript
from IndicTransToolkit import IndicProcessor
import sacrebleu


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

BASE_MODEL = "krpraveen/indictrans2-sanskrit-en-finetuned"

LORA_ADAPTER = os.path.join(
    PROJECT_ROOT,
    "models",
    "indictrans2_lora_25k_full"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "final",
    "test.csv"
)

RESULT_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "indictrans2_lora_25k_full",
    "lora_test_results.csv"
)

BATCH_SIZE = 2
MAX_SOURCE_LENGTH = 256
MAX_NEW_TOKENS = 128


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("IndicTrans2 + LoRA 25K - HELD-OUT TEST EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Base model:", BASE_MODEL)
print("LoRA adapter:", LORA_ADAPTER)
print("Test file:", TEST_FILE)
print("=" * 70)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

df = pd.read_csv(TEST_FILE)

df = df.dropna(
    subset=["sanskrit", "english"]
).reset_index(drop=True)

print("Test sentences:", len(df))


# ============================================================
# IAST → DEVANAGARI
# ============================================================

print("\nConverting IAST → Devanagari...")


def to_devanagari(text):
    return sanscript.transliterate(
        str(text),
        sanscript.IAST,
        sanscript.DEVANAGARI
    )


source_texts = [
    to_devanagari(text)
    for text in df["sanskrit"]
]

references = df["english"].astype(str).tolist()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base IndicTrans2 model...")

dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=dtype
)


# ============================================================
# LOAD LoRA ADAPTER
# ============================================================

print("\nLoading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER
)

model = model.to(DEVICE)

model.eval()

print("LoRA adapter loaded successfully.")


# ============================================================
# INDIC PROCESSOR
# ============================================================

processor = IndicProcessor(
    inference=True
)


# ============================================================
# TRANSLATION
# ============================================================

predictions = []

print("\nStarting translation...")

for start in tqdm(
    range(0, len(source_texts), BATCH_SIZE)
):

    batch = source_texts[
        start:start + BATCH_SIZE
    ]

    processed_batch = processor.preprocess_batch(
        batch,
        src_lang="san_Deva",
        tgt_lang="eng_Latn"
    )

    inputs = tokenizer(
        processed_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=2,
            early_stopping=True
        )

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    decoded = processor.postprocess_batch(
        decoded,
        lang="eng_Latn"
    )

    predictions.extend(decoded)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

result_df = pd.DataFrame(
    {
        "sanskrit": df["sanskrit"],
        "reference": references,
        "prediction": predictions
    }
)

result_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# BLEU
# ============================================================

bleu = sacrebleu.corpus_bleu(
    predictions,
    [references]
)


# ============================================================
# chrF
# ============================================================

chrf = sacrebleu.corpus_chrf(
    predictions,
    [references]
)


# ============================================================
# REPETITION
# ============================================================

def repetition_rate(text):

    words = text.lower().split()

    if len(words) < 2:
        return 0.0

    repeated = 0

    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeated += 1

    return repeated / len(words)


repetition_values = [
    repetition_rate(text)
    for text in predictions
]

average_repetition = (
    sum(repetition_values)
    / len(repetition_values)
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL LoRA TEST RESULTS")
print("=" * 70)

print("Test sentences :", len(predictions))
print("BLEU           :", round(bleu.score, 4))
print("chrF           :", round(chrf.score, 4))
print(
    "Repetition     :",
    round(average_repetition, 2),
    "%"
)

print("=" * 70)

print("\nResults saved to:")
print(RESULT_FILE)


# ============================================================
# SAMPLE TRANSLATIONS
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE TRANSLATIONS")
print("=" * 70)

for i in range(min(10, len(result_df))):

    print("\nSanskrit:")
    print(result_df.iloc[i]["sanskrit"])

    print("\nReference:")
    print(result_df.iloc[i]["reference"])

    print("\nLoRA:")
    print(result_df.iloc[i]["prediction"])

    print("-" * 70)



