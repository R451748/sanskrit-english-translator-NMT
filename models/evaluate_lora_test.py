
import os
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from peft import PeftModel

from indic_transliteration import sanscript
from IndicTransToolkit import IndicProcessor

from sacrebleu import corpus_bleu, corpus_chrf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

BASE_MODEL = "krpraveen/indictrans2-sanskrit-en-finetuned"

LORA_MODEL = os.path.join(
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

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "lora_test_results.csv"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
MAX_SOURCE_LENGTH = 256
MAX_NEW_TOKENS = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("IndicTrans2 LoRA - HELD-OUT TEST EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Test file:", TEST_FILE)
print("LoRA model:", LORA_MODEL)


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
# LOAD PROCESSOR
# ============================================================

print("\nLoading IndicProcessor...")

processor = IndicProcessor(
    inference=True
)


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

print("\nLoading base IndicTrans2...")

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
    LORA_MODEL
)

model = model.to(DEVICE)

model.eval()

print("LoRA model loaded successfully.")


# ============================================================
# TRANSLATION
# ============================================================

predictions = []

references = []

sanskrit_original = []

total = len(df)

print("\nStarting translation...")
print("Total:", total)


for start in range(0, total, BATCH_SIZE):

    batch_df = df.iloc[
        start:start + BATCH_SIZE
    ]

    iast_sentences = (
        batch_df["sanskrit"]
        .astype(str)
        .tolist()
    )

    english_references = (
        batch_df["english"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # IAST → DEVANAGARI
    # --------------------------------------------------------

    devanagari = []

    for sentence in iast_sentences:

        try:

            text = sanscript.transliterate(
                sentence,
                sanscript.IAST,
                sanscript.DEVANAGARI
            )

        except Exception:

            text = sentence

        devanagari.append(text)

    # --------------------------------------------------------
    # INDIC PROCESSOR
    # --------------------------------------------------------

    processed = processor.preprocess_batch(
        devanagari,
        src_lang="san_Deva",
        tgt_lang="eng_Latn"
    )

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        processed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1
        )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )

    # --------------------------------------------------------
    # POSTPROCESS
    # --------------------------------------------------------

    decoded = processor.postprocess_batch(
        decoded,
        lang="eng_Latn"
    )

    predictions.extend(decoded)

    references.extend(
        english_references
    )

    sanskrit_original.extend(
        iast_sentences
    )

    current = min(
        start + BATCH_SIZE,
        total
    )

    if current % 100 == 0 or current == total:

        print(
            f"Processed {current}/{total}"
        )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df = pd.DataFrame({

    "sanskrit": sanskrit_original,

    "reference": references,

    "prediction": predictions

})

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nPredictions saved to:")
print(OUTPUT_FILE)


# ============================================================
# BLEU
# ============================================================

print("\nCalculating BLEU...")

bleu = corpus_bleu(
    predictions,
    [references]
)


# ============================================================
# chrF
# ============================================================

print("Calculating chrF...")

chrf = corpus_chrf(
    predictions,
    [references]
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("LoRA TEST RESULTS")
print("=" * 70)

print(
    f"Test sentences : {len(predictions)}"
)

print(
    f"BLEU           : {bleu.score:.4f}"
)

print(
    f"chrF           : {chrf.score:.4f}"
)

print("=" * 70)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\nSample translations:")
print("=" * 70)

for i in range(min(10, len(results_df))):

    print("\nSanskrit:")
    print(results_df.iloc[i]["sanskrit"])

    print("\nReference:")
    print(results_df.iloc[i]["reference"])

    print("\nLoRA:")
    print(results_df.iloc[i]["prediction"])

    print("-" * 70)
