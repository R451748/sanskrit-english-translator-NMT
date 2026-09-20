import os
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from indic_transliteration import sanscript
from IndicTransToolkit import IndicProcessor

from sacrebleu import corpus_bleu, corpus_chrf


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

MODEL_NAME = "krpraveen/indictrans2-sanskrit-en-finetuned"

# IMPORTANT: 25K LoRA model
LORA_PATH = os.path.join(
    PROJECT_ROOT,
    "models/indictrans2_lora_25k"
)

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset/final/validation.csv"
)

# Evaluate same 100 validation sentences
NUM_SAMPLES = 100

# RTX 3050 6 GB
BATCH_SIZE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("IndicTrans2 25K LoRA Evaluation")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading validation data...")

df = pd.read_csv(DATA_FILE).head(NUM_SAMPLES)

print("Samples:", len(df))

sources_ia = df["sanskrit"].astype(str).tolist()
references = df["english"].astype(str).tolist()


# ============================================================
# IAST -> DEVANAGARI
# ============================================================

print("\nConverting IAST -> Devanagari...")

sources_deva = [
    sanscript.transliterate(
        text,
        sanscript.IAST,
        sanscript.DEVANAGARI
    )
    for text in sources_ia
]


print("\nExample:")
print("IAST :", sources_ia[0])
print("Deva :", sources_deva[0])
print("REF  :", references[0])


# ============================================================
# INDIC PROCESSOR
# ============================================================

print("\nLoading IndicProcessor...")

ip = IndicProcessor(inference=True)

processed_sources = ip.preprocess_batch(
    sources_deva,
    src_lang="san_Deva",
    tgt_lang="eng_Latn"
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_model(model, model_name):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    predictions = []

    model.eval()

    for start in range(
        0,
        len(processed_sources),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(processed_sources)
        )

        batch = processed_sources[start:end]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,

                # Same decoding for every model
                num_beams=1,

                max_new_tokens=64,

                no_repeat_ngram_size=3
            )

        decoded = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        decoded = [
            text.strip()
            for text in decoded
        ]

        predictions.extend(decoded)

        print(
            f"Processed {end}/{len(processed_sources)}"
        )

    return predictions


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading BASE IndicTrans2 model...")

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE
)

base_model.to(DEVICE)

print("Base model loaded.")


# ============================================================
# BASE TRANSLATION
# ============================================================

base_predictions = translate_model(
    base_model,
    "BASE MODEL"
)


# ============================================================
# FREE BASE MODEL
# ============================================================

del base_model

if DEVICE == "cuda":
    torch.cuda.empty_cache()

print("\nBase model unloaded.")


# ============================================================
# LOAD 25K LoRA MODEL
# ============================================================

print("\nLoading 25K LoRA adapter...")

lora_base = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE
)

lora_model = PeftModel.from_pretrained(
    lora_base,
    LORA_PATH
)

lora_model.to(DEVICE)

print("25K LoRA model loaded.")


# ============================================================
# LoRA TRANSLATION
# ============================================================

lora_predictions = translate_model(
    lora_model,
    "25K LoRA MODEL"
)


# ============================================================
# CALCULATE METRICS
# ============================================================

print("\nCalculating metrics...")


base_bleu = corpus_bleu(
    base_predictions,
    [references]
).score

base_chrf = corpus_chrf(
    base_predictions,
    [references]
).score


lora_bleu = corpus_bleu(
    lora_predictions,
    [references]
).score

lora_chrf = corpus_chrf(
    lora_predictions,
    [references]
).score


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS - 25K LoRA")
print("=" * 60)


print("\nBASE MODEL")

print(
    f"BLEU : {base_bleu:.4f}"
)

print(
    f"chrF : {base_chrf:.4f}"
)


print("\n25K LoRA MODEL")

print(
    f"BLEU : {lora_bleu:.4f}"
)

print(
    f"chrF : {lora_chrf:.4f}"
)


print("\nCHANGE")

print(
    f"BLEU change : {lora_bleu - base_bleu:+.4f}"
)

print(
    f"chrF change : {lora_chrf - base_chrf:+.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(
    LORA_PATH,
    "evaluation_100.txt"
)


with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "IndicTrans2 25K LoRA Evaluation\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Samples: {NUM_SAMPLES}\n"
    )

    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    f.write(
        "Beams: 1\n"
    )

    f.write(
        "Max new tokens: 64\n\n"
    )


    f.write("BASE MODEL\n")

    f.write(
        f"BLEU: {base_bleu:.4f}\n"
    )

    f.write(
        f"chrF: {base_chrf:.4f}\n\n"
    )


    f.write("25K LoRA MODEL\n")

    f.write(
        f"BLEU: {lora_bleu:.4f}\n"
    )

    f.write(
        f"chrF: {lora_chrf:.4f}\n\n"
    )


    f.write("CHANGE\n")

    f.write(
        f"BLEU: {lora_bleu - base_bleu:+.4f}\n"
    )

    f.write(
        f"chrF: {lora_chrf - base_chrf:+.4f}\n\n"
    )


    f.write("=" * 60 + "\n")
    f.write("SAMPLE TRANSLATIONS\n")
    f.write("=" * 60 + "\n")


    for i in range(len(references)):

        f.write(
            f"\n[{i + 1}]\n"
        )

        f.write(
            f"Sanskrit: {sources_ia[i]}\n"
        )

        f.write(
            f"Reference: {references[i]}\n"
        )

        f.write(
            f"Base: {base_predictions[i]}\n"
        )

        f.write(
            f"25K LoRA: {lora_predictions[i]}\n"
        )


print("\nSaved:")
print(result_file)

print("\nEvaluation complete.")