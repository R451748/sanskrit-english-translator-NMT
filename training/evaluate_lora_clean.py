import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indic_transliteration import sanscript
from IndicTransToolkit import IndicProcessor
from sacrebleu import corpus_bleu, corpus_chrf

PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset/final/lora_clean_eval.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments/lora_clean_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL = "krpraveen/indictrans2-sanskrit-en-finetuned"

MODELS = {
    "BASE": BASE_MODEL,
    "LORA_5K": os.path.join(
        PROJECT_ROOT,
        "models/indictrans2_lora_5k"
    ),
    "LORA_25K": os.path.join(
        PROJECT_ROOT,
        "models/indictrans2_lora_25k"
    )
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FAST SETTINGS
NUM_SAMPLES = 100
BATCH_SIZE = 8
MAX_NEW_TOKENS = 64
NUM_BEAMS = 1

print("=" * 70)
print("FAST CLEAN EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

df = pd.read_csv(DATA_FILE)

df = df.head(NUM_SAMPLES).copy()

print("Evaluation sentences:", len(df))


# ---------------------------------------------------------
# PREPROCESS
# ---------------------------------------------------------

print("\nPreprocessing Sanskrit...")

ip = IndicProcessor(inference=True)

sources = []

for text in tqdm(df["sanskrit"].tolist()):
    iast = str(text)

    deva = sanscript.transliterate(
        iast,
        sanscript.IAST,
        sanscript.DEVANAGARI
    )

    processed = ip.preprocess_batch(
        [deva],
        src_lang="san_Deva",
        tgt_lang="eng_Latn"
    )[0]

    sources.append(processed)


references = df["english"].astype(str).tolist()


# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_model(name, model_path):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model = model.to(DEVICE)
    model.eval()

    predictions = []

    with torch.inference_mode():

        for start in tqdm(
            range(0, len(sources), BATCH_SIZE),
            desc=name
        ):

            batch = sources[start:start + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            inputs = {
                k: v.to(DEVICE)
                for k, v in inputs.items()
            }

            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )

            decoded = ip.postprocess_batch(
                decoded,
                lang="eng_Latn"
            )

            predictions.extend(decoded)

    bleu = corpus_bleu(
        predictions,
        [references]
    ).score

    chrf = corpus_chrf(
        predictions,
        [references]
    ).score

    print("\nRESULT")
    print("BLEU :", round(bleu, 4))
    print("chrF :", round(chrf, 4))

    # Save predictions immediately
    result_df = pd.DataFrame({
        "sanskrit": df["sanskrit"],
        "reference": references,
        "prediction": predictions
    })

    result_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{name.lower()}_100_predictions.csv"
        ),
        index=False
    )

    # Free GPU memory
    del model
    del tokenizer

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return bleu, chrf


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

results = {}

for name, path in MODELS.items():

    bleu, chrf = evaluate_model(
        name,
        path
    )

    results[name] = {
        "BLEU": bleu,
        "chrF": chrf
    }


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results).T

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fast_clean_100_results.csv"
    )
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(results_df)

print("\nSaved to:")
print(OUTPUT_DIR)