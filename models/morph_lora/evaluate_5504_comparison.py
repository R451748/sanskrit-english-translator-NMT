# ============================================================
# EVALUATE BASELINE vs MORPHOLOGY LoRA
#
# Same 550 validation sentences
#
# BASELINE:
#   Sanskrit -> IndicTrans2 + LoRA
#
# PROPOSED:
#   Sanskrit + Morphology -> IndicTrans2 + LoRA
#
# Metrics:
#   BLEU
#   chrF
#   Repetition rate
# ============================================================

import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from peft import PeftModel

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from sacrebleu import corpus_bleu, corpus_chrf


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"
)

DATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "morphology_train_5504.json"
)

BASELINE_DIR = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "baseline_5504"
)

MORPH_DIR = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "morphology_5504"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "evaluation_5504"
)

# IMPORTANT:
# Always use the original base model for the tokenizer.
BASE_MODEL = (
    "krpraveen/"
    "indictrans2-sanskrit-en-finetuned"
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

MAX_SOURCE_LENGTH = 256

MAX_NEW_TOKENS = 128

NUM_BEAMS = 2

BATCH_SIZE = 4


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("BASELINE vs MORPHOLOGY LoRA EVALUATION")
print("=" * 70)

print()

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

print()


# ============================================================
# 4. CHECK PATHS
# ============================================================

print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

if not BASELINE_DIR.exists():

    raise FileNotFoundError(
        f"Baseline LoRA directory not found:\n"
        f"{BASELINE_DIR}"
    )

if not MORPH_DIR.exists():

    raise FileNotFoundError(
        f"Morphology LoRA directory not found:\n"
        f"{MORPH_DIR}"
    )

print(
    "Dataset:",
    DATA_FILE
)

print(
    "Baseline:",
    BASELINE_DIR
)

print(
    "Morphology:",
    MORPH_DIR
)

print()


# ============================================================
# 5. IAST → DEVANAGARI
# ============================================================

def iast_to_devanagari(text):

    if not text:

        return ""

    return transliterate(
        text,
        sanscript.IAST,
        sanscript.DEVANAGARI
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    records = json.load(f)


print(
    "Total records:",
    len(records)
)

print()


# ============================================================
# 7. SAME TRAIN / VALIDATION SPLIT
#
# IMPORTANT:
# This reproduces the exact split used during training.
# ============================================================

indices = list(
    range(len(records))
)

random.Random(SEED).shuffle(
    indices
)

val_size = int(
    len(indices) * 0.10
)

val_indices = indices[
    :val_size
]

validation_records = [
    records[i]
    for i in val_indices
]


print("=" * 70)
print("VALIDATION SET")
print("=" * 70)

print(
    "Validation sentences:",
    len(validation_records)
)

print(
    "Expected:",
    550
)

print()


# ============================================================
# 8. BUILD BASELINE SOURCE
# ============================================================

def build_baseline_source(record):

    sanskrit_iast = record[
        "sanskrit"
    ]

    sanskrit_devanagari = (
        iast_to_devanagari(
            sanskrit_iast
        )
    )

    source = (
        "san_Deva "
        "eng_Latn "
        + sanskrit_devanagari
    )

    return source


# ============================================================
# 9. BUILD MORPHOLOGY SOURCE
# ============================================================

def build_morphology_source(record):

    sanskrit_iast = record[
        "sanskrit"
    ]

    sanskrit_devanagari = (
        iast_to_devanagari(
            sanskrit_iast
        )
    )

    morphology_tags = record.get(
        "morphology_tags",
        []
    )

    morphology = " ".join(
        morphology_tags
    )

    source = (
        "san_Deva "
        "eng_Latn "
        + sanskrit_devanagari
        + " MORPHOLOGY "
        + morphology
    )

    return source


# ============================================================
# 10. LOAD LoRA MODEL
# ============================================================

def load_lora_model(
    model_dir,
    model_name
):

    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print(
        "Model:",
        model_name
    )

    print(
        "Adapter:",
        model_dir
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # DO NOT load tokenizer from the LoRA directory.
    #
    # The custom IndicTrans2 tokenizer gives:
    #
    # TypeError:
    # PreTrainedTokenizer.__init__()
    # got multiple values for keyword argument
    # 'src_vocab_file'
    #
    # Therefore load tokenizer from the original
    # IndicTrans2 model.
    # --------------------------------------------------------

    print(
        "Loading tokenizer from base model..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    print(
        "Tokenizer loaded."
    )

    # --------------------------------------------------------
    # Load original base model
    # --------------------------------------------------------

    print(
        "Loading base model..."
    )

    base_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=(
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )
        )
    )

    print(
        "Base model loaded."
    )

    # --------------------------------------------------------
    # Load LoRA adapter
    # --------------------------------------------------------

    print(
        "Loading LoRA adapter..."
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(model_dir)
    )

    print(
        "LoRA adapter loaded."
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model ready."
    )

    print()

    return (
        tokenizer,
        model
    )


# ============================================================
# 11. GENERATE TRANSLATIONS
# ============================================================

def generate_translations(
    model,
    tokenizer,
    records,
    morphology=False
):

    predictions = []

    references = []

    if morphology:

        title = (
            "GENERATING MORPHOLOGY "
            "PREDICTIONS"
        )

    else:

        title = (
            "GENERATING BASELINE "
            "PREDICTIONS"
        )

    print("=" * 70)
    print(title)
    print("=" * 70)

    print()

    for start in tqdm(
        range(
            0,
            len(records),
            BATCH_SIZE
        ),
        desc="Translation"
    ):

        batch_records = records[
            start:
            start + BATCH_SIZE
        ]

        sources = []

        batch_references = []

        for record in batch_records:

            if morphology:

                source = (
                    build_morphology_source(
                        record
                    )
                )

            else:

                source = (
                    build_baseline_source(
                        record
                    )
                )

            sources.append(
                source
            )

            batch_references.append(
                record["english"]
            )

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        inputs = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.no_grad():

            generated_ids = (
                model.generate(

                    **inputs,

                    max_new_tokens=(
                        MAX_NEW_TOKENS
                    ),

                    num_beams=NUM_BEAMS,

                    no_repeat_ngram_size=3,

                    repetition_penalty=1.1,

                    early_stopping=True
                )
            )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        batch_predictions = (
            tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        )

        predictions.extend(
            batch_predictions
        )

        references.extend(
            batch_references
        )

    return (
        predictions,
        references
    )


# ============================================================
# 12. REPETITION DETECTION
# ============================================================

def has_repetition(text):

    words = re.findall(
        r"\S+",
        text.lower()
    )

    if len(words) < 4:

        return False

    # --------------------------------------------------------
    # Repeated single word 3+ times
    # --------------------------------------------------------

    counts = {}

    for word in words:

        counts[word] = (
            counts.get(word, 0) + 1
        )

        if counts[word] >= 3:

            return True

    # --------------------------------------------------------
    # Repeated consecutive bigram
    # --------------------------------------------------------

    for i in range(
        len(words) - 3
    ):

        if (
            words[i]
            == words[i + 2]
            and
            words[i + 1]
            == words[i + 3]
        ):

            return True

    return False


def repetition_rate(
    predictions
):

    if not predictions:

        return 0.0

    repeated = sum(
        has_repetition(
            prediction
        )
        for prediction in predictions
    )

    return (
        repeated
        / len(predictions)
        * 100
    )


# ============================================================
# 13. METRICS
# ============================================================

def calculate_metrics(
    predictions,
    references
):

    bleu_result = corpus_bleu(
        predictions,
        [references]
    )

    chrf_result = corpus_chrf(
        predictions,
        [references]
    )

    repetition = (
        repetition_rate(
            predictions
        )
    )

    return {

        "BLEU":
            bleu_result.score,

        "chrF":
            chrf_result.score,

        "repetition_rate":
            repetition,

        "sentences":
            len(predictions)
    }


# ============================================================
# 14. LOAD BASELINE
# ============================================================

baseline_tokenizer, baseline_model = (
    load_lora_model(
        BASELINE_DIR,
        "Baseline LoRA 5504"
    )
)


# ============================================================
# 15. EVALUATE BASELINE
# ============================================================

(
    baseline_predictions,
    references
) = generate_translations(

    baseline_model,

    baseline_tokenizer,

    validation_records,

    morphology=False
)


baseline_metrics = calculate_metrics(
    baseline_predictions,
    references
)


# ------------------------------------------------------------
# Free GPU memory
# ------------------------------------------------------------

del baseline_model

del baseline_tokenizer

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 16. LOAD MORPHOLOGY MODEL
# ============================================================

morph_tokenizer, morph_model = (
    load_lora_model(
        MORPH_DIR,
        "Morphology LoRA 5504"
    )
)


# ============================================================
# 17. EVALUATE MORPHOLOGY
# ============================================================

(
    morph_predictions,
    morph_references
) = generate_translations(

    morph_model,

    morph_tokenizer,

    validation_records,

    morphology=True
)


morph_metrics = calculate_metrics(
    morph_predictions,
    morph_references
)


# ============================================================
# 18. FINAL COMPARISON
# ============================================================

print()

print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print()

print(
    "BASELINE LoRA — 5504"
)

print(
    f"BLEU: "
    f"{baseline_metrics['BLEU']:.4f}"
)

print(
    f"chrF: "
    f"{baseline_metrics['chrF']:.4f}"
)

print(
    f"Repetition: "
    f"{baseline_metrics['repetition_rate']:.2f}%"
)

print(
    f"Sentences: "
    f"{baseline_metrics['sentences']}"
)

print()

print(
    "MORPHOLOGY LoRA — 5504"
)

print(
    f"BLEU: "
    f"{morph_metrics['BLEU']:.4f}"
)

print(
    f"chrF: "
    f"{morph_metrics['chrF']:.4f}"
)

print(
    f"Repetition: "
    f"{morph_metrics['repetition_rate']:.2f}%"
)

print(
    f"Sentences: "
    f"{morph_metrics['sentences']}"
)

print()


# ============================================================
# 19. DIFFERENCES
# ============================================================

bleu_difference = (
    morph_metrics["BLEU"]
    - baseline_metrics["BLEU"]
)

chrf_difference = (
    morph_metrics["chrF"]
    - baseline_metrics["chrF"]
)

repetition_difference = (
    morph_metrics["repetition_rate"]
    - baseline_metrics["repetition_rate"]
)


print("=" * 70)
print("MORPHOLOGY EFFECT")
print("=" * 70)

print()

print(
    f"BLEU difference: "
    f"{bleu_difference:+.4f}"
)

print(
    f"chrF difference: "
    f"{chrf_difference:+.4f}"
)

print(
    f"Repetition difference: "
    f"{repetition_difference:+.2f}%"
)

print()


# ============================================================
# 20. SAVE RESULTS
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


comparison = {

    "experiment":
        "Baseline vs Morphology LoRA 5504",

    "validation_sentences":
        len(validation_records),

    "seed":
        SEED,

    "generation": {

        "num_beams":
            NUM_BEAMS,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "no_repeat_ngram_size":
            3,

        "repetition_penalty":
            1.1
    },

    "baseline":
        baseline_metrics,

    "morphology":
        morph_metrics,

    "difference": {

        "BLEU":
            bleu_difference,

        "chrF":
            chrf_difference,

        "repetition_rate":
            repetition_difference
    }
}


metrics_file = (
    RESULT_DIR
    / "comparison_metrics.json"
)


with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 21. SAVE PREDICTIONS
# ============================================================

results = []

for i, record in enumerate(
    validation_records
):

    results.append({

        "index":
            record.get("index"),

        "sanskrit":
            record["sanskrit"],

        "reference":
            references[i],

        "baseline":
            baseline_predictions[i],

        "morphology":
            morph_predictions[i],

        "morphology_tags":
            record.get(
                "morphology_tags",
                []
            )
    })


predictions_file = (
    RESULT_DIR
    / "comparison_predictions.json"
)


with open(
    predictions_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 22. QUALITATIVE EXAMPLES
# ============================================================

print("=" * 70)
print("QUALITATIVE EXAMPLES")
print("=" * 70)

for i in range(
    min(10, len(results))
):

    item = results[i]

    print()

    print("-" * 70)

    print(
        "Example:",
        i + 1
    )

    print(
        "Index:",
        item["index"]
    )

    print()

    print(
        "SANSKRIT:"
    )

    print(
        item["sanskrit"]
    )

    print()

    print(
        "MORPHOLOGY:"
    )

    print(
        " ".join(
            item["morphology_tags"]
        )
    )

    print()

    print(
        "REFERENCE:"
    )

    print(
        item["reference"]
    )

    print()

    print(
        "BASELINE:"
    )

    print(
        item["baseline"]
    )

    print()

    print(
        "MORPHOLOGY MODEL:"
    )

    print(
        item["morphology"]
    )


# ============================================================
# 23. FINISHED
# ============================================================

print()

print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print()

print(
    "Metrics saved:"
)

print(
    metrics_file
)

print()

print(
    "Predictions saved:"
)

print(
    predictions_file
)

print()

print("=" * 70)