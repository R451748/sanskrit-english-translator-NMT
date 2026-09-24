# ============================================================
# HELD-OUT TEST EVALUATION
#
# Baseline LoRA 5504 vs Morphology LoRA 5504
#
# Test set:
#   dataset/final/test.csv
#
# Metrics:
#   BLEU
#   chrF
#   Repetition rate
#
# IMPORTANT:
# The test set is NEVER used for training.
# ============================================================

import csv
import json
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

# Final held-out test set
TEST_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "final"
    / "test.csv"
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
    / "test_5504_evaluation"
)

BASE_MODEL = (
    "krpraveen/"
    "indictrans2-sanskrit-en-finetuned"
)


# ============================================================
# 2. SETTINGS
# ============================================================

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
print("HELD-OUT TEST EVALUATION")
print("5504 BASELINE vs 5504 MORPHOLOGY")
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
# 4. CHECK FILES
# ============================================================

print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}"
    )

if not BASELINE_DIR.exists():

    raise FileNotFoundError(
        f"Baseline model not found:\n{BASELINE_DIR}"
    )

if not MORPH_DIR.exists():

    raise FileNotFoundError(
        f"Morphology model not found:\n{MORPH_DIR}"
    )

print(
    "Test file:",
    TEST_FILE
)

print(
    "Baseline model:",
    BASELINE_DIR
)

print(
    "Morphology model:",
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
# 6. LOAD TEST CSV
# ============================================================

print("=" * 70)
print("LOADING HELD-OUT TEST SET")
print("=" * 70)

test_records = []

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        sanskrit = (
            row.get("sanskrit")
            or ""
        ).strip()

        english = (
            row.get("english")
            or ""
        ).strip()

        if not sanskrit:
            continue

        if not english:
            continue

        test_records.append({

            "sanskrit":
                sanskrit,

            "english":
                english
        })


print(
    "Test sentences:",
    len(test_records)
)

print()

if len(test_records) != 5056:

    print(
        "WARNING:"
    )

    print(
        "Expected 5056 based on your "
        "final test sequence files."
    )

    print(
        "Actual CSV records:",
        len(test_records)
    )

    print()


# ============================================================
# 7. VERIFY NO MORPHOLOGY TRAINING DATA
# ============================================================

print("=" * 70)
print("TEST SET CHECK")
print("=" * 70)

test_sanskrit = set(
    record["sanskrit"]
    for record in test_records
)

print(
    "Unique Sanskrit test sentences:",
    len(test_sanskrit)
)

print()


# ============================================================
# 8. BUILD BASELINE SOURCE
# ============================================================

def build_baseline_source(
    record
):

    sanskrit_iast = (
        record["sanskrit"]
    )

    sanskrit_devanagari = (
        iast_to_devanagari(
            sanskrit_iast
        )
    )

    return (
        "san_Deva "
        "eng_Latn "
        + sanskrit_devanagari
    )


# ============================================================
# 9. MORPHOLOGY LOOKUP
#
# The 5504 morphology records correspond to a subset of
# training data, NOT the held-out test set.
#
# Therefore we do NOT invent morphology for test sentences.
#
# For a fair morphology-model test, we need morphology
# features for the test sentences.
# ============================================================

MORPHOLOGY_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "test_morphology_5056.json"
)

# ============================================================
# 10. LOAD MORPHOLOGY RECORDS
# ============================================================

print("=" * 70)
print("CHECKING MORPHOLOGY COVERAGE")
print("=" * 70)

morphology_lookup = {}

if MORPHOLOGY_FILE.exists():

    with open(
        MORPHOLOGY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        morphology_records = json.load(f)

    for record in morphology_records:

        key = (
            record.get(
                "sanskrit",
                ""
            ).strip()
        )

        if key:

            morphology_lookup[key] = (
                record.get(
                    "morphology_tags",
                    []
                )
            )

else:

    morphology_records = []

print(
    "Morphology records:",
    len(morphology_records)
)

covered = sum(
    1
    for record in test_records
    if record["sanskrit"] in morphology_lookup
)

print(
    "Test sentences with morphology:",
    covered
)

print()


# ============================================================
# IMPORTANT SAFETY CHECK
# ============================================================

if covered != len(test_records):

    print(
        "IMPORTANT:"
    )

    print(
        "The current morphology dataset does NOT "
        "cover the held-out test set."
    )

    print(
        f"Coverage: {covered}/{len(test_records)}"
    )

    print()

    print(
        "We must NOT evaluate the morphology model "
        "by inventing or copying morphology tags."
    )

    print(
        "The correct next research step is to generate "
        "morphology features for the test set using "
        "the same local morphology model."
    )

    print()

    raise RuntimeError(
        "Missing morphology features for held-out test set."
    )


# ============================================================
# 11. BUILD MORPHOLOGY SOURCE
# ============================================================

def build_morphology_source(
    record
):

    sanskrit_iast = (
        record["sanskrit"]
    )

    sanskrit_devanagari = (
        iast_to_devanagari(
            sanskrit_iast
        )
    )

    morphology_tags = (
        morphology_lookup[
            sanskrit_iast
        ]
    )

    morphology = " ".join(
        morphology_tags
    )

    return (
        "san_Deva "
        "eng_Latn "
        + sanskrit_devanagari
        + " MORPHOLOGY "
        + morphology
    )


# ============================================================
# 12. LOAD MODEL
# ============================================================

def load_lora_model(
    model_dir,
    name
):

    print("=" * 70)
    print(
        "LOADING:",
        name
    )
    print("=" * 70)

    print(
        "Adapter:",
        model_dir
    )

    print()

    # IMPORTANT:
    # Load tokenizer from BASE MODEL.
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    print(
        "Tokenizer loaded."
    )

    # Load original base model.
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

    # Load LoRA adapter.
    model = PeftModel.from_pretrained(
        base_model,
        str(model_dir)
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "LoRA adapter loaded."
    )

    print()

    return (
        tokenizer,
        model
    )


# ============================================================
# 13. GENERATION
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

        print(
            "Generating MORPHOLOGY predictions..."
        )

    else:

        print(
            "Generating BASELINE predictions..."
        )

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
# 14. REPETITION
# ============================================================

def has_repetition(text):

    words = re.findall(
        r"\S+",
        text.lower()
    )

    if len(words) < 4:

        return False

    counts = {}

    for word in words:

        counts[word] = (
            counts.get(word, 0) + 1
        )

        if counts[word] >= 3:

            return True

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
# 15. METRICS
# ============================================================

def calculate_metrics(
    predictions,
    references
):

    bleu = corpus_bleu(
        predictions,
        [references]
    ).score

    chrf = corpus_chrf(
        predictions,
        [references]
    ).score

    repetition = (
        repetition_rate(
            predictions
        )
    )

    return {

        "BLEU":
            bleu,

        "chrF":
            chrf,

        "repetition_rate":
            repetition,

        "sentences":
            len(predictions)
    }


# ============================================================
# 16. BASELINE TEST
# ============================================================

baseline_tokenizer, baseline_model = (
    load_lora_model(
        BASELINE_DIR,
        "Baseline LoRA 5504"
    )
)

(
    baseline_predictions,
    references
) = generate_translations(
    baseline_model,
    baseline_tokenizer,
    test_records,
    morphology=False
)

baseline_metrics = (
    calculate_metrics(
        baseline_predictions,
        references
    )
)


del baseline_model
del baseline_tokenizer

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 17. MORPHOLOGY TEST
# ============================================================

morph_tokenizer, morph_model = (
    load_lora_model(
        MORPH_DIR,
        "Morphology LoRA 5504"
    )
)

(
    morph_predictions,
    morph_references
) = generate_translations(
    morph_model,
    morph_tokenizer,
    test_records,
    morphology=True
)

morph_metrics = (
    calculate_metrics(
        morph_predictions,
        morph_references
    )
)


# ============================================================
# 18. DIFFERENCES
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


# ============================================================
# 19. FINAL RESULTS
# ============================================================

print()

print("=" * 70)
print("FINAL HELD-OUT TEST COMPARISON")
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
        "Held-out Test: Baseline vs Morphology LoRA 5504",

    "test_sentences":
        len(test_records),

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


with open(
    RESULT_DIR
    / "test_comparison_metrics.json",
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
    test_records
):

    results.append({

        "sanskrit":
            record["sanskrit"],

        "reference":
            references[i],

        "baseline":
            baseline_predictions[i],

        "morphology":
            morph_predictions[i],

        "morphology_tags":
            morphology_lookup[
                record["sanskrit"]
            ]
    })


with open(
    RESULT_DIR
    / "test_comparison_predictions.json",
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
# 22. SAMPLE RESULTS
# ============================================================

print("=" * 70)
print("SAMPLE TEST RESULTS")
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

    print()

    print(
        "SANSKRIT:"
    )

    print(
        item["sanskrit"]
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
        "MORPHOLOGY:"
    )

    print(
        item["morphology"]
    )

    print()

    print(
        "TAGS:"
    )

    print(
        " ".join(
            item["morphology_tags"]
        )
    )


# ============================================================
# 23. FINISHED
# ============================================================

print()

print("=" * 70)
print("HELD-OUT TEST EVALUATION COMPLETED")
print("=" * 70)

print()

print(
    "Metrics:"
)

print(
    RESULT_DIR
    / "test_comparison_metrics.json"
)

print()

print(
    "Predictions:"
)

print(
    RESULT_DIR
    / "test_comparison_predictions.json"
)

print("=" * 70)
